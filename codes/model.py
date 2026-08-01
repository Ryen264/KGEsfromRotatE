import logging

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataloader import TestDataset
from loss import compute_kge_loss, UniGammaController, is_learnable_kgau_gammas
from metrics.classification import classification_metrics
from metrics.ranking import ranks_from_score_matrix, rotate_ranking_metrics_from_ranks
from strategy import get_strategy


class KGEBase(object):
    def score(self, head, relation, tail, mode):
        raise NotImplementedError

    def query_encoder(self, head, relation, tail, mode):
        raise NotImplementedError

    def target_encoder(self, head, relation, tail, mode):
        raise NotImplementedError


class ComplEx(KGEBase):
    @staticmethod
    def _split_complex(x):
        return torch.chunk(x, 2, dim=-1)

    @staticmethod
    def _merge_complex(re_part, im_part):
        return torch.cat([re_part, im_part], dim=-1)

    @classmethod
    def _complex_mult(cls, a, b):
        re_a, im_a = cls._split_complex(a)
        re_b, im_b = cls._split_complex(b)
        re_part = re_a * re_b - im_a * im_b
        im_part = re_a * im_b + im_a * re_b
        return cls._merge_complex(re_part, im_part)

    @classmethod
    def _complex_conj_mult(cls, a, b):
        re_a, im_a = cls._split_complex(a)
        re_b, im_b = cls._split_complex(b)
        re_part = re_a * re_b + im_a * im_b
        im_part = re_a * im_b - im_a * re_b
        return cls._merge_complex(re_part, im_part)

    @classmethod
    def _hermitian_dot(cls, a, b):
        re_a, im_a = cls._split_complex(a)
        re_b, im_b = cls._split_complex(b)
        return (re_a * re_b + im_a * im_b).sum(dim=-1)

    def query_encoder(self, head, relation, tail, mode):
        if mode == 'head-batch':
            return self._complex_conj_mult(relation, tail)
        return self._complex_mult(head, relation)

    def target_encoder(self, head, relation, tail, mode):
        if mode == 'head-batch':
            return head
        return tail

    def score(self, head, relation, tail, mode):
        query = self.query_encoder(head, relation, tail, mode)
        target = self.target_encoder(head, relation, tail, mode)
        return self._hermitian_dot(query, target)

    def score_query_entities(self, query, entity_embedding):
        '''
        Score queries against all entity embeddings without expanding to [B, E, D].
        query: [B, D], entity_embedding: [E, D] -> scores [B, E]
        '''
        re_q, im_q = self._split_complex(query)
        re_e, im_e = self._split_complex(entity_embedding)
        return re_q @ re_e.transpose(0, 1) + im_q @ im_e.transpose(0, 1)


class RotatE(KGEBase):
    '''
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

    Entities are complex (double_entity_embedding); relations are real phases
    (not doubled). Score = margin_gamma - ||h ◦ r - t||_1 over complex moduli.
    '''

    def __init__(self, embedding_range, margin_gamma):
        self.embedding_range = embedding_range
        self.margin_gamma = margin_gamma
        self.pi = 3.14159265358979323846

    @staticmethod
    def _split_complex(x):
        return torch.chunk(x, 2, dim=-1)

    def _relation_rotation(self, relation):
        # Map relation embeddings to phases in [-pi, pi], then to unit complex.
        phase_relation = relation / (self.embedding_range.item() / self.pi)
        return torch.cos(phase_relation), torch.sin(phase_relation)

    def query_encoder(self, head, relation, tail, mode):
        re_relation, im_relation = self._relation_rotation(relation)
        if mode == 'head-batch':
            # Inverse rotation: r^{-1} ◦ t  (since |r|=1, r^{-1}=conj(r))
            re_tail, im_tail = self._split_complex(tail)
            re_query = re_relation * re_tail + im_relation * im_tail
            im_query = re_relation * im_tail - im_relation * re_tail
        else:
            # Forward rotation: h ◦ r
            re_head, im_head = self._split_complex(head)
            re_query = re_head * re_relation - im_head * im_relation
            im_query = re_head * im_relation + im_head * re_relation
        return torch.cat([re_query, im_query], dim=-1)

    def target_encoder(self, head, relation, tail, mode):
        if mode == 'head-batch':
            return head
        return tail

    @staticmethod
    def _complex_distance(re_diff, im_diff):
        # Per-dim complex modulus, then L1 over dims.
        # Use stack(...).norm (finite grad at 0) — NOT hypot, which yields NaN
        # grads when re=im=0. That happens once 1vsAll/KvsAll fits positives.
        return torch.stack([re_diff, im_diff], dim=0).norm(dim=0).sum(dim=-1)

    def score(self, head, relation, tail, mode):
        query = self.query_encoder(head, relation, tail, mode)
        target = self.target_encoder(head, relation, tail, mode)
        re_query, im_query = self._split_complex(query)
        re_target, im_target = self._split_complex(target)
        return self.margin_gamma.item() - self._complex_distance(
            re_query - re_target, im_query - im_target
        )

    def score_query_entities(self, query, entity_embedding):
        '''
        Score queries against all entity embeddings without expanding to [B, E, D]
        via forward()/index_select. Same contract as ComplEx.score_query_entities:
        query [B, D], entity_embedding [E, D] -> scores [B, E].

        ComplEx uses a bilinear matmul (autograd saves only [B,D]/[E,D]).
        RotatE distance is not bilinear, so a custom autograd saves the same
        embeddings and recomputes entity chunks in backward. Chunk budget is
        fixed (512MiB), not adaptive to free VRAM.
        '''
        re_q, im_q = self._split_complex(query)
        re_e, im_e = self._split_complex(entity_embedding)
        gamma = self.margin_gamma.detach().to(dtype=query.dtype, device=query.device)
        return _RotatEScoreQueryEntities.apply(re_q, im_q, re_e, im_e, gamma)


class _RotatEScoreQueryEntities(torch.autograd.Function):
    '''
    All-entity RotatE scores with ComplEx-like saved-tensor footprint:
    save query/entity embeddings only; recompute distance chunk-wise in backward.
    '''

    # Same fixed budget style as KGEModel._score_all_entities_chunked.
    _BYTES_BUDGET = 512 * 1024 * 1024

    @staticmethod
    def _chunk_size(batch_size, half_dim, nentity):
        # Peak temps ≈ re_diff + im_diff + stacked + norm ~ 5x [B, C, d] float32.
        per_entity = max(batch_size * half_dim * 4 * 5, 1)
        return max(64, min(nentity, _RotatEScoreQueryEntities._BYTES_BUDGET // per_entity))

    @staticmethod
    def _distance(re_diff, im_diff):
        return torch.stack([re_diff, im_diff], dim=0).norm(dim=0).sum(dim=-1)

    @staticmethod
    def forward(ctx, re_q, im_q, re_e, im_e, gamma):
        ctx.save_for_backward(re_q, im_q, re_e, im_e)
        ctx.gamma = float(gamma)
        batch_size, half_dim = re_q.shape
        nentity = re_e.size(0)
        chunk = _RotatEScoreQueryEntities._chunk_size(batch_size, half_dim, nentity)
        ctx.chunk = chunk

        scores = re_q.new_empty(batch_size, nentity)
        for start in range(0, nentity, chunk):
            end = min(start + chunk, nentity)
            re_diff = re_q.unsqueeze(1) - re_e[start:end].unsqueeze(0)
            im_diff = im_q.unsqueeze(1) - im_e[start:end].unsqueeze(0)
            scores[:, start:end] = ctx.gamma - _RotatEScoreQueryEntities._distance(
                re_diff, im_diff
            )
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        re_q, im_q, re_e, im_e = ctx.saved_tensors
        chunk = ctx.chunk
        nentity = re_e.size(0)

        grad_re_q = torch.zeros_like(re_q)
        grad_im_q = torch.zeros_like(im_q)
        grad_re_e = torch.zeros_like(re_e)
        grad_im_e = torch.zeros_like(im_e)

        for start in range(0, nentity, chunk):
            end = min(start + chunk, nentity)
            re_diff = re_q.unsqueeze(1) - re_e[start:end].unsqueeze(0)
            im_diff = im_q.unsqueeze(1) - im_e[start:end].unsqueeze(0)
            # Finite subgradient at 0: clamp only the divisor (re_diff=0 => grad=0).
            moduli = torch.stack([re_diff, im_diff], dim=0).norm(dim=0).clamp_min(1e-12)
            grad = grad_scores[:, start:end].unsqueeze(-1)
            # score = gamma - sum moduli; d(score)/d(re_diff) = -re_diff/moduli
            g_re = -grad * (re_diff / moduli)
            g_im = -grad * (im_diff / moduli)
            grad_re_q += g_re.sum(dim=1)
            grad_im_q += g_im.sum(dim=1)
            grad_re_e[start:end] -= g_re.sum(dim=0)
            grad_im_e[start:end] -= g_im.sum(dim=0)

        return grad_re_q, grad_im_q, grad_re_e, grad_im_e, None


KGE_SCORERS = {
    'ComplEx': ComplEx,
    'RotatE': RotatE,
}


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, dim, margin_gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.dim = dim
        self.epsilon = 2.0
        
        self.margin_gamma = nn.Parameter(
            torch.Tensor([margin_gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin_gamma.item() + self.epsilon) / dim]), 
            requires_grad=False
        )
        
        self.entity_dim = dim*2 if double_entity_embedding else dim
        self.relation_dim = dim*2 if double_relation_embedding else dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name not in KGE_SCORERS:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE':
            self.kge_scorer = RotatE(
                embedding_range=self.embedding_range,
                margin_gamma=self.margin_gamma,
            )
        else:
            self.kge_scorer = KGE_SCORERS[model_name]()
        
    def query_encoder(self, head, relation, tail=None, mode='tail-batch'):
        return self.kge_scorer.query_encoder(head, relation, tail, mode)

    def target_encoder(self, tail, head=None, relation=None, mode='tail-batch'):
        return self.kge_scorer.target_encoder(head, relation, tail, mode)
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.reshape(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.reshape(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)

        if mode == 'single':
            score_mode = 'tail-batch'
        else:
            score_mode = mode

        return self.kge_scorer.score(head, relation, tail, score_mode)

    def score_all_entities(self, positive_sample, mode='tail-batch'):
        '''
        Score each positive query against all entities.

        Avoids materializing [B, nentity, dim] candidate embeddings (OOM on
        AllNeg / 1vsAll / KvsAll). Uses a query-[B,D] x entity-[E,D] path when
        the scorer supports it; otherwise falls back to chunked forward().
        '''
        head = self.entity_embedding[positive_sample[:, 0]].unsqueeze(1)
        relation = self.relation_embedding[positive_sample[:, 1]].unsqueeze(1)
        tail = self.entity_embedding[positive_sample[:, 2]].unsqueeze(1)

        if mode == 'head-batch':
            query = self.kge_scorer.query_encoder(head, relation, tail, mode)
        else:
            query = self.kge_scorer.query_encoder(head, relation, tail, 'tail-batch')

        query = query.squeeze(1)  # [B, D]
        scorer = self.kge_scorer
        if hasattr(scorer, 'score_query_entities'):
            return scorer.score_query_entities(query, self.entity_embedding)
        return self._score_all_entities_chunked(positive_sample, mode)

    def _score_all_entities_chunked(self, positive_sample, mode, chunk_size=None):
        '''Fallback: score entity id chunks through forward().'''
        nentity = self.nentity
        batch_size = positive_sample.size(0)
        device = positive_sample.device
        if chunk_size is None:
            entity_dim = self.entity_embedding.size(1)
            # Keep ~512MiB peak for [B, C, D] float32 (+ mul intermediates).
            bytes_budget = 512 * 1024 * 1024
            per_entity = max(batch_size * entity_dim * 4 * 2, 1)
            chunk_size = max(256, min(nentity, bytes_budget // per_entity))

        chunks = []
        for start in range(0, nentity, chunk_size):
            end = min(start + chunk_size, nentity)
            candidates = torch.arange(start, end, device=device, dtype=torch.long)
            negative_sample = candidates.unsqueeze(0).expand(batch_size, -1)
            chunks.append(self((positive_sample, negative_sample), mode=mode))
        return torch.cat(chunks, dim=1)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        strategy = get_strategy(args)
        batch = next(train_iterator)
        (
            positive_score, negative_score, subsampling_weight,
            positive_sample, mode, negative_weights, scores, labels,
        ) = strategy.prepare_train_batch(batch, model)

        loss, log = compute_kge_loss(
            positive_score, negative_score, subsampling_weight, model, args,
            positive_sample=positive_sample, mode=mode,
            negative_weights=negative_weights,
            scores=scores, labels=labels,
        )

        loss.backward()

        optimizer.step()

        if is_learnable_kgau_gammas(args):
            UniGammaController(args).clamp_log_gammas(model)

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            cls_metrics = classification_metrics(
                y_true,
                (y_score > 0).astype(int),
                y_prob=y_score,
            )
            metrics = {'auc_pr': cls_metrics['pr_auc']}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=4, 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=4, 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            ranks = []
            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        ranks.extend(ranks_from_score_matrix(score, positive_arg))

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1
            metrics = rotate_ranking_metrics_from_ranks(ranks)
        return metrics