from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from dataloader import BidirectionalOneShotIterator, TrainDataset


STRATEGY_CHOICES = ('uniform', 'bernoulli', 'selfadv', '1vsall', 'kvsall', 'kgau')


def suggested_max_workers():
    '''PyTorch DataLoader soft cap: CPUs available to this process.'''
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def resolve_num_workers(args, num_loaders=1):
    '''
    Clamp DataLoader workers to the system-suggested max.

    When several loaders are alive at once (e.g. head+tail), split the budget
    so total workers stay within the suggested limit.
    '''
    requested = int(getattr(args, 'cpu_num', 4) or 0)
    suggested = suggested_max_workers()
    budget = max(0, suggested // max(int(num_loaders), 1))
    if requested <= 0:
        return budget
    return min(requested, budget if budget > 0 else requested)


class KGEStrategy(object):
    '''
    Abstract training strategy.

    NegSamp strategies sample a fixed number of negatives per positive.
    AllNeg strategies score all entities as candidates for each query.
    KGAU optimizes alignment-uniformity on positives only (no negatives).
    '''

    name = None
    family = None  # 'negsamp' | 'allneg' | 'kgau'

    def __init__(self, args):
        self.args = args

    @property
    def uses_negative_samples(self):
        return self.family == 'negsamp'

    @property
    def uses_all_entities(self):
        return self.family == 'allneg'

    def weight_negatives(self, negative_score):
        '''
        Per-negative weights for NegSamp losses. Shape: [batch, n_neg].
        Uniform / Bernoulli use equal weights (mean via sum of 1/n).
        '''
        raise NotImplementedError

    def build_train_iterator(self, train_triples, nentity, nrelation):
        raise NotImplementedError

    def prepare_train_batch(self, batch, model):
        '''
        Convert a dataloader batch into
        (positive_score, negative_score, subsampling_weight,
         positive_sample, mode, negative_weights, scores, labels).

        NegSamp: scores/labels are None; use positive/negative scores.
        1vsAll / KvsAll: scores is [B, E]; labels are one-hot / multi-hot
        (also keep pos/neg split for pairwise losses).
        KGAU: scores unused.
        '''
        raise NotImplementedError


class KGAUStrategy(KGEStrategy):
    '''
    KGAU training strategy: positives only.

    No negative sampling and no all-entity scoring. The KGAU-family loss uses
    query/target (and optional entity) embeddings from positive triples.
    '''

    name = 'kgau'
    family = 'kgau'

    def weight_negatives(self, negative_score):
        raise RuntimeError('KGAU strategy does not use negative samples')

    def build_train_iterator(self, train_triples, nentity, nrelation):
        batch_size = self.args.batch_size
        num_workers = resolve_num_workers(self.args, num_loaders=2)

        train_dataloader_head = DataLoader(
            PositiveOnlyTrainDataset(train_triples, nentity, nrelation, 'head-batch'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=PositiveOnlyTrainDataset.collate_fn,
        )
        train_dataloader_tail = DataLoader(
            PositiveOnlyTrainDataset(train_triples, nentity, nrelation, 'tail-batch'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=PositiveOnlyTrainDataset.collate_fn,
        )
        return BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    def prepare_train_batch(self, batch, model):
        positive_sample, subsampling_weight, mode = batch
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        return None, None, subsampling_weight, positive_sample, mode, None, None, None


class NegSampStrategy(KGEStrategy):
    '''
    NegSamp family (uniform / bernoulli / selfadv).

    Memory: scoring all N negatives at once materializes [B, N, D] candidate
    embeddings. When N is large, score negatives in chunks and use gradient
    checkpointing so only one [B, C, D] block is live at a time (same ~512MiB
    budget heuristic as AllNeg's chunked fallback).
    '''

    family = 'negsamp'

    def weight_negatives(self, negative_score):
        # Equal weights; sum_j w_j * L_j with w_j = 1/n equals mean.
        n_neg = negative_score.size(1)
        return negative_score.new_full(negative_score.shape, 1.0 / n_neg)

    def resolve_negative_chunk_size(self, batch_size, n_neg, entity_dim):
        '''Return configured chunk width C (default 256). C<=0 disables chunking (use all N).'''
        if n_neg <= 0:
            return 0
        chunk = int(getattr(self.args, 'negative_chunk_size', 256) or 0)
        if chunk <= 0:
            return n_neg
        return min(chunk, n_neg)

    def score_negatives(self, model, positive_sample, negative_sample, mode):
        '''Score [B, N] negatives, chunking (+ checkpoint) when N exceeds the budget.'''
        n_neg = negative_sample.size(1)
        if n_neg == 0:
            return torch.zeros(
                positive_sample.size(0), 0,
                device=positive_sample.device,
                dtype=model.entity_embedding.dtype,
            )

        chunk_size = self.resolve_negative_chunk_size(
            positive_sample.size(0),
            n_neg,
            model.entity_embedding.size(1),
        )
        if chunk_size >= n_neg:
            return model((positive_sample, negative_sample), mode=mode)

        score_chunks = []
        for start in range(0, n_neg, chunk_size):
            end = min(start + chunk_size, n_neg)
            neg_chunk = negative_sample[:, start:end]

            def _forward_chunk(pos, neg, _mode=mode):
                return model((pos, neg), mode=_mode)

            if model.training:
                # Recompute chunk activations on backward → peak ≈ one chunk.
                chunk_score = checkpoint(
                    _forward_chunk,
                    positive_sample,
                    neg_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_score = _forward_chunk(positive_sample, neg_chunk)
            score_chunks.append(chunk_score)
        return torch.cat(score_chunks, dim=1)

    def build_train_iterator(self, train_triples, nentity, nrelation):
        neg_size = self.args.negative_sample_size
        batch_size = self.args.batch_size
        num_workers = resolve_num_workers(self.args, num_loaders=2)

        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, neg_size, 'head-batch'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=TrainDataset.collate_fn,
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, neg_size, 'tail-batch'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=TrainDataset.collate_fn,
        )
        return BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    def prepare_train_batch(self, batch, model):
        positive_sample, negative_sample, subsampling_weight, mode = batch
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = self.score_negatives(
            model, positive_sample, negative_sample, mode,
        )
        positive_score = model(positive_sample)
        negative_weights = self.weight_negatives(negative_score)
        return (
            positive_score, negative_score, subsampling_weight,
            positive_sample, mode, negative_weights, None, None,
        )


class UniformNS(NegSampStrategy):
    '''Uniform negative sampling: corrupt head/tail with uniformly drawn entities.'''

    name = 'uniform'


class SelfAdvNS(NegSampStrategy):
    '''
    Self-adversarial negative sampling (RotatE): candidates are still uniform,
    but negatives are weighted by detached softmax(alpha * score).
    '''

    name = 'selfadv'

    def weight_negatives(self, negative_score):
        temperature = getattr(self.args, 'adversarial_temperature', 1.0)
        return F.softmax(negative_score * temperature, dim=1).detach()


class BernoulliNS(NegSampStrategy):
    '''
    Bernoulli negative sampling (TransE): for relation r, corrupt the head with
    probability tph/(tph+hpt) and the tail with probability hpt/(tph+hpt).

    Uses separate head/tail loaders (same batch shape as UniformNS) so each
    training step scores a full batch in one mode — fair peak-memory comparison.
    '''

    name = 'bernoulli'

    def build_train_iterator(self, train_triples, nentity, nrelation):
        neg_size = self.args.negative_sample_size
        batch_size = self.args.batch_size
        num_workers = resolve_num_workers(self.args, num_loaders=2)

        train_dataloader_head = DataLoader(
            BernoulliTrainDataset(
                train_triples, nentity, nrelation, neg_size, 'head-batch'
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=BernoulliTrainDataset.collate_fn,
        )
        train_dataloader_tail = DataLoader(
            BernoulliTrainDataset(
                train_triples, nentity, nrelation, neg_size, 'tail-batch'
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=BernoulliTrainDataset.collate_fn,
        )
        return BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)


class AllNegStrategy(KGEStrategy):
    family = 'allneg'

    def weight_negatives(self, negative_score):
        n_neg = negative_score.size(1)
        return negative_score.new_full(negative_score.shape, 1.0 / n_neg)

    def build_train_iterator(self, train_triples, nentity, nrelation):
        batch_size = self.args.batch_size
        num_workers = resolve_num_workers(self.args, num_loaders=2)
        dataset_cls = self._dataset_class()

        train_dataloader_head = DataLoader(
            dataset_cls(train_triples, nentity, nrelation, 'head-batch'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=dataset_cls.collate_fn,
        )
        train_dataloader_tail = DataLoader(
            dataset_cls(train_triples, nentity, nrelation, 'tail-batch'),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=dataset_cls.collate_fn,
        )
        return BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    def _dataset_class(self):
        raise NotImplementedError

    def _candidate_entities(self, nentity, device):
        return torch.arange(nentity, device=device, dtype=torch.long)

    def _score_all_entities(self, model, positive_sample, mode):
        '''Full [B, E] scores without building [B, E, dim] embeddings.'''
        if hasattr(model, 'score_all_entities'):
            return model.score_all_entities(positive_sample, mode=mode)
        # Legacy fallback (may OOM on large graphs / dims).
        nentity = model.nentity
        candidates = self._candidate_entities(nentity, positive_sample.device)
        negative_sample = candidates.unsqueeze(0).expand(positive_sample.size(0), -1)
        return model((positive_sample, negative_sample), mode=mode)

    def prepare_train_batch(self, batch, model):
        positive_sample, label_matrix, subsampling_weight, mode = batch
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            label_matrix = label_matrix.cuda()
            subsampling_weight = subsampling_weight.cuda()

        all_scores = self._score_all_entities(model, positive_sample, mode)
        # Pairwise losses use pos/neg; BCE/CE use full scores + labels
        # (one-hot for 1vsAll, multi-hot for KvsAll).
        labels = self._labels_for_loss(all_scores, positive_sample, label_matrix, mode)
        loss_name = getattr(self.args, 'loss', '')
        # BCE/CE already consume [B, E] scores+labels — skip materializing
        # [B, E-1] negatives (and KvsAll's Python pad loop).
        if loss_name in ('bce', 'ce'):
            return (
                None, None, subsampling_weight,
                positive_sample, mode, None, all_scores, labels,
            )

        # Pairwise path: use expanded labels (not compact dataloader indices).
        positive_score, negative_score = self._split_scores(
            all_scores, positive_sample, labels, mode
        )
        negative_weights = self.weight_negatives(negative_score)
        return (
            positive_score, negative_score, subsampling_weight,
            positive_sample, mode, negative_weights, all_scores, labels,
        )

    def _labels_for_loss(self, all_scores, positive_sample, label_matrix, mode):
        '''Default: multi-hot / one-hot matrix from the dataset.'''
        return label_matrix

    def _split_scores(self, all_scores, positive_sample, label_matrix, mode):
        raise NotImplementedError


class OneVsAll(AllNegStrategy):
    '''
    1vsAll: for a positive (h, r, t), all other entities are negatives for that
    instance (single positive label among |E| candidates).
    '''

    name = '1vsall'

    def _dataset_class(self):
        return OneVsAllTrainDataset

    @staticmethod
    def _positive_indices(positive_sample, label_matrix, mode):
        # Dataset may pass dense one-hot [B, E] or compact target indices [B].
        if label_matrix is not None and label_matrix.dim() == 1:
            return label_matrix.long()
        if mode == 'tail-batch':
            return positive_sample[:, 2].long()
        return positive_sample[:, 0].long()

    def _labels_for_loss(self, all_scores, positive_sample, label_matrix, mode):
        loss_name = getattr(self.args, 'loss', '')
        pos_idx = self._positive_indices(positive_sample, label_matrix, mode)
        # CE single-label prefers class indices; BCE uses one-hot matrix.
        if loss_name == 'ce':
            return pos_idx
        if label_matrix is not None and label_matrix.dim() == 2:
            return label_matrix
        labels = all_scores.new_zeros(all_scores.shape)
        labels.scatter_(1, pos_idx.view(-1, 1), 1.0)
        return labels

    def _split_scores(self, all_scores, positive_sample, label_matrix, mode):
        batch_size = all_scores.size(0)
        pos_idx = self._positive_indices(positive_sample, label_matrix, mode)

        positive_score = all_scores.gather(1, pos_idx.view(-1, 1))
        mask = torch.ones_like(all_scores, dtype=torch.bool)
        mask.scatter_(1, pos_idx.view(-1, 1), False)
        negative_score = all_scores[mask].view(batch_size, -1)
        return positive_score, negative_score


class KvsAll(AllNegStrategy):
    '''
    KvsAll: for (h, r, *) [or (*, r, t)], all known training completions are
    positives (label 1); remaining entities are negatives (label 0).

    BCE/CE use full [B, E] scores with multi-hot labels. Pairwise losses still
    receive a pos/neg split (instance positive vs label-0 entities).
    '''

    name = 'kvsall'

    def _dataset_class(self):
        return KvsAllTrainDataset

    def _labels_for_loss(self, all_scores, positive_sample, label_matrix, mode):
        # Dense multi-hot [B, E], or compact padded indices [B, K] with -1 pad.
        if label_matrix.dim() == 2 and label_matrix.dtype in (
            torch.float16, torch.float32, torch.float64,
        ):
            return label_matrix
        labels = all_scores.new_zeros(all_scores.shape)
        valid = label_matrix >= 0
        if valid.any():
            rows = (
                torch.arange(label_matrix.size(0), device=label_matrix.device)
                .unsqueeze(1)
                .expand_as(label_matrix)
            )
            labels[rows[valid], label_matrix[valid].long()] = 1.0
        return labels

    def _split_scores(self, all_scores, positive_sample, label_matrix, mode):
        if mode == 'tail-batch':
            pos_idx = positive_sample[:, 2]
        else:
            pos_idx = positive_sample[:, 0]

        positive_score = all_scores.gather(1, pos_idx.view(-1, 1))
        neg_mask = label_matrix < 0.5
        if not neg_mask.any():
            neg_mask = torch.ones_like(all_scores, dtype=torch.bool)
            neg_mask.scatter_(1, pos_idx.view(-1, 1), False)

        # Vectorized pad: gather with a dense index matrix instead of a
        # Python row loop (only used for pairwise losses; BCE/CE skip this).
        neg_counts = neg_mask.sum(dim=1)
        max_neg = int(neg_counts.max().item())
        # Descending sort puts True(1) first; take the leading max_neg cols.
        neg_idx = neg_mask.to(torch.int8).argsort(dim=1, descending=True)[:, :max_neg]
        negative_score = all_scores.gather(1, neg_idx)
        # Rows with fewer than max_neg true negatives: fill padded slots with
        # that row's mean over true negatives (matches previous semantics).
        pad_mask = (
            torch.arange(max_neg, device=all_scores.device).unsqueeze(0)
            >= neg_counts.unsqueeze(1)
        )
        if pad_mask.any():
            safe_counts = neg_counts.clamp_min(1).to(all_scores.dtype)
            row_sum = (all_scores * neg_mask.to(all_scores.dtype)).sum(dim=1)
            fill = (row_sum / safe_counts).unsqueeze(1).expand_as(negative_score)
            negative_score = torch.where(pad_mask, fill, negative_score)
        return positive_score, negative_score

    def prepare_multilabel_batch(self, batch, model):
        '''Full [B, E] scores and multi-hot labels for multi-label BCE/CE.'''
        positive_sample, label_matrix, subsampling_weight, mode = batch
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            label_matrix = label_matrix.cuda()
            subsampling_weight = subsampling_weight.cuda()

        all_scores = self._score_all_entities(model, positive_sample, mode)
        labels = self._labels_for_loss(all_scores, positive_sample, label_matrix, mode)
        return all_scores, labels, subsampling_weight, positive_sample, mode


# ---------------------------------------------------------------------------
# Datasets for Bernoulli / AllNeg / KGAU
# ---------------------------------------------------------------------------

class PositiveOnlyTrainDataset(Dataset):
    '''Positive triples only (no negative entity ids). Used by KGAUStrategy.'''

    def __init__(self, triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.count = TrainDataset.count_frequency(triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        return (
            torch.LongTensor(positive_sample),
            subsampling_weight,
            self.mode,
        )

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        subsample_weight = torch.cat([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, subsample_weight, mode


class BernoulliTrainDataset(Dataset):
    '''
    Fixed-mode Bernoulli dataset for fair head/tail loaders.

    Each item corrupts only ``mode`` (head-batch or tail-batch). Triples are
    accepted with TransE Bernoulli probability for that side (rejection
    sampling), so batches stay full-size and single-mode like UniformNS.
    '''

    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        if mode not in ('head-batch', 'tail-batch'):
            raise ValueError('BernoulliTrainDataset mode %s not supported' % mode)
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = TrainDataset.count_frequency(triples)
        self.true_head, self.true_tail = TrainDataset.get_true_head_and_tail(triples)
        self.tph, self.hpt = self._relation_tph_hpt(triples, nrelation)

    @staticmethod
    def _relation_tph_hpt(triples, nrelation):
        '''Average tails-per-head (tph) and heads-per-tail (hpt) per relation.'''
        heads_for_r = [set() for _ in range(nrelation)]
        tails_for_r = [set() for _ in range(nrelation)]
        n_triples_r = np.zeros(nrelation, dtype=np.float64)
        for h, r, t in triples:
            heads_for_r[r].add(h)
            tails_for_r[r].add(t)
            n_triples_r[r] += 1

        tph = np.ones(nrelation, dtype=np.float64)
        hpt = np.ones(nrelation, dtype=np.float64)
        for r in range(nrelation):
            n_heads = len(heads_for_r[r])
            n_tails = len(tails_for_r[r])
            if n_heads > 0:
                tph[r] = n_triples_r[r] / float(n_heads)
            if n_tails > 0:
                hpt[r] = n_triples_r[r] / float(n_tails)
        return tph, hpt

    def __len__(self):
        return self.len

    def _sample_negatives(self, head, relation, tail):
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.isin(negative_sample, self.true_head[(relation, tail)], invert=True)
            else:
                mask = np.isin(negative_sample, self.true_tail[(head, relation)], invert=True)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        return np.concatenate(negative_sample_list)[: self.negative_sample_size]

    def _bernoulli_accepts_mode(self, relation):
        p_corrupt_head = self.tph[relation] / (self.tph[relation] + self.hpt[relation])
        corrupt_head = np.random.rand() < p_corrupt_head
        if self.mode == 'head-batch':
            return corrupt_head
        return not corrupt_head

    def __getitem__(self, idx):
        positive_sample = None
        for offset in range(self.len):
            i = (idx + offset) % self.len
            candidate = self.triples[i]
            head, relation, tail = candidate
            if self._bernoulli_accepts_mode(relation):
                positive_sample = candidate
                break
        if positive_sample is None:
            positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample = self._sample_negatives(head, relation, tail)

        return (
            torch.LongTensor(positive_sample),
            torch.LongTensor(negative_sample),
            subsampling_weight,
            self.mode,
        )

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode


class _AllNegTrainDatasetBase(Dataset):
    def __init__(self, triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.count = TrainDataset.count_frequency(triples)
        self.true_head, self.true_tail = TrainDataset.get_true_head_and_tail(triples)

    def __len__(self):
        return self.len

    def _subsampling_weight(self, head, relation, tail):
        weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        return torch.sqrt(1 / torch.Tensor([weight]))

    def _label_vector(self, head, relation, tail):
        raise NotImplementedError

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        labels = self._label_vector(head, relation, tail)
        return (
            torch.LongTensor(positive_sample),
            labels,
            self._subsampling_weight(head, relation, tail),
            self.mode,
        )

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        label_matrix = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, label_matrix, subsample_weight, mode


class OneVsAllTrainDataset(_AllNegTrainDatasetBase):
    def _label_vector(self, head, relation, tail):
        # Compact target index [ ]; one-hot is built on-device in OneVsAll.
        if self.mode == 'tail-batch':
            return torch.tensor(tail, dtype=torch.long)
        return torch.tensor(head, dtype=torch.long)


class KvsAllTrainDataset(_AllNegTrainDatasetBase):
    def _label_vector(self, head, relation, tail):
        # Compact positive entity ids; multi-hot is built on-device in KvsAll.
        if self.mode == 'tail-batch':
            ents = self.true_tail[(head, relation)]
        else:
            ents = self.true_head[(relation, tail)]
        return torch.tensor([int(e) for e in ents], dtype=torch.long)

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        pos_lists = [_[1] for _ in data]
        max_k = max((int(x.numel()) for x in pos_lists), default=0)
        # Pad with -1 so KvsAll._labels_for_loss can scatter valid ids only.
        label_idx = torch.full((len(data), max(max_k, 1)), -1, dtype=torch.long)
        for i, ids in enumerate(pos_lists):
            if ids.numel() > 0:
                label_idx[i, : ids.numel()] = ids
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, label_idx, subsample_weight, mode


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    'uniform': UniformNS,
    'bernoulli': BernoulliNS,
    'selfadv': SelfAdvNS,
    '1vsall': OneVsAll,
    'kvsall': KvsAll,
    'kgau': KGAUStrategy,
}


def normalize_strategy_name(name):
    if name is None:
        return 'uniform'
    key = str(name).strip().lower()
    aliases = {
        'uniformns': 'uniform',
        'bernoullins': 'bernoulli',
        'selfadvns': 'selfadv',
        'self-adv': 'selfadv',
        'self_adv': 'selfadv',
        '1vsall': '1vsall',
        'kvsall': 'kvsall',
        'kgmau': 'kgau',
        'kgmamu': 'kgau',
    }
    key = aliases.get(key, key)
    if key not in STRATEGY_REGISTRY:
        raise ValueError(
            'Unknown strategy: {}. Choose from {}'.format(name, ', '.join(STRATEGY_CHOICES))
        )
    return key


def resolve_strategy_name(args):
    '''
    Prefer explicit args.strategy.
    Defaults: kgau-family loss -> kgau; sans / -adv -> selfadv; else uniform.
    '''
    explicit = getattr(args, 'strategy', None)
    if explicit is not None and str(explicit).strip() != '':
        return normalize_strategy_name(explicit)

    loss_name = getattr(args, 'loss', '')
    if loss_name in ('kgau', 'kgmau', 'kgmamu'):
        return 'kgau'
    if getattr(args, 'negative_adversarial_sampling', False):
        return 'selfadv'
    if loss_name == 'sans':
        return 'selfadv'
    return 'uniform'


def get_strategy(args):
    name = resolve_strategy_name(args)
    args.strategy = name
    # Keep legacy flag in sync for SelfAdversarialNegativeSamplingLoss.
    if name == 'selfadv':
        args.negative_adversarial_sampling = True
    elif not getattr(args, 'negative_adversarial_sampling', False):
        args.negative_adversarial_sampling = False
    return STRATEGY_REGISTRY[name](args)


def is_negsamp_strategy(args):
    return resolve_strategy_name(args) in ('uniform', 'bernoulli', 'selfadv')


def is_allneg_strategy(args):
    return resolve_strategy_name(args) in ('1vsall', 'kvsall')


def is_kgau_strategy(args):
    return resolve_strategy_name(args) == 'kgau'
