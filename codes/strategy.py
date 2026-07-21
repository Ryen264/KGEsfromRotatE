from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from dataloader import BidirectionalOneShotIterator, TrainDataset


STRATEGY_CHOICES = ('uniform', 'bernoulli', 'selfadv', '1vsall', 'kvsall', 'kgau')


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
         positive_sample, mode, negative_weights) for compute_kge_loss.

        negative_weights come from weight_negatives() (once); KGAU returns None.
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
        num_workers = getattr(self.args, 'cpu_num', 4)

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
        # Scores / negative weights unused by KGAU-family losses.
        return None, None, subsampling_weight, positive_sample, mode, None


class NegSampStrategy(KGEStrategy):
    family = 'negsamp'

    def weight_negatives(self, negative_score):
        # Equal weights; sum_j w_j * L_j with w_j = 1/n equals mean.
        n_neg = negative_score.size(1)
        return negative_score.new_full(negative_score.shape, 1.0 / n_neg)

    def build_train_iterator(self, train_triples, nentity, nrelation):
        neg_size = self.args.negative_sample_size
        batch_size = self.args.batch_size
        num_workers = getattr(self.args, 'cpu_num', 4)

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

        negative_score = model((positive_sample, negative_sample), mode=mode)
        positive_score = model(positive_sample)
        negative_weights = self.weight_negatives(negative_score)
        return (
            positive_score, negative_score, subsampling_weight,
            positive_sample, mode, negative_weights,
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
    probability tph/(tph+hpt) and the tail with probability hpt/(tph+hpt),
    where tph / hpt are average tails-per-head and heads-per-tail for r.
    '''

    name = 'bernoulli'

    def build_train_iterator(self, train_triples, nentity, nrelation):
        neg_size = self.args.negative_sample_size
        batch_size = self.args.batch_size
        num_workers = getattr(self.args, 'cpu_num', 4)

        dataset = BernoulliTrainDataset(
            train_triples, nentity, nrelation, neg_size
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=BernoulliTrainDataset.collate_fn,
        )
        return BernoulliOneShotIterator(loader)

    def prepare_train_batch(self, batch, model):
        '''
        A Bernoulli batch may mix head-batch and tail-batch rows. Score each
        mode subset separately, then concatenate scores in original order.
        '''
        positive_sample, negative_sample, subsampling_weight, modes = batch
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        batch_size = positive_sample.size(0)
        device = positive_sample.device
        positive_score = model(positive_sample)
        negative_score = positive_score.new_zeros(batch_size, negative_sample.size(1))

        modes_list = list(modes)
        head_idx = [i for i, m in enumerate(modes_list) if m == 'head-batch']
        tail_idx = [i for i, m in enumerate(modes_list) if m == 'tail-batch']

        if head_idx:
            idx = torch.LongTensor(head_idx).to(device)
            neg = model(
                (positive_sample.index_select(0, idx), negative_sample.index_select(0, idx)),
                mode='head-batch',
            )
            negative_score[idx] = neg
        if tail_idx:
            idx = torch.LongTensor(tail_idx).to(device)
            neg = model(
                (positive_sample.index_select(0, idx), negative_sample.index_select(0, idx)),
                mode='tail-batch',
            )
            negative_score[idx] = neg

        # Representative mode for KGAU query/target encoders (prefer majority).
        mode = 'head-batch' if len(head_idx) >= len(tail_idx) else 'tail-batch'
        negative_weights = self.weight_negatives(negative_score)
        return (
            positive_score, negative_score, subsampling_weight,
            positive_sample, mode, negative_weights,
        )


class AllNegStrategy(KGEStrategy):
    family = 'allneg'

    def weight_negatives(self, negative_score):
        n_neg = negative_score.size(1)
        return negative_score.new_full(negative_score.shape, 1.0 / n_neg)

    def build_train_iterator(self, train_triples, nentity, nrelation):
        batch_size = self.args.batch_size
        num_workers = getattr(self.args, 'cpu_num', 4)
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

    def prepare_train_batch(self, batch, model):
        positive_sample, label_matrix, subsampling_weight, mode = batch
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            label_matrix = label_matrix.cuda()
            subsampling_weight = subsampling_weight.cuda()

        nentity = label_matrix.size(1)
        candidates = self._candidate_entities(nentity, positive_sample.device)
        # [B, E] entity ids for full ranking
        negative_sample = candidates.unsqueeze(0).expand(positive_sample.size(0), -1)
        all_scores = model((positive_sample, negative_sample), mode=mode)
        positive_score, negative_score = self._split_scores(
            all_scores, positive_sample, label_matrix, mode
        )
        negative_weights = self.weight_negatives(negative_score)
        return (
            positive_score, negative_score, subsampling_weight,
            positive_sample, mode, negative_weights,
        )

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

    def _split_scores(self, all_scores, positive_sample, label_matrix, mode):
        batch_size = all_scores.size(0)
        if mode == 'tail-batch':
            pos_idx = positive_sample[:, 2]
        else:
            pos_idx = positive_sample[:, 0]

        positive_score = all_scores.gather(1, pos_idx.view(-1, 1))
        # Mask out the positive column to form negatives
        mask = torch.ones_like(all_scores, dtype=torch.bool)
        mask.scatter_(1, pos_idx.view(-1, 1), False)
        negative_score = all_scores[mask].view(batch_size, -1)
        return positive_score, negative_score


class KvsAll(AllNegStrategy):
    '''
    KvsAll: for (h, r, *) [or (*, r, t)], all known training completions are
    positives (label 1); remaining entities are negatives (label 0).

    Scores are reduced to one positive column (true entity of the instance) and
    negatives = all entities with label 0, so existing pointwise/pairwise losses
    still apply. Multi-label BCE over the full matrix is available via
    prepare_multilabel_batch().
    '''

    name = 'kvsall'

    def _dataset_class(self):
        return KvsAllTrainDataset

    def _split_scores(self, all_scores, positive_sample, label_matrix, mode):
        batch_size = all_scores.size(0)
        if mode == 'tail-batch':
            pos_idx = positive_sample[:, 2]
        else:
            pos_idx = positive_sample[:, 0]

        positive_score = all_scores.gather(1, pos_idx.view(-1, 1))
        # Negatives: entities that are not known true completions for this query
        neg_mask = label_matrix < 0.5
        # Ensure at least one negative slot by falling back to "not the instance positive"
        if not neg_mask.any():
            neg_mask = torch.ones_like(all_scores, dtype=torch.bool)
            neg_mask.scatter_(1, pos_idx.view(-1, 1), False)

        # Variable number of negatives per row — pad by sampling from masked scores
        # via a dense [B, E] mask fill with a large negative for ignored positions
        # then take all label-0 columns. Use gather of flattened indices per row.
        nentity = all_scores.size(1)
        neg_counts = neg_mask.sum(dim=1)
        max_neg = int(neg_counts.max().item())
        # Build padded negative scores
        negative_score = all_scores.new_full((batch_size, max_neg), 0.0)
        for i in range(batch_size):
            row = all_scores[i][neg_mask[i]]
            negative_score[i, : row.numel()] = row
            if row.numel() < max_neg:
                # repeat last / mean fill unused slots so .mean(dim=1) is stable
                fill = row.mean() if row.numel() > 0 else all_scores.new_zeros(())
                negative_score[i, row.numel():] = fill
        return positive_score, negative_score

    def prepare_multilabel_batch(self, batch, model):
        '''Full [B, E] scores and multi-hot labels for multi-label BCE.'''
        positive_sample, label_matrix, subsampling_weight, mode = batch
        if self.args.cuda:
            positive_sample = positive_sample.cuda()
            label_matrix = label_matrix.cuda()
            subsampling_weight = subsampling_weight.cuda()

        nentity = label_matrix.size(1)
        candidates = self._candidate_entities(nentity, positive_sample.device)
        negative_sample = candidates.unsqueeze(0).expand(positive_sample.size(0), -1)
        all_scores = model((positive_sample, negative_sample), mode=mode)
        return all_scores, label_matrix, subsampling_weight, positive_sample, mode


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
    def __init__(self, triples, nentity, nrelation, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
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

    def _sample_negatives(self, mode, head, relation, tail):
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if mode == 'head-batch':
                mask = np.isin(negative_sample, self.true_head[(relation, tail)], invert=True)
            else:
                mask = np.isin(negative_sample, self.true_tail[(head, relation)], invert=True)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        return np.concatenate(negative_sample_list)[: self.negative_sample_size]

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        p_corrupt_head = self.tph[relation] / (self.tph[relation] + self.hpt[relation])
        mode = 'head-batch' if np.random.rand() < p_corrupt_head else 'tail-batch'
        negative_sample = self._sample_negatives(mode, head, relation, tail)

        return (
            torch.LongTensor(positive_sample),
            torch.LongTensor(negative_sample),
            subsampling_weight,
            mode,
        )

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        modes = [_[3] for _ in data]
        return positive_sample, negative_sample, subsample_weight, modes


class BernoulliOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        return next(self.iterator)

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data


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
        labels = torch.zeros(self.nentity, dtype=torch.float32)
        if self.mode == 'tail-batch':
            labels[tail] = 1.0
        else:
            labels[head] = 1.0
        return labels


class KvsAllTrainDataset(_AllNegTrainDatasetBase):
    def _label_vector(self, head, relation, tail):
        labels = torch.zeros(self.nentity, dtype=torch.float32)
        if self.mode == 'tail-batch':
            for e in self.true_tail[(head, relation)]:
                labels[int(e)] = 1.0
        else:
            for e in self.true_head[(relation, tail)]:
                labels[int(e)] = 1.0
        return labels


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
