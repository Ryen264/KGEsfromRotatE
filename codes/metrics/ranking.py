"""Ranking metrics for evaluating KG models."""

from typing import List, Sequence, Tuple

import torch


def topk_accuracy(output, target, topk=(1,)):
    """Compute top-k classification accuracy (percentage) for each k in topk."""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def accuracy(output, target, topk=(1,)):
    """Backward-compatible alias for top-k accuracy."""

    return topk_accuracy(output, target, topk=topk)


def ranks_from_score_matrix(
    score,
    target_indices,
    *,
    tie_handling='rounded_mean_rank',
    tie_rtol=1e-4,
    tie_atol=1e-5,
):
    """Compute 1-based filtered ranks with LibKGE-style tie handling."""

    scores = score.clone()
    scores[torch.isnan(scores)] = float('-inf')
    target_scores = scores.gather(1, target_indices.unsqueeze(1))
    target_scores = target_scores.clone()
    target_scores[torch.isnan(target_scores)] = float('-inf')

    is_close = torch.isclose(scores, target_scores, rtol=tie_rtol, atol=tie_atol)
    is_greater = scores > target_scores
    num_ties = torch.sum(is_close, dim=1, dtype=torch.long)
    rank_zero = torch.sum(is_greater & ~is_close, dim=1, dtype=torch.long)

    if tie_handling == 'rounded_mean_rank':
        ranks_zero = rank_zero + num_ties // 2
    elif tie_handling == 'best_rank':
        ranks_zero = rank_zero
    elif tie_handling == 'worst_rank':
        ranks_zero = rank_zero + num_ties - 1
    else:
        raise ValueError('Unsupported tie_handling={!r}'.format(tie_handling))

    return ranks_zero.add(1).tolist()


def ranking_metrics_from_ranks(ranks, round_digits=4):
    """Compute link-prediction metrics from 1-based ranks."""

    ranks_list = list(ranks)
    if not ranks_list:
        raise ValueError('ranks must not be empty')

    total = float(len(ranks_list))
    mr = sum(ranks_list) / total
    mrr = sum(1.0 / rank for rank in ranks_list) / total
    hit_at_1 = sum(1 for rank in ranks_list if rank <= 1) / total
    hit_at_3 = sum(1 for rank in ranks_list if rank <= 3) / total
    hit_at_10 = sum(1 for rank in ranks_list if rank <= 10) / total

    metrics = {
        'mr': mr,
        'mrr': mrr,
        'hit@1': hit_at_1,
        'hit@3': hit_at_3,
        'hit@10': hit_at_10,
    }
    if round_digits is not None:
        metrics = {key: round(value, round_digits) for key, value in metrics.items()}
    return metrics


def rotate_ranking_metrics_from_ranks(ranks, round_digits=None):
    """Aggregate ranks into RotatE-style metric keys (MRR, MR, HITS@k)."""

    metrics = ranking_metrics_from_ranks(ranks, round_digits=round_digits)
    return {
        'MRR': metrics['mrr'],
        'MR': metrics['mr'],
        'HITS@1': metrics['hit@1'],
        'HITS@3': metrics['hit@3'],
        'HITS@10': metrics['hit@10'],
    }


def ranking_metrics_from_scores(scores, targets, topk=(1, 3, 10)):
    """Compute link-prediction metrics from a score matrix and target indices."""

    with torch.no_grad():
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.view(-1)
        elif targets.dim() != 1:
            raise ValueError('targets must have shape (batch_size,) or (batch_size, 1)')

        maxk = max(topk)
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        target_rank = torch.nonzero(
            sorted_indices.eq(targets.unsqueeze(-1)).long(), as_tuple=False
        )
        if target_rank.size(0) != scores.size(0):
            raise RuntimeError('Unable to locate one target rank per example')

        ranks = ranks_from_score_matrix(scores, targets.view(-1))
        metrics = ranking_metrics_from_ranks(ranks)
        topk_scores = sorted_scores[:, :maxk].tolist()
        topk_indices = sorted_indices[:, :maxk].tolist()
        return topk_scores, topk_indices, metrics, ranks


def link_prediction_metrics(ranks):
    """Alias for ranking_metrics_from_ranks for link prediction tasks."""

    return ranking_metrics_from_ranks(ranks)
