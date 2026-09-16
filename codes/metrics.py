"""Metrics for evaluating Knowledge Graph Embedding models.

Includes ranking metrics for Link Prediction and classification metrics 
for Value Classification (Triple Classification) tasks.
"""

from typing import List, Sequence, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# =============================================================================
# Ranking Metrics (Link Prediction)
# =============================================================================

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


# =============================================================================
# Classification Metrics (Value / Triple Classification)
# =============================================================================

def classification_metrics(y_true, y_pred, y_prob=None, zero_division=0):
    """Compute standard binary classification metrics for Value Classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_classes = np.unique(y_true)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=zero_division),
        'recall': recall_score(y_true, y_pred, zero_division=zero_division),
        'f1': f1_score(y_true, y_pred, zero_division=zero_division),
    }

    if y_prob is None or unique_classes.size < 2:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
        return metrics

    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['roc_auc'] = 0.0

    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['pr_auc'] = 0.0

    return metrics

def find_global_threshold(y_true, y_prob, n_thresholds=100):
    """Find a global threshold that maximizes validation accuracy."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.size == 0 or y_prob.size == 0:
        raise ValueError('y_true and y_prob must not be empty')

    if np.allclose(y_prob.min(), y_prob.max()):
        return float(y_prob[0])

    thresholds = np.linspace(float(y_prob.min()), float(y_prob.max()), n_thresholds)
    best_acc = -1.0
    best_f1 = -1.0
    best_t = float(thresholds[0])
    
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        acc = float((y_pred == y_true).mean())
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_t = float(t)
    return best_t

def find_relation_specific_thresholds(y_true, y_prob, relations, n_thresholds=100):
    """Find optimal thresholds for each relation individually with global fallback."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    relations = np.asarray(relations)
    
    thresholds = {}
    unique_relations = np.unique(relations)
    
    # Tính sẵn global threshold làm fallback cho các relation bị thiếu data
    global_t = find_global_threshold(y_true, y_prob, n_thresholds)
    
    for r in unique_relations:
        mask = (relations == r)
        r_true = y_true[mask]
        r_prob = y_prob[mask]
        
        # Nếu relation không có đủ cả 2 class (0 và 1) để tìm mốc phân chia, dùng global_t
        if len(np.unique(r_true)) < 2:
            thresholds[r] = global_t
        else:
            thresholds[r] = find_global_threshold(r_true, r_prob, n_thresholds)
            
    return thresholds, global_t

def triple_classification_metrics(y_true, y_prob, relations, threshold_mode='relation', n_thresholds=100):
    """Compute metrics using either global or relation-specific thresholds."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    relations = np.asarray(relations)
    
    y_pred = np.zeros_like(y_true)
    
    if threshold_mode == 'relation':
        thresholds, _ = find_relation_specific_thresholds(y_true, y_prob, relations, n_thresholds)
        for r, t in thresholds.items():
            mask = (relations == r)
            y_pred[mask] = (y_prob[mask] > t).astype(int)
    else: # global
        t = find_global_threshold(y_true, y_prob, n_thresholds)
        y_pred = (y_prob > t).astype(int)
        
    return classification_metrics(y_true, y_pred, y_prob=y_prob)