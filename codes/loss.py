from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F


def regularization(model, coeff, p):
    '''
    Lp embedding regularization: coeff * (||E||_p^p + ||R||_p^p).
    '''
    if coeff == 0.0:
        return None, {}

    regularization_term = coeff * (
        model.entity_embedding.norm(p = p)**p +
        model.relation_embedding.norm(p = p)**p
    )
    return regularization_term, {'regularization': regularization_term.item()}


def weighted_mean(loss, subsampling_weight, uni_weight):
    if uni_weight:
        return loss.mean()
    weight = subsampling_weight.view(-1)
    loss = loss.view(-1)
    return (weight * loss).sum() / weight.sum()


class KGELoss:
    def __init__(self, args):
        self.args = args

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        raise NotImplementedError


class SelfAdversarialNegativeSamplingLoss(KGELoss):
    '''
    Margin negative sampling loss - a listwise margin-based log-sigmoid loss
    L = 1/2 * (L_positive + L_negative) + regularization
    L_positive = logsigmoid(positive_score)
    L_negative = logsigmoid(-negative_score)
    '''

    def _weighted_mean(self, score, subsampling_weight):
        if self.args.uni_weight:
            return - score.mean()
        return - (subsampling_weight * score).sum() / subsampling_weight.sum()

    def _positive_sample_loss(self, positive_score, subsampling_weight):
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)
        return self._weighted_mean(positive_score, subsampling_weight)

    def _negative_sample_loss(self, negative_score, subsampling_weight):
        if self.args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim = 1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        return self._weighted_mean(negative_score, subsampling_weight)

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        positive_sample_loss_val = self._positive_sample_loss(positive_score, subsampling_weight)
        negative_sample_loss_val = self._negative_sample_loss(negative_score, subsampling_weight)

        loss = (positive_sample_loss_val + negative_sample_loss_val) / 2

        regularization_term, regularization_log = regularization(
            model, self.args.regularization_coeff, p=self.args.regularization_p
        )
        if regularization_term is not None:
            loss = loss + regularization_term

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss_val.item(),
            'negative_sample_loss': negative_sample_loss_val.item(),
            'loss': loss.item()
        }
        return loss, log


class CrossEntropyLoss(KGELoss):
    '''
    Cross-entropy loss - a listwise cross-entropy loss
    L = cross_entropy([positive_score; negative_scores], target=0) + regularization
    '''

    def _weighted_mean(self, loss, subsampling_weight):
        return weighted_mean(loss, subsampling_weight, self.args.uni_weight)

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        scores = torch.cat([positive_score, negative_score], dim=1)
        target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        loss = self._weighted_mean(
            F.cross_entropy(scores, target, reduction='none'),
            subsampling_weight,
        )
        regularization_term, regularization_log = regularization(
            model, self.args.regularization_coeff, p=self.args.regularization_p
        )
        if regularization_term is not None:
            loss = loss + regularization_term

        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return loss, log

class MarginRankingLoss(KGELoss):
    '''
    Margin ranking loss - a pairwise margin-based ranking loss
    L = max(0, margin - positive_score + negative_score) + regularization
    '''

    def _weighted_mean(self, loss, subsampling_weight):
        return weighted_mean(loss, subsampling_weight, self.args.uni_weight)

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        positive_expanded = positive_score.expand_as(negative_score)
        target = torch.ones_like(negative_score)
        per_sample_loss = F.margin_ranking_loss(
            positive_expanded, negative_score, target,
            margin=self.args.gamma, reduction='none',
        ).mean(dim=1)
        loss = self._weighted_mean(per_sample_loss, subsampling_weight)
        regularization_term, regularization_log = regularization(
            model, self.args.regularization_coeff, p=self.args.regularization_p
        )
        if regularization_term is not None:
            loss = loss + regularization_term

        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return loss, log

class BinaryCrossEntropyLoss(KGELoss):
    '''
    Binary cross-entropy loss - treats positive/negative scores as binary labels
    L = BCE(sigmoid(positive), 1) + BCE(sigmoid(negative), 0) + regularization
    '''

    def _weighted_mean(self, loss, subsampling_weight):
        return weighted_mean(loss, subsampling_weight, self.args.uni_weight)

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        positive_loss = F.binary_cross_entropy_with_logits(
            positive_score.squeeze(dim=1),
            torch.ones(positive_score.size(0), device=positive_score.device),
            reduction='none',
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            negative_score,
            torch.zeros_like(negative_score),
            reduction='none',
        ).mean(dim=1)
        loss = self._weighted_mean((positive_loss + negative_loss) / 2, subsampling_weight)

        regularization_term, regularization_log = regularization(
            model, self.args.regularization_coeff, p=self.args.regularization_p
        )
        if regularization_term is not None:
            loss = loss + regularization_term
        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return loss, log

class MeanSquaredErrorLoss(KGELoss):
    '''
    Mean squared error on logits with targets y=1 (positive) and y=0 (negative)
    L = 1/2 * (f(x+)-1)^2 + 1/2 * mean((f(x-)-0)^2) + regularization
    '''

    def _weighted_mean(self, loss, subsampling_weight):
        return weighted_mean(loss, subsampling_weight, self.args.uni_weight)

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        positive_loss = (positive_score.squeeze(dim=1) - 1).pow(2)
        negative_loss = negative_score.pow(2).mean(dim=1)
        loss = self._weighted_mean((positive_loss + negative_loss) / 2, subsampling_weight)

        regularization_term, regularization_log = regularization(
            model, self.args.regularization_coeff, p=self.args.regularization_p
        )
        if regularization_term is not None:
            loss = loss + regularization_term

        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return loss, log


class BayesianPersonalizedRankingLoss(KGELoss):
    '''
    Bayesian personalized ranking loss
    L = -log(sigmoid(f(x+) - f(x-))) + regularization
    '''

    def _weighted_mean(self, loss, subsampling_weight):
        return weighted_mean(loss, subsampling_weight, self.args.uni_weight)

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        positive_expanded = positive_score.expand_as(negative_score)
        per_sample_loss = -F.logsigmoid(positive_expanded - negative_score).mean(dim=1)
        loss = self._weighted_mean(per_sample_loss, subsampling_weight)

        regularization_term, regularization_log = regularization(
            model, self.args.regularization_coeff, p=self.args.regularization_p
        )
        if regularization_term is not None:
            loss = loss + regularization_term

        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return loss, log


class InfoNCELoss(KGELoss):
    '''
    InfoNCE loss with temperature tau
    L = -log(exp(f(x+)/tau) / (exp(f(x+)/tau) + sum_i exp(f(x-_i)/tau))) + regularization
    '''

    def _weighted_mean(self, loss, subsampling_weight):
        return weighted_mean(loss, subsampling_weight, self.args.uni_weight)

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        tau = getattr(self.args, 'infonce_temperature', 1.0)
        scores = torch.cat([positive_score, negative_score], dim=1) / tau
        target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        loss = self._weighted_mean(
            F.cross_entropy(scores, target, reduction='none'),
            subsampling_weight,
        )

        regularization_term, regularization_log = regularization(
            model, self.args.regularization_coeff, p=self.args.regularization_p
        )
        if regularization_term is not None:
            loss = loss + regularization_term

        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return loss, log


class AlignmentUniformityLoss(KGELoss):
    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x, tuni=2):
        x = F.normalize(x, dim=-1)
        if x.size(0) < 2:
            return x.new_zeros(())
        return torch.pdist(x, p=2).pow(2).mul(-tuni).exp().mean().log()

    def calculate_loss(self, head, relation, tail, model, mode):
        tuni = getattr(self.args, 'tuni', 2)
        gamma_q = getattr(self.args, 'gamma_q', 1.0)
        gamma_t = getattr(self.args, 'gamma_t', 1.0)

        query_e = model.query_encoder(head, relation, tail, mode=mode)
        target_e = model.target_encoder(tail, head=head, relation=relation, mode=mode)
        align_loss = self.alignment(query_e, target_e)

        uniform_loss = query_e.new_zeros(())
        uniform_count = 0
        if gamma_q > 0:
            uniform_loss = uniform_loss + gamma_q * self.uniformity(query_e, tuni=tuni)
            uniform_count += 1
        if gamma_t > 0:
            uniform_loss = uniform_loss + gamma_t * self.uniformity(target_e, tuni=tuni)
            uniform_count += 1

        if uniform_count > 0:
            loss = align_loss + uniform_loss / uniform_count
            uniform_loss_val = (uniform_loss / uniform_count).item()
        else:
            loss = align_loss
            uniform_loss_val = 0.0

        regularization_term, regularization_log = regularization(
            model, self.args.regularization_coeff, p=self.args.regularization_p
        )
        if regularization_term is not None:
            loss = loss + regularization_term

        log = {
            **regularization_log,
            'align_loss': align_loss.item(),
            'uniform_loss': uniform_loss_val,
            'loss': loss.item(),
        }
        return loss, log

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        raise NotImplementedError(
            'AlignmentUniformityLoss requires positive triple embeddings; '
            'use compute_kge_loss with positive_sample and mode.'
        )


LOSS_REGISTRY = {
    'self_adv': SelfAdversarialNegativeSamplingLoss,
    'ce': CrossEntropyLoss,
    'mr': MarginRankingLoss,
    'bce': BinaryCrossEntropyLoss,
    'mse': MeanSquaredErrorLoss,
    'bpr': BayesianPersonalizedRankingLoss,
    'infonce': InfoNCELoss,
    'au': AlignmentUniformityLoss,
}

def get_loss(args):
    loss_name = getattr(args, 'loss', 'self_adv')
    if loss_name not in LOSS_REGISTRY:
        raise ValueError('Unknown loss: {}'.format(loss_name))
    return LOSS_REGISTRY[loss_name](args)

def compute_kge_loss(positive_score, negative_score, subsampling_weight, model, args,
                     positive_sample=None, mode=None):
    loss_name = getattr(args, 'loss', 'self_adv')
    if loss_name == 'au':
        if positive_sample is None or mode is None:
            raise ValueError('AlignmentUniformityLoss requires positive_sample and mode')
        head = model.entity_embedding[positive_sample[:, 0]]
        relation = model.relation_embedding[positive_sample[:, 1]]
        tail = model.entity_embedding[positive_sample[:, 2]]
        return get_loss(args).calculate_loss(head, relation, tail, model, mode)
    return get_loss(args)(positive_score, negative_score, subsampling_weight, model)