from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
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


class SquaredErrorLoss(KGELoss):
    '''
    Mean squared error on logits with targets y=1 (positive) and y=0 (negative)
    L = 1/2 * (f(x)-y)^2 + regularization
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


class PointwiseHingeLoss(KGELoss):
    '''
    Pointwise Hinge loss - a pointwise max-margin loss with labels l=1 (positive) and l=-1 (negative)
    L = max(0, margin - l * score) + regularization
    '''

    def _weighted_mean(self, loss, subsampling_weight):
        return weighted_mean(loss, subsampling_weight, self.args.uni_weight)

    def __call__(self, positive_score, negative_score, subsampling_weight, model):
        margin = self.args.gamma
        positive_loss = F.relu(margin - positive_score.squeeze(dim=1))
        negative_loss = F.relu(margin + negative_score).mean(dim=1)
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


class BayesianPersonalizedRankingLoss(KGELoss):
    '''
    Bayesian personalized ranking loss - a pairwise logistic loss
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


class CrossEntropyLoss(KGELoss):
    '''
    Cross-entropy loss - a listwise softmax loss
    L = -log(softmax(positive_score)) + regularization
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


class SelfAdversarialNegativeSamplingLoss(KGELoss):
    '''
    Self-adversarial negative sampling loss (RotatE Eq. 6).

    L = -log(sigmoid(positive_score))
        - sum_j softmax(alpha * negative_score_j).detach() * log(sigmoid(-negative_score_j))
        + regularization
    '''

    def _weighted_mean(self, loss, subsampling_weight):
        return weighted_mean(loss, subsampling_weight, self.args.uni_weight)

    def _positive_sample_loss(self, positive_score, subsampling_weight):
        # RotatE: positive_sample_loss = -weighted_mean(logsigmoid(positive_score))
        positive_log_prob = F.logsigmoid(positive_score).squeeze(dim=1)
        return -self._weighted_mean(positive_log_prob, subsampling_weight)

    def _negative_sample_loss(self, negative_score, subsampling_weight):
        if self.args.negative_adversarial_sampling:
            # RotatE: softmax weights are detached from the computation graph.
            negative_log_prob = (
                F.softmax(negative_score * self.args.adversarial_temperature, dim=1).detach()
                * F.logsigmoid(-negative_score)
            ).sum(dim=1)
        else:
            negative_log_prob = F.logsigmoid(-negative_score).mean(dim=1)

        return -self._weighted_mean(negative_log_prob, subsampling_weight)

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


AU_UNIFORM_TERMS = [
    ('query', 'uni_gamma_query'),
    ('target', 'uni_gamma_target'),
    ('entity', 'uni_gamma_entity'),
]

AU_GAMMA_KEY_BY_TERM = {term_key: gamma_key for term_key, gamma_key in AU_UNIFORM_TERMS}


def is_learnable_au_gammas(args):
    return getattr(args, 'loss', '') == 'au' and getattr(args, 'learnable_au_gammas', False)


def get_au_uniform_embeddings(head, relation, tail, query_e, target_e, term_key):
    if term_key == 'query':
        return query_e
    if term_key == 'target':
        return target_e
    if term_key == 'head':
        return head
    if term_key == 'tail':
        return tail
    if term_key == 'entity':
        return torch.cat([head, tail], dim=0)
    if term_key == 'relation':
        return relation
    raise ValueError('Unknown AU uniform term key: {}'.format(term_key))


class UniGammaController(object):
    def __init__(self, args):
        self.args = args

    def active_terms(self):
        active = []
        for term_key, gamma_key in AU_UNIFORM_TERMS:
            if getattr(self.args, gamma_key, 0.0) > 0:
                active.append(term_key)
        return active

    def gamma_init(self, term_key):
        gamma_key = AU_GAMMA_KEY_BY_TERM[term_key]
        return getattr(self.args, gamma_key, 0.0)

    def schedule_mult(self, epoch):
        if not getattr(self.args, 'gamma_linear_schedule', False):
            return 1.0
        start_epoch = getattr(self.args, 'gamma_schedule_start_epoch', 0)
        span = getattr(self.args, 'gamma_schedule_epochs', 0) or getattr(self.args, 'epochs', 1)
        if span <= 0:
            return 1.0
        progress = float(epoch - start_epoch) / float(span)
        progress = max(0.0, min(1.0, progress))
        end_mult = getattr(self.args, 'gamma_schedule_end', 0.1)
        return 1.0 + progress * (end_mult - 1.0)

    def effective_gamma(self, model, term_key, epoch):
        if not hasattr(model, 'au_log_gamma_adj'):
            self.ensure_model_params(model)
        gamma_init = self.gamma_init(term_key)
        schedule = self.schedule_mult(epoch)
        log_adj = torch.clamp(model.au_log_gamma_adj[term_key], max=0.0)
        return gamma_init * schedule * torch.exp(log_adj)

    def ensure_model_params(self, model):
        if hasattr(model, 'au_log_gamma_adj'):
            return
        active = self.active_terms()
        if not active:
            return
        device = model.entity_embedding.device
        model.au_log_gamma_adj = nn.ParameterDict({
            term_key: nn.Parameter(torch.zeros((), device=device))
            for term_key in active
        })

    def clamp_log_gammas(self, model):
        for param in model.au_log_gamma_adj.parameters():
            param.data.clamp_(max=0.0)

    def log_effective_gammas(self, model, epoch):
        log = {}
        for term_key in self.active_terms():
            eff = self.effective_gamma(model, term_key, epoch)
            if isinstance(eff, torch.Tensor):
                log['uni_gamma_eff_{}'.format(term_key)] = eff.item()
            else:
                log['uni_gamma_eff_{}'.format(term_key)] = float(eff)
        return log


def update_au_gamma_schedule(args):
    if not is_learnable_au_gammas(args):
        return 1.0
    controller = UniGammaController(args)
    epoch = getattr(args, 'current_epoch', 0)
    args.au_schedule_mult = controller.schedule_mult(epoch)
    return args.au_schedule_mult


def build_training_optimizer(model, args):
    lr = args.learning_rate
    if not is_learnable_au_gammas(args):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )

    gamma_lr = getattr(args, 'log_au_gamma_lr', None) or lr
    embedding_params = []
    gamma_params = []
    for name, param in model.named_parameters():
        if name.startswith('au_log_gamma_adj.'):
            gamma_params.append(param)
        else:
            embedding_params.append(param)
    param_groups = [{'params': embedding_params, 'lr': lr}]
    if gamma_params:
        param_groups.append({'params': gamma_params, 'lr': gamma_lr})
    return torch.optim.Adam(param_groups)


def set_optimizer_learning_rates(optimizer, lr, gamma_lr=None):
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = gamma_lr if gamma_lr is not None else lr


class AlignmentUniformityLoss(KGELoss):
    '''
    Alignment-uniformity loss - a pairwise uniformity-based loss
    L = alignment_loss + Avg(uniformity_loss)_terms + regularization
    alignment_loss = (query_e - target_e).norm(p=2, dim=1).pow(2).mean()
    uniformity_loss = torch.pdist(x, p=2).pow(2).mul(-tuni).exp().mean().log()
    '''
    
    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x, tuni=4):
        x = F.normalize(x, dim=-1)
        if x.size(0) < 2:
            return (x * 0).sum()
        return torch.pdist(x, p=2).pow(2).mul(-tuni).exp().mean().log()

    def _compute_uniform_terms(self, head, relation, tail, query_e, target_e, model, tuni):
        uniform_loss_sum = query_e.new_zeros(())
        uniform_count = 0
        uniform_log = {}
        epoch = getattr(self.args, 'current_epoch', 0)
        learnable = is_learnable_au_gammas(self.args)
        controller = UniGammaController(self.args) if learnable else None

        for term_key, gamma_key in AU_UNIFORM_TERMS:
            gamma_init = getattr(self.args, gamma_key, 0.0)
            if gamma_init <= 0:
                continue
            embeddings = get_au_uniform_embeddings(
                head, relation, tail, query_e, target_e, term_key,
            )
            uniform_val = self.uniformity(embeddings, tuni=tuni)
            if learnable:
                gamma_weight = controller.effective_gamma(model, term_key, epoch)
            else:
                gamma_weight = gamma_init
            uniform_loss_sum = uniform_loss_sum + gamma_weight * uniform_val
            uniform_count += 1
            uniform_log['uniform_{}'.format(term_key)] = uniform_val.item()
            if learnable:
                eff_item = (
                    gamma_weight.item()
                    if isinstance(gamma_weight, torch.Tensor)
                    else float(gamma_weight)
                )
                uniform_log['uni_gamma_eff_{}'.format(term_key)] = eff_item

        return uniform_loss_sum, uniform_count, uniform_log

    def calculate_loss(self, head, relation, tail, model, mode):
        tuni = getattr(self.args, 'tuni', 4)

        query_e = model.query_encoder(head, relation, tail, mode=mode)
        target_e = model.target_encoder(tail, head=head, relation=relation, mode=mode)
        align_loss = self.alignment(query_e, target_e)

        uniform_loss_sum, uniform_count, uniform_log = self._compute_uniform_terms(
            head, relation, tail, query_e, target_e, model, tuni,
        )

        if uniform_count > 0:
            loss = align_loss + uniform_loss_sum / uniform_count
            uniform_loss_val = (uniform_loss_sum / uniform_count).item()
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
            **uniform_log,
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
    'se': SquaredErrorLoss,
    'hinge': PointwiseHingeLoss,
    'bce': BinaryCrossEntropyLoss,
    'mr': MarginRankingLoss,
    'bpr': BayesianPersonalizedRankingLoss,
    'ce': CrossEntropyLoss,
    'sans': SelfAdversarialNegativeSamplingLoss,
    'au': AlignmentUniformityLoss,
}

def get_loss(args):
    loss_name = getattr(args, 'loss', 'sans')
    if loss_name not in LOSS_REGISTRY:
        raise ValueError('Unknown loss: {}'.format(loss_name))
    return LOSS_REGISTRY[loss_name](args)

def compute_kge_loss(positive_score, negative_score, subsampling_weight, model, args,
                     positive_sample=None, mode=None):
    loss_name = getattr(args, 'loss', 'sans')
    if loss_name == 'au':
        if positive_sample is None or mode is None:
            raise ValueError('AlignmentUniformityLoss requires positive_sample and mode')
        head = model.entity_embedding[positive_sample[:, 0]]
        relation = model.relation_embedding[positive_sample[:, 1]]
        tail = model.entity_embedding[positive_sample[:, 2]]
        return get_loss(args).calculate_loss(head, relation, tail, model, mode)
    return get_loss(args)(positive_score, negative_score, subsampling_weight, model)
