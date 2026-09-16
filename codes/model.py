"""Knowledge Graph Embedding (KGE) models.

This module implements:
  - KGEBase: common interface for a KGE score function.
  - ComplEx, RotatE: score functions, each in its own class, exposing
    `query_encoder` / `target_encoder` / `score` so that training
    strategies (NegSamp, AllNeg, KGAU, ...) can reuse the same query /
    target embeddings the score is built from.
  - KGEModel: the nn.Module wrapping entity/relation embedding tables
    around a KGE score function, plus the train/eval loop entry points
    (`train_step`, `test_step`).

RotatE's all-entity scoring path (`score_query_entities`) is backed by a
custom autograd.Function (`_RotatEScoreQueryEntities`) with an optional
Triton-accelerated kernel, so it never materializes a [batch, nentity, dim]
tensor. The Triton kernels are defined as plain module-level functions
because `@triton.jit` requires that (it cannot decorate bound methods);
they are private implementation details of RotatE and are not meant to be
used directly.
"""

import logging

import numpy as np
import random # Thêm module random để sinh negative samples

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataloader import TestDataset
from loss import (
    compute_kge_loss,
    UniGammaController,
    is_kgau_family_loss,
    is_learnable_kgau_gammas,
)
from metrics import (
    triple_classification_metrics,
    ranks_from_score_matrix,
    rotate_ranking_metrics_from_ranks
)
from strategy import get_strategy, resolve_strategy_name

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def resolve_rotate_score_mode(args):
    '''
    RotatE scoring mode from training setup.

    KGAU-family (loss or strategy) aligns L2-normalized query/target
    embeddings, so ranking uses cosine similarity. NegSamp / AllNeg keep
    RotatE distance.
    '''
    if is_kgau_family_loss(args):
        return 'cosine'
    try:
        if resolve_strategy_name(args) == 'kgau':
            return 'cosine'
    except (AttributeError, KeyError, ValueError):
        pass
    return 'distance'


# =============================================================================
# Base class
# =============================================================================

class KGEBase(object):
    '''
    Common interface for a KGE score function.

    `score(head, relation, tail, mode)` is the training/eval-facing entry
    point. It is decomposed into `query_encoder` (the head/relation or
    relation/tail side, depending on `mode`) and `target_encoder` (the
    remaining entity) so that strategies which need the query and target
    embeddings separately (e.g. KGAU-style alignment losses) can reuse
    them instead of recomputing.
    '''

    @staticmethod
    def _split_complex(x):
        '''Split a [..., 2*d] embedding into its (real, imaginary) halves.'''
        return torch.chunk(x, 2, dim=-1)

    def score(self, head, relation, tail, mode):
        raise NotImplementedError

    def query_encoder(self, head, relation, tail, mode):
        raise NotImplementedError

    def target_encoder(self, head, relation, tail, mode):
        raise NotImplementedError


# =============================================================================
# ComplEx
# =============================================================================

class ComplEx(KGEBase):
    '''
    ComplEx: Complex Embeddings for Simple Link Prediction.

    Entities and relations are complex-valued (stored as concatenated
    [real, imaginary] halves). The score is the Hermitian dot product
    Re(<h * r, conj(t)>).
    '''

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
        Score queries against all entity embeddings without expanding to
        [B, E, D]. query: [B, D], entity_embedding: [E, D] -> scores [B, E].
        '''
        re_q, im_q = self._split_complex(query)
        re_e, im_e = self._split_complex(entity_embedding)
        return re_q @ re_e.transpose(0, 1) + im_q @ im_e.transpose(0, 1)


# =============================================================================
# RotatE
# =============================================================================
#
# The Triton kernels below implement the forward/backward of RotatE's
# all-entity distance score (used by `score_query_entities`). They are kept
# as module-level functions (required by `@triton.jit`) and are only ever
# invoked through `_RotatEScoreQueryEntities`, which is in turn a private
# implementation detail of `RotatE.score_query_entities`.

if _TRITON_AVAILABLE:

    @triton.jit
    def _rotate_allneg_fwd_kernel(
        re_q_ptr, im_q_ptr, re_e_ptr, im_e_ptr, scores_ptr,
        B, E, D,
        stride_qb, stride_qd,
        stride_ee, stride_ed,
        stride_sb, stride_se,
        gamma,
        BLOCK_E: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_e = tl.program_id(1)
        if pid_b >= B:
            return

        e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
        e_mask = e_offs < E
        acc = tl.zeros((BLOCK_E,), dtype=tl.float32)

        for d0 in range(0, D, BLOCK_D):
            d_offs = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            rq = tl.load(
                re_q_ptr + pid_b * stride_qb + d_offs * stride_qd,
                mask=d_mask, other=0.0,
            ).to(tl.float32)
            iq = tl.load(
                im_q_ptr + pid_b * stride_qb + d_offs * stride_qd,
                mask=d_mask, other=0.0,
            ).to(tl.float32)

            re = tl.load(
                re_e_ptr + e_offs[:, None] * stride_ee + d_offs[None, :] * stride_ed,
                mask=e_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            ie = tl.load(
                im_e_ptr + e_offs[:, None] * stride_ee + d_offs[None, :] * stride_ed,
                mask=e_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            re_diff = rq[None, :] - re
            im_diff = iq[None, :] - ie
            mod = tl.sqrt(re_diff * re_diff + im_diff * im_diff)
            mod = tl.where(d_mask[None, :], mod, 0.0)
            acc += tl.sum(mod, axis=1)

        tl.store(
            scores_ptr + pid_b * stride_sb + e_offs * stride_se,
            gamma - acc,
            mask=e_mask,
        )

    @triton.jit
    def _rotate_allneg_bwd_query_kernel(
        re_q_ptr, im_q_ptr, re_e_ptr, im_e_ptr, grad_scores_ptr,
        grad_re_q_ptr, grad_im_q_ptr,
        B, E, D,
        stride_qb, stride_qd,
        stride_ee, stride_ed,
        stride_sb, stride_se,
        BLOCK_E: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        '''One program per batch row; owns that query gradient (no atomics).'''
        pid_b = tl.program_id(0)
        if pid_b >= B:
            return

        for d0 in range(0, D, BLOCK_D):
            d_offs = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            acc_re = tl.zeros((BLOCK_D,), dtype=tl.float32)
            acc_im = tl.zeros((BLOCK_D,), dtype=tl.float32)

            rq = tl.load(
                re_q_ptr + pid_b * stride_qb + d_offs * stride_qd,
                mask=d_mask, other=0.0,
            ).to(tl.float32)
            iq = tl.load(
                im_q_ptr + pid_b * stride_qb + d_offs * stride_qd,
                mask=d_mask, other=0.0,
            ).to(tl.float32)

            for e0 in range(0, E, BLOCK_E):
                e_offs = e0 + tl.arange(0, BLOCK_E)
                e_mask = e_offs < E
                gscore = tl.load(
                    grad_scores_ptr + pid_b * stride_sb + e_offs * stride_se,
                    mask=e_mask, other=0.0,
                ).to(tl.float32)

                re = tl.load(
                    re_e_ptr + e_offs[:, None] * stride_ee + d_offs[None, :] * stride_ed,
                    mask=e_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                ie = tl.load(
                    im_e_ptr + e_offs[:, None] * stride_ee + d_offs[None, :] * stride_ed,
                    mask=e_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                re_diff = rq[None, :] - re
                im_diff = iq[None, :] - ie
                mod = tl.maximum(tl.sqrt(re_diff * re_diff + im_diff * im_diff), 1e-12)
                g_re = -gscore[:, None] * (re_diff / mod)
                g_im = -gscore[:, None] * (im_diff / mod)
                g_re = tl.where(e_mask[:, None] & d_mask[None, :], g_re, 0.0)
                g_im = tl.where(e_mask[:, None] & d_mask[None, :], g_im, 0.0)
                acc_re += tl.sum(g_re, axis=0)
                acc_im += tl.sum(g_im, axis=0)

            tl.store(
                grad_re_q_ptr + pid_b * stride_qb + d_offs * stride_qd,
                acc_re, mask=d_mask,
            )
            tl.store(
                grad_im_q_ptr + pid_b * stride_qb + d_offs * stride_qd,
                acc_im, mask=d_mask,
            )

    @triton.jit
    def _rotate_allneg_bwd_entity_kernel(
        re_q_ptr, im_q_ptr, re_e_ptr, im_e_ptr, grad_scores_ptr,
        grad_re_e_ptr, grad_im_e_ptr,
        B, E, D,
        stride_qb, stride_qd,
        stride_ee, stride_ed,
        stride_sb, stride_se,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        '''One program per entity; owns that entity gradient (no atomics).'''
        pid_e = tl.program_id(0)
        if pid_e >= E:
            return

        for d0 in range(0, D, BLOCK_D):
            d_offs = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            acc_re = tl.zeros((BLOCK_D,), dtype=tl.float32)
            acc_im = tl.zeros((BLOCK_D,), dtype=tl.float32)

            re = tl.load(
                re_e_ptr + pid_e * stride_ee + d_offs * stride_ed,
                mask=d_mask, other=0.0,
            ).to(tl.float32)
            ie = tl.load(
                im_e_ptr + pid_e * stride_ee + d_offs * stride_ed,
                mask=d_mask, other=0.0,
            ).to(tl.float32)

            for b0 in range(0, B, BLOCK_B):
                b_offs = b0 + tl.arange(0, BLOCK_B)
                b_mask = b_offs < B
                gscore = tl.load(
                    grad_scores_ptr + b_offs * stride_sb + pid_e * stride_se,
                    mask=b_mask, other=0.0,
                ).to(tl.float32)

                rq = tl.load(
                    re_q_ptr + b_offs[:, None] * stride_qb + d_offs[None, :] * stride_qd,
                    mask=b_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                iq = tl.load(
                    im_q_ptr + b_offs[:, None] * stride_qb + d_offs[None, :] * stride_qd,
                    mask=b_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                re_diff = rq - re[None, :]
                im_diff = iq - ie[None, :]
                mod = tl.maximum(tl.sqrt(re_diff * re_diff + im_diff * im_diff), 1e-12)
                g_re = -gscore[:, None] * (re_diff / mod)
                g_im = -gscore[:, None] * (im_diff / mod)
                g_re = tl.where(b_mask[:, None] & d_mask[None, :], g_re, 0.0)
                g_im = tl.where(b_mask[:, None] & d_mask[None, :], g_im, 0.0)
                # d(re_diff)/d(re_e)=-1 => accumulate -g_re over batch
                acc_re += -tl.sum(g_re, axis=0)
                acc_im += -tl.sum(g_im, axis=0)

            tl.store(
                grad_re_e_ptr + pid_e * stride_ee + d_offs * stride_ed,
                acc_re, mask=d_mask,
            )
            tl.store(
                grad_im_e_ptr + pid_e * stride_ee + d_offs * stride_ed,
                acc_im, mask=d_mask,
            )


class _RotatEScoreQueryEntities(torch.autograd.Function):
    '''
    All-entity RotatE scores with ComplEx-like saved-tensor footprint: save
    query/entity embeddings only, recompute distance in backward.

    On CUDA + Triton: fused kernels (no [B, C, D] intermediates).
    Otherwise: chunked PyTorch stack+norm path.

    Private implementation detail of `RotatE.score_query_entities`.
    '''

    _BYTES_BUDGET = 512 * 1024 * 1024
    _TRITON_BLOCK_E = 64
    _TRITON_BLOCK_D = 64
    # Sticky disable after a launch failure (e.g. bad CUDA device context).
    _triton_disabled = False

    @staticmethod
    def _chunk_size(batch_size, half_dim, nentity):
        # Peak temps ~= re_diff + im_diff + stacked + norm ~ 5x [B, C, d] float32.
        per_entity = max(batch_size * half_dim * 4 * 5, 1)
        return max(64, min(nentity, _RotatEScoreQueryEntities._BYTES_BUDGET // per_entity))

    @staticmethod
    def _distance(re_diff, im_diff):
        # stack+norm is faster than sqrt(re^2+im^2) on GPU and finite at 0.
        return torch.stack([re_diff, im_diff], dim=0).norm(dim=0).sum(dim=-1)

    @staticmethod
    def _use_triton(re_q):
        return (
            _TRITON_AVAILABLE
            and not _RotatEScoreQueryEntities._triton_disabled
            and re_q.is_cuda
            and re_q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )

    @staticmethod
    def _disable_triton(reason):
        if not _RotatEScoreQueryEntities._triton_disabled:
            logging.warning(
                'Disabling Triton RotatE AllNeg kernels (%s); using chunked PyTorch.',
                str(reason),
                exc_info=True
            )
        _RotatEScoreQueryEntities._triton_disabled = True

    @staticmethod
    def _forward_triton(re_q, im_q, re_e, im_e, gamma):
        batch_size, half_dim = re_q.shape
        nentity = re_e.size(0)
        scores = re_q.new_empty(batch_size, nentity)
        re_q_c = re_q.contiguous()
        im_q_c = im_q.contiguous()
        re_e_c = re_e.contiguous()
        im_e_c = im_e.contiguous()
        scores_c = scores.contiguous()

        block_e = _RotatEScoreQueryEntities._TRITON_BLOCK_E
        block_d = min(_RotatEScoreQueryEntities._TRITON_BLOCK_D, triton.next_power_of_2(half_dim))
        grid = (batch_size, triton.cdiv(nentity, block_e))
        with torch.cuda.device(re_q.device):
            _rotate_allneg_fwd_kernel[grid](
                re_q_c, im_q_c, re_e_c, im_e_c, scores_c,
                batch_size, nentity, half_dim,
                re_q_c.stride(0), re_q_c.stride(1),
                re_e_c.stride(0), re_e_c.stride(1),
                scores_c.stride(0), scores_c.stride(1),
                float(gamma),
                BLOCK_E=block_e,
                BLOCK_D=block_d,
            )
        return scores_c

    @staticmethod
    def _backward_triton(re_q, im_q, re_e, im_e, grad_scores):
        batch_size, half_dim = re_q.shape
        nentity = re_e.size(0)
        grad_re_q = torch.zeros_like(re_q)
        grad_im_q = torch.zeros_like(im_q)
        grad_re_e = torch.zeros_like(re_e)
        grad_im_e = torch.zeros_like(im_e)

        re_q_c = re_q.contiguous()
        im_q_c = im_q.contiguous()
        re_e_c = re_e.contiguous()
        im_e_c = im_e.contiguous()
        grad_scores_c = grad_scores.contiguous()
        grad_re_q_c = grad_re_q.contiguous()
        grad_im_q_c = grad_im_q.contiguous()
        grad_re_e_c = grad_re_e.contiguous()
        grad_im_e_c = grad_im_e.contiguous()

        block_e = _RotatEScoreQueryEntities._TRITON_BLOCK_E
        block_b = _RotatEScoreQueryEntities._TRITON_BLOCK_E
        block_d = min(_RotatEScoreQueryEntities._TRITON_BLOCK_D, triton.next_power_of_2(half_dim))

        with torch.cuda.device(re_q.device):
            _rotate_allneg_bwd_query_kernel[(batch_size,)](
                re_q_c, im_q_c, re_e_c, im_e_c, grad_scores_c,
                grad_re_q_c, grad_im_q_c,
                batch_size, nentity, half_dim,
                re_q_c.stride(0), re_q_c.stride(1),
                re_e_c.stride(0), re_e_c.stride(1),
                grad_scores_c.stride(0), grad_scores_c.stride(1),
                BLOCK_E=block_e,
                BLOCK_D=block_d,
            )
            _rotate_allneg_bwd_entity_kernel[(nentity,)](
                re_q_c, im_q_c, re_e_c, im_e_c, grad_scores_c,
                grad_re_e_c, grad_im_e_c,
                batch_size, nentity, half_dim,
                re_q_c.stride(0), re_q_c.stride(1),
                re_e_c.stride(0), re_e_c.stride(1),
                grad_scores_c.stride(0), grad_scores_c.stride(1),
                BLOCK_B=block_b,
                BLOCK_D=block_d,
            )
        return grad_re_q_c, grad_im_q_c, grad_re_e_c, grad_im_e_c

    @staticmethod
    def _forward_chunked(re_q, im_q, re_e, im_e, gamma):
        batch_size, half_dim = re_q.shape
        nentity = re_e.size(0)
        chunk = _RotatEScoreQueryEntities._chunk_size(batch_size, half_dim, nentity)
        scores = re_q.new_empty(batch_size, nentity)
        for start in range(0, nentity, chunk):
            end = min(start + chunk, nentity)
            re_diff = re_q.unsqueeze(1) - re_e[start:end].unsqueeze(0)
            im_diff = im_q.unsqueeze(1) - im_e[start:end].unsqueeze(0)
            scores[:, start:end] = float(gamma) - _RotatEScoreQueryEntities._distance(
                re_diff, im_diff
            )
        return scores, chunk

    @staticmethod
    def _backward_chunked(re_q, im_q, re_e, im_e, grad_scores, chunk):
        nentity = re_e.size(0)
        grad_re_q = torch.zeros_like(re_q)
        grad_im_q = torch.zeros_like(im_q)
        grad_re_e = torch.zeros_like(re_e)
        grad_im_e = torch.zeros_like(im_e)

        for start in range(0, nentity, chunk):
            end = min(start + chunk, nentity)
            re_diff = re_q.unsqueeze(1) - re_e[start:end].unsqueeze(0)
            im_diff = im_q.unsqueeze(1) - im_e[start:end].unsqueeze(0)
            moduli = torch.stack([re_diff, im_diff], dim=0).norm(dim=0).clamp_min(1e-12)
            grad = grad_scores[:, start:end].unsqueeze(-1)
            g_re = -grad * (re_diff / moduli)
            g_im = -grad * (im_diff / moduli)
            grad_re_q += g_re.sum(dim=1)
            grad_im_q += g_im.sum(dim=1)
            grad_re_e[start:end] -= g_re.sum(dim=0)
            grad_im_e[start:end] -= g_im.sum(dim=0)
        return grad_re_q, grad_im_q, grad_re_e, grad_im_e

    @staticmethod
    def forward(ctx, re_q, im_q, re_e, im_e, gamma):
        ctx.save_for_backward(re_q, im_q, re_e, im_e)
        ctx.gamma = float(gamma)
        use_triton = _RotatEScoreQueryEntities._use_triton(re_q)
        if use_triton:
            try:
                scores = _RotatEScoreQueryEntities._forward_triton(
                    re_q, im_q, re_e, im_e, ctx.gamma
                )
                ctx.use_triton = True
                ctx.chunk = None
                return scores
            except Exception as exc:
                _RotatEScoreQueryEntities._disable_triton(exc)
        ctx.use_triton = False
        scores, chunk = _RotatEScoreQueryEntities._forward_chunked(
            re_q, im_q, re_e, im_e, ctx.gamma
        )
        ctx.chunk = chunk
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        re_q, im_q, re_e, im_e = ctx.saved_tensors
        if ctx.use_triton:
            try:
                grads = _RotatEScoreQueryEntities._backward_triton(
                    re_q, im_q, re_e, im_e, grad_scores
                )
                return (*grads, None)
            except Exception as exc:
                _RotatEScoreQueryEntities._disable_triton(exc)
                # Recompute with chunked path (same inputs / grad_scores).
                chunk = _RotatEScoreQueryEntities._chunk_size(
                    re_q.size(0), re_q.size(1), re_e.size(0),
                )
                grads = _RotatEScoreQueryEntities._backward_chunked(
                    re_q, im_q, re_e, im_e, grad_scores, chunk,
                )
                return (*grads, None)
        grads = _RotatEScoreQueryEntities._backward_chunked(
            re_q, im_q, re_e, im_e, grad_scores, ctx.chunk
        )
        return (*grads, None)


class RotatE(KGEBase):
    '''
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

    Entities are complex (double_entity_embedding); relations are real
    phases (not doubled).

    score_mode:
      - 'distance' (NegSamp / AllNeg): margin_gamma - ||h o r - t||_1
      - 'cosine' (KGAU-family): cosine(query, target) after L2-normalize,
        matching KGAU alignment on normalized query/target embeddings.
    '''

    SCORE_MODES = ('distance', 'cosine')

    def __init__(self, embedding_range, margin_gamma, score_mode='distance'):
        if score_mode not in self.SCORE_MODES:
            raise ValueError(
                'RotatE score_mode must be one of {}, got {}'.format(
                    self.SCORE_MODES, score_mode
                )
            )
        self.embedding_range = embedding_range
        self.margin_gamma = margin_gamma
        self.score_mode = score_mode
        self.pi = 3.14159265358979323846

    def _relation_rotation(self, relation):
        # Map relation embeddings to phases in [-pi, pi], then to unit complex.
        phase_relation = relation / (self.embedding_range.item() / self.pi)
        return torch.cos(phase_relation), torch.sin(phase_relation)

    def query_encoder(self, head, relation, tail, mode):
        re_relation, im_relation = self._relation_rotation(relation)
        if mode == 'head-batch':
            # Inverse rotation: r^{-1} o t  (since |r|=1, r^{-1}=conj(r))
            re_tail, im_tail = self._split_complex(tail)
            re_query = re_relation * re_tail + im_relation * im_tail
            im_query = re_relation * im_tail - im_relation * re_tail
        else:
            # Forward rotation: h o r
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
        # Use stack(...).norm (finite grad at 0) -- NOT hypot, which yields
        # NaN grads when re=im=0. That happens once 1vsAll/KvsAll fits positives.
        return torch.stack([re_diff, im_diff], dim=0).norm(dim=0).sum(dim=-1)

    @staticmethod
    def _cosine_score(query, target):
        '''Cosine similarity after L2-normalize along the embedding dim.'''
        query = F.normalize(query, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        return (query * target).sum(dim=-1)

    def score(self, head, relation, tail, mode):
        query = self.query_encoder(head, relation, tail, mode)
        target = self.target_encoder(head, relation, tail, mode)
        if self.score_mode == 'cosine':
            return self._cosine_score(query, target)
        re_query, im_query = self._split_complex(query)
        re_target, im_target = self._split_complex(target)
        return self.margin_gamma.item() - self._complex_distance(
            re_query - re_target, im_query - im_target
        )

    def score_query_entities(self, query, entity_embedding):
        '''
        Score queries against all entity embeddings without expanding to
        [B, E, D] via forward()/index_select. Same contract as
        ComplEx.score_query_entities: query [B, D], entity_embedding [E, D]
        -> scores [B, E].

        cosine: L2-normalize then matmul (KGAU).
        distance: custom autograd / Triton path (NegSamp / AllNeg-style
        full-entity).
        '''
        if self.score_mode == 'cosine':
            query_n = F.normalize(query, p=2, dim=-1)
            entity_n = F.normalize(entity_embedding, p=2, dim=-1)
            return query_n @ entity_n.transpose(0, 1)

        re_q, im_q = self._split_complex(query)
        re_e, im_e = self._split_complex(entity_embedding)
        gamma = self.margin_gamma.detach().to(dtype=query.dtype, device=query.device)
        return _RotatEScoreQueryEntities.apply(re_q, im_q, re_e, im_e, gamma)


KGE_SCORERS = {
    'ComplEx': ComplEx,
    'RotatE': RotatE,
}


# =============================================================================
# KGEModel: embedding tables + train/eval loop around a KGE score function
# =============================================================================

class KGEModel(nn.Module):
    '''
    Holds the entity/relation embedding tables and delegates scoring to a
    `KGEBase` score function (`self.kge_scorer`) selected via `model_name`.
    '''

    def __init__(self, model_name, nentity, nrelation, dim, margin_gamma,
                 double_entity_embedding=False, double_relation_embedding=False,
                 score_mode='distance'):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.dim = dim
        self.epsilon = 2.0
        self.score_mode = score_mode

        self.margin_gamma = nn.Parameter(
            torch.Tensor([margin_gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin_gamma.item() + self.epsilon) / dim]),
            requires_grad=False
        )

        self.entity_dim = dim * 2 if double_entity_embedding else dim
        self.relation_dim = dim * 2 if double_relation_embedding else dim

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
            # KGAU uses cosine; NegSamp/AllNeg keep RotatE distance.
            self.kge_scorer = RotatE(
                embedding_range=self.embedding_range,
                margin_gamma=self.margin_gamma,
                score_mode=score_mode,
            )
        else:
            self.kge_scorer = KGE_SCORERS[model_name]()
            
        # Placeholder cho mô hình Generator (được sử dụng riêng bởi KBGAN Strategy)
        self.generator = None

    def query_encoder(self, head, relation, tail=None, mode='tail-batch'):
        return self.kge_scorer.query_encoder(head, relation, tail, mode)

    def target_encoder(self, tail, head=None, relation=None, mode='tail-batch'):
        return self.kge_scorer.target_encoder(head, relation, tail, mode)

    def _lookup_triple(self, sample, mode):
        '''Index into the embedding tables for `sample`, per `mode`.

        Returns (head, relation, tail) broadcastable as
        [batch, 1, dim] / [batch, negative_sample_size, dim], and the
        score-function mode to use ('head-batch' or 'tail-batch').
        '''
        if mode == 'single':
            head = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 0]
            ).unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding, dim=0, index=sample[:, 1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 2]
            ).unsqueeze(1)
            return head, relation, tail, 'tail-batch'

        if mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part.reshape(-1)
            ).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(
                self.relation_embedding, dim=0, index=tail_part[:, 1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part[:, 2]
            ).unsqueeze(1)
            return head, relation, tail, mode

        if mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part[:, 0]
            ).unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding, dim=0, index=head_part[:, 1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part.reshape(-1)
            ).view(batch_size, negative_sample_size, -1)
            return head, relation, tail, mode

        raise ValueError('mode %s not supported' % mode)

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculates the score of a batch of triples.
        In the 'single' mode, sample is a batch of triples.
        In the 'head-batch' or 'tail-batch' mode, sample consists of two
        parts. The first part is usually the positive sample. And the
        second part is the entities in the negative samples. Because
        negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        head, relation, tail, score_mode = self._lookup_triple(sample, mode)
        return self.kge_scorer.score(head, relation, tail, score_mode)

    def score_all_entities(self, positive_sample, mode='tail-batch'):
        '''
        Score each positive query against all entities.

        Avoids materializing [B, nentity, dim] candidate embeddings (OOM on
        AllNeg / 1vsAll / KvsAll). Uses a query-[B,D] x entity-[E,D] path
        when the scorer supports it; otherwise falls back to chunked
        forward().
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
        A single train step. Apply back-propagation and return the loss.
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
    def _test_step_countries(model, test_triples, args):
        '''Countries S* datasets are evaluated on AUC-PR.'''
        sample = list()
        y_true = list()
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
        return {'auc_pr': cls_metrics['pr_auc']}

    @staticmethod
    def _test_step_ranking(model, test_triples, all_true_triples, args):
        '''Standard (filtered) MRR, MR, HITS@1, HITS@3, HITS@10 evaluation.'''
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

        return rotate_ranking_metrics_from_ranks(ranks)

    @staticmethod
    def _test_step_triple_classification(model, test_triples, all_true_triples, args):
        '''
        Evaluate Value Classification (Triple Classification) task.
        Supports both 'global' and 'relation' threshold modes via args.threshold_mode.
        '''
        y_true = []
        samples = []
        relations = []
        
        for head, relation, tail in test_triples:
            samples.append((head, relation, tail))
            y_true.append(1)
            relations.append(relation)
            
            if random.random() < 0.5:
                neg_head = random.randint(0, args.nentity - 1)
                samples.append((neg_head, relation, tail))
            else:
                neg_tail = random.randint(0, args.nentity - 1)
                samples.append((head, relation, neg_tail))
            y_true.append(0)
            relations.append(relation) # Negative sample giữ nguyên relation
            
        sample_tensor = torch.LongTensor(samples)
        if args.cuda:
            sample_tensor = sample_tensor.cuda()
            
        with torch.no_grad():
            scores = model(sample_tensor, mode='single').squeeze(-1).cpu().numpy()
            
        # Lấy chế độ threshold từ args, mặc định là 'relation' vì nó cho kết quả tốt hơn
        t_mode = getattr(args, 'threshold_mode', 'relation')
        return triple_classification_metrics(y_true, scores, relations, threshold_mode=t_mode)

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets.
        '''
        model.eval()

        if args.countries:
            return KGEModel._test_step_countries(model, test_triples, args)
        elif getattr(args, 'triple_classification', False):
            return KGEModel._test_step_triple_classification(model, test_triples, all_true_triples, args)
        return KGEModel._test_step_ranking(model, test_triples, all_true_triples, args)