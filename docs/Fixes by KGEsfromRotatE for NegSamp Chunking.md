# Fixes by KGEsfromRotatE for NegSamp Chunking

Áp dụng cho: Uniform / Bernoulli / Self-Adversarial (SANS) negative sampling.

---

## 0. Mục tiêu (spec)

**Vấn đề:** Score một lần cả `N` negatives → materialize `[B, N, D]` → peak GPU cao.

**Yêu cầu:**

1. Vẫn sample đủ `N` negatives (sampler không đổi).
2. Score theo chunk chiều N, độ rộng `C`.
3. Khi train và `C < N`: dùng `torch.utils.checkpoint` mỗi chunk.
4. `cat` → `neg_scores [B, N]`.
5. SelfAdv: `softmax` trên **đủ N** (không theo từng chunk).
6. **Một** scalar loss → **một** `backward`.

Semantics loss phải giữ nguyên so với không chunk (cùng N, cùng công thức weight).

---

## 1. File cần điều chỉnh

| File | Việc cần làm |
|---|---|
| **`negsamp_strategy.py`** | Chỗ chính: resolve `C`, `score_negatives` (chunk + checkpoint + cat), gọi loss một lần |
| **`adversarial_bce_loss.py`** / loss utilities | Loss chỉ nhận full `[B,N]` scores + weights; bỏ path “chunk bên trong loss” / partial losses nếu đang dùng |
| `filtered_1_to_n_sampler.py` | **Không cần sửa** (vẫn trả `[B, N]`) |
| `complex.py` | **Không cần sửa** |
| `lookup_embedder.py` | **Không cần sửa** |

---

## 2. Các điều chỉnh cụ thể (A–F)

| # | Điều chỉnh | Hành động |
|---|---|---|
| **A** | Score negatives **một lần** rồi `cat` | Xóa pattern: score `no_grad` full N lấy adversarial weight → score lại từng chunk. Thay bằng: chunk → checkpoint (train) → `cat` → weight → loss. |
| **B** | Chunk chỉ ở strategy | Thêm / gom `resolve_negative_chunk_size` + `score_negatives`. `train_batch` gọi `score_negatives` rồi `loss_fn(pos, neg_full, weights)`. Loss không tự chia chunk. |
| **C** | Convention `negative_chunk_size` | **Default = 256**. **`≤ 0` = tắt chunk** (`C = N`). Không dùng `0` cho “auto VRAM”. Muốn auto thì thêm flag riêng (ví dụ `negative_chunk_auto`). |
| **D** | SelfAdv weight sau `cat` | `w = softmax(neg_scores * temperature, dim=1).detach()` với `neg_scores` shape `[B, N]`. Uniform: `w = 1/N`. |
| **E** | Một backward | Bỏ `loss_parts` + `retain_graph` cho NegSamp. `loss.backward()` một lần mỗi batch. |
| **F** *(tuỳ chọn)* | Uniform BCE | Nếu còn path cộng `-logsigmoid(-neg).sum` theo chunk mà không chia `1/N`: đổi sang weight `1/N` hoặc `.mean(dim=1)` tương đương. Bỏ qua nếu chỉ chạy SelfAdv. |

---

## 3. Pseudo-code triển khai

```python
# negative_chunk_size: int, default 256
# <= 0  → tắt chunk (C = N)
# >  0  → C = min(negative_chunk_size, N)

def resolve_negative_chunk_size(n_neg, negative_chunk_size):
    if n_neg <= 0:
        return 0
    c = int(negative_chunk_size or 0)
    if c <= 0:
        return n_neg          # tắt chunk
    return min(c, n_neg)


def score_negatives(model, pos, neg, mode, training):
    # pos: positive batch; neg: [B, N] entity ids
    B, N = neg.shape
    C = resolve_negative_chunk_size(N, negative_chunk_size)

    if C >= N:
        return model_score(pos, neg, mode)   # một forward full N

    chunks = []
    for start in range(0, N, C):
        end = min(start + C, N)
        neg_c = neg[:, start:end]

        def _fwd(p, n, _mode=mode):
            return model_score(p, n, _mode)

        if training:
            from torch.utils.checkpoint import checkpoint
            s = checkpoint(_fwd, pos, neg_c, use_reentrant=False)
        else:
            s = _fwd(pos, neg_c)
        chunks.append(s)
    return torch.cat(chunks, dim=1)          # [B, N]


def train_step(...):
    pos_batch, neg_batch, subsample_w, mode = sample(...)
    pos_scores = score_positives(...)
    neg_scores = score_negatives(model, pos_batch, neg_batch, mode, training=True)

    if self_adv:
        w = softmax(neg_scores * temperature, dim=1).detach()
    else:
        w = full_like(neg_scores, 1.0 / neg_scores.size(1))

    loss = sans_or_bce_loss(pos_scores, neg_scores, w, subsample_w)
    loss.backward()   # một lần
    optimizer.step()
```

`model_score` = API score head/tail-batch sẵn có (`score_emb` / `score_hrt` …) — giữ nguyên, chỉ bọc chunk bên ngoài.

---

## 4. Cách tự kiểm tra

| # | Cách verify | Đạt khi |
|---|---|---|
| **A** | Counter / hook số lần score negatives mỗi batch | ≈ `ceil(N/C)`, không ≈ `2×ceil(N/C)` |
| **B** | Đọc `train_batch` | Có `cat` → một `loss_fn(...)`; không loop `loss_parts` |
| **C** | Chạy với `negative_chunk_size=0` và `=256` | `0` = 1 forward full N; `256` = nhiều chunk |
| **D** | `w.sum(dim=1) ≈ 1` | Softmax trên N, không trên C |
| **E** | Trace backward | Một `backward` / batch, không chuỗi `retain_graph=True` |
| **F** | Chỉ nếu còn non-adv chunked path | Tương đương weight `1/N` |

### Smoke test

1. `N=1024`, `C=256` → đúng 4 chunk; vài step không NaN/Inf.
2. Cùng seed: `C=0` vs `C=256` → loss step đầu gần bằng (sai số số học nhỏ OK).
3. SelfAdv: `w.shape[-1] == N`.

### Không cần chỉnh thêm nếu

- Sampler / ComplEx / embedder không liên quan chunking.
- Chỉ train SelfAdv → có thể bỏ **F**.
- Không cần auto theo VRAM nếu đã dùng default `C=256`.

---

## 5. Kỳ vọng sau khi chỉnh

| Metric | Kỳ vọng | Ghi chú |
|---|---|---|
| **Peak GPU memory** | **Giảm** so với forward full `[B,N,D]` không chunk | Peak ~ một chunk `[B,C,D]` (+ checkpoint recompute) |
| | Có thể **cao hơn một chút** so với bản cũ (double-score + partial backward) | Đổi lấy code đơn giản + 1 lần score |
| **Time / epoch** | **Giảm** so với bản cũ SelfAdv (bỏ lần score `no_grad` thứ hai) | |
| | Hơi **chậm hơn** so với một forward full N không chunk | Tradeoff bình thường |
| **MRR / Hits** | **Không đổi chủ đích** | Cùng N, cùng full-N softmax, cùng loss |

**Tóm lại:** Đúng semantics + chunk/checkpoint đơn giản; không tối ưu peak tuyệt đối. Kỳ vọng: nhanh hơn path SelfAdv cũ, peak ổn định hơn full-N, chất lượng giữ nguyên.
