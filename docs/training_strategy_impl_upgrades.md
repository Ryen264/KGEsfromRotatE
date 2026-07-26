# Training Strategy Implementation Upgrades

Giảm peak GPU memory, **giữ nguyên semantics loss** (so sánh strategy vẫn fair).

Setup tham chiếu: ComplEx `dim=500` (`D=1000` với double-emb), `batch_size=512`, NegSamp `N=1024`, WN18RR `|E|≈40943`.

---

## 1. NegSamp — Chunked Negative Scoring with Gradient Checkpointing

**Strategies:** `uniform`, `bernoulli`, `selfadv`

**Vấn đề:** materialize `[B, N, D]` một lần → peak cao (~5.5 GB).

**Các bước:**
1. Sample đủ `N` negatives như cũ.
2. Score theo chunk chiều N, độ rộng `C`.
3. Khi train và `C < N`: `torch.utils.checkpoint` mỗi chunk.
4. `cat` → `negative_score [B, N]`; weight như cũ (Self-Adv softmax trên đủ N).

**Code:** `codes/strategy.py` (`NegSampStrategy.score_negatives`)

| Parameter | Default | Ý nghĩa |
|---|---|---|
| `negative_chunk_size` | **256** | Max negatives / chunk; `<=0` = không chunk (dùng hết N) |

`C = min(negative_chunk_size, N)`. Khi `C ≥ N` → không chunk.

---

## 2. AllNeg — Full-Entity Query–Entity Matmul Scoring

**Strategies:** `1vsall`, `kvsall`

**Vấn đề:** expand `[B, E, D]` → OOM.

**Các bước:**
1. Labels: 1vsAll one-hot; KvsAll multi-hot.
2. Encode query `[B, D]` một lần.
3. `scores[B,E] =` matmul query × full entity table (`score_query_entities`).
4. Loss trên `[B, E]` + labels.

**Code:** `codes/model.py` (`score_all_entities`), `codes/strategy.py` (`AllNegStrategy`)

| Parameter | Default | Ý nghĩa |
|---|---|---|
| *(không có)* | luôn bật với ComplEx | — |

**Lưu ý:** `negative_sample_size` / `negative_chunk_size` **không dùng**. Candidates = `nentity`, không phải `N`.

---

## 3. KGAU — Chunked Pairwise Uniformity (Exact i&lt;j Reduction)

**Loss/strategy:** `kgau` / `kgmau` / `kgmamu` + strategy `kgau`

**Vấn đề:** `torch.pdist` backward ~`O(n²·D)` (entity term `n=2B=1024` → ~3.9 GB OOM).

**Các bước:**
1. Positives-only (không score negatives / full E).
2. Thay `pdist` bằng block cặp `[C,C]`, chỉ `i < j`.
3. Sau normalize: `‖a−b‖² = 2 − 2 a·b`.
4. Cộng dồn `mean` rồi `log` — **exact** cùng công thức cũ.

**Code:** `codes/loss.py` (`chunked_pairwise_uniformity`, …)

| Parameter | Default | Ý nghĩa |
|---|---|---|
| `uniform_pair_chunk_size` | **256** | Độ rộng block cặp; `<=0` = không block (dùng hết n, dễ OOM) |

`C = min(uniform_pair_chunk_size, n)`.

---

## Tóm tắt

| Nhóm | Tên kỹ thuật | Hyperparam | Default |
|---|---|---|---|
| NegSamp | Chunked Negative Scoring + Checkpoint | `negative_chunk_size` | **256** |
| AllNeg | Query–Entity Matmul Scoring | — | luôn bật |
| KGAU | Chunked Pairwise Uniformity | `uniform_pair_chunk_size` | **256** |

Ví dụ trong JSON:

```json
"negative_chunk_size": 256,
"uniform_pair_chunk_size": 256
```
