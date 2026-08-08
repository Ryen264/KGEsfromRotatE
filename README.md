# KGEs from RotatE

## Giới thiệu

Mã nguồn này phát triển từ [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) (ICLR 2019) để thử nghiệm và mở rộng các mô hình nhúng đồ thị tri thức cho **khung huấn luyện Knowledge Graph Alignment - Uniformity (KGAU)**.

Ngoài cài đặt khung huấn luyện KGAU cho ComplEx và RotatE, mã nguồn bổ sung nhiều hàm mất mát, chiến lược huấn luyện (lấy mẫu âm / toàn mẫu âm), tối ưu bộ nhớ (chunking), cùng công cụ huấn luyện theo tệp cấu hình JSON và trực quan hóa quá trình học.

## Tính năng đã triển khai

### Mô hình

- [x] **ComplEx** (cần `-de` và `-dr`: nhân đôi embedding thực thể và quan hệ)
- [x] **RotatE** (cần `-de`; quan hệ là pha thực, không nhân đôi `-dr`)

### Hàm mất mát (`--loss`)

| Mã | Tên | Ghi chú |
|---|---|---|
| `se` | Bình phương sai số (*Squared Error*) | |
| `hinge` | Hinge theo điểm | |
| `bce` | Entropy chéo nhị phân (*Binary Cross-Entropy*) | Thường kèm `1vsall` / `kvsall` |
| `mr` | Xếp hạng biên (*Margin Ranking*) | |
| `bpr` | Bayesian Personalized Ranking | |
| `ce` | Entropy chéo nhiều lớp (*Cross-Entropy*) | |
| `sans` | Lấy mẫu âm đối kháng tự thích nghi (*Self-Adversarial NS*) | Mặc định gốc RotatE khi bật `-adv` |
| `kgau` | Căn chỉnh–Độ đều (*Alignment–Uniformity*) | Họ KGAU |
| `kgmau` | KGAU với biên căn chỉnh | |
| `kgmamu` | KGAU với biên căn chỉnh và biên độ đều | |

### Chiến lược huấn luyện (`--strategy`)

| Nhóm | Giá trị | Mô tả ngắn |
|---|---|---|
| Lấy mẫu âm (NegSamp) | `uniform`, `bernoulli`, `selfadv` | Mỗi bộ ba dương kèm \(N\) thực thể âm |
| Toàn bộ âm (AllNeg) | `1vsall`, `kvsall` | Chấm điểm trên toàn bộ tập thực thể |
| KGAU | `kgau` | Chiến lược buấn luyện KGAU |

### Chỉ số đánh giá

- Dự đoán liên kết (*link prediction*, filtered): **MRR**, **MR**, **HITS@1**, **HITS@3**, **HITS@10**
- Phân loại bộ ba (*triple classification*): accuracy, precision, recall, F1, PR-AUC, ROC-AUC

### Tối ưu hiệu năng / bộ nhớ

- **Chunking lấy mẫu âm** (`--negative_chunk_size`, mặc định `256`): chia việc chấm điểm mẫu âm, kết hợp gradient checkpointing
- **Chunking cặp độ đều** (`--uniform_pair_chunk_size`, mặc định `256`): tính độ phủ đều theo khối cặp, giữ đúng mọi cặp \(i < j\)
- RotatE AllNeg: nhân kernel Triton (kèm fallback PyTorch) để tăng tốc chấm điểm toàn thực thể

## Cấu trúc thư mục

```
KGEsfromRotatE/
├── codes/                 # Huấn luyện và đánh giá chính
│   ├── run.py             # CLI: đọc args, vòng lặp train/valid/test
│   ├── loss.py            # Các hàm mất mát (gồm hàm mất mát KGAU)
│   ├── strategy.py        # Chiến lược NegSamp / AllNeg / KGAU
│   ├── dataloader.py      # Dataset / DataLoader
│   └── metrics/           # Độ đo xếp hạng và phân loại
├── configs/               # Tệp cấu hình JSON (ComplEx/RotatE × loss × dataset)
├── data/                  # Bộ dữ liệu KGE
├── visualization/         # Huấn luyện theo config + biểu đồ / t-SNE
│   ├── main.py            # Đường cong học: căn chỉnh–độ đều, loss, MRR
│   └── emb_tsne.py        # Ảnh chụp t-SNE embedding theo epoch
├── nohup/                 # Chạy nền (log, pid)
├── run.sh                 # Tiện ích tìm siêu tham số (kiểu RotatE gốc)
└── best_config.sh         # Cấu hình tái hiện bài báo RotatE
```

## Định dạng dữ liệu

Mỗi bộ dữ liệu trong `data/<tên>/` gồm:

| Tệp | Vai trò |
|---|---|
| `entities.dict` | Ánh xạ thực thể → id |
| `relations.dict` | Ánh xạ quan hệ → id |
| `train.txt` | Bộ ba huấn luyện |
| `valid.txt` | Bộ ba kiểm định (tạo tệp rỗng nếu không có) |
| `test.txt` | Bộ ba kiểm thử |

Các bộ có sẵn ví dụ: `wn18rr`, `fb15k_237`, `fb15k`, `wn18`, `countries_s1/s2/s3`, `yago3_10`, `hetionet`.

## Cài đặt

1. Tạo môi trường ảo và kích hoạt:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
```

2. Cài phụ thuộc:

```bash
pip install -r requirements.txt
```

## Huấn luyện

### Cách 1 — Theo tệp cấu hình JSON (khuyến nghị)

Các thí nghiệm trong repo thường dùng `visualization/main.py` (vẽ đường cong học) hoặc chạy nền qua `nohup/`:

```bash
# Huấn luyện + lưu biểu đồ căn chỉnh/độ đều và loss/MRR
.venv/bin/python -u visualization/main.py configs/ComplEx64_WN18RR_kgau4_100e.json --gpu 0 --no-show

# Chạy nền
./nohup/run.sh configs/ComplEx64_WN18RR_kgau4_100e.json --gpu 0 --no-show
```

Ví dụ cấu hình KGAU (`configs/ComplEx64_WN18RR_kgau4_100e.json`):

```json
{
  "model": "ComplEx",
  "data_path": "data/wn18rr",
  "loss": "kgau",
  "strategy": "kgau",
  "dim": 64,
  "batch_size": 512,
  "margin_gamma": 200.0,
  "learning_rate": 0.002,
  "epochs": 100,
  "double_entity_embedding": true,
  "double_relation_embedding": true,
  "uniform_t": 4,
  "uniform_pair_chunk_size": 256,
  "uniform_gamma_q": 1.0,
  "uniform_gamma_y": 1.0,
  "uniform_gamma_e": 1.0
}
```

### Cách 2 — CLI `codes/run.py`

Ví dụ huấn luyện RotatE trên FB15k với GPU 0 (phong cách gốc RotatE):

```bash
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
  --cuda \
  --do_valid \
  --do_test \
  --data_path data/fb15k \
  --model RotatE \
  -n 256 -b 1024 -d 1000 \
  -g 24.0 -a 1.0 -adv \
  -lr 0.0001 --epochs 150 \
  --test_batch_size 16 -de
```

Ví dụ ComplEx + KGAU trên WN18RR:

```bash
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train --cuda \
  --do_valid --do_test \
  --data_path data/wn18rr \
  --model ComplEx -de -dr \
  --loss kgau --strategy kgau \
  -d 64 -b 512 -g 200.0 -lr 0.002 --epochs 100 \
  --uniform_t 4 --uniform_pair_chunk_size 256 \
  --uniform-gamma-q 1.0 --uniform-gamma-y 1.0 --uniform-gamma-e 1.0 \
  -r 5e-6 -rp 3
```

Xem đầy đủ tham số trong `codes/run.py` (`parse_args`).

### Tái hiện kết quả bài báo RotatE

Chạy các lệnh trong `best_config.sh` để tái hiện RotatE / ComplEx trên các bộ chuẩn (theo mã nguồn gốc). Tìm siêu tham số nhanh:

```bash
bash run.sh train RotatE fb15k 0 0 1024 256 1000 24.0 1.0 0.0001 150 16 -de
```

## Kiểm thử từ checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_test --cuda
```

Checkpoint tốt nhất theo chỉ số kiểm định (best-valid) được ưu tiên khi đánh giá cuối (xem log `Test checkpoint: best-valid`).

## Trực quan hóa bản nhúng (t-SNE)

`visualization/emb_tsne.py` huấn luyện theo config và lưu ảnh t-SNE tại các epoch mốc (mặc định: epoch 1, giữa, cuối — với 200 epoch là 1 / 100 / 200). Mỗi màu là một query `(head, relation)`; mỗi điểm là một trong top-\(k\) đuôi theo **điểm số mô hình tại đúng epoch đó** (không phải gold label).

```bash
.venv/bin/python -u visualization/emb_tsne.py \
  configs/ComplEx64_WN18RR_kgau4_200e.json \
  --tsne-epochs 1 100 200 --gpu 0

# Chạy nền
./nohup/run_emb_tsne.sh configs/ComplEx64_WN18RR_kgau4_200e.json --tsne-epochs 1 100 200 --gpu 0
```

Kết quả mặc định nằm trong `visualization/outputs/<tên_config>_<timestamp>/`.

## Tham số quan trọng

| Tham số | Ý nghĩa |
|---|---|
| `--model` | `ComplEx` hoặc `RotatE` |
| `--loss` / `--strategy` | Hàm mất mát và chiến lược (bảng trên) |
| `-d` / `--dim` | Số chiều embedding |
| `-b` / `--batch_size` | Kích thước batch |
| `-n` / `--negative_sample_size` | Số mẫu âm (NegSamp) |
| `--negative_chunk_size` | Kích thước chunk khi chấm điểm mẫu âm (`≤0` = tắt) |
| `-g` / `--margin_gamma` | Biên / hệ số gamma của mô hình |
| `--uniform_t` | Nhiệt độ \(t\) trong độ đều |
| `--uniform_pair_chunk_size` | Chunk cặp khi tính độ đều |
| `--uniform-gamma-q/y/e` | Trọng số độ đều trên query / target / entity |
| `-r` / `-rp` | Hệ số và bậc chuẩn hóa \(L_p\) embedding |
| `--epochs` | Số chu kỳ huấn luyện |
| `-de` / `-dr` | Nhân đôi embedding thực thể / quan hệ |

## Kiến trúc mã nguồn

Ba đối tượng chính:

- **TrainDataset / TestDataset** (`dataloader.py`, và dataset trong `strategy.py`): luồng dữ liệu huấn luyện và đánh giá
- **KGEModel** (`model.py`): embedding, bộ chấm điểm (query/target encoder), API `train_step` / `test_step`
- **Chiến lược + hàm mất mát** (`strategy.py`, `loss.py`): chuẩn bị batch và tính loss

`codes/run.py` chứa hàm chính: phân tích tham số, đọc dữ liệu, khởi tạo mô hình và vòng lặp huấn luyện.

Thêm scorers mới trong `model.py` (đăng ký vào `KGE_SCORERS`), ví dụ khung:

```python
def score(self, head, relation, tail, mode):
    # trả về điểm số bộ ba; điểm cao = hợp lệ hơn
    ...
```

## Trích dẫn

Nếu sử dụng mã nguồn dựa trên RotatE, hãy trích dẫn [bài báo](https://openreview.net/forum?id=HkgEQnRqYQ):

```
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
```
