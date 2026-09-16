# KGEs from RotatE

## Giới thiệu

Repository này được phát triển và mở rộng từ mã nguồn gốc [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) (ICLR 2019). Mục tiêu chính của dự án là thử nghiệm, thiết kế và tích hợp **khung huấn luyện KGAU (Knowledge Graph Alignment - Uniformity)**, đồng thời hỗ trợ các cơ chế sinh mẫu đối nghịch nâng cao (như **KBGAN**) cho các tác vụ biểu diễn Đồ thị Tri thức (Knowledge Graph Embeddings).

Ngoài việc cài đặt khung huấn luyện KGAU cho ComplEx và RotatE, mã nguồn còn được bổ sung hàng loạt hàm mất mát, chiến lược huấn luyện (lấy mẫu âm / toàn mẫu âm), tối ưu hóa bộ nhớ (chunking), cùng hệ thống cấu hình linh hoạt qua JSON và trực quan hóa quá trình học.

## Tính năng đã triển khai

### Kiến trúc Mô hình
- [x] **ComplEx** (Yêu cầu bật cờ `-de` và `-dr` để nhân đôi không gian nhúng của thực thể và quan hệ).
- [x] **RotatE** (Yêu cầu bật cờ `-de`; riêng quan hệ được biểu diễn dưới dạng pha thực nên không dùng `-dr`).

### Hàm mất mát (`--loss`)

| Mã Loss | Tên gọi | Ghi chú |
|---|---|---|
| `se` | Bình phương sai số (*Squared Error*) | |
| `hinge` | Hinge Loss theo điểm | |
| `bce` | Entropy chéo nhị phân (*Binary Cross-Entropy*) | Thường dùng kèm chiến lược `1vsall` / `kvsall`. |
| `mr` | Xếp hạng biên (*Margin Ranking*) | |
| `bpr` | Bayesian Personalized Ranking | |
| `ce` | Entropy chéo nhiều lớp (*Cross-Entropy*) | |
| `sans` | Lấy mẫu âm đối kháng tự thích nghi (*Self-Adversarial NS*) | Mặc định của RotatE (kích hoạt qua cờ `-adv`). |
| `kgau` | Căn chỉnh – Độ đều (*Alignment – Uniformity*) | Thuộc họ KGAU. |
| `kgmau` | KGAU kết hợp biên căn chỉnh | |
| `kgmamu`| KGAU kết hợp biên căn chỉnh và biên độ đều | |

### Chiến lược huấn luyện (`--strategy`)

| Nhóm | Giá trị (`--strategy`) | Mô tả |
|---|---|---|
| Lấy mẫu âm (NegSamp) | `uniform`, `bernoulli`, `selfadv` | Mỗi bộ ba dương (positive) đi kèm N thực thể âm (negative). |
| Sinh mẫu đối nghịch | `kbgan` | Sử dụng Generator mồi để trích xuất các *hard negatives* chất lượng cao. |
| Toàn bộ âm (AllNeg) | `1vsall`, `kvsall` | Chấm điểm ứng viên trên toàn bộ không gian tập thực thể. |
| KGAU | `kgau` | Chiến lược huấn luyện đặc thù sử dụng positive samples để tối ưu Alignment-Uniformity. |

### Tác vụ & Chỉ số đánh giá
- **Dự đoán liên kết (Link prediction, filtered):** MRR, MR, HITS@1, HITS@3, HITS@10.
- **Phân loại Giá trị / Phân loại Bộ ba (Value / Triple Classification):** Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC. 
  *(Tích hợp cơ chế tự động dò ngưỡng Hybrid: Relation-specific thresholds kết hợp Global fallback).*

### Tối ưu Hiệu năng / Bộ nhớ
- **Chunking lấy mẫu âm** (`--negative_chunk_size`, mặc định `256`): Chia nhỏ quá trình chấm điểm mẫu âm kết hợp Gradient Checkpointing để chống tràn RAM GPU (OOM).
- **Chunking độ đều KGAU** (`--uniform_pair_chunk_size`, mặc định `256`): Tính toán độ phủ đều theo khối cặp, tối ưu bộ nhớ nhưng vẫn đảm bảo trọn vẹn thuật toán.
- **Tăng tốc RotatE AllNeg**: Tích hợp nhân kernel Triton (với fallback PyTorch) để tăng tốc độ chấm điểm trên toàn bộ thực thể.

## Cấu trúc thư mục

```text
KGEsfromRotatE/
├── codes/                 # Mã nguồn lõi (Kiến trúc, Huấn luyện, Đánh giá)
│   ├── __init__.py
│   ├── dataloader.py      # Xử lý Dataset và DataLoader
│   ├── loss.py            # Cài đặt các hàm mất mát (KGAU, BCE, Margin, SANS,...)
│   ├── metrics.py         # Tổng hợp các độ đo xếp hạng và phân loại
│   ├── model.py           # Kiến trúc KGEModel, query/target encoders, hàm train/test
│   └── strategy.py        # Chiến lược huấn luyện (NegSamp, AllNeg, KBGAN, KGAU)
├── configs/               # Các tệp cấu hình JSON mẫu cho từng bộ dữ liệu và mô hình
├── data/                  # Thư mục chứa các bộ dữ liệu Đồ thị tri thức
├── docs/                  # Tài liệu và ghi chú của dự án
├── nohup/                 # Scripts hỗ trợ chạy nền và lưu trữ log
├── visualization/         # Công cụ trực quan hóa (Learning curves, t-SNE)
│   ├── emb_tsne.py        
│   └── main.py            
├── .gitignore
├── LICENSE
├── main.py                # ENTRY POINT: File chạy chính của toàn bộ dự án
└── requirements.txt       # Danh sách thư viện phụ thuộc
```

## Định dạng dữ liệu

Mỗi bộ dữ liệu đặt trong thư mục `data/<tên_dataset>/` bắt buộc phải bao gồm 5 tệp:

| Tệp | Vai trò |
|---|---|
| `entities.dict` | Từ điển ánh xạ: Tên thực thể → ID |
| `relations.dict` | Từ điển ánh xạ: Tên quan hệ → ID |
| `train.txt` | Tập bộ ba dùng để huấn luyện |
| `valid.txt` | Tập kiểm định (Tạo tệp rỗng nếu không sử dụng) |
| `test.txt` | Tập kiểm thử (Đánh giá mô hình cuối cùng) |

*Một số bộ dataset tiêu biểu: `wn18rr`, `fb15k_237`, `countries_s1/s2/s3`, `yago3_10`, `hetionet`.*

## Cài đặt Môi trường

1. Tạo và kích hoạt môi trường ảo (Virtual Environment):

```bash
python -m venv .venv

# Trên Linux/macOS
source .venv/bin/activate

# Trên Windows (PowerShell)
.venv\Scripts\activate
```

2. Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

## Hướng dẫn Chạy mô hình (Huấn luyện & Đánh giá)

Tất cả các lệnh thực thi đều được gọi từ thư mục gốc (root) thông qua file `main.py`.

### Cách 1 — Chạy theo tệp cấu hình JSON (Khuyến nghị)
Sử dụng công cụ trong thư mục `visualization` hoặc `nohup` để chạy dựa trên config định sẵn, tự động vẽ đường cong học tập (Learning curves).

```bash
# Huấn luyện + lưu biểu đồ căn chỉnh/độ đều, loss, MRR
python -u visualization/main.py configs/ComplEx64_WN18RR_kgau4_100e.json --gpu 0 --no-show

# Hoặc chạy nền bằng script nohup
./nohup/run.sh configs/ComplEx64_WN18RR_kgau4_100e.json --gpu 0 --no-show
```

### Cách 2 — Chạy trực tiếp qua CLI `main.py`

**Ví dụ 1: Huấn luyện RotatE trên tập FB15k (Phong cách gốc):**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --do_train --do_valid --do_test --cuda \
  --data_path data/fb15k \
  --model RotatE -de \
  -n 256 -b 1024 -d 1000 \
  -g 24.0 -a 1.0 -adv \
  -lr 0.0001 --epochs 150 \
  --test_batch_size 16 
```

**Ví dụ 2: Huấn luyện ComplEx với Loss KGAU trên WN18RR:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --do_train --do_valid --do_test --cuda \
  --data_path data/wn18rr \
  --model ComplEx -de -dr \
  --loss kgau --strategy kgau \
  -d 64 -b 512 -g 200.0 -lr 0.002 --epochs 100 \
  --uniform_t 4 --uniform_pair_chunk_size 256 \
  --uniform-gamma-q 1.0 --uniform-gamma-y 1.0 --uniform-gamma-e 1.0 \
  -r 5e-6 -rp 3
```

**Ví dụ 3: Đánh giá Value/Triple Classification sử dụng KBGAN:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --do_train --do_test --cuda \
  --data_path data/wn18rr \
  --model ComplEx -de -dr \
  --strategy kbgan \
  --generator_checkpoint path/to/generator/checkpoint \
  --triple_classification \
  --threshold_mode relation
```

## Kiểm thử từ Checkpoint đã lưu

Sử dụng cờ `--init_checkpoint` trỏ tới thư mục chứa file `checkpoint` hoặc `checkpoint_best`. Hệ thống sẽ ưu tiên nạp trọng số tốt nhất dựa trên tập validation.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --do_test --cuda --init_checkpoint path/to/save_folder
```

## Trực quan hóa Không gian Nhúng (t-SNE)

Chạy script `emb_tsne.py` để chụp lại không gian nhúng của đồ thị tri thức tại các mốc epoch cụ thể. Tính năng này giúp quan sát sự dịch chuyển của các cụm thực thể (clusters) trong quá trình mô hình học.

```bash
python -u visualization/emb_tsne.py \
  configs/ComplEx64_WN18RR_kgau4_200e.json \
  --tsne-epochs 1 100 200 --gpu 0
```
*Kết quả ảnh xuất ra mặc định nằm trong: `visualization/outputs/<tên_config>_<timestamp>/`.*

## Các Tham số Command-Line Quan trọng

| Cờ (Flag) | Ý nghĩa |
|---|---|
| `--model` | Chọn kiến trúc: `ComplEx` hoặc `RotatE`. |
| `--loss` | Chỉ định hàm mất mát (VD: `sans`, `kgau`, `bce`,...). |
| `--strategy` | Chiến lược huấn luyện (VD: `uniform`, `kbgan`, `1vsall`, `kgau`). |
| `-d` / `--dim` | Số chiều (Dimension) của vector nhúng. |
| `-b` / `--batch_size` | Kích thước Batch Size huấn luyện. |
| `-n` / `--negative_sample_size` | Số lượng mẫu âm sinh ra cho mỗi mẫu dương (áp dụng cho NegSamp). |
| `--triple_classification`| Bật chế độ đánh giá Value Classification thay vì Link Prediction. |
| `--generator_checkpoint`| Bắt buộc cung cấp đường dẫn checkpoint mô hình mồi nếu dùng `--strategy kbgan`. |
| `--uniform_t` | Siêu tham số nhiệt độ T dùng trong uniformity loss. |
| `-de` / `-dr` | Cờ nhân đôi kích thước không gian biểu diễn cho thực thể / quan hệ. |

## Trích dẫn (Citation)

Nếu bạn sử dụng một phần hoặc toàn bộ mã nguồn liên quan đến RotatE, vui lòng trích dẫn bài báo gốc theo định dạng sau:

```bibtex
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
```