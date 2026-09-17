# KGEs from RotatE

Mở rộng từ [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) (ICLR 2019), repository này tích hợp khung huấn luyện **KGAU (Alignment - Uniformity)** và cơ chế sinh mẫu đối nghịch **KBGAN** cho Knowledge Graph Embeddings. Hệ thống hỗ trợ cấu hình tự động qua JSON, đa dạng hàm loss/strategy và tối ưu bộ nhớ cấp cao.

## Tính năng cốt lõi

*   **Kiến trúc mô hình:** ComplEx (yêu cầu bật `-de`, `-dr`), RotatE (yêu cầu bật `-de`).
*   **Tác vụ & Đánh giá:**
    *   Link Prediction (filtered): MRR, MR, HITS@1, 3, 10.
    *   Value / Triple Classification: Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC (Tích hợp cơ chế dò ngưỡng Hybrid tự động).
*   **Tối ưu hiệu năng:** Hỗ trợ Negative chunking (`--negative_chunk_size`), Uniformity chunking (`--uniform_pair_chunk_size`), và Triton kernel cho RotatE AllNeg nhằm chống tràn RAM GPU (OOM).

### Hàm mất mát (`--loss`)
| Mã Loss | Tính năng |
|---|---|
| `se`, `hinge`, `bce`, `ce` | Các hàm loss phân loại cơ bản (thường kết hợp chiến lược `1vsall`, `kvsall`). |
| `mr`, `bpr` | Xếp hạng biên (Margin Ranking) và Bayesian Personalized Ranking. |
| `sans` | Self-Adversarial Negative Sampling (Được bật tự động khi dùng cờ `-adv`). |
| `kgau`, `kgmau`, `kgmamu` | Họ loss KGAU (Căn chỉnh - Độ đều) kết hợp các biến thể về biên (margin). |

### Chiến lược huấn luyện (`--strategy`)
| Nhóm | Tham số | Mô tả |
|---|---|---|
| NegSamp | `uniform`, `bernoulli`, `selfadv` | Các chiến thuật lấy mẫu âm tiêu chuẩn. |
| Adversarial | `kbgan` | Dùng mô hình Generator mồi để trích xuất các *hard negatives* chất lượng cao. |
| AllNeg | `1vsall`, `kvsall` | Chấm điểm ứng viên trên toàn bộ không gian thực thể. |
| KGAU | `kgau` | Tối ưu hóa đặc thù Alignment-Uniformity chỉ trên mẫu dương. |

## Cấu trúc & Dữ liệu

Yêu cầu định dạng 5 tệp trong `data/<tên_dataset>/`: `entities.dict`, `relations.dict`, `train.txt`, `valid.txt`, `test.txt`.

```text
KGEsfromRotatE/
├── codes/                 # dataloader.py, loss.py, metrics.py, model.py, strategy.py
├── configs/               # Tệp cấu hình JSON mẫu
├── data/                  # Dataset (VD: wn18rr, fb15k_237,...)
├── nohup/                 # Scripts quản lý tác vụ chạy nền
├── visualization/         # Vẽ đường cong học tập, t-SNE
└── main.py                # Entry point (Chạy chính)
```

## Cài đặt & Sử dụng

**1. Khởi tạo môi trường:**
```bash
python -m venv .venv
source .venv/bin/activate  # Trên Windows dùng: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Huấn luyện qua JSON (Khuyến nghị):**
Hệ thống tự động sử dụng GPU, cấp đường dẫn lưu model và chạy trọn vẹn chu trình (Train/Valid/Test). Bạn có thể ghi đè tham số trực tiếp qua CLI (VD: đổi `--epochs 50` so với config gốc).
```bash
# Chạy trực tiếp (lưu log và biểu đồ)
python main.py configs/ComplEx64_WN18RR_kgau4_100e.json --epochs 50

# Chạy nền tự động quản lý qua bash script
./nohup/run.sh configs/ComplEx64_WN18RR_kgau4_100e.json --epochs 50
```

**3. Huấn luyện bằng CLI thuần:**
```bash
# Link Prediction với RotatE (SANS)
python main.py --do_train --do_valid --do_test \
  --data_path data/fb15k --model RotatE -de -adv \
  -n 256 -b 1024 -d 1000 -g 24.0 -lr 0.0001 --epochs 150

# Value / Triple Classification với KBGAN
python main.py --do_train --do_test \
  --data_path data/wn18rr --model ComplEx -de -dr \
  --strategy kbgan --generator_checkpoint path/to/generator_model \
  --triple_classification --threshold_mode relation
```

**4. Kiểm thử từ Checkpoint đã lưu:**
Hệ thống ưu tiên nạp trọng số tốt nhất (`checkpoint_best`) dựa trên tập validation.
```bash
python main.py --do_test --init_checkpoint path/to/save_folder
```

**5. Trực quan hóa t-SNE:**
Chụp lại sự phân cụm của không gian biểu diễn nhúng tại các mốc epoch cụ thể (VD: 1, 100, 200).
```bash
python -u visualization/emb_tsne.py configs/ComplEx64_WN18RR_kgau4_200e.json --tsne-epochs 1 100 200
```

## Trích dẫn
Nếu bạn sử dụng một phần hoặc toàn bộ mã nguồn liên quan đến RotatE, vui lòng trích dẫn [bài báo gốc](https://openreview.net/forum?id=HkgEQnRqYQ):
```bibtex
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019}
}
```