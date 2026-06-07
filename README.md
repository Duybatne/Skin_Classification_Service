# 🩺 Skin Classification API

Dịch vụ REST API phân loại bệnh da liễu sử dụng **FastAPI** và mô hình **EfficientNet-B3** được fine-tune trên tập dữ liệu ảnh da thật.

> **Val F1 (Macro):** 0.749 · **Macro AUC-ROC:** 0.972 · **4 nhãn phân loại**

---

## 🏷️ Nhãn phân loại

| Nhãn | Mô tả |
|------|-------|
| `acne` | Mụn trứng cá |
| `cancer` | Ung thư da |
| `scar` | Sẹo |
| `normal` | Da bình thường |

---

## 📁 Cấu trúc thư mục

```text
Skin_Classification_Service/
├── app/
│   ├── main.py                 # Khởi tạo FastAPI, mount router
│   ├── api/
│   │   ├── predict.py          # POST /api/predict  (standard + TTA)
│   │   └── health.py           # GET  /health, GET /classes
│   ├── core/
│   │   └── config.py           # Settings: MODEL_PATH, CLASSES, IMG_SIZE, ...
│   └── services/
│       ├── model.py            # EfficientNetB3Classifier + SkinClassifier
│       ├── preprocess.py       # Resize → ToTensor → Normalize (300×300)
│       └── tta.py              # Test-Time Augmentation (10 augmented views)
│
├── weights/
│   └── best_model.pth          # ← Đặt checkpoint từ Kaggle vào đây
│
├── tests/
│   ├── test_predict.py         # Integration tests (endpoints + schema)
│   └── test_model.py           # Unit tests (forward shape, preprocess)
│
├── docs/
│   └── skin-classification.ipynb   # Notebook huấn luyện gốc
│
├── .env                        # Biến môi trường (MODEL_PATH, PORT, ...)
├── Dockerfile                  # Containerization
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Cài đặt & Khởi chạy

### 1. Cài dependencies

**Conda (khuyên dùng):**
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
pip install fastapi uvicorn[standard] python-multipart python-dotenv Pillow
```

**Hoặc pip:**
```bash
pip install -r requirements.txt
```

### 2. Đặt file model

Tải `best_model.pth` từ Kaggle/checkpoint sau khi train và đặt vào:
```
weights/best_model.pth
```

> Nếu thiếu file, server vẫn khởi động nhưng sẽ chạy với trọng số ngẫu nhiên.

### 3. Cấu hình `.env` (tuỳ chọn)

```env
MODEL_PATH=weights/best_model.pth
HOST=0.0.0.0
PORT=8000
CONFIDENCE_THRESHOLD=0.5
```

### 4. Chạy server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi start, log sẽ hiển thị:
```
[SkinClassifier] Loaded 'weights/best_model.pth' (epoch=27, val_f1=0.749)
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/` | Thông tin service |
| `GET` | `/health` | Trạng thái model + device |
| `GET` | `/classes` | Danh sách nhãn phân loại |
| `POST` | `/api/predict` | Phân loại ảnh da |
| `POST` | `/api/predict?tta=true` | Phân loại với Test-Time Augmentation |

### POST `/api/predict`

**Request:**
```bash
curl -X POST http://localhost:8000/api/predict \
     -F "file=@skin_image.jpg"
```

**Response (standard):**
```json
{
  "success": true,
  "filename": "skin_image.jpg",
  "tta_enabled": false,
  "prediction": {
    "label": "acne",
    "confidence": 0.9123
  },
  "probabilities": {
    "acne": 0.9123,
    "cancer": 0.0412,
    "scar": 0.0312,
    "normal": 0.0153
  },
  "low_confidence": false
}
```

**Response (TTA — `?tta=true`):**
```bash
curl -X POST "http://localhost:8000/api/predict?tta=true" \
     -F "file=@skin_image.jpg"
```
```json
{
  "success": true,
  "tta_enabled": true,
  "prediction": { "label": "acne", "confidence": 0.9089 },
  "probabilities": { "acne": 0.9089, "cancer": 0.0421, "scar": 0.0338, "normal": 0.0152 },
  "low_confidence": false,
  "std_per_class": {
    "acne": 0.0214,
    "cancer": 0.0178,
    "scar": 0.0102,
    "normal": 0.0091
  }
}
```

> **`low_confidence: true`** — khi `confidence < 0.5` (cấu hình qua `CONFIDENCE_THRESHOLD`)  
> **`std_per_class`** — độ lệch chuẩn qua 10 augmentation; std cao = model không chắc chắn

### GET `/health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "weights/best_model.pth",
  "device": "cpu"
}
```

---

## 🧪 Kiểm thử

```bash
python -m pytest tests/ -v
```

**13 tests** bao gồm:
- Unit tests: forward shape, preprocess output, softmax sum
- Integration tests: schema validation, TTA mode, error cases, system endpoints

---

## 🐳 Docker

```bash
# Build
docker build -t skin-classifier-api .

# Run
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  --name skin-api \
  skin-classifier-api
```

> Volume mount `-v` đảm bảo checkpoint không bị COPY vào image (tránh làm nặng image).

---

## 📖 Tài liệu tương tác

| URL | Mô tả |
|-----|-------|
| http://localhost:8000/docs | Swagger UI — upload ảnh trực tiếp |
| http://localhost:8000/redoc | ReDoc — tài liệu đọc |

---

## 🔬 Thông số mô hình

| Tham số | Giá trị |
|---------|---------|
| Architecture | EfficientNet-B3 (ImageNet pretrained) |
| Fine-tuning | 2-phase: freeze → unfreeze 3 blocks cuối |
| Loss | Focal Loss (γ=2, class-weighted) |
| Input size | 300 × 300 px |
| Normalization | ImageNet mean/std |
| Optimizer | AdamW (differential LR) |
| Scheduler | CosineAnnealingLR |
| Best Val F1 (Macro) | 0.749 |
| Macro AUC-ROC | 0.972 |
| Training set | 8,056 ảnh |
| Test set | 1,727 ảnh |
