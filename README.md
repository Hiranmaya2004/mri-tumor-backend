# 🧠 MRI Brain Tumor Classifier — Backend API

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Hiranmaya2004/mri-tumor-backend)

A production-ready **FastAPI** backend that classifies brain MRI images into four tumor categories using a fine-tuned **ResNet-50** model served via **ONNX Runtime**.

## 🔬 Model Details

| Property | Value |
|---|---|
| **Architecture** | ResNet-50 (ImageNet pre-trained) |
| **Classes** | Glioma, Meningioma, No Tumor, Pituitary |
| **Test Accuracy** | 94.19% (on 1,600 images) |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Framework** | PyTorch → ONNX export |
| **Inference** | ONNX Runtime (CPU) |

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Hiranmaya2004/mri-tumor-backend.git
cd mri-tumor-backend
```

### 2. Set up Python environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Run the server

```bash
python app.py
```

The API will be available at **http://localhost:8000**

### 4. Test it

```bash
# Health check
curl http://localhost:8000/health

# Classify an MRI image
curl -X POST http://localhost:8000/predict -F "file=@your_mri_scan.png"
```

## 📡 API Endpoints

### `GET /`
Welcome message with API information.

### `GET /health`
Health check — confirms the model is loaded and ready.

**Response:**
```json
{
  "status": "healthy",
  "model": "brain_tumor_classifier.onnx",
  "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
}
```

### `POST /predict`
Classify a brain MRI image.

**Request:** Multipart form upload with key `file`

**Response:**
```json
{
  "prediction": "Glioma",
  "confidence": 0.9412,
  "probabilities": {
    "Glioma": 0.9412,
    "Meningioma": 0.0321,
    "No Tumor": 0.0156,
    "Pituitary": 0.0111
  }
}
```

## 🐳 Docker

```bash
# Build
docker build -t mri-backend .

# Run
docker run -p 8000:8000 mri-backend
```

## ☁️ Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo `Hiranmaya2004/mri-tumor-backend`
4. Render will auto-detect the `render.yaml` configuration
5. Click **Deploy** — your API will be live at `https://mri-tumor-backend.onrender.com`

> **Note:** On the free tier, the service spins down after 15 min of inactivity. First request after idle takes ~30s to cold-start.

## 📁 Project Structure

```
mri-tumor-backend/
├── app.py                # FastAPI application + ONNX inference
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container deployment
├── render.yaml           # Render.com deployment config
├── .gitignore            # Python ignores
├── .gitattributes        # Git LFS for model files
├── README.md             # This file
└── model/
    ├── brain_tumor_classifier.onnx       # ONNX model header
    └── brain_tumor_classifier.onnx.data  # ONNX model weights (~94 MB)
```

## 🛠️ Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — Modern async Python web framework
- **[ONNX Runtime](https://onnxruntime.ai/)** — High-performance ML inference engine
- **[Pillow](https://pillow.readthedocs.io/)** — Image preprocessing
- **[NumPy](https://numpy.org/)** — Tensor operations
- **[Uvicorn](https://www.uvicorn.org/)** — ASGI server

## 🔗 Related

- **Frontend Demo**: [MRI Saliency Stability Analyzer](https://github.com/Hiranmaya2004/mri-saliency-demo) — Interactive web demo that calls this API

## 👥 Research Team

Department of CSE, ITER, Siksha 'O' Anusandhan University, Bhubaneswar, India

- Prakash Chandra Sahoo (Supervisor)
- Hiranmaya Panda
- Yagya Saini
- Naman Kumar
- Gourab Panda
- Janmejaya Panda

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
