"""
MRI Brain Tumor Classifier — Backend API
=========================================
FastAPI server that loads a ResNet-50 ONNX model and serves
real-time brain tumor classification predictions.

Classes: Glioma | Meningioma | No Tumor | Pituitary
"""

import io
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "brain_tumor_classifier.onnx"

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ImageNet normalization (used during training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 224

# ── Global session holder ──────────────────────────────────
ort_session: ort.InferenceSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX model once at startup."""
    global ort_session

    if not MODEL_PATH.exists():
        logger.error("Model file not found at %s", MODEL_PATH)
        raise FileNotFoundError(
            f"ONNX model not found at {MODEL_PATH}. "
            "Place brain_tumor_classifier.onnx (and .onnx.data) in the model/ directory."
        )

    logger.info("Loading ONNX model from %s …", MODEL_PATH)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Use CPU provider (safe default; add CUDAExecutionProvider for GPU)
    providers = ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(
        str(MODEL_PATH), sess_options, providers=providers
    )
    logger.info(
        "Model loaded successfully. Input: %s  Output: %s",
        ort_session.get_inputs()[0].name,
        ort_session.get_outputs()[0].name,
    )
    yield
    logger.info("Shutting down — releasing ONNX session.")
    ort_session = None


# ── FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title="MRI Brain Tumor Classifier API",
    description=(
        "Upload a brain MRI image and receive a tumor classification prediction. "
        "Powered by a fine-tuned ResNet-50 model exported to ONNX."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow the frontend to call this API ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Preprocessing ─────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image bytes into a model-ready float32 tensor.

    Pipeline (mirrors training):
        1. Open image → convert to RGB
        2. Resize to 224×224 (bilinear)
        3. Scale pixel values to [0, 1]
        4. Normalize with ImageNet mean & std
        5. Transpose to NCHW layout: (1, 3, 224, 224)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not open image: {exc}",
        )

    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

    # HWC float32 in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Normalize
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW → NCHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


# ── Routes ─────────────────────────────────────────────────
@app.get("/", tags=["General"])
async def root():
    """Welcome message with API info."""
    return {
        "service": "MRI Brain Tumor Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload an MRI image for classification",
            "GET /health": "Health check",
        },
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check — confirms the model is loaded."""
    return {
        "status": "healthy" if ort_session is not None else "model_not_loaded",
        "model": str(MODEL_PATH.name),
        "classes": CLASS_NAMES,
    }


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Classify a brain MRI image.

    **Accepts:** PNG, JPG, JPEG image uploads  
    **Returns:** Predicted class, confidence, and per-class probabilities.
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got {file.content_type}",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Preprocess
    input_tensor = preprocess_image(image_bytes)

    # Inference
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    raw_output = ort_session.run([output_name], {input_name: input_tensor})

    logits = raw_output[0]  # shape: (1, 4)
    probabilities = softmax(logits)[0]  # shape: (4,)

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    logger.info(
        "Prediction: %s (%.2f%%) | Probs: %s",
        predicted_class,
        confidence * 100,
        {c: f"{p:.4f}" for c, p in zip(CLASS_NAMES, probabilities)},
    )

    return JSONResponse(
        content={
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(prob), 4)
                for name, prob in zip(CLASS_NAMES, probabilities)
            },
        }
    )


# ── Run directly ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
