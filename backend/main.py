
import io
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import base64
import asyncio
import tempfile
import time
import json
import uuid
import warnings
from pathlib import Path
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()  

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# ─── Globals ──────────────────────────────────────────────────────────────────
_prediction_history: deque = deque(maxlen=100)
_session_stats = {
    "total_predictions": 0,
    "confident_predictions": 0,
    "session_start": datetime.now().isoformat(),
    "label_counts": {}
}
_model: Optional[tf.keras.Model] = None
_landmark_model: Optional[tf.keras.Model] = None
_class_names: Optional[List[str]] = None
_config: Optional[Dict] = None
_model_load_error: Optional[str] = None


class PatchedInputLayer(tf.keras.layers.InputLayer):
    """
    Compatibility shim for older TensorFlow/Keras versions.

    Some models saved with newer Keras include `optional` and `batch_shape`
    keys in the InputLayer config, which older deserializers may not accept.
    """

    def __init__(
        self,
        input_shape=None,
        batch_shape=None,
        optional=None,  # newer Keras keyword; ignored for inference
        **kwargs,
    ):
        # Convert `batch_shape=[None, H, W, C]` -> input_shape=(H,W,C)
        if input_shape is None and batch_shape is not None:
            try:
                input_shape = tuple(batch_shape[1:])
            except Exception:
                pass

        # Strip newer/unrecognized keys before delegating.
        kwargs.pop("batch_shape", None)
        kwargs.pop("optional", None)

        super().__init__(input_shape=input_shape, **kwargs)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _landmark_model, _class_names, _config, _model_load_error

    print("[Backend] Starting up...")
    base_dir = Path(__file__).resolve().parent
    model_dir = (base_dir / os.getenv("MODEL_DIR", "model_v1")).resolve()
    labels_path = (base_dir / os.getenv("LABELS_PATH", "labels.npy")).resolve()
    config_path = (base_dir / os.getenv("CONFIG_PATH", "training_config.json")).resolve()

    _model_load_error = None

    # Load model
    try:
        # Support both `.keras` and `.h5` exports.
        # Priority: best.* then model.*; otherwise fall back to a single file if present.
        # Prefer `.h5` first (your latest model change), then fall back to `.keras`.
        candidates = [
            model_dir / "best.h5",
            model_dir / "model.h5",
            model_dir / "best.keras",
            model_dir / "model.keras",
        ]
        load_path = next((p for p in candidates if p.exists()), None)
        if load_path is None:
            h5_files = sorted(model_dir.glob("*.h5"))
            keras_files = sorted(model_dir.glob("*.keras"))
            all_files = keras_files + h5_files
            load_path = all_files[0] if len(all_files) == 1 else model_dir

        print(f"[Backend] Loading model from: {load_path}")
        # 1) Normal load first
        _model = tf.keras.models.load_model(str(load_path), compile=False)
        _model.trainable = False

        # Warm-up pass
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _model.predict(dummy, verbose=0)
        print("[Backend] Model loaded and warmed up.")

    except Exception as e:
        # 2) Retry with compatibility shim
        # Your reported failure is a config kwarg mismatch on InputLayer.
        try:
            _model = tf.keras.models.load_model(
                str(load_path),
                compile=False,
                safe_mode=False,
                custom_objects={"InputLayer": PatchedInputLayer},
            )
            _model.trainable = False

            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _model.predict(dummy, verbose=0)
            _model_load_error = None
            print("[Backend] Model loaded and warmed up (after retry).")
        except Exception as e2:
            _model = None
            _model_load_error = f"{type(e2).__name__}: {e2}"
            print(f"[Backend] WARNING: Image model load failed: {_model_load_error}")

    # Load landmark model (Stage A)
    landmark_model_path = base_dir / "landmark_model.keras"
    if landmark_model_path.exists():
        try:
            _landmark_model = tf.keras.models.load_model(str(landmark_model_path), compile=False)
            _landmark_model.trainable = False
            dummy = np.zeros((1, 63), dtype=np.float32)
            _landmark_model.predict(dummy, verbose=0)
            print("[Backend] Landmark model loaded and warmed up.")
        except Exception as e:
            print(f"[Backend] WARNING: Landmark model load failed: {e}")
    else:
        print("[Backend] Landmark model not found yet (waiting for training).")

    # Load labels
    try:
        _class_names = np.load(str(labels_path), allow_pickle=True).tolist()
        # Avoid non-ASCII characters in logs (some Windows consoles may choke),
        # which would otherwise trigger the fallback to default labels.
        print(f"[Backend] Labels loaded: {len(_class_names)} classes")
    except Exception as e:
        _class_names = [str(i) for i in range(10)] + \
                       [chr(c) for c in range(ord('A'), ord('Z') + 1)]
        print(f"[Backend] Using default labels. Error: {e}")

    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _config = json.load(f)
    except:
        _config = {"backbone": "mobilenetv2", "confidence_threshold": 0.60}

    print(f"[Backend] Ready — {len(_class_names)} classes.")
    yield
    print("[Backend] Shutdown.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ISL Sign Language Chatbot API",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class Base64PredictRequest(BaseModel):
    image: str
    confidence_threshold: float = 0.60


class LandmarksPredictRequest(BaseModel):
    landmarks: List[float] = Field(..., description="63 normalized landmark floats")
    confidence_threshold: float = 0.85


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "hi"


class TTSRequest(BaseModel):
    text: str
    rate: int = 150
    volume: float = 1.0


# ─── Inference Helper ─────────────────────────────────────────────────────────
def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    backbone = _config.get("backbone", "mobilenetv2") if _config else "mobilenetv2"
    img_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img      = tf.image.resize(img_rgb, (224, 224))
    img      = tf.cast(img, tf.float32)
    if backbone == "mobilenetv2":
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    else:
        img = tf.keras.applications.efficientnet.preprocess_input(img)
    return tf.expand_dims(img, 0).numpy()


def _run_inference(image_bgr: np.ndarray, confidence_threshold: float = 0.60) -> Dict:
    import random

    if _model is None:
        # Demo mode
        label = random.choice(_class_names or ["A"])
        conf  = random.uniform(0.65, 0.99)
        return {
            "label": label, "confidence": round(conf, 4),
            "top3": [[label, round(conf, 4)]],
            "latency_ms": 10.0, "above_threshold": True,
            "raw_predicted_label": label,
            "prediction_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat()
        }

    processed = _preprocess(image_bgr)
    t0    = time.perf_counter()
    probs = _model.predict(processed, verbose=0)[0]
    ms    = (time.perf_counter() - t0) * 1000

    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    top3  = [[_class_names[i], round(float(probs[i]), 4)]
             for i in np.argsort(probs)[::-1][:3]]
    above = conf >= confidence_threshold
    label = _class_names[idx] if above else "UNCERTAIN"

    _session_stats["total_predictions"] += 1
    if above:
        _session_stats["confident_predictions"] += 1
        _session_stats["label_counts"][_class_names[idx]] = \
            _session_stats["label_counts"].get(_class_names[idx], 0) + 1

    result = {
        "label": label, "confidence": round(conf, 4),
        "top3": top3, "latency_ms": round(ms, 2),
        "above_threshold": above,
        "raw_predicted_label": _class_names[idx],
        "prediction_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat()
    }
    _prediction_history.append(result)
    return result


def _run_landmark_inference(features: List[float], confidence_threshold: float = 0.60) -> Dict:
    import random
    if _landmark_model is None:
        # Demo mode if model not built yet
        label = random.choice(_class_names or ["A"])
        conf  = random.uniform(0.65, 0.99)
        return {
            "label": label, "confidence": round(conf, 4),
            "top3": [[label, round(conf, 4)]],
            "latency_ms": 1.0, "above_threshold": True,
            "raw_predicted_label": label,
            "prediction_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "source": "demo_landmarks"
        }

    # Input format: (1, 63)
    input_data = np.array(features, dtype=np.float32).reshape(1, -1)
    
    t0 = time.perf_counter()
    probs = _landmark_model.predict(input_data, verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    top3 = [[_class_names[i], round(float(probs[i]), 4)]
             for i in np.argsort(probs)[::-1][:3]]
    
    above = conf >= confidence_threshold
    label = _class_names[idx] if above else "UNCERTAIN"

    _session_stats["total_predictions"] += 1
    if above:
        _session_stats["confident_predictions"] += 1
        _session_stats["label_counts"][_class_names[idx]] = \
            _session_stats["label_counts"].get(_class_names[idx], 0) + 1

    result = {
        "label": label, "confidence": round(conf, 4),
        "top3": top3, "latency_ms": round(ms, 2),
        "above_threshold": above,
        "raw_predicted_label": _class_names[idx],
        "prediction_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "source": "landmarks"
    }
    _prediction_history.append(result)
    return result


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "num_classes": len(_class_names) if _class_names else 0,
        "model_load_error": _model_load_error,
        "session_stats": _session_stats
    }


@app.get("/labels")
async def labels():
    return {"labels": _class_names, "count": len(_class_names)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(default=0.60)
):
    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image.")
    return _run_inference(img, confidence_threshold)


@app.post("/predict/base64")
async def predict_base64(payload: Base64PredictRequest):
    b64 = payload.image
    if "," in b64:
        b64 = b64.split(",")[1]
    try:
        img_bytes = base64.b64decode(b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Decode failed")
    except Exception as e:
        raise HTTPException(400, f"Base64 error: {e}")
    return _run_inference(img, payload.confidence_threshold)


@app.post("/predict/landmarks")
async def predict_landmarks(payload: LandmarksPredictRequest):
    """
    Inference endpoint using extracted MediaPipe hand landmarks.
    Expects exactly 63 floats (21 landmarks * 3 coords).
    """
    if len(payload.landmarks) != 63:
        raise HTTPException(400, f"Expected 63 floats, got {len(payload.landmarks)}")
    
    return _run_landmark_inference(payload.landmarks, payload.confidence_threshold)


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Cohere chatbot for ISL users.
    Supports multi-turn conversation using Cohere API.
    """
    api_key = os.getenv("COHERE_API_KEY", "").strip()
    
    # Debug logging
    print(f"[Chat] Cohere API key present: {bool(api_key)}")
    if api_key:
        print(f"[Chat] API key length: {len(api_key)}")
        print(f"[Chat] API key prefix: {api_key[:10]}...")
    else:
        print("[Chat] WARNING: No Cohere API key found in environment")
    
    if not api_key:
        # Demo mode response (no API key)
        return {
            "response": (
                f'I received your message: "{request.message}".\n\n'
                "To enable AI responses, please set your Cohere API key in the .env file.\n"
                "Get a key at: https://dashboard.cohere.com/"
            ),
            "model": "demo-mode",
            "requires_api_key": True
        }

    try:
        # Import Cohere
        import cohere
        
        # Initialize Cohere client
        client = cohere.Client(api_key)
        
        # Prepare conversation history for Cohere
        chat_history = []
        
        # Convert existing history to Cohere format
        for msg in (request.history or []):
            # Cohere expects "USER" and "CHATBOT" roles (uppercase)
            role = "USER" if msg.role.lower() == "user" else "CHATBOT"
            chat_history.append({
                "role": role,
                "message": msg.content
            })
        
        # System prompt for context
        system_prompt = (
            "You are a helpful AI assistant for Indian Sign Language (ISL) users. "
            "Users communicate via ISL gestures converted to text — input may be "
            "individual letters, short words, or abbreviations. "
            "Always try to interpret the intended meaning and respond naturally. "
            "Keep responses concise and conversational. "
            "If input is unclear, ask a short clarifying question."
        )
        
        # Use Cohere's chat endpoint with a current live model
        response = client.chat(
            model="command-a-03-2025",  # Live model (updated Sep 2025)
            message=request.message,
            chat_history=chat_history,
            preamble=system_prompt,
            temperature=0.7,
            max_tokens=512,
        )
        
        # Safely extract response data
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        return {
            "response": response_text,
            "model": "cohere-command-a-03-2025",
            "conversation_id": getattr(response, 'conversation_id', None),
            "response_id": getattr(response, 'response_id', None)
        }
        
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Cohere library not installed. Run: pip install cohere"
        )
    except Exception as e:
        # Extract detailed error if available
        error_msg = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            try:
                error_detail = json.loads(e.response.text)
                error_msg = error_detail.get('message', error_msg)
            except:
                pass
        print(f"[Chat] Cohere error: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Cohere API error: {error_msg}"
        )


@app.post("/translate")
async def translate(request: TranslateRequest):
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(
            source=request.source_lang,
            target=request.target_lang
        ).translate(request.text)
        return {
            "original": request.text,
            "translated": translated,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang
        }
    except ImportError:
        raise HTTPException(500, "Install: pip install deep-translator")
    except Exception as e:
        raise HTTPException(500, f"Translation error: {e}")


@app.post("/tts")
async def tts(request: TTSRequest):
    try:
        import pyttsx3
        loop = asyncio.get_event_loop()

        def _synth(text, rate, volume):
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd) # Crucial: close the handle so Windows allows pyttsx3 to write
            
            engine.save_to_file(text, path)
            engine.runAndWait()
            engine.stop()
            
            return path

        path = await loop.run_in_executor(
            None, _synth, request.text, request.rate, request.volume
        )
        return FileResponse(path, media_type="audio/wav", filename="speech.wav")

    except ImportError:
        raise HTTPException(500, "Install: pip install pyttsx3")
    except Exception as e:
        raise HTTPException(500, f"TTS error: {e}")


# ─── Test endpoint for Cohere API ────────────────────────────────────────────
@app.get("/test-cohere")
async def test_cohere():
    """
    Quick test to verify Cohere API connection.
    """
    api_key = os.getenv("COHERE_API_KEY", "").strip()
    
    if not api_key:
        return {
            "status": "error",
            "message": "COHERE_API_KEY not found in environment variables"
        }
    
    try:
        import cohere
        client = cohere.Client(api_key)
        response = client.chat(
            model="command-a-03-2025",
            message="Hello! Please respond with 'Cohere API is working correctly!'"
        )
        return {
            "status": "success",
            "response": response.text,
            "model_used": "command-a-03-2025"
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Cohere library not installed. Run: pip install cohere"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cohere API error: {str(e)}"
        }