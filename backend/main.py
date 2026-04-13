
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
from pathlib import Path
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
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
_class_names: Optional[List[str]] = None
_config: Optional[Dict] = None


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _class_names, _config

    print("[Backend] Starting up...")
    model_dir   = os.getenv("MODEL_DIR",   "model_v1")
    labels_path = os.getenv("LABELS_PATH", "labels.npy")
    config_path = os.getenv("CONFIG_PATH", "training_config.json")

    # Load model
    try:
        keras_path = os.path.join(model_dir, "model.keras")
        best_path  = os.path.join(model_dir, "best.keras")

        if os.path.exists(keras_path):
            load_path = keras_path
        elif os.path.exists(best_path):
            load_path = best_path
        else:
            load_path = model_dir

        print(f"[Backend] Loading model from: {load_path}")
        _model = tf.keras.models.load_model(load_path, compile=False)
        _model.trainable = False

        # Warm-up pass
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _model.predict(dummy, verbose=0)
        print("[Backend] Model loaded and warmed up.")

    except Exception as e:
        print(f"[Backend] WARNING: Model load failed: {e}")
        print("[Backend] Running in demo mode.")

    # Load labels
    try:
        _class_names = np.load(labels_path, allow_pickle=True).tolist()
        print(f"[Backend] Labels: {len(_class_names)} classes → {_class_names}")
    except Exception as e:
        _class_names = [str(i) for i in range(10)] + \
                       [chr(c) for c in range(ord('A'), ord('Z') + 1)]
        print(f"[Backend] Using default labels. Error: {e}")

    # Load config
    try:
        with open(config_path) as f:
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


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "num_classes": len(_class_names) if _class_names else 0,
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
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            engine.save_to_file(text, tmp.name)
            engine.runAndWait()
            engine.stop()
            return tmp.name

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
