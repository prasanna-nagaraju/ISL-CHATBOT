// ISLChatbot.jsx
// Merged recognizer + chatbot interface
// Recognized signs → directly sent to Cohere AI

import { useState, useEffect, useRef, useCallback } from "react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import "./ISLChatbot.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const INDIAN_LANGS = [
  { code: "hi",  label: "Hindi",     native: "हिन्दी" },
  { code: "ta",  label: "Tamil",     native: "தமிழ்" },
  { code: "te",  label: "Telugu",    native: "తెలుగు" },
  { code: "kn",  label: "Kannada",   native: "ಕನ್ನಡ" },
  { code: "ml",  label: "Malayalam", native: "മലയാളം" },
  { code: "mr",  label: "Marathi",   native: "मराठी" },
  { code: "bn",  label: "Bengali",   native: "বাংলা" },
  { code: "gu",  label: "Gujarati",  native: "ગુજરાતી" },
  { code: "pa",  label: "Punjabi",   native: "ਪੰਜਾਬੀ" },
  { code: "or",  label: "Odia",      native: "ଓଡ଼ିଆ" },
  { code: "en",  label: "English",   native: "English" },
];

const INFERENCE_MS = 600;

export default function ISLChatbot({ onBack }) {
  // Camera
  const videoRef   = useRef(null);
  const streamRef  = useRef(null);
  const intervalRef = useRef(null);
  const audioRef   = useRef(null);

  const [isCamOn,     setIsCamOn]     = useState(false);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [camError,    setCamError]    = useState(null);

  // Recognition
  const [currentSign, setCurrentSign]     = useState(null);
  const [signedText,  setSignedText]      = useState("");
  const lastSignRef = useRef({ label: null, count: 0 });

  // MediaPipe Tasks Vision HandLandmarker (same model as Python training)
  const landmarkerRef = useRef(null);
  const [isHandPresent, setIsHandPresent] = useState(false);

  const inferenceInFlightRef = useRef(false);

  // Chat
  const [messages,  setMessages]  = useState([
    {
      role: "model",
      content: "Namaste! 🙏 I'm your ISL AI assistant. Start signing to build your message, then press Send. I'll respond to your Indian Sign Language input.",
      timestamp: Date.now()
    }
  ]);
  const [isThinking, setIsThinking] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(85);
  const chatEndRef = useRef(null);

  // Translation + TTS
  const [targetLang,  setTargetLang]  = useState("hi");
  const [isSpeaking,  setIsSpeaking]  = useState(false);
  const [translating, setTranslating] = useState(null); // message index being translated
  const [translations, setTranslations] = useState({}); // { msgIndex: translatedText }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  // ── Camera ────────────────────────────────────────────────────────

  // ── Initialize MediaPipe Tasks Vision HandLandmarker ────────────────
  // This uses the EXACT SAME model as Python's mp.tasks.vision.HandLandmarker
  // so landmarks are guaranteed to match what the classifier was trained on.
  useEffect(() => {
    let cancelled = false;
    async function initLandmarker() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const lm = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5
        });
        if (!cancelled) {
          landmarkerRef.current = lm;
          console.log("[ISL] HandLandmarker (Tasks Vision) initialized.");
        }
      } catch (e) {
        console.error("[ISL] HandLandmarker init failed:", e);
      }
    }
    initLandmarker();
    return () => { cancelled = true; };
  }, []);

  const startCamera = async () => {
    setCamError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setIsCamOn(true);
    } catch (e) {
      setCamError("Camera access denied. Please allow camera permission.");
    }
  };

  const stopCamera = () => {
    stopRecognizing();
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCamOn(false);
    setCurrentSign(null);
    setIsHandPresent(false);
  };

  // ── Recognition loop (Tasks Vision Landmarks) ──────────────────────

  // Normalize landmarks exactly how Python train_landmarks.py does
  const normalizeLandmarks = useCallback((landmarks) => {
    // landmarks is an array of {x, y, z} from @mediapipe/tasks-vision
    // This is IDENTICAL format to Python's mp.tasks.vision output
    const wrist = landmarks[0];

    // 1. Translate: center on wrist
    const translated = landmarks.map(lm => ({
      x: lm.x - wrist.x,
      y: lm.y - wrist.y,
      z: lm.z - wrist.z
    }));

    // 2. Flatten Z to 0 (matches Python training)
    for (const lm of translated) lm.z = 0;

    // 3. Scale: normalize by max distance from wrist (2D)
    let maxDist = 0;
    for (const lm of translated) {
      const dist = Math.sqrt(lm.x * lm.x + lm.y * lm.y);
      if (dist > maxDist) maxDist = dist;
    }

    // 4. Flatten to 63 floats
    const features = [];
    for (const lm of translated) {
      if (maxDist > 0) {
        features.push(lm.x / maxDist);
        features.push(lm.y / maxDist);
        features.push(0);
      } else {
        features.push(0, 0, 0);
      }
    }
    return features;
  }, []);

  // Hidden canvas used to flip the webcam frame before feeding to MediaPipe
  const flipCanvasRef = useRef(null);

  const runInference = useCallback(async () => {
    if (inferenceInFlightRef.current) return;
    const v = videoRef.current;
    if (!v || v.readyState < 2 || !landmarkerRef.current) return;

    // The webcam is in selfie/mirror mode. The Kaggle dataset images are NOT mirrored.
    // We must flip the frame horizontally before running landmark detection so
    // the extracted coordinates match the non-mirrored training data.
    let canvas = flipCanvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      flipCanvasRef.current = canvas;
    }
    const vw = v.videoWidth || 640;
    const vh = v.videoHeight || 480;
    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext("2d");
    ctx.translate(vw, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(v, 0, 0, vw, vh);
    ctx.setTransform(1, 0, 0, 1, 0, 0); // reset

    // Run the Tasks Vision HandLandmarker on the FLIPPED frame
    const result = landmarkerRef.current.detectForVideo(canvas, performance.now());

    if (!result.landmarks || result.landmarks.length === 0) {
      setIsHandPresent(false);
      setCurrentSign(null);
      lastSignRef.current = { label: null, count: 0 };
      return;
    }

    setIsHandPresent(true);
    const handLm = result.landmarks[0]; // array of {x, y, z}

    inferenceInFlightRef.current = true;
    const features = normalizeLandmarks(handLm);

    try {
      const res = await fetch(`${API}/predict/landmarks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ landmarks: features, confidence_threshold: confidenceThreshold / 100 })
      });
      if (!res.ok) return;

      const data = await res.json();
      setCurrentSign(data);

      if (data.above_threshold) {
        const last = lastSignRef.current;
        if (data.label === last.label) {
          last.count++;
          if (last.count === 2) {
            setSignedText(prev => prev + data.label);
            last.count = 0;
          }
        } else {
          lastSignRef.current = { label: data.label, count: 1 };
        }
      } else {
        lastSignRef.current = { label: null, count: 0 };
      }
    } catch { /* ignore network errors */ }
    finally {
      inferenceInFlightRef.current = false;
    }
  }, [normalizeLandmarks]);

  const startRecognizing = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setIsRecognizing(true);
    intervalRef.current = setInterval(runInference, 300);
  }, [runInference]);

  const stopRecognizing = () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    setIsRecognizing(false);
  };

  useEffect(() => () => stopCamera(), []);

  // ── Send signed message to chatbot ────────────────────────────────

  const sendToChatbot = async () => {
    const text = signedText.trim();
    if (!text || isThinking) return;

    const userMsg = { role: "user", content: text, timestamp: Date.now() };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setSignedText("");
    setIsThinking(true);

    try {
      // Build history (exclude first greeting)
      const history = updated.slice(1, -1).map(m => ({
        role: m.role,
        content: m.content
      }));

      const res  = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history })
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }

      const data = await res.json();
      setMessages(prev => [...prev, {
        role: "model",
        content: data.response || "Sorry, I could not process that.",
        timestamp: Date.now(),
        model: data.model
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: "model",
        content: `⚠️ Error: ${e.message}. Make sure COHERE_API_KEY is set in your .env file.`,
        timestamp: Date.now(),
        isError: true
      }]);
    } finally {
      setIsThinking(false);
    }
  };

  // ── Translation ───────────────────────────────────────────────────

  const translateMessage = async (text, msgIndex) => {
    setTranslating(msgIndex);
    try {
      const res  = await fetch(`${API}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, source_lang: "en", target_lang: targetLang })
      });
      const data = await res.json();
      setTranslations(prev => ({ ...prev, [msgIndex]: data.translated }));
    } catch {
      setTranslations(prev => ({ ...prev, [msgIndex]: "Translation failed." }));
    } finally {
      setTranslating(null);
    }
  };

  const translateSignedText = async () => {
    if (!signedText.trim()) return;
    setTranslating("input");
    try {
      const res  = await fetch(`${API}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: signedText, source_lang: "en", target_lang: targetLang })
      });
      const data = await res.json();
      setTranslations(prev => ({ ...prev, input: data.translated }));
    } catch {
      setTranslations(prev => ({ ...prev, input: "Translation failed." }));
    } finally {
      setTranslating(null);
    }
  };

  // ── TTS ───────────────────────────────────────────────────────────

  const speak = async (text) => {
    if (!text || isSpeaking) return;
    setIsSpeaking(true);
    try {
      const res  = await fetch(`${API}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, rate: 150, volume: 1.0 })
      });
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      audioRef.current.src = url;
      audioRef.current.play();
      audioRef.current.onended = () => { setIsSpeaking(false); URL.revokeObjectURL(url); };
    } catch {
      // Fallback to browser TTS
      if ("speechSynthesis" in window) {
        const u = new SpeechSynthesisUtterance(text);
        u.onend = () => setIsSpeaking(false);
        window.speechSynthesis.speak(u);
      } else {
        setIsSpeaking(false);
      }
    }
  };

  // ── Confidence color ──────────────────────────────────────────────
  const confColor = (c) => c >= 0.75 ? "#22c55e" : c >= 0.5 ? "#f59e0b" : "#ef4444";

  const formatTime = (ts) =>
    new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <div className="isl-chatbot">
      <audio ref={audioRef} style={{ display: "none" }} />

      {/* ── Header ────────────────────────────────────────────── */}
      <header className="chatbot-header">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <div className="header-title">
          <span className="header-logo">🤟</span>
          <div>
            <div className="header-name">ISL Chatbot</div>
            <div className="header-sub">Indian Sign Language · AI Assistant</div>
          </div>
        </div>
        <div className="lang-selector">
          <label>Translate to</label>
          <select
            className="lang-select"
            value={targetLang}
            onChange={e => setTargetLang(e.target.value)}
          >
            {INDIAN_LANGS.map(l => (
              <option key={l.code} value={l.code}>
                {l.native} ({l.label})
              </option>
            ))}
          </select>
        </div>
      </header>

      {/* ── Main layout ───────────────────────────────────────── */}
      <div className="chatbot-body">

        {/* Left: Camera panel */}
        <div className="camera-panel">

          {/* Video feed */}
          <div className={`video-wrap ${isRecognizing ? "active" : ""}`}>
            {!isCamOn && (
              <div className="video-placeholder">
                <span className="placeholder-hand">🤟</span>
                <p>Camera is off</p>
              </div>
            )}
            <video
              ref={videoRef}
              className={`video-feed ${isCamOn ? "show" : ""}`}
              muted playsInline autoPlay
            />
            {isRecognizing && (
              <div className="scan-corners">
                <div className="sc tl" /><div className="sc tr" />
                <div className="sc bl" /><div className="sc br" />
                <div className="scan-bar" />
                <div className="roi-hint">
                  {isHandPresent ? "Hand detected" : "Place hand in view"}
                </div>
              </div>
            )}
          </div>

          {/* Camera controls */}
          <div className="cam-controls">
            {!isCamOn ? (
              <button className="btn btn-primary" onClick={startCamera}>
                📷 Start Camera
              </button>
            ) : (
              <>
                <button
                  className={`btn ${isRecognizing ? "btn-danger" : "btn-primary"}`}
                  onClick={isRecognizing ? stopRecognizing : startRecognizing}
                >
                  {isRecognizing ? "⏹ Stop" : "🎯 start showing gesture"}
                </button>
                <button className="btn btn-ghost" onClick={stopCamera}>
                  ✕ Camera
                </button>
              </>
            )}
          </div>

          {camError && <div className="cam-error">{camError}</div>}

          {/* Current sign display */}
          <div className="current-sign-card">
            <div className="sign-label">Detecting</div>
            <div
              className="sign-letter"
              style={{ color: currentSign?.above_threshold ? "var(--saffron)" : "var(--text-muted)" }}
            >
              {currentSign?.above_threshold ? currentSign.label : "—"}
            </div>
            {currentSign && (
              <div className="sign-conf" style={{ color: confColor(currentSign.confidence) }}>
                {(currentSign.confidence * 100).toFixed(1)}% confidence
              </div>
            )}
          </div>

          {/* Message builder — type OR sign */}
          <div className="text-builder">
            <div className="text-builder-header">
              <span className="tb-label">Message <span className="tb-hint">(type or sign)</span></span>
              <button className="icon-btn danger" onClick={() => { setSignedText(""); setTranslations(prev => ({ ...prev, input: "" })); }} title="Clear">✕ Clear</button>
            </div>

            <textarea
              className="text-input"
              value={signedText}
              onChange={e => setSignedText(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendToChatbot();
                }
              }}
              placeholder="Sign gestures to build text, or type directly here...&#10;Press Enter to send, Shift+Enter for new line."
              rows={3}
            />

            {/* Translation of input */}
            {translations.input && (
              <div className="translation-bubble">
                <span className="trans-lang">{INDIAN_LANGS.find(l => l.code === targetLang)?.native}</span>
                {translations.input}
              </div>
            )}

            {/* Actions */}
            <div className="tb-bottom">
              <div className="tb-tool-row">
                <button className="tool-btn" onClick={translateSignedText}
                  disabled={!signedText || translating === "input"}>
                  {translating === "input" ? "..." : "🌐 Translate"}
                </button>
                <button className="tool-btn" onClick={() => speak(signedText)}
                  disabled={!signedText || isSpeaking}>
                  {isSpeaking ? "🔊 Speaking..." : "🔊 Speak"}
                </button>
                <button className="tool-btn" onClick={() => navigator.clipboard.writeText(signedText)}
                  disabled={!signedText}>
                  📋 Copy
                </button>
              </div>
              <button
                className="btn btn-send"
                onClick={sendToChatbot}
                disabled={!signedText.trim() || isThinking}
              >
                {isThinking ? <span className="spinner" /> : "Send to AI →"}
              </button>
            </div>
          </div>
        </div>

        {/* Right: Chat panel */}
        <div className="chat-panel">
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`msg-wrap ${msg.role === "user" ? "user" : "bot"}`}>
                {msg.role === "model" && (
                  <div className="msg-avatar">🤖</div>
                )}
                <div className={`msg-bubble ${msg.role === "user" ? "user-bubble" : "bot-bubble"} ${msg.isError ? "error-bubble" : ""}`}>
                  <div className="msg-text">{msg.content}</div>

                  {/* Translation of this message */}
                  {translations[i] && (
                    <div className="translation-bubble">
                      <span className="trans-lang">{INDIAN_LANGS.find(l => l.code === targetLang)?.native}</span>
                      {translations[i]}
                    </div>
                  )}

                  {/* Message actions */}
                  <div className="msg-actions">
                    <span className="msg-time">{formatTime(msg.timestamp)}</span>
                    <button
                      className="msg-action-btn"
                      onClick={() => translateMessage(msg.content, i)}
                      disabled={translating === i}
                      title="Translate"
                    >
                      {translating === i ? "..." : "🌐"}
                    </button>
                    <button
                      className="msg-action-btn"
                      onClick={() => speak(translations[i] || msg.content)}
                      disabled={isSpeaking}
                      title="Speak"
                    >
                      🔊
                    </button>
                  </div>
                </div>
                {msg.role === "user" && (
                  <div className="msg-avatar user-av">👤</div>
                )}
              </div>
            ))}

            {isThinking && (
              <div className="msg-wrap bot">
                <div className="msg-avatar">🤖</div>
                <div className="bot-bubble msg-bubble">
                  <div className="typing-dots">
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* ISL reference & Settings strip at bottom */}
          <div className="gesture-ref">
            
            <div className="ref-left">
              <span className="ref-label">ISL Reference</span>
              <div className="gesture-chips">
              {"ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").map(l => (
                <span
                  key={l}
                  className={`g-chip ${currentSign?.label === l ? "active" : ""}`}
                >
                  {l}
                </span>
              ))}
              {"0123456789".split("").map(n => (
                <span
                  key={n}
                  className={`g-chip num ${currentSign?.label === n ? "active" : ""}`}
                >
                  {n}
                </span>
              ))}
            </div>
            </div>

            <div className="ref-right">
              <div className="bottom-slider-wrap">
                <span className="ref-label">Sensitivity ({confidenceThreshold}%)</span>
                <input
                  type="range"
                  min="50"
                  max="99"
                  title={confidenceThreshold < 70 ? "More jittery" : confidenceThreshold > 90 ? "Requires perfect sign" : "Balanced"}
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                  className="conf-slider bottom-slider"
                />
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
