// LandingPage.jsx
import "./LandingPage.css";

const FEATURES = [
  {
    icon: "🤟",
    title: "ISL Recognition",
    desc: "Real-time Indian Sign Language gesture detection using MobileNetV2 deep learning model trained on A–Z and 0–9 signs."
  },
  {
    icon: "🤖",
    title: "AI Chatbot",
    desc: "Gemini Flash powered assistant that understands your signed input and responds intelligently in natural language."
  },
  {
    icon: "🌐",
    title: "Indian Languages",
    desc: "Translate responses into Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali and more."
  },
  {
    icon: "🔊",
    title: "Voice Output",
    desc: "Text-to-speech playback for both recognized text and AI responses — bridging the communication gap."
  }
];

const STEPS = [
  { num: "01", title: "Open Camera", desc: "Allow webcam access and position your hand in front of the camera." },
  { num: "02", title: "Sign Gestures", desc: "Form ISL signs — letters and numbers are recognized in real time." },
  { num: "03", title: "Build Message", desc: "Signed letters build up into words. Press Send when ready." },
  { num: "04", title: "Get AI Response", desc: "Gemini AI reads your signed message and replies instantly." },
];

export default function LandingPage({ onGetStarted }) {
  return (
    <div className="landing">

      {/* Hero */}
      <section className="hero">
        <div className="hero-badge">Indian Sign Language · AI Powered</div>
        <h1 className="hero-title">
          Speak with your<br />
          <span className="hero-accent">hands.</span>
        </h1>
        <p className="hero-desc">
          A real-time ISL recognition system that converts Indian Sign Language gestures
          into text and feeds them directly into an AI chatbot — breaking communication barriers
          for the deaf and hard-of-hearing community.
        </p>
        <button className="btn-start" onClick={onGetStarted}>
          Start Signing →
        </button>
      </section>

      {/* Features */}
      <section className="features">
        <h2 className="section-title">What it does</h2>
        <div className="features-grid">
          {FEATURES.map(f => (
            <div className="feature-card" key={f.title}>
              <div className="feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="how-it-works">
        <h2 className="section-title">How it works</h2>
        <div className="steps">
          {STEPS.map(s => (
            <div className="step" key={s.num}>
              <div className="step-num">{s.num}</div>
              <div className="step-content">
                <h3>{s.title}</h3>
                <p>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="cta">
        <h2>Ready to communicate?</h2>
        <p>No setup required — just your webcam and your hands.</p>
        <button className="btn-start" onClick={onGetStarted}>
          Open ISL Chatbot →
        </button>
      </section>

      <footer className="landing-footer">
        Built with MobileNetV2 · Gemini Flash · FastAPI · React
      </footer>
    </div>
  );
}
