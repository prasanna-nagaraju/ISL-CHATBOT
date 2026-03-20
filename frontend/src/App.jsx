// App.jsx
import { useState } from "react";
import LandingPage from "./components/LandingPage";
import ISLChatbot  from "./components/ISLChatbot";
import "./App.css";

export default function App() {
  const [page, setPage] = useState("landing"); // "landing" | "chatbot"

  return (
    <div className="app">
      {page === "landing" ? (
        <LandingPage onGetStarted={() => setPage("chatbot")} />
      ) : (
        <ISLChatbot onBack={() => setPage("landing")} />
      )}
    </div>
  );
}