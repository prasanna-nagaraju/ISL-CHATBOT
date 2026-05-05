# 🌐 Multilingual Chatbot with Indic Languages Support

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Cohere](https://img.shields.io/badge/Cohere-LLM-red.svg)
![IndicTrans2](https://img.shields.io/badge/IndicTrans2-NMT-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

### AI-powered multilingual chatbot supporting Indian languages with translation and text-to-speech

**Built for seamless communication across diverse linguistic communities in India**

</div>

---

# 📌 Project Overview

The **Multilingual Chatbot with Indic Languages Support** is an AI-powered conversational platform designed to bridge language barriers by enabling users to interact in multiple Indian languages.

This system integrates:

- **Cohere API** for intelligent natural language understanding and chatbot responses  
- **IndicTrans2** for real-time translation between English and major Indic languages  
- **pyttsx3** for offline text-to-speech synthesis  

The chatbot allows users to ask questions in their preferred regional language (Telugu, Hindi, Tamil, Bengali, etc.), processes them intelligently, translates where necessary, and responds both as text and speech.

---

# 🚀 Key Features

## 🤖 AI Chatbot
- Natural language conversation using Cohere LLM  
- Context-aware intelligent responses  
- Fast and scalable REST API support  

## 🌍 Multilingual Translation
- Supports 10+ Indic languages  
- Real-time bidirectional translation  
- Powered by IndicTrans2 neural machine translation  

## 🔊 Text-to-Speech
- Offline speech synthesis using pyttsx3  
- Audio replay support  
- Accessible for wider user adoption  

## 💻 User Interface
- Responsive web design  
- Mobile and desktop compatible  
- Easy language selection dropdown  

## 🐳 Deployment Ready
- Docker containerization  
- Cloud deployment support (AWS / GCP / Heroku)  
- Secure API key handling with `.env`

---

# 🛠️ Tech Stack

## Core Technologies

| Component | Technology | Purpose |
|----------|-------------|---------|
| Chatbot Engine | Cohere API | Conversational AI |
| Translation | IndicTrans2 | Indic language translation |
| Text-to-Speech | pyttsx3 | Voice output |
| Backend | Flask / FastAPI | API & server logic |
| Frontend | HTML, CSS, JavaScript | User interface |
| Deployment | Docker | Containerization |

---

# 📂 Project Structure

```bash
multilingual-chatbot/
│
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── .env                        # Environment variables
├── Dockerfile                  # Docker configuration
│
├── static/
│   ├── css/style.css           # Frontend styling
│   ├── js/main.js              # Frontend functionality
│   └── assets/images/          # Images & icons
│
├── templates/
│   └── index.html              # Main web UI
│
├── src/
│   ├── chatbot/
│   │   ├── cohere_client.py
│   │   └── response_handler.py
│   │
│   ├── translation/
│   │   ├── indic_translator.py
│   │   └── language_mapping.py
│   │
│   ├── tts/
│   │   └── speech_engine.py
│   │
│   └── utils/
│       ├── helpers.py
│       └── constants.py
│
├── scripts/
│   ├── download_models.py
│   └── preprocess_data.py
│
├── tests/
│   ├── test_chatbot.py
│   ├── test_translation.py
│   └── test_tts.py
│
└── README.md
