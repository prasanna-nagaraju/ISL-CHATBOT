# 🚀 Signbot: Indian Sign Language Chatbot

Full-stack ISL recognition app with MediaPipe hand tracking, TensorFlow model inference, and AI chat.

## 🏗️ Local Development

```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend  
cd frontend
npm install
npm run dev
```

**Full-stack Docker:**
```bash
# Copy .env.example → .env, add COHERE_API_KEY
docker compose up --build
```
- Backend: http://localhost:8000/health
- Frontend: http://localhost:3000

## ☁️ Production Deployment

### Backend (Render.com - Free)
1. Push to GitHub: `gh repo create signbot --public`
2. render.com → New → Docker → Connect GitHub/backend
3. Build auto-starts (Dockerfile detected)
4. Dashboard → Environment → Add `COHERE_API_KEY`
5. Live: `https://your-app.onrender.com/health`

### Frontend (Vercel - Free)
1. vercel.com → Import GitHub repo
2. Project Settings → Environment Variables:
   ```
   VITE_API_URL=https://your-backend.onrender.com
   ```
3. Deploy → Live: `https://your-app.vercel.app`

## 📊 API Endpoints
| Endpoint | Description |
|----------|-------------|
`/health` | Status + model info |
`/labels` | 35 ISL classes (1-9, A-Z) |
`/predict` | Image upload → sign prediction |
`/predict/base64` | Base64 image |
`/predict/landmarks` | 63-float landmarks |
`/chat` | Cohere AI conversation |

## 🤖 Models
- **Main**: Image-based (EfficientNet/MobileNetV2)
- **Landmarks**: Dense net on MediaPipe 21-handmarks (63 features)
- Classes: 1-9, A-Z (35 total)
- Acc: ~99% val

## 🔑 Setup Cohere Chat
1. cohere.com → API key
2. `.env`: `COHERE_API_KEY=xxxx`

## 📁 Structure
```
signbot/
├── backend/          # FastAPI + TF models
├── frontend/         # React + MediaPipe
├── docker-compose.yml
└── TODO.md           # Deployment progress
```

**Demo**: [TBD after deploy]

