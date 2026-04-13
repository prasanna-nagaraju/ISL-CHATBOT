    # Signbot Deployment TODO
Status: ✅ Config Files Complete

## Completed Steps
### Step 1: ✅ Plan Confirmation
### Step 2: ✅ Deployment Config Files Created
- `backend/render.yaml`
- `.env.example`
- `docker-compose.yml` (incl. frontend Dockerfile)
- `frontend/Dockerfile` + `nginx.conf`
- `README.md`

## 🚀 Next User Actions

### Step 3: GitHub Repo
```bash
git init
git add .
git commit -m "Deploy-ready: Signbot ISL chatbot w/ Render+Vercel"
gh repo create signbot --public --push
```
*Install GitHub CLI: https://cli.github.com*

### Step 4: Backend → Render.com (5 mins)
1. render.com → Sign up (GitHub login)
2. New → Web Service → Docker → Connect `signbot/backend`
3. Settings → Environment → `COHERE_API_KEY` = [your key]
4. Deploy → Live URL ready!

### Step 5: Frontend → Vercel (3 mins)  
1. vercel.com → Import `signbot` repo
2. Settings → Env Vars → `VITE_API_URL` = [your render backend URL]
3. Deploy!

### Step 6: Test
```bash
# Backend health
curl https://your-app.onrender.com/health

# Local full-stack
docker compose up
```

**Your app will be live at:**
- Backend: `https://*.onrender.com`
- Frontend: `https://*.vercel.app`

**Reply "deployed" with URLs when done → I'll verify & complete!**

## Files Created:
- 6 deployment configs
- Full README w/ instructions
- Docker for prod/local

