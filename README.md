# AI-Translate (MVP)

## Quick start (local)

### Backend
1. Go to backend/app
2. Create virtualenv and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (default is 1234 for demo)
   ```bash
   cp .env.example .env
   ```
4. Run:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend
1. Go to frontend
2. Install and start:
   ```bash
   npm install
   npm start
   ```
3. Open http://localhost:3000

## Docker
Build and run:
```
docker-compose up --build
```

## Notes
- OPENAI_API_KEY in `.env.example` is set to `1234` as a placeholder. Replace with your OpenRouter/OpenAI key.
- Whisper requires ffmpeg.
- PaddleOCR and paddlepaddle may need platform-specific installation steps.

