# Cache

A persistent AI character born online. FastAPI + Gemini + FAISS memory (he remembers what you told him days ago).

## Run locally
```
pip install -r requirements.txt
set GEMINI_API_KEY=your_key
uvicorn main:app --reload
```
Open http://127.0.0.1:8000

## Deploy (no Docker needed on Render-style hosts)
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Env var: `GEMINI_API_KEY`

Memory is stored on the host's disk, which resets on restart.
