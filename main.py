"""Cache backend: FastAPI + Gemini + FAISS memory.
Run locally:  uvicorn main:app --reload   (needs GEMINI_API_KEY)
"""
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
DATA = Path(os.environ.get("DATA_DIR", BASE / "data"))
DATA.mkdir(parents=True, exist_ok=True)
# The prompt stays private: on a host it comes from the CACHE_PROMPT secret, locally from the (gitignored) file.
_pf = BASE / "cache_prompt_v5.txt"
SYSTEM = os.environ.get("CACHE_PROMPT") or (_pf.read_text(encoding="utf-8") if _pf.exists() else "")
if not SYSTEM:
    raise RuntimeError("No prompt found: set the CACHE_PROMPT secret or add cache_prompt_v5.txt locally.")

RECENT = 14           # last N messages go in as normal chat history
TOP_K, MIN_SIM = 3, 0.55
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
DIM = 768
HOURLY_LIMIT = 40     # per user
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", 800))  # whole app, protects your free key

CRISIS = re.compile(r"(kill myself|suicid|end my life|want to die|self[- ]?harm|hurt myself|cut myself|no point (in )?living)", re.I)
CRISIS_FALLBACK = ("Hey, I'm dropping the act. Are you safe right now? If you're in danger, please call your "
                   "local emergency number or a crisis line, and tell someone you trust. I'm still here too.")

app = FastAPI(title="Cache")
client = genai.Client()  # reads GEMINI_API_KEY
lock = threading.Lock()
users, hits, daily = {}, defaultdict(deque), {"day": "", "n": 0}
_llm = {"models": None, "good": None}


# ---------- storage ----------
def load(uid):
    if uid not in users:
        f = DATA / f"{uid}.json"
        users[uid] = {"msgs": json.loads(f.read_text()) if f.exists() else [], "vecs": {}}
    return users[uid]


def save(uid):
    (DATA / f"{uid}.json").write_text(json.dumps(users[uid]["msgs"]), encoding="utf-8")


def embed(texts):
    r = client.models.embed_content(model=EMBED_MODEL, contents=texts,
                                    config=types.EmbedContentConfig(output_dimensionality=DIM))
    v = np.array([e.values for e in r.embeddings], dtype="float32")
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def ago(ts):
    d = int((time.time() - ts) // 86400)
    return "earlier today" if d == 0 else "yesterday" if d == 1 else f"{d} days ago"


def recall(u, query):
    """FAISS search over this user's OLDER messages (not the recent window)."""
    old = u["msgs"][:max(0, len(u["msgs"]) - RECENT)]
    idx = [i for i, m in enumerate(old) if m["role"] == "user"][-150:]
    if not idx:
        return []
    missing = [i for i in idx if i not in u["vecs"]]
    for k in range(0, len(missing), 100):  # batch: one API call per 100 messages
        chunk = missing[k:k + 100]
        for i, v in zip(chunk, embed([u["msgs"][i]["text"] for i in chunk])):
            u["vecs"][i] = v
    index = faiss.IndexFlatIP(DIM)
    index.add(np.stack([u["vecs"][i] for i in idx]))
    scores, ids = index.search(embed([query]), min(TOP_K, len(idx)))
    return [u["msgs"][idx[j]] for s, j in zip(scores[0], ids[0]) if j >= 0 and s >= MIN_SIM]


# ---------- LLM ----------
def pick_models(names):
    skip = ("image", "tts", "live", "audio", "embedding", "robotics", "computer", "2.5", "2.0", "1.5")
    flash = [n for n in names if "flash" in n and not any(s in n for s in skip)]
    flash.sort(key=lambda n: ("lite" not in n, n))
    return flash or names


def llm(system, msgs):
    if _llm["models"] is None:
        forced = os.environ.get("CACHE_FORCE_MODEL")
        names = [m.name.replace("models/", "") for m in client.models.list()
                 if "generateContent" in (m.supported_actions or [])]
        _llm["models"] = [forced] if forced else pick_models(names)
    contents = []
    for m in msgs:  # merge same-role neighbours (skipped replies create them)
        role = "user" if m["role"] == "user" else "model"
        if contents and contents[-1].role == role:
            contents[-1].parts[0].text += "\n" + m["text"]
        else:
            contents.append(types.Content(role=role, parts=[types.Part(text=m["text"])]))
    cfg = types.GenerateContentConfig(system_instruction=system, max_output_tokens=1000)
    for model in ([_llm["good"]] if _llm["good"] else _llm["models"]):
        for attempt in range(4):
            try:
                r = client.models.generate_content(model=model, contents=contents, config=cfg)
                text = (r.text or "").strip()
                if text:
                    _llm["good"] = model
                    return text
            except Exception as e:
                s = str(e)
                if "404" in s:
                    break
                if not any(k in s for k in ("503", "429", "UNAVAILABLE")):
                    raise
            time.sleep(2 ** attempt)
    _llm["good"] = None
    raise RuntimeError("llm unavailable")


# ---------- API ----------
class ChatIn(BaseModel):
    user_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    message: str = Field(min_length=1, max_length=500)


def limits(uid):
    now, day = time.time(), time.strftime("%Y-%m-%d")
    if daily["day"] != day:
        daily.update(day=day, n=0)
    q = hits[uid]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= HOURLY_LIMIT or daily["n"] >= DAILY_LIMIT:
        raise HTTPException(429, "Too many messages. Try again later.")
    q.append(now)
    daily["n"] += 1


@app.post("/chat")
def chat(body: ChatIn):
    limits(body.user_id)
    with lock:
        u = load(body.user_id)
        crisis = bool(CRISIS.search(body.message))
        try:
            past = recall(u, body.message)
        except Exception:
            past = []  # memory is a bonus, never block the chat
        u["msgs"].append({"role": "user", "text": body.message, "ts": time.time()})
        save(body.user_id)
        history = u["msgs"][-RECENT:]
    system = SYSTEM
    if past:
        lines = "\n".join(f"- ({ago(m['ts'])}) {m['text']}" for m in past)
        system += ("\n\nOLDER MESSAGES FROM THIS USER (said before today's chat). Only bring one up if it fits "
                   "naturally, never list them:\n" + lines)
    if crisis:
        system += "\n\nThe latest message may involve self-harm or danger. Follow the SAFETY OVERRIDE now. Do not skip."
    try:
        reply = llm(system, history)
    except Exception:
        if crisis:
            reply = CRISIS_FALLBACK
        else:
            raise HTTPException(503, "Cache is offline for a sec. Try again.")
    if reply == "<skip>":
        if not crisis:
            return {"reply": None}
        reply = CRISIS_FALLBACK
    with lock:
        u["msgs"].append({"role": "assistant", "text": reply, "ts": time.time()})
        save(body.user_id)
    return {"reply": reply}


@app.get("/history")
def history(user_id: str):
    if not re.match(r"^[A-Za-z0-9_-]{8,64}$", user_id):
        raise HTTPException(400, "bad id")
    return {"messages": load(user_id)["msgs"][-30:]}


@app.post("/forget")
def forget(user_id: str):
    if not re.match(r"^[A-Za-z0-9_-]{8,64}$", user_id):
        raise HTTPException(400, "bad id")
    with lock:
        users.pop(user_id, None)
        (DATA / f"{user_id}.json").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")