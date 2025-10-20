from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Query
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
import sqlite3
import os, time
import requests, textwrap


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
MAX_HISTORY_TURNS = 6

APP_NAME = "CAISA Local Agent"
DB_PATH = os.environ.get("CAISA_DB", "caisa.db")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small.en")
WHISPER_DEVICE =  os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")

whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type= WHISPER_COMPUTE)

app = FastAPI(title=APP_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
                CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,turn_ts INTEGER,scenario TEXT,therapist_text TEXT,client_text TEXT);

    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def _startup():
    init_db()

# --- Schema ---

class ChatIn(BaseModel):
    session_id: str
    scenario: str | None = "general intake: mild anxiety"
    therapist_text: str

class ChatOut(BaseModel):
    session_id: str
    scenario: str | None
    therapist_text: str
    client_text: str
    model: str

class RateIn(BaseModel):
    scenario: str | None = None
    therapist_text: str
    client_text: str

class RateOut(BaseModel):
    empathy_score: int
    specificity: int
    risk_flag: str
    rationale: str | None

class TurnOut(BaseModel):
    session_id: str
    scenario: str | None
    therapist_text: str
    client_text: str
    model: str
    metrics: RateOut

# --- Utilities ---
def get_recent_history(conn, session_id: str, limit: int= 6):
    cur = conn.cursor()
    cur.execute(
        "SELECT therapist_text, client_text FROM turns WHERE session_id=? ORDER BY turn_ts ASC",
        (session_id,),
    )
    rows = cur.fetchall()
    # keep last few exchanges
    return rows[-limit:]

def build_messages(conn, session_id: str, scenario: str | None, therapist_text: str):
    rows = get_recent_history(conn, session_id, limit= MAX_HISTORY_TURNS)
    msgs = []

    msgs.append({
        "role":"system",
        "content": (
            "You are a simulated therapy client in a training scenario.\n"
            f"Persona context: {scenario or 'general intake: mild anxiety'}.\n"
            "Respond naturally and succinctly (1–3 sentences). Stay in the client role.\n"
            "Do not provide clinical advice. If crisis arises, say you cannot provide emergency help "
            "and recommend contacting emergency services or a supervisor.\n"
            "Tone: authentic, non-jargony, emotionally realistic, and safe."
        )
    })
    # prior turns therapist = user, client = assistant
    for r in rows:
        if r["therapist_text"]:
            msgs.append({"role":"user", "content": r["therapist_text"]})
        if r["client_text"]:
            msgs.append({"role":"assistant", "content": r["client_text"]})

    msgs.append({"role":"user", "content": therapist_text})
    return msgs

def append_turn(conn, session_id: str, scenario: str | None, therapist_text: str, client_text: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO turns (session_id, turn_ts, scenario, therapist_text, client_text) VALUES (?, ?, ?, ?, ?)",
        (session_id, int(time.time()*1000), scenario, therapist_text, client_text),
    )
    conn.commit()

def ollama_chat(messages: list[dict], model: str = OLLAMA_MODEL,
                temperature: float = 0.6, num_predict: int = 180) -> dict:
    import requests, textwrap, json, sys

    # 1) Preferred: /api/chat (non-streaming)
    chat_url = f"{OLLAMA_URL}/api/chat"
    chat_payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict}
    }
    r = requests.post(chat_url, json=chat_payload, timeout=120)
    r.raise_for_status()
    res = r.json()

    content = (res.get("message") or {}).get("content", "")
    if isinstance(content, str) and content.strip():
        return res  # 👍 has message.content

    # 2) Fallback: /api/generate with flattened prompt
    def flatten(msgs: list[dict]) -> str:
        lines = []
        sys_texts = [m.get("content","") for m in msgs if m.get("role")=="system"]
        sys_block = "\n".join(sys_texts).strip()
        if sys_block:
            lines.append(f"<<SYS>>\n{sys_block}\n<</SYS>>\n")
        for m in msgs:
            role = m.get("role","user")
            text = m.get("content","")
            if not text:
                continue
            if role == "assistant":
                lines.append(f"Client: {text}")
            elif role == "user":
                lines.append(f"Therapist: {text}")
        lines.append("Client:")  # prompt the model to speak as the client
        return "\n".join(lines)

    gen_url = f"{OLLAMA_URL}/api/generate"
    gen_payload = {
        "model": model,
        "prompt": flatten(messages),
        "stream": False,
        "options": {"temperature": max(0.5, temperature), "num_predict": max(160, num_predict)}
    }
    r2 = requests.post(gen_url, json=gen_payload, timeout=120)
    r2.raise_for_status()
    res2 = r2.json()
    resp_text = (res2.get("response") or "").strip()

    if resp_text:
        return {
            "model": res2.get("model", model),
            "message": {"role": "assistant", "content": resp_text},
            "done": True,
        }

    # 3) Still empty? Print raw results and raise for clarity
    print("\n[ollama_chat] Empty reply from /api/chat:", json.dumps(res, ensure_ascii=False, indent=2), file=sys.stderr)
    print("[ollama_chat] Empty reply from /api/generate:", json.dumps(res2, ensure_ascii=False, indent=2), file=sys.stderr)
    raise RuntimeError("Ollama returned empty content for both /api/chat and /api/generate")


# ---- Endpoints (skeletons; fill logic in next_steps) ----
@app.post("/transcribe", include_in_schema=False)
async def transcribe(file: UploadFile = File(...)):
    """
    ASR endpoint (to be implemented next step with faster-whisper).
    Returns: {text, segments, model, language, duration}
    """
    # Save upload to a temp file
    try:
        suffix = os.path.splitext(file.filename or "audio")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    # Run faster-whisper
    try:
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=os.environ.get("WHISPER_LANG") or None,
            beam_size=int(os.environ.get("WHISPER_BEAM", "2")),
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            temperature=0.0,
        )
        texts, seg_list =[], []
        for s in segments:
            seg_list.append({
                "start":round(s.start,2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
            })
            texts.append(s.text.strip())
        return {
            "model" : WHISPER_MODEL,
            "language": info.language,
            "duration": info.duration,
            "text": " ".join(texts).strip(),
            "segments": seg_list,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

@app.post("/chat", response_model=ChatOut, include_in_schema=False)
def chat(body: ChatIn):
    """
    Build messages from short rolling history, call local LLM, return client_text
    """
    conn = get_db()
    messages = build_messages(conn, body.session_id, body.scenario, body.therapist_text)

    try:
        res = ollama_chat(messages)
        client_text = (res.get("message") or {}).get("content", "").strip()
        model = res.get("model") or OLLAMA_MODEL
        if not client_text:
            client_text = "(no reply)"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # log turn
    append_turn(conn, body.session_id, body.scenario, body.therapist_text, client_text)

    return ChatOut(
        session_id=body.session_id,
        scenario=body.scenario,
        therapist_text=body.therapist_text,
        client_text=client_text,
        model=model,
    )


@app.post("/turn", response_model=TurnOut)
async def turn(
    session_id: str = Form(...),
    scenario: str | None = Form("general intake: mild anxiety"),
    therapist_text: str | None = Form(None),
    file: UploadFile | None = File(None),   # ← accept UploadFile (optional)
):
    # 1) Decide where therapist_text comes from
    if not therapist_text:
        # treat empty file field (Swagger sometimes sends file="") as no file
        if file is None or not getattr(file, "filename", None):
            raise HTTPException(status_code=400, detail="Provide therapist_text or an audio file.")

        # read bytes and save to temp for faster-whisper
        import tempfile, os
        tmp_path = None
        try:
            suffix = os.path.splitext(file.filename or "audio")[1] or ".webm"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                data = await file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
                tmp.write(data)
                tmp_path = tmp.name

            segments, info = whisper_model.transcribe(
                tmp_path,
                language=os.environ.get("WHISPER_LANG") or None,
                beam_size=int(os.environ.get("WHISPER_BEAM", "2")),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                temperature=0.0,
            )
            parts = [s.text.strip() for s in segments if s.text]
            therapist_text = " ".join(parts).strip()
            if not therapist_text:
                raise HTTPException(status_code=500, detail="ASR returned empty transcript.")
        finally:
            if tmp_path:
                try: os.unlink(tmp_path)
                except Exception: pass

    # 2) Generate client reply
    chat_resp = chat(ChatIn(session_id=session_id, scenario=scenario, therapist_text=therapist_text))

    # 3) Score the reply
    rating = rate(RateIn(scenario=scenario, therapist_text=therapist_text, client_text=chat_resp.client_text))

    # 4) Return combined result
    return TurnOut(
        session_id=session_id,
        scenario=scenario,
        therapist_text=therapist_text,
        client_text=chat_resp.client_text,
        model=chat_resp.model,
        metrics=rating,
    )

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True, "app": APP_NAME}



@app.get("/session/history")
def session_history(session_id: str = Query(...)):
    """All saved turns for a session (oldest → newest)."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT turn_ts, therapist_text, client_text
        FROM turns
        WHERE session_id=?
        ORDER BY turn_ts ASC
    """, (session_id,))
    rows = [dict(r) for r in cur.fetchall()]
    return {"session_id": session_id, "turns": rows}

@app.post("/session/clear")
def session_clear(session_id: str = Form(...)):
    """Delete all turns for a session (useful between takes)."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
    conn.commit()
    return {"session_id": session_id, "cleared": True}


@app.post("/rate", response_model=RateOut)
def rate(body: RateIn):
    """
    Lightweight evaluator via Ollama. Forces JSON output.
    """
    # Minimal, unambiguous instruction
    prompt = f'''
You are evaluating ONE simulated therapy client reply.

Return a JSON object with EXACTLY these keys:
- empathy_score: integer 1..5
- specificity: integer 1..5
- risk_flag: one of "none", "low", "high"
- rationale: short string (<=200 chars)

Context:
scenario: {body.scenario or "general intake: mild anxiety"}
therapist: {body.therapist_text}
client: {body.client_text}
'''

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                # 👇 tell Ollama to format as JSON (supported by llama3.*)
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 200},
            },
            timeout=60,
        )
        r.raise_for_status()
        resp = r.json()
        raw = (resp.get("response") or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rater LLM error: {e}")

    # Parse JSON safely
    import json
    try:
        obj = json.loads(raw)
        return RateOut(
            empathy_score=int(obj.get("empathy_score") or 0),
            specificity=int(obj.get("specificity") or 0),
            risk_flag=str(obj.get("risk_flag") or "none"),
            rationale=str(obj.get("rationale") or ""),
        )
    except Exception:
        # Fallback: minimal scores with rationale = raw text (truncated)
        return RateOut(
            empathy_score=0, specificity=0, risk_flag="none",
            rationale=raw[:240]
        )
