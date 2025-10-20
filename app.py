from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import os, tempfile, uvicorn
from faster_whisper import WhisperModel

# --- Config ---
ASR_SECRET = os.getenv("ASR_SECRET", "change-me")  # shared header secret
MODEL_NAME = os.getenv("WHISPER_MODEL", "small.en")  # e.g., tiny, base, small, small.en, medium, large-v3
DEVICE = os.getenv("DEVICE", "cpu")                 # "cpu" or "cuda"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")     # cpu: int8 / int8_float16; gpu: float16

# Initialize model once on startup
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

app = FastAPI(title="CAISA Local ASR", version="0.1.0")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    x_asr_key: Optional[str] = Header(default=None)
):
    # Simple shared-secret guard
    if ASR_SECRET and (not x_asr_key or x_asr_key != ASR_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized: bad x-asr-key")

    # Persist to temp file so ffmpeg can read
    try:
        suffix = os.path.splitext(file.filename or "audio")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    # Transcribe
    try:
        segments, info = model.transcribe(
            tmp_path,
            language=os.getenv("LANGUAGE", None),  # None = auto
            beam_size=int(os.getenv("BEAM_SIZE", "2")),
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            temperature=0.0,
        )
        full_text = []
        seg_list = []
        for seg in segments:
            seg_list.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "avg_logprob": float(getattr(seg, "avg_logprob", 0.0)),
                "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0)),
            })
            full_text.append(seg.text.strip())
        out = {
            "model": MODEL_NAME,
            "language": info.language,
            "duration": info.duration,
            "text": " ".join(full_text).strip(),
            "segments": seg_list,
        }
        return JSONResponse(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)

