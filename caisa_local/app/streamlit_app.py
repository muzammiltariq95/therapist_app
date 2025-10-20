
import io
import json
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import requests
import soundfile as sf
import streamlit as st

# Optional live mode via WebRTC
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase
    import av  # PyAV for audio frames
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False

st.set_page_config(page_title="CAISA · Therapist Mic → Local Agent (Streamlit)", page_icon="🎙️", layout="wide")

# -------------------------
# UI: Sidebar configuration
# -------------------------
with st.sidebar:
    st.title(" CAISA Mic → Agent")
    st.caption("Record → /turn → ASR → AI reply → history")
    agent_url = st.text_input("Agent URL", value="http://127.0.0.1:8001/turn")
    api_key = st.text_input("x-caisa-key (optional header)", value="", type="password")
    session_id = st.text_input("Session ID", value=f"demo-{int(time.time())}")
    scenario = st.text_input("Scenario (persona brief)", value="general intake: mild anxiety")
    st.markdown("---")
    live_mode = st.toggle("Live mode (auto-turn on silence)", value=False, disabled=not HAS_WEBRTC)
    if not HAS_WEBRTC and live_mode:
        st.warning("streamlit-webrtc not available. Install it or turn off Live mode.")

    st.markdown("#### Audio encoding")
    mime_choice = st.selectbox(
        "Upload as",
        options=[
            "audio/wav",
            "audio/webm;codecs=opus (requires ffmpeg, optional)"
        ],
        index=0,
        help="Your backend should accept common audio types. WAV is simplest. WEBM/Opus needs ffmpeg to encode."
    )
    st.markdown("---")
    st.caption("Tip: use HTTPS on public deployments so browsers allow mic access.")

# -------------------------
# Helpers
# -------------------------
def post_to_agent(audio_bytes: bytes, filename: str, content_type: str) -> Tuple[bool, str, dict]:
    """
    Send audio to /turn endpoint with multipart/form-data and return (ok, text_payload, json_payload_if_any).
    """
    if not agent_url.strip():
        return False, "Please set Agent URL", {}

    files = {"file": (filename, audio_bytes, content_type)}
    data = {"session_id": session_id, "scenario": scenario}
    headers = {}
    if api_key.strip():
        headers["x-caisa-key"] = api_key.strip()

    try:
        resp = requests.post(agent_url.strip(), files=files, data=data, headers=headers, timeout=60)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            payload = resp.json()
            return resp.ok, json.dumps(payload, indent=2), payload
        else:
            text = resp.text
            return resp.ok, text, {}
    except Exception as e:
        return False, f"Network error: {e}", {}

def fetch_history() -> str:
    """
    GET /session/history?session_id=... (derive api root from /turn url)
    """
    base = agent_url.strip().rstrip("/")
    if not base.endswith("/turn"):
        return "(set a valid /turn URL first)"
    api_root = base[:-5]  # remove '/turn'
    url = f"{api_root}/session/history"
    try:
        r = requests.get(url, params={"session_id": session_id}, timeout=30)
        if not r.ok:
            return f"Error {r.status_code}: {r.text}"
        data = r.json()
        lines = []
        for i, t in enumerate(data.get("turns", []), start=1):
            therapist_text = t.get("therapist_text", "")
            client_text = t.get("client_text", "")
            lines.append(f"[{i}] Therapist: {therapist_text}")
            if client_text:
                lines.append(f"    Client  : {client_text}")
            lines.append("")
        return "\n".join(lines) if lines else "(no turns yet)"
    except Exception as e:
        return f"{e}"

def clear_history() -> str:
    base = agent_url.strip().rstrip("/")
    if not base.endswith("/turn"):
        return "Set a valid /turn URL first"
    api_root = base[:-5]
    url = f"{api_root}/session/clear"
    try:
        r = requests.post(url, data={"session_id": session_id}, timeout=30)
        if not r.ok:
            return f"Error {r.status_code}: {r.text}"
        return "(cleared)"
    except Exception as e:
        return f"{e}"

def wav_bytes_from_np(np_audio: np.ndarray, samplerate: int) -> bytes:
    """
    Encode a mono float32 numpy signal to 16-bit PCM WAV in-memory.
    """
    # Ensure float32 in [-1,1]
    x = np.clip(np_audio.astype(np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, x, samplerate, subtype="PCM_16", format="WAV")
    return buf.getvalue()

# Optional: convert WAV to WEBM/Opus using ffmpeg if available
def wav_to_webm_opus(wav_bytes: bytes, bitrate: str = "48k") -> Optional[bytes]:
    """
    Convert WAV -> WEBM/Opus via ffmpeg if installed. Returns None if failed.
    """
    import subprocess, tempfile, os
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            f_in.write(wav_bytes)
            in_path = f_in.name
        out_path = in_path.replace(".wav", ".webm")
        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-c:a", "libopus",
            "-b:a", bitrate,
            out_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with open(out_path, "rb") as f_out:
            data = f_out.read()
        try:
            os.remove(in_path)
            os.remove(out_path)
        except Exception:
            pass
        return data
    except Exception:
        return None

# -------------------------
# Live mode (silence based)
# -------------------------
@dataclass
class LiveBuffer:
    samplerate: int = 48000
    buf: List[np.ndarray] = field(default_factory=list)
    RMS_SILENCE_THRESHOLD: float = 0.015  # reserved if you want to surface RMS meter
    END_SPEECH_MS: int = 1200
    last_audio_time: float = field(default_factory=lambda: time.time())
    last_sent_time: float = 0.0
    MIN_TURN_GAP: float = 0.7

    def add_chunk(self, chunk: np.ndarray):
        # chunk expected mono float32 [-1,1]
        self.buf.append(chunk)
        self.last_audio_time = time.time()

    def should_finalize(self) -> bool:
        now = time.time()
        dt_sil = now - self.last_audio_time
        return dt_sil * 1000.0 >= self.END_SPEECH_MS and (now - self.last_sent_time) >= self.MIN_TURN_GAP

    def flush_audio(self) -> Optional[np.ndarray]:
        if not self.buf:
            return None
        x = np.concatenate(self.buf, axis=0)
        self.buf.clear()
        return x

# Streamlit session state
if "req_preview" not in st.session_state:
    st.session_state.req_preview = ""
if "res_preview" not in st.session_state:
    st.session_state.res_preview = ""
if "history_text" not in st.session_state:
    st.session_state.history_text = ""

st.title("CAISA · Therapist Mic → Local Agent (Streamlit)")
st.caption("Streamlit edition: record mic → send to /turn → show JSON → session history")

col_l, col_r = st.columns([0.65, 0.35], gap="large")

with col_l:
    st.subheader("🎤 Microphone")
    if live_mode and not HAS_WEBRTC:
        st.info("Live mode requires 'streamlit-webrtc'. Falling back to manual upload/record.")
        live_mode = False

    if not live_mode:
        st.markdown("**Manual turn** – upload a clip, then send it as one turn.")
        uploaded = st.file_uploader("Upload audio (wav/mp3/webm/ogg/m4a)", type=["wav", "mp3", "webm", "ogg", "m4a"], accept_multiple_files=False)
        st.caption("Prefer in-app recording? Enable **Live mode** in the sidebar (WebRTC).")

        audio_bytes = None
        mime_used = None
        if uploaded is not None:
            audio_bytes = uploaded.read()
            mime_used = uploaded.type or "application/octet-stream"
            st.audio(audio_bytes, format=mime_used)

        if audio_bytes is not None:
            st.markdown("**Send to Agent**")
            req_preview = f"file: {getattr(uploaded, 'name', 'upload')} ({len(audio_bytes)//1024} KB)\nsession_id: {session_id}\nscenario: {scenario}"
            st.code(req_preview, language="text")
            if st.button("⬆️ Send now", use_container_width=True):
                out_bytes = audio_bytes
                out_mime = mime_used

                if mime_choice.startswith("audio/wav"):
                    try:
                        data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
                        if data.ndim > 1:
                            data = data.mean(axis=1)  # mono
                        wav_bytes = wav_bytes_from_np(data, sr)
                        out_bytes, out_mime = wav_bytes, "audio/wav"
                    except Exception:
                        out_bytes, out_mime = audio_bytes, mime_used
                else:
                    try:
                        data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
                        if data.ndim > 1: data = data.mean(axis=1)
                        wav_bytes = wav_bytes_from_np(data, sr)
                        webm = wav_to_webm_opus(wav_bytes)  # may be None
                        if webm:
                            out_bytes, out_mime = webm, "audio/webm"
                        else:
                            st.warning("ffmpeg conversion failed; sending WAV instead.")
                            out_bytes, out_mime = wav_bytes, "audio/wav"
                    except Exception:
                        out_bytes, out_mime = audio_bytes, mime_used

                ok, text_resp, payload = post_to_agent(out_bytes, "therapist_turn", out_mime)
                st.session_state.req_preview = req_preview
                st.session_state.res_preview = text_resp
                if ok:
                    st.success("Uploaded")
                else:
                    st.error(text_resp[:300])
    else:
        st.markdown("**Live mode** – auto-sends a turn when ~1.2s of silence is detected.")
        st.caption("We buffer your speech and end a turn on silence, then POST it to the Agent.")

        RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        live = LiveBuffer()
        status_box = st.empty()
        meter = st.progress(0)
        sent_counter = st.empty()

        class MicProcessor(AudioProcessorBase):
            def __init__(self) -> None:
                self.q = queue.Queue()

            def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
                samples = frame.to_ndarray()
                if samples.ndim == 2:
                    samples = samples.mean(axis=0)
                # normalize if integer
                if samples.dtype != np.float32:
                    if np.issubdtype(samples.dtype, np.integer):
                        maxv = np.iinfo(samples.dtype).max
                        samples = samples.astype(np.float32) / float(maxv)
                    else:
                        samples = samples.astype(np.float32)
                self.q.put(samples)
                return frame

        ctx = webrtc_streamer(
            key="mic-live",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=MicProcessor,
            rtc_configuration=RTC_CFG,
            media_stream_constraints={"audio": True, "video": False},
        )

        def worker():
            turns_sent = 0
            while True:
                if not ctx.state.playing:
                    time.sleep(0.05)
                    continue
                proc: MicProcessor = ctx.audio_processor
                if proc is None:
                    time.sleep(0.05)
                    continue
                # Drain queue
                joined = []
                try:
                    while True:
                        chunk = proc.q.get_nowait()
                        joined.append(chunk)
                except queue.Empty:
                    pass

                if joined:
                    block = np.concatenate(joined, axis=0)
                    live.add_chunk(block)
                    rms = float(np.sqrt(np.mean(block**2))) if block.size else 0.0
                    meter.progress(int(min(100, max(1, rms * 200))))

                if live.should_finalize():
                    audio = live.flush_audio()
                    if audio is not None and audio.size > 2000:
                        wav_bytes = wav_bytes_from_np(audio, samplerate=live.samplerate)
                        if mime_choice.startswith("audio/webm"):
                            webm = wav_to_webm_opus(wav_bytes)
                            out = (webm, "audio/webm") if webm else (wav_bytes, "audio/wav")
                        else:
                            out = (wav_bytes, "audio/wav")

                        ok, text_resp, payload = post_to_agent(out[0], f"therapist_{int(time.time())}", out[1])
                        st.session_state.req_preview = f"(live fragment) size={len(out[0])//1024} KB\nsession_id:{session_id}\nscenario:{scenario}"
                        st.session_state.res_preview = text_resp
                        turns_sent += 1
                        live.last_sent_time = time.time()
                        sent_counter.info(f"Turns sent in this session: **{turns_sent}**")
                time.sleep(0.05)

        if ctx.state.playing:
            status_box.success("Live… listening for silence to end a turn")
            threading.Thread(target=worker, daemon=True).start()
        else:
            status_box.info("Click **Start** above to grant mic access.")

with col_r:
    st.subheader("📦 Request (FormData) / Response (JSON)")
    st.markdown("**Request preview**")
    st.code(st.session_state.req_preview or "(none yet)", language="text")
    st.markdown("**Response (raw)**")
    st.code(st.session_state.res_preview or "(none yet)", language="json")

    st.markdown("---")
    st.subheader("🗂️ Session History")
    c1, c2 = st.columns(2)
    if c1.button("⟳ Refresh history", use_container_width=True):
        st.session_state.history_text = fetch_history()
    if c2.button("✖ Clear history", use_container_width=True):
        st.session_state.history_text = clear_history()
    st.text_area("History", value=st.session_state.history_text, height=260)

st.markdown("---")
st.caption("Built with Streamlit. Live mode uses WebRTC (optional). WAV upload works by default; WEBM/Opus requires ffmpeg.")
