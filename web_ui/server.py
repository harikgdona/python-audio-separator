"""
Karaoke Pipeline — Web UI Backend
Uses threading + subprocess.Popen for Windows compatibility (no asyncio subprocess).
"""

import asyncio
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import torchaudio
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Windows fix: ProactorEventLoop supports subprocesses
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI()

BASE_DIR     = Path(__file__).parent.parent
PIPELINE_DIR = BASE_DIR / "karaoke_pipeline"
STATIC_DIR   = Path(__file__).parent / "static"
SONGS_DIR    = Path(r"D:\My Apps\Karoke\Songs")
VENV_PYTHON  = BASE_DIR / ".venv" / "Scripts" / "python.exe"


# ── Root redirect ─────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Serve any local audio/stem file ──────────────────────────────────────────

@app.get("/file")
async def serve_file(path: str):
    p = Path(path)
    if not p.exists():
        return {"error": "not found"}
    media = "audio/wav" if p.suffix.lower() == ".wav" else "audio/mpeg"
    return FileResponse(str(p), media_type=media)


# ── Waveform peak data ────────────────────────────────────────────────────────

@app.get("/waveform")
async def get_waveform(path: str, points: int = 800):
    p = Path(path)
    if not p.exists():
        return {"error": "not found"}
    waveform, sr = torchaudio.load(str(p))
    data     = waveform.mean(0).abs().numpy()
    step     = max(1, len(data) // points)
    peaks    = [float(data[i*step:(i+1)*step].max()) for i in range(points)]
    duration = round(len(data) / sr, 3)
    return {"peaks": peaks, "duration": duration, "sr": sr}


# ── Song list ─────────────────────────────────────────────────────────────────

@app.get("/songs")
async def list_songs():
    if not SONGS_DIR.exists():
        return {"songs": [], "dir": str(SONGS_DIR)}
    songs = [
        {"name": f.name, "path": str(f)}
        for f in sorted(SONGS_DIR.iterdir())
        if f.suffix.lower() in {".mp3", ".wav"}
    ]
    return {"songs": songs}


# ── Pipeline runner (thread-based, Windows-safe) ──────────────────────────────

class ProcessRequest(BaseModel):
    input_path: str
    remove: str = "male"


@app.post("/process")
async def process(req: ProcessRequest):
    stems_dir = Path(tempfile.mkdtemp(prefix="karaoke_ui_"))
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    cmd = [
        str(VENV_PYTHON),
        str(PIPELINE_DIR / "main.py"),
        "--input",     req.input_path,
        "--remove",    req.remove,
        "--stems-dir", str(stems_dir),
    ]

    def run_in_thread():
        """Run pipeline subprocess in a background thread, push lines to queue."""
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PIPELINE_DIR),
            )
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    loop.call_soon_threadsafe(queue.put_nowait, line)
            proc.wait()
            rc = proc.returncode
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, f"JSON:{{\"event\":\"error\",\"msg\":\"{exc}\"}}")
            rc = 1
        loop.call_soon_threadsafe(queue.put_nowait, f"__DONE__{rc}__")

    thread = threading.Thread(target=run_in_thread, daemon=True)

    async def stream():
        yield _sse("start", f"Starting pipeline for: {Path(req.input_path).name}")
        yield _sse("info",  f"Removing: {req.remove} voices")

        thread.start()

        while True:
            line = await queue.get()

            # Sentinel — pipeline finished
            if line.startswith("__DONE__"):
                rc = int(line.replace("__DONE__","").replace("__",""))
                if rc != 0:
                    yield _sse("error", "Pipeline failed — see log above.")
                break

            if line.startswith("JSON:"):
                try:
                    ev = json.loads(line[5:])
                    yield _sse_rich(ev)
                except Exception:
                    yield _sse("log", line)
            else:
                yield _sse("log", line)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(type_: str, message: str) -> str:
    return f"data: {json.dumps({'type': type_, 'message': message})}\n\n"


def _sse_rich(ev: dict) -> str:
    etype = ev.get("event", "log")

    if etype == "stage":
        name, status = ev.get("name", ""), ev.get("status", "")
        msg     = f"[{name.upper()}] {status}"
        payload = {"type": "stage", "name": name, "status": status,
                   "message": msg,
                   **{k: v for k, v in ev.items() if k not in ("event","name","status")}}

    elif etype == "segment":
        g, s, e, c, rem = (ev.get("gender","?"), ev.get("start",0),
                           ev.get("end",0), ev.get("confidence",0),
                           ev.get("remove", False))
        msg     = f"[SEGMENT] {s:.1f}s–{e:.1f}s  {g}  {c:.0%}  {'→ REMOVE' if rem else '→ keep'}"
        payload = {"type": "segment", "message": msg, **ev}

    elif etype == "done":
        payload = {"type": "done", "message": "Pipeline complete!",
                   "output": ev.get("output", "")}

    elif etype == "device":
        payload = {"type": "info",
                   "message": f"[DEVICE] {ev.get('mode','').upper()} {ev.get('name','')}"}

    elif etype == "info":
        payload = {"type": "info",
                   "message": f"Input: {ev.get('input','')} | Remove: {ev.get('remove','')}"}

    elif etype == "error":
        payload = {"type": "error", "message": f"[ERROR] {ev.get('msg', str(ev))}"}

    else:
        payload = {"type": "log", "message": str(ev)}

    return f"data: {json.dumps(payload)}\n\n"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
