"""
Karaoke Pipeline — Web UI Backend
Includes pipeline endpoints + labeling tool endpoints.
"""
import asyncio, json, shutil, subprocess, sys, tempfile, threading, uuid
from pathlib import Path

import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import db as DB

app = FastAPI()
DB.init_db()

BASE_DIR     = Path(__file__).parent.parent
PIPELINE_DIR = BASE_DIR / "karaoke_pipeline"
STATIC_DIR   = Path(__file__).parent / "static"
SONGS_DIR    = Path(r"D:\My Apps\Karoke\Songs")
VENV_PYTHON  = BASE_DIR / ".venv" / "Scripts" / "python.exe"
DATA_DIR     = BASE_DIR / "data"
UPLOADS_DIR  = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Serve local files ─────────────────────────────────────────────────────────
@app.get("/file")
async def serve_file(path: str):
    p = Path(path)
    if not p.exists(): return {"error": "not found"}
    media = "audio/wav" if p.suffix.lower()==".wav" else "audio/mpeg"
    return FileResponse(str(p), media_type=media)

@app.get("/waveform")
async def get_waveform(path: str, points: int = 800):
    p = Path(path)
    if not p.exists(): return {"error": "not found"}
    waveform, sr = torchaudio.load(str(p))
    data  = waveform.mean(0).abs().numpy()
    step  = max(1, len(data)//points)
    peaks = [float(data[i*step:(i+1)*step].max()) for i in range(points)]
    return {"peaks": peaks, "duration": round(len(data)/sr, 3), "sr": sr}

@app.get("/songs")
async def list_karaoke_songs():
    if not SONGS_DIR.exists(): return {"songs": []}
    songs = [{"name": f.name, "path": str(f)} for f in sorted(SONGS_DIR.iterdir())
             if f.suffix.lower() in {".mp3",".wav"}]
    return {"songs": songs}

# ── Pipeline SSE ──────────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    input_path: str
    remove: str = "male"

@app.post("/process")
async def process(req: ProcessRequest):
    stems_dir = Path(tempfile.mkdtemp(prefix="karaoke_ui_"))
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    cmd  = [str(VENV_PYTHON), str(PIPELINE_DIR/"main.py"),
            "--input", req.input_path, "--remove", req.remove,
            "--stems-dir", str(stems_dir)]
    def run():
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, cwd=str(PIPELINE_DIR))
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if line: loop.call_soon_threadsafe(queue.put_nowait, line)
            proc.wait(); rc = proc.returncode
        except Exception as exc:
            rc = 1; loop.call_soon_threadsafe(queue.put_nowait, f"JSON:{{\"event\":\"error\",\"msg\":\"{exc}\"}}")
        loop.call_soon_threadsafe(queue.put_nowait, f"__DONE__{rc}__")
    threading.Thread(target=run, daemon=True).start()
    async def stream():
        yield _sse("start", f"Starting: {Path(req.input_path).name}")
        yield _sse("info",  f"Removing: {req.remove}")
        while True:
            line = await queue.get()
            if line.startswith("__DONE__"):
                rc = int(line.replace("__DONE__","").replace("__",""))
                if rc != 0: yield _sse("error","Pipeline failed — see log.")
                break
            if line.startswith("JSON:"):
                try: yield _sse_rich(json.loads(line[5:]))
                except: yield _sse("log", line)
            else: yield _sse("log", line)
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Label: Upload song ────────────────────────────────────────────────────────
@app.post("/label/songs")
async def label_upload_song(
    file: UploadFile = File(...),
    singer_name: str = Form(""),
    year: str = Form(""),
    notes: str = Form("")
):
    song_id   = str(uuid.uuid4())[:8]
    dest_dir  = UPLOADS_DIR / song_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename
    with open(dest_path, "wb") as f:
        f.write(await file.read())
    db_id = DB.create_song(file.filename, str(dest_path),
                            singer_name, int(year) if year.isdigit() else None, notes)
    return {"id": db_id, "upload_id": song_id, "path": str(dest_path)}

@app.get("/label/songs")
async def label_list_songs():
    return {"songs": DB.list_songs()}

# ── Label: Separate stems for a song ─────────────────────────────────────────
class SeparateRequest(BaseModel):
    song_id: int

@app.post("/label/separate")
async def label_separate(req: SeparateRequest):
    song     = DB.get_song(req.song_id)
    if not song: return {"error": "song not found"}
    src      = Path(song["original_path"])
    stems_dir = UPLOADS_DIR / str(req.song_id) / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    cmd  = [str(VENV_PYTHON), "-m", "demucs",
            "--name", "htdemucs_6s",
            "--out",  str(stems_dir),
            "--device", "cpu", str(src)]
    def run():
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, cwd=str(PIPELINE_DIR))
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if line: loop.call_soon_threadsafe(queue.put_nowait, line)
            proc.wait(); rc = proc.returncode
        except Exception as exc:
            rc = 1; loop.call_soon_threadsafe(queue.put_nowait, f"ERROR:{exc}")
        loop.call_soon_threadsafe(queue.put_nowait, f"__DONE__{rc}__")
    threading.Thread(target=run, daemon=True).start()
    async def stream():
        yield _sse("info", f"Separating: {src.name}")
        while True:
            line = await queue.get()
            if line.startswith("__DONE__"):
                rc = int(line.replace("__DONE__","").replace("__",""))
                if rc == 0:
                    track_dir = stems_dir / "htdemucs_6s" / src.stem
                    if track_dir.exists():
                        for wav in track_dir.glob("*.wav"):
                            wf, sr = torchaudio.load(str(wav))
                            dur    = round(wf.shape[1] / sr, 3)
                            DB.create_stem(req.song_id, wav.stem, str(wav), dur)
                        stems = DB.get_stems(req.song_id)
                        yield _sse("stems_ready", json.dumps(stems))
                    yield _sse("done", "Stems ready for labeling!")
                else:
                    yield _sse("error", "Stem separation failed.")
                break
            yield _sse("log", line)
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Label: Stems / Segments ───────────────────────────────────────────────────
@app.get("/label/songs/{song_id}/stems")
async def label_get_stems(song_id: int):
    return {"stems": DB.get_stems(song_id)}

@app.get("/label/songs/{song_id}/segments")
async def label_get_segments(song_id: int):
    return {"segments": DB.get_segments(song_id)}

class SegmentIn(BaseModel):
    stem_id:      int
    start_sec:    float
    end_sec:      float
    label_type:   str = "instrument"
    labels:       list = []
    purity:       str  = "solo"
    singer_name:  str  = ""
    singer_gender:str  = ""
    notes:        str  = ""

@app.post("/label/segments")
async def label_create_segment(s: SegmentIn):
    seg_id = DB.create_segment(s.stem_id, s.start_sec, s.end_sec, s.label_type,
                                s.labels, s.purity, s.singer_name, s.singer_gender, s.notes)
    if s.singer_name: DB.add_singer(s.singer_name, s.singer_gender)
    return {"id": seg_id}

class SegmentUpdate(BaseModel):
    start_sec:    float
    end_sec:      float
    label_type:   str = "instrument"
    labels:       list = []
    purity:       str  = "solo"
    singer_name:  str  = ""
    singer_gender:str  = ""
    notes:        str  = ""

@app.put("/label/segments/{seg_id}")
async def label_update_segment(seg_id: int, s: SegmentUpdate):
    DB.update_segment(seg_id, s.start_sec, s.end_sec, s.label_type,
                      s.labels, s.purity, s.singer_name, s.singer_gender, s.notes)
    return {"ok": True}

@app.delete("/label/segments/{seg_id}")
async def label_delete_segment(seg_id: int):
    DB.delete_segment(seg_id)
    return {"ok": True}

# ── Instruments / Singers ─────────────────────────────────────────────────────
@app.get("/label/instruments")
async def label_get_instruments():
    return {"instruments": DB.get_instruments()}

class InstrumentIn(BaseModel):
    name: str
    category: str = "Custom"

@app.post("/label/instruments")
async def label_add_instrument(i: InstrumentIn):
    DB.add_instrument(i.name, i.category)
    return {"instruments": DB.get_instruments()}

@app.get("/label/singers")
async def label_get_singers():
    return {"singers": DB.get_singers()}

# ── Export / GitHub sync ──────────────────────────────────────────────────────
@app.get("/label/export/{song_id}")
async def label_export(song_id: int):
    data      = DB.export_song_json(song_id)
    out_dir   = DATA_DIR / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file  = out_dir / f"song_{song_id}.json"
    out_file.write_text(json.dumps(data, indent=2))
    return {"path": str(out_file), "data": data}

@app.post("/label/sync-github")
async def label_sync_github():
    repo_dir  = BASE_DIR
    out_dir   = DATA_DIR / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_songs = DB.list_songs()
    for s in all_songs:
        d = DB.export_song_json(s["id"])
        (out_dir / f"song_{s['id']}.json").write_text(json.dumps(d, indent=2))
    try:
        subprocess.run(["git","checkout","-B","training-data"], cwd=repo_dir, capture_output=True)
        subprocess.run(["git","add", str(out_dir)], cwd=repo_dir, capture_output=True)
        res = subprocess.run(["git","commit","-m","chore: update training labels"],
                              cwd=repo_dir, capture_output=True, text=True)
        subprocess.run(["git","push","origin","training-data","--force"],
                        cwd=repo_dir, capture_output=True)
        return {"ok": True, "message": res.stdout.strip() or "Synced to training-data branch"}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}

@app.get("/label/stats")
async def label_stats():
    return DB.stats()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _sse(t, msg):
    return f"data: {json.dumps({'type':t,'message':msg})}\n\n"

def _sse_rich(ev):
    t = ev.get("event","log")
    if t=="stage":
        n,s=ev.get("name",""),ev.get("status","")
        p={"type":"stage","name":n,"status":s,"message":f"[{n.upper()}] {s}",
           **{k:v for k,v in ev.items() if k not in("event","name","status")}}
    elif t=="segment":
        g,s,e,c,rem=ev.get("gender","?"),ev.get("start",0),ev.get("end",0),ev.get("confidence",0),ev.get("remove",False)
        p={"type":"segment","message":f"[SEG] {s:.1f}s–{e:.1f}s {g} {c:.0%} {'→REMOVE' if rem else '→keep'}",**ev}
    elif t=="done":  p={"type":"done","message":"Pipeline complete!","output":ev.get("output","")}
    elif t=="device":p={"type":"info","message":f"[DEVICE] {ev.get('mode','').upper()} {ev.get('name','')}"}
    elif t=="info":  p={"type":"info","message":f"Input: {ev.get('input','')} | Remove: {ev.get('remove','')}"}
    elif t=="error": p={"type":"error","message":f"[ERROR] {ev.get('msg',str(ev))}"}
    else:            p={"type":"log","message":str(ev)}
    return f"data: {json.dumps(p)}\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
