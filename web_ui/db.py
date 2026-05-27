"""
SQLite database layer for the labeling tool.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "labels.db"

DEFAULT_INSTRUMENTS = [
    ("Tabla",            "Percussion"),
    ("Mridangam",        "Percussion"),
    ("Ghatam",           "Percussion"),
    ("Kanjira",          "Percussion"),
    ("Dholak",           "Percussion"),
    ("Dhol",             "Percussion"),
    ("Pakhawaj",         "Percussion"),
    ("Thavil",           "Percussion"),
    ("Naal",             "Percussion"),
    ("Drum Kit",         "Percussion"),
    ("Sitar",            "Strings (Plucked)"),
    ("Saraswati Veena",  "Strings (Plucked)"),
    ("Tanpura",          "Strings (Plucked)"),
    ("Sarod",            "Strings (Plucked)"),
    ("Mandolin",         "Strings (Plucked)"),
    ("Gottuvadyam",      "Strings (Plucked)"),
    ("Guitar (Acoustic)","Strings (Plucked)"),
    ("Guitar (Electric)","Strings (Plucked)"),
    ("Bass Guitar",      "Strings (Plucked)"),
    ("Violin",           "Strings (Bowed)"),
    ("Sarangi",          "Strings (Bowed)"),
    ("Dilruba",          "Strings (Bowed)"),
    ("Esraj",            "Strings (Bowed)"),
    ("Cello",            "Strings (Bowed)"),
    ("Bansuri",          "Wind (Indian)"),
    ("Nadaswaram",       "Wind (Indian)"),
    ("Shehnai",          "Wind (Indian)"),
    ("Venu",             "Wind (Indian)"),
    ("Flute",            "Wind (Western)"),
    ("Trumpet",          "Wind (Western)"),
    ("Saxophone",        "Wind (Western)"),
    ("Clarinet",         "Wind (Western)"),
    ("Trombone",         "Wind (Western)"),
    ("Harmonium",        "Keyboard"),
    ("Piano",            "Keyboard"),
    ("Keyboard",         "Keyboard"),
    ("Synthesizer",      "Keyboard"),
    ("Organ",            "Keyboard"),
    ("Male Solo",        "Vocals"),
    ("Female Solo",      "Vocals"),
    ("Male Chorus",      "Vocals"),
    ("Female Chorus",    "Vocals"),
    ("Background Vocals","Vocals"),
    ("Humming",          "Vocals"),
    ("Synthesizer Pad",  "Electronic"),
    ("Electronic Beats", "Electronic"),
    ("Strings Section",  "Orchestra"),
    ("Brass Section",    "Orchestra"),
    ("Full Orchestra",   "Orchestra"),
]

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS songs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        filename     TEXT NOT NULL,
        original_path TEXT,
        singer_name  TEXT,
        year         INTEGER,
        notes        TEXT,
        created_at   TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS stems (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        song_id     INTEGER NOT NULL,
        stem_name   TEXT NOT NULL,
        file_path   TEXT,
        duration_sec REAL,
        FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS segments (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        stem_id      INTEGER NOT NULL,
        start_sec    REAL NOT NULL,
        end_sec      REAL NOT NULL,
        label_type   TEXT,
        labels       TEXT,
        purity       TEXT,
        singer_name  TEXT,
        singer_gender TEXT,
        notes        TEXT,
        created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (stem_id) REFERENCES stems(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS instruments (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        name      TEXT UNIQUE NOT NULL,
        category  TEXT,
        is_custom INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS singers (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        name   TEXT UNIQUE NOT NULL,
        gender TEXT
    );
    """)
    for name, cat in DEFAULT_INSTRUMENTS:
        c.execute("INSERT OR IGNORE INTO instruments (name,category) VALUES (?,?)", (name, cat))
    conn.commit()
    conn.close()

# ── Songs ────────────────────────────────────────────────────────────────────

def create_song(filename, original_path, singer_name="", year=None, notes=""):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO songs (filename,original_path,singer_name,year,notes) VALUES (?,?,?,?,?)",
              (filename, original_path, singer_name, year, notes))
    song_id = c.lastrowid
    conn.commit(); conn.close()
    return song_id

def list_songs():
    conn = get_conn()
    rows = conn.execute("""
        SELECT s.*, COUNT(DISTINCT st.id) as stem_count,
               COUNT(DISTINCT sg.id) as segment_count
        FROM songs s
        LEFT JOIN stems st ON st.song_id = s.id
        LEFT JOIN segments sg ON sg.stem_id = st.id
        GROUP BY s.id ORDER BY s.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_song(song_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

# ── Stems ────────────────────────────────────────────────────────────────────

def create_stem(song_id, stem_name, file_path, duration_sec=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO stems (song_id,stem_name,file_path,duration_sec) VALUES (?,?,?,?)",
              (song_id, stem_name, file_path, duration_sec))
    stem_id = c.lastrowid
    conn.commit(); conn.close()
    return stem_id

def get_stems(song_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM stems WHERE song_id=? ORDER BY stem_name", (song_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── Segments ─────────────────────────────────────────────────────────────────

def create_segment(stem_id, start_sec, end_sec, label_type, labels, purity,
                   singer_name="", singer_gender="", notes=""):
    conn = get_conn()
    c = conn.cursor()
    labels_json = json.dumps(labels) if isinstance(labels, list) else labels
    c.execute("""INSERT INTO segments
        (stem_id,start_sec,end_sec,label_type,labels,purity,singer_name,singer_gender,notes)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (stem_id, start_sec, end_sec, label_type, labels_json, purity, singer_name, singer_gender, notes))
    seg_id = c.lastrowid
    conn.commit(); conn.close()
    return seg_id

def update_segment(seg_id, start_sec, end_sec, label_type, labels, purity,
                   singer_name="", singer_gender="", notes=""):
    conn = get_conn()
    labels_json = json.dumps(labels) if isinstance(labels, list) else labels
    conn.execute("""UPDATE segments SET
        start_sec=?,end_sec=?,label_type=?,labels=?,purity=?,
        singer_name=?,singer_gender=?,notes=? WHERE id=?""",
        (start_sec, end_sec, label_type, labels_json, purity,
         singer_name, singer_gender, notes, seg_id))
    conn.commit(); conn.close()

def delete_segment(seg_id):
    conn = get_conn()
    conn.execute("DELETE FROM segments WHERE id=?", (seg_id,))
    conn.commit(); conn.close()

def get_segments(song_id):
    conn = get_conn()
    rows = conn.execute("""
        SELECT sg.*, st.stem_name FROM segments sg
        JOIN stems st ON st.id = sg.stem_id
        WHERE st.song_id = ? ORDER BY st.stem_name, sg.start_sec
    """, (song_id,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        try: d["labels"] = json.loads(d["labels"] or "[]")
        except: d["labels"] = []
        result.append(d)
    return result

# ── Instruments ───────────────────────────────────────────────────────────────

def get_instruments():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM instruments ORDER BY category, name").fetchall()
    conn.close()
    by_cat = {}
    for r in rows:
        cat = r["category"] or "Other"
        by_cat.setdefault(cat, []).append({"id": r["id"], "name": r["name"], "is_custom": r["is_custom"]})
    return by_cat

def add_instrument(name, category="Custom"):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO instruments (name,category,is_custom) VALUES (?,?,1)",
                 (name.strip(), category))
    conn.commit(); conn.close()

# ── Singers ───────────────────────────────────────────────────────────────────

def get_singers():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM singers ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def add_singer(name, gender=""):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO singers (name,gender) VALUES (?,?)", (name.strip(), gender))
    conn.commit(); conn.close()

# ── Export ────────────────────────────────────────────────────────────────────

def export_song_json(song_id):
    song     = get_song(song_id)
    stems    = get_stems(song_id)
    segments = get_segments(song_id)
    return {"song": song, "stems": stems, "segments": segments,
            "exported_at": datetime.utcnow().isoformat()}

def stats():
    conn = get_conn()
    songs    = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    segments = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    singers  = conn.execute("SELECT COUNT(DISTINCT singer_name) FROM segments WHERE singer_name!=''").fetchone()[0]
    top_labels = conn.execute("""
        SELECT value as label, COUNT(*) as cnt FROM segments,
        json_each(segments.labels) GROUP BY value ORDER BY cnt DESC LIMIT 10
    """).fetchall()
    conn.close()
    return {"songs": songs, "segments": segments, "singers": singers,
            "top_labels": [dict(r) for r in top_labels]}
