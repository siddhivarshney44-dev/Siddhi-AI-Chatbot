import json
import os
import time

SESSIONS_DIR = "chat_sessions"


def _ensure_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _path(session_id):
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def new_session_id():
    return str(int(time.time() * 1000))


def list_sessions():
    _ensure_dir()
    files = [f[:-5] for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
    files.sort(key=lambda sid: os.path.getmtime(_path(sid)), reverse=True)
    return files


def load_session(session_id):
    try:
        with open(_path(session_id), "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_session(session_id, messages):
    _ensure_dir()
    with open(_path(session_id), "w") as f:
        json.dump(messages, f)


def delete_session(session_id):
    try: