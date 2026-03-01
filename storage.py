"""
storage.py
File-based session storage only. No user accounts, no user files.
Sessions live in data/sessions/{session_id}.json
"""
import json
import os
from datetime import datetime
from typing import Optional
from models import DialogState

DATA_DIR     = "data"
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")

os.makedirs(SESSIONS_DIR, exist_ok=True)


def get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def load_session(session_id: str) -> Optional[DialogState]:
    path = get_session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DialogState(**data)


def save_session(session_id: str, state: DialogState):
    path = get_session_path(session_id)
    data = state.model_dump()
    data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def delete_session(session_id: str):
    path = get_session_path(session_id)
    if os.path.exists(path):
        os.remove(path)