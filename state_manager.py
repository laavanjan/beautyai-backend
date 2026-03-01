"""
state_manager.py
Thin async wrapper around file-based storage.
Sessions are stored as JSON files in data/sessions/.
No user accounts — session lives as long as the browser tab is open.
"""
from models import DialogState
from storage import load_session, save_session


async def get_dialog_state(session_id: str) -> DialogState:
    state = load_session(session_id)

    if state is None:
        # Brand-new session — create and persist immediately
        state = DialogState(session_id=session_id)
        save_session(session_id, state)

    return state


async def save_dialog_state(session_id: str, state: DialogState) -> None:
    save_session(session_id, state)