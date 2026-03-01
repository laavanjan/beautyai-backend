"""
app.py  —  Beauty Mart Chatbot
Session-based only. No user accounts.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional

from state_manager import get_dialog_state, save_dialog_state
from conversational_engine import process_message
from models import DialogState

app = FastAPI(title="Beauty Mart Chatbot")

# ── CORS ─────────────────────────────────────────────────────────────────────
# Using wildcard to allow all origins — safe for a public chatbot with no auth
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message_text: str = ""
    input_type: Literal["text", "button"]
    button_payload: Optional[Dict[str, Any]] = None


@app.get("/")
async def health():
    return {"status": "ok", "service": "BeautyAI Backend"}


@app.post("/chat")
async def handle_chat(req: ChatRequest):
    state = await get_dialog_state(req.session_id)

    if req.input_type == "button" and req.button_payload:
        slot  = req.button_payload.get("slot")
        value = req.button_payload.get("value")
        if slot == "_text":
            user_message = value
        elif slot and slot != "_action":
            state.slots[slot] = value
            user_message = f"I selected: {value}"
        else:
            user_message = value or req.message_text
    else:
        user_message = req.message_text.strip()

    if user_message.lower() in ("restart", "start over", "reset", "begin again"):
        state = DialogState(session_id=req.session_id)
        user_message = "Hello, I'd like to start fresh."

    response = await process_message(state, user_message)
    await save_dialog_state(req.session_id, state)
    return response


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    state = await get_dialog_state(session_id)
    return {
        "session_id":     state.session_id,
        "slots":          state.slots,
        "products_count": len(state.products),
        "history_turns":  len(state.conversation_history),
    }


@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    from storage import delete_session
    delete_session(session_id)
    return {"status": "reset", "session_id": session_id}