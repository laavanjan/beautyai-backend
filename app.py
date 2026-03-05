"""
app.py — Beauty Mart Chatbot
Session-based only. No user accounts.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional, List

from state_manager import get_dialog_state, save_dialog_state
from conversational_engine import process_message
from models import DialogState

app = FastAPI(title="Beauty Mart Chatbot")

# ── CORS ──────────────────────────────────────────────────────────────────────
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id:     str
    message_text:   str = ""
    input_type:     Literal["text", "button"]
    button_payload: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# Typed so the frontend always gets a predictable shape.
# Optional fields are None when not relevant — frontend checks before rendering.
# ─────────────────────────────────────────────────────────────────────────────

class SuggestedOption(BaseModel):
    label:   str
    payload: Dict[str, Any]


class RankingSignals(BaseModel):
    rating:         Optional[float] = None
    review_count:   Optional[int]   = None
    purchase_count: Optional[int]   = None


class ProductAttributes(BaseModel):
    skin_types: List[str] = Field(default_factory=list)
    hair_types: List[str] = Field(default_factory=list)
    concerns:   List[str] = Field(default_factory=list)
    texture:    Optional[str] = None
    free_from:  List[str] = Field(default_factory=list)


class ProductDetail(BaseModel):
    """Full product object returned by the product detail agent."""
    product_id:      str
    name:            str
    brand:           str
    image_url:       str
    price:           float
    description:     str                   = ""
    how_to_use:      str                   = ""
    key_ingredients: List[str]             = Field(default_factory=list)
    attributes:      ProductAttributes     = Field(default_factory=ProductAttributes)
    ranking_signals: RankingSignals        = Field(default_factory=RankingSignals)
    in_stock:        bool                  = True


class RoutineStep(BaseModel):
    """One step in a personalised routine."""
    step_number:  int
    routine_step: str          # internal key e.g. "cleanser"
    step_name:    str          # display name e.g. "Cleanser"
    purpose:      str          # one-line description shown on the card
    product:      Dict[str, Any]


class ChatResponse(BaseModel):
    reply_text:            str
    suggested_options:     List[SuggestedOption]        = Field(default_factory=list)
    current_node:          str                          = "conversation"

    # Product card carousel — shown when search agent runs
    products:              List[Dict[str, Any]]         = Field(default_factory=list)

    # Routine cards — shown when routine agent runs
    routine:               List[RoutineStep]            = Field(default_factory=list)

    # Product detail view — shown when product detail agent runs
    show_product_detail:   bool                         = False
    product_detail:        Optional[ProductDetail]      = None
    complementary_products:List[Dict[str, Any]]         = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {"status": "ok", "service": "BeautyAI Backend"}

@app.options("/chat")
async def options_chat():
    return {}

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest):
    state = await get_dialog_state(req.session_id)

    # ── Resolve user message from text or button payload ─────────────────────
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

    if not user_message:
        raise HTTPException(status_code=400, detail="message_text or button_payload.value is required")

    # ── Hard reset ───────────────────────────────────────────────────────────
    if user_message.lower() in ("restart", "start over", "reset", "begin again"):
        state = DialogState(session_id=req.session_id)
        user_message = "Hello, I'd like to start fresh."

    # ── Run engine ───────────────────────────────────────────────────────────
    response = await process_message(state, user_message)
    await save_dialog_state(req.session_id, state)

    # ── Validate + return typed response ─────────────────────────────────────
    # Pydantic will coerce and validate — any missing fields get defaults,
    # any type mismatches surface immediately instead of silently breaking
    return ChatResponse(**response)


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG ROUTES
# ─────────────────────────────────────────────────────────────────────────────

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