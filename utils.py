from typing import List, Dict, Any
from models import DialogState, ChatResponse
import random

GREETING_MESSAGES = [
    "Welcome to Beauty Mart. ✨ How can I assist you with your skincare or beauty routine today?",
    "Hi there, it’s great to have you here at Beauty Mart. What are you looking for today?",
    "Hello and welcome. 🧴 Are you exploring skincare, hair care, body care, or baby products today?",
    "Hi, I’m your Beauty Mart assistant. 😊 How can I help you refine your routine or discover something new?",
    "Welcome back to Beauty Mart. ✨ Is there a particular concern or category you’d like to focus on today?",
    "Hello! I’m here to help you find products that truly fit your needs. What would you like to start with?",
    "Hi there. 🌿 Tell me a bit about what you’re hoping to improve in your skin or hair today.",
    "Welcome to Beauty Mart. 🛍️ Are you browsing for yourself or shopping for someone else today?",
    "Good to see you at Beauty Mart. ✨ Do you have a specific concern, like dryness or sensitivity, that you’d like to work on today?",
    "Hello, and thanks for stopping by. 😊 Are you in the mood to update your routine or just explore a few options?"
]



def build_button_options(labels_and_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standard way to create suggested_options / quick reply buttons
    Example input:
    [
        {"label": "Oily", "payload": {"value": "oily", "slot": "skin_type"}},
        {"label": "Dry",  "payload": {"value": "dry",  "slot": "skin_type"}},
    ]
    """
    return [
        {
            "label": item["label"],
            "payload": item["payload"]
        }
        for item in labels_and_payloads
    ]


def create_standard_response(
    state: DialogState,
    text: str,
    buttons: List[Dict[str, Any]] | None = None,
    products: List[Dict] | None = None,
    trigger_recommender: bool = False,
    recommender_context: Dict | None = None
) -> Dict:
    """
    Helper to create consistent response shape
    """
    resp = {
        "reply_text": text,
        "suggested_options": buttons or [],
        "current_node": state.current_node,
        "products": products or state.products,
        "trigger_recommender": trigger_recommender,
        "recommender_context": recommender_context or state.recommender_context
    }
    return resp


def get_category_emoji(category: str | None) -> str:
    """Just a little UX sugar"""
    emojis = {
        "Face": "✨",
        "Hair": "💆",
        "Body": "🧴",
        "Baby": "👶"
    }
    return emojis.get(category, "🛍️")


def has_all_required_slots(state: DialogState, required: List[str]) -> bool:
    """Quick check if we have enough info to recommend"""
    return all(slot in state.slots and state.slots[slot] is not None for slot in required)

def handle_product_question(state: DialogState, full_message: str) -> str:
    q = full_message.lower()
    if "retinol" in q:
        return (
            "Yes, retinol is excellent for anti-aging! It helps reduce fine lines, boost collagen, "
            "and improve skin texture. For dry skin, start with a low strength (0.1–0.3%) at night, "
            "2–3 times a week, and always pair with a good moisturizer + SPF during the day."
        )
    elif "niacinamide" in q:
        return "Niacinamide is great for oil control, pore minimization, and fading dark spots — perfect if you have oily or acne-prone skin."
    else:
        # Fallback — generic or use LLM
        return "That's a great question! Could you clarify what you'd like to know more about?"

