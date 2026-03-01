# nlu/__init__.py
"""
Expose the main NLU functions so they can be imported cleanly like:
    from nlu import detect_intent, extract_slots_from_text, check_contradictions
"""

from .intent_detector import detect_intent
from .slot_extractor import extract_slots_from_text
from .contradiction_checker import (
    check_contradictions,
    generate_confirmation_prompt
)

# Optional: you can add a small convenience function if you want
# (not required — just for nicer usage sometimes)

def process_text_input(message: str) -> dict:
    """
    Quick helper: run the full lightweight NLU pipeline in one call.
    Useful mainly for debugging / testing.
    
    Returns dict like:
    {
        "intent": "skincare_input",
        "slots": {...},
        "contradiction": PendingConfirmation or None
    }
    """
    intent = detect_intent(message)
    slots = {}
    contradiction = None

    if intent in ["skincare_input", "mixed"]:
        slots = extract_slots_from_text(message)
        # Note: contradiction check needs state → usually done in main.py
        # Here we just return raw slots for simplicity

    return {
        "intent": intent,
        "slots": slots,
        "contradiction": contradiction  # usually None here
    }