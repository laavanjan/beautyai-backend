import json
import logging
from groq import Groq
from config import GROQ_API_KEY, LLM_PROVIDER, MODEL
from typing import Dict, Any
from models import ExtractedSlots

client = Groq(api_key=GROQ_API_KEY)
logging.basicConfig(level=logging.INFO)  # Add for production debugging

SLOT_EXTRACTION_PROMPT = """You are extracting structured slot values from a beauty/skincare shopping conversation.

Extract ANY of the following slots that are mentioned, implied, or can be confidently inferred from the user's message.
Be helpful and reasonably infer — if the intent is obvious, include the slot.
Do NOT require exact wording — infer from common phrases.

Supported slots:
- main_category     : "Face", "Hair", "Body", "Baby"
- skin_type         : "oily", "dry", "combination", "normal", "sensitive"
- skin_concern      : list e.g. ["acne", "dryness", "aging", "oil control"]
- hair_type         : "straight", "wavy", "curly", "coily"
- hair_concern      : list e.g. ["dandruff", "hair fall", "dryness"]
- age_range         : "teen", "20s", "30s", "40+", "50+"
- sensitivity       : "very sensitive", "sensitive", "normal"
- goal              : e.g. "brightening", "hydration", "anti-aging"
- baby_section      : "Baby Bath & Shampoo", "Baby Lotions & creams", "Baby Milk Powder"

Examples:
"I need baby bath and shampoo" → {"main_category": "Baby", "baby_section": "Baby Bath & Shampoo"}
"lotion for baby" → {"main_category": "Baby", "baby_section": "Baby Lotions & creams"}
"baby products" → {"main_category": "Baby"}
"face for acne oily skin" → {"main_category": "Face", "skin_concern": ["acne"], "skin_type": "oily"}

Return valid JSON only. Use null if a slot is not present.
No explanations, no extra text."""

def extract_slots_from_text(message: str) -> Dict[str, Any]:
    if not message.strip():
        return {}

    try:
        print(f"[SLOT_DEBUG] Sending to {LLM_PROVIDER} - {MODEL}: {message}")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_beauty_slots",
                    "description": "Extract structured beauty/skincare slots from user message",
                    "parameters": ExtractedSlots.model_json_schema()
                }
            }
        ]
        system_content = (
            "You are a precise slot extractor for a beauty shopping assistant. "
            "Always look for category, concern, type, and especially baby_section when Baby is mentioned. "
            "Infer reasonably from context. "
            "If user mentions a specific baby section like 'bath', 'lotion', 'milk powder' — map it to the exact value. "
            "NEVER respond with text — ONLY output the tool call with filled slots."
        )
        response = client.chat.completions.create(
        model=MODEL,  # ← change here
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": message}
        ],
        tools=tools,
        tool_choice="required",  # forces tool use
        temperature=0.2,         # slightly higher than 0.1 for better compliance
        max_tokens=300
        )
        
        # Groq returns tool call in response
        tool_call = response.choices[0].message.tool_calls[0]
        function_args = tool_call.function.arguments
        
        print(f"[SLOT_RAW_ARGS] {function_args}")

        # Parse the arguments as our Pydantic model
        extracted = ExtractedSlots.model_validate_json(function_args)
        
        # Convert to plain dict, remove nulls
        slots_dict = extracted.model_dump(exclude_none=True)
        
        print(f"[SLOT_PARSED] {slots_dict}")
        return slots_dict
        

    except Exception as e:
        print(f"[SLOT_ERROR] {type(e).__name__}: {e}")
        return {}
