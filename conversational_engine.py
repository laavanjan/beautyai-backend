"""
conversational_engine.py — BeautyAI v2

═══════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
═══════════════════════════════════════════════════════════════════

Every message passes through an INTENT AGENT first.
The intent agent classifies the message into one of two activities:

┌─────────────────────────────────────────────────────────────────┐
│  ACTIVITY 1 — Guided Profile Collection + Personalised Output   │
│                                                                 │
│  Steps (tracked in slots["_step"]):                             │
│    ask_concern  → ask user their main skin/hair concern         │
│    ask_type     → ask skin type (oily/dry…) or hair type        │
│    ask_allergy  → ask for ingredients to avoid                  │
│    ask_output   → "Do you want a ROUTINE or a PRODUCT LINE?"    │
│                                                                 │
│  Output branches:                                               │
│    "routine"      → Routine Agent builds personalised steps     │
│    "product_line" → Section Agent shows the best-fit section    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ACTIVITY 2 — Direct Product Discovery                          │
│                                                                 │
│  2A  Section Browse   — user names a section/product type       │
│      → Section Agent: show ALL products in that section         │
│                                                                 │
│  2B  Product Detail   — user mentions a specific product name   │
│      → Detail Agent:  show full product info + complementary    │
└─────────────────────────────────────────────────────────────────┘

AGENTS (5):
  1. Intent Agent      — classifies every message (no LLM on casual)
  2. Collection Agent  — guided Activity 1 flow (chat only, no products)
  3. Routine Agent     — builds personalised step-by-step routine
  4. Section Agent     — shows products in a section (browse or profile-filtered)
  5. Detail Agent      — shows full product details + complementary items

SHARED PIPELINE (process_message):
  ① Slot extractor    — pulls profile values from every user message
  ② Intent Agent      — classifies intent
  ③ Dispatch          — routes to correct agent
  ④ State save        — persists slots + conversation history
"""

import json
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from groq import Groq

from config import GROQ_API_KEY, MODEL
from models import DialogState, ConversationTurn
from products import (
    build_routine, format_routine_intro,
    search_products, get_section_products,
    PRODUCT_CATALOG,
)
from catalog_index import (
    CATALOG_INDEX,
    build_search_context,
    get_valid_values,
    has_filter_qualifier,
    detect_section,
    get_section_valid_filters,
)

client = Groq(api_key=GROQ_API_KEY)
CATALOG_SNAPSHOT = build_search_context()

# ═════════════════════════════════════════════════════════════════════════════
# INTENT TYPES
# ═════════════════════════════════════════════════════════════════════════════

INTENT_COLLECT        = "collect"         # continue guided profile collection
INTENT_SECTION_BROWSE = "section_browse"  # show products in a section
INTENT_PRODUCT_DETAIL = "product_detail"  # show details for a specific product
INTENT_CASUAL         = "casual"          # greeting / chitchat
INTENT_OFF_TOPIC      = "off_topic"       # non-beauty topic

# Activity 1 collection steps — tracked in slots["_step"]
STEP_ASK_CONCERN  = "ask_concern"
STEP_ASK_TYPE     = "ask_type"
STEP_ASK_ALLERGY  = "ask_allergy"
STEP_ASK_OUTPUT   = "ask_output"    # NEW — "routine or product line?"
STEP_DONE         = "done"

# Output choices from ask_output step
OUTPUT_ROUTINE      = "routine"
OUTPUT_PRODUCT_LINE = "product_line"


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _merge_slots(existing: Dict, incoming: Dict) -> Dict:
    merged = dict(existing)
    for key, value in incoming.items():
        if value is None:
            continue
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = list(dict.fromkeys(merged[key] + value))
        else:
            merged[key] = value
    return merged


def _profile_summary(slots: Dict) -> str:
    lines = []
    if slots.get("main_category"):  lines.append(f"Category : {slots['main_category']}")
    if slots.get("skin_type"):      lines.append(f"Skin type: {slots['skin_type']}")
    if slots.get("skin_concern"):
        c = slots["skin_concern"]
        lines.append(f"Concerns : {', '.join(c) if isinstance(c, list) else c}")
    if slots.get("hair_type"):      lines.append(f"Hair type: {slots['hair_type']}")
    if slots.get("hair_concern"):
        c = slots["hair_concern"]
        lines.append(f"Concerns : {', '.join(c) if isinstance(c, list) else c}")
    if slots.get("baby_section"):   lines.append(f"Baby section: {slots['baby_section']}")
    if slots.get("exclusions"):     lines.append(f"Avoid    : {', '.join(slots['exclusions'])}")
    return "\n".join(lines) if lines else "No profile yet."


def _llm_json(messages: List[Dict], temperature: float = 0.4, max_tokens: int = 500) -> Dict:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"[LLM] attempt {attempt+1} error: {e}")
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
    return {"message": "Sorry, could you repeat that? 😊", "buttons": []}


def _llm_text(messages: List[Dict], temperature: float = 0.5, max_tokens: int = 300) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM] attempt {attempt+1} error: {e}")
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
    return "Sorry, could you repeat that? 😊"


# ═════════════════════════════════════════════════════════════════════════════
# SLOT EXTRACTOR
# Runs on EVERY message — pulls profile values before any routing decision.
# ═════════════════════════════════════════════════════════════════════════════

def _detect_message_category(message: str) -> str:
    """
    Detect which product category a message is about.
    Uses CATALOG_INDEX section keywords + per-category concern lists
    so it automatically stays in sync with products_db.json.
    """
    msg = message.lower()

    # Baby — explicit word is always reliable
    if "baby" in msg:
        return "baby"

    # Build category signal words from live catalog on first call
    # (cached at module level after first invocation)
    cat_signals = _get_category_signals()

    # Check body before hair/face because "body wash" > "wash" ambiguity
    if any(w in msg for w in cat_signals["body"]):
        return "body"
    if any(w in msg for w in cat_signals["hair"]):
        return "hair"
    if any(w in msg for w in cat_signals["face"]):
        return "face"
    return "unknown"


def _get_category_signals() -> Dict[str, set]:
    """
    Build category → signal words from CATALOG_INDEX.
    Combines:
      - per-category subcategory names (lowercased words)
      - per-category concerns
      - per-category hair/skin types
    Cached at module level after first build.
    """
    if hasattr(_get_category_signals, "_cache"):
        return _get_category_signals._cache  # type: ignore[attr-defined]

    signals: Dict[str, set] = {"face": set(), "hair": set(), "body": set(), "baby": set()}

    for cat in ("face", "hair", "body", "baby"):
        bc = CATALOG_INDEX.get("by_category", {}).get(cat, {})

        # Section keyword map already scoped — pull subcategory name words
        for subcat in bc.get("subcategories", []):
            for word in subcat.lower().split():
                if len(word) > 3:
                    signals[cat].add(word)

        # Concerns are strong signals
        signals[cat].update(bc.get("concerns", []))

        # Hair/skin types
        signals[cat].update(bc.get("skin_types", []))
        signals[cat].update(bc.get("hair_types", []))

    # Add high-value explicit signals that might be too short or missed above
    signals["hair"].update({"hair", "scalp", "frizz", "dandruff", "shampoo",
                             "conditioner", "curly", "wavy", "coily", "straight"})
    signals["face"].update({"skin", "face", "acne", "pimple", "serum",
                             "toner", "cleanser", "moisturis", "sunscreen", "spf"})
    signals["body"].update({"body", "deodorant", "shower", "lotion"})
    signals["baby"].add("baby")

    # Prevent cross-contamination: remove "hair" words from face signals, etc.
    signals["face"] -= signals["hair"]
    signals["body"] -= signals["hair"]
    signals["body"] -= signals["face"]

    _get_category_signals._cache = signals  # type: ignore[attr-defined]
    return signals


def _build_slot_prompt(message: str, profile: str) -> str:
    msg_cat = _detect_message_category(message)

    # ── Pull ALL values live from CATALOG_INDEX — zero hardcoding ─────────────
    skin_types  = ", ".join(get_valid_values("skin_types"))       or "oily, dry, combination, normal, sensitive"
    hair_types  = ", ".join(get_valid_values("hair_types"))       or "straight, wavy, curly, coily"

    # Use per-category concern lists — catalog_index already separates them
    face_concerns = get_valid_values("concerns", "face")
    hair_concerns = get_valid_values("concerns", "hair")
    body_concerns = get_valid_values("concerns", "body") or face_concerns  # body re-uses face concerns

    # Fallback: if catalog doesn't have per-category concerns, split globally
    if not face_concerns or not hair_concerns:
        all_concerns = get_valid_values("concerns")
        hair_kw = {"frizz", "breakage", "dandruff", "hair", "scalp", "split",
                   "porosity", "shine", "volume", "sebum", "oily scalp"}
        hair_concerns = hair_concerns or [c for c in all_concerns if any(h in c.lower() for h in hair_kw)]
        face_concerns = face_concerns or [c for c in all_concerns if c not in hair_concerns]

    face_str = ", ".join(face_concerns) or "acne, dryness, dullness, hyperpigmentation, redness, wrinkles"
    hair_str = ", ".join(hair_concerns) or "frizz, dryness, dandruff, breakage, hair loss"
    body_str = ", ".join(body_concerns) or "dryness, irritation, sensitivity"

    # Sensitivity tags — strip _free suffix for readability in prompts
    def _excl_tags(cat: Optional[str] = None) -> str:
        tags = get_valid_values("sensitivity_safe", cat) or get_valid_values("sensitivity_safe")
        # Remove _free suffix so LLM extracts "sulfate" not "sulfate_free"
        bare = [t.replace("_free", "").replace("_", " ") for t in tags]
        return ", ".join(bare) or "sulfate, fragrance, paraben, alcohol, silicone"

    # Baby sections live from catalog
    baby_sections = ", ".join(
        CATALOG_INDEX.get("by_category", {}).get("baby", {}).get("subcategories", [])
    ) or "Baby Bath & Shampoo, Baby Lotions & Creams, Baby Milk Powder"

    if msg_cat == "baby":
        schema = (
            '{\n'
            '  "main_category": "Baby",\n'
            f'  "baby_section" : "{baby_sections}  or null",\n'
            f'  "exclusions"   : ["{_excl_tags("baby")}"] or null\n'
            '}\n'
            'Rules:\n'
            '- "baby shampoo" / "baby wash" → baby_section: first baby section\n'
            '- "baby lotion" / "baby cream" → baby_section: second baby section\n'
            '- "baby milk" / "formula"      → baby_section: third baby section\n'
            '- NEVER set skin_type, skin_concern, hair_type, hair_concern'
        )
    elif msg_cat == "hair":
        schema = (
            '{\n'
            '  "main_category": "Hair",\n'
            f'  "hair_type"    : "{hair_types}  or null",\n'
            f'  "hair_concern" : [{hair_str}]  or null,\n'
            f'  "exclusions"   : ["{_excl_tags("hair")}"] or null\n'
            '}\n'
            'Rules:\n'
            '- "frizzy" → hair_concern: ["frizz"]\n'
            '- "dry hair" → hair_concern: ["dryness"]\n'
            '- "hair fall" / "hair loss" → hair_concern: ["hair loss"]\n'
            '- "dandruff" / "itchy scalp" → hair_concern: ["dandruff"]\n'
            '- concerns MUST be arrays. NEVER set skin_type or skin_concern.'
        )
    elif msg_cat == "body":
        schema = (
            '{\n'
            '  "main_category": "Body",\n'
            f'  "skin_type"    : "{skin_types}  or null",\n'
            f'  "skin_concern" : [{body_str}]  or null,\n'
            f'  "exclusions"   : ["{_excl_tags("body")}"] or null\n'
            '}\n'
            'Rules: Only extract if explicitly mentioned. NEVER set hair fields.'
        )
    else:
        # Face (default)
        schema = (
            '{\n'
            '  "main_category": "Face | Hair | Body | Baby  or null",\n'
            f'  "skin_type"    : "{skin_types}  or null",\n'
            f'  "skin_concern" : [{face_str}]  or null,\n'
            f'  "exclusions"   : ["{_excl_tags("face")}"] or null\n'
            '}\n'
            'Rules:\n'
            '- "oily skin" → skin_type: "oily"\n'
            '- "sensitive skin" → skin_type: "sensitive", skin_concern: ["sensitivity"]\n'
            '- "acne" / "pimples" / "breakouts" → skin_concern: ["acne"]\n'
            '- "dark spots" / "pigmentation"    → skin_concern: ["hyperpigmentation"]\n'
            '- "wrinkles" / "fine lines"        → skin_concern: ["aging"]\n'
            '- concerns MUST be arrays. NEVER set hair fields.'
        )

    return (
        f'Extract beauty profile slots.\nCurrent profile: {profile}\nMessage: "{message}"\n\n'
        'Return ONLY valid JSON matching this schema. Use null for anything not mentioned.\n'
        + schema +
        '\n\nCRITICAL: Only extract what is EXPLICITLY stated. Never infer or assume.\n'
        '"no allergies" / "none" / "nothing" → exclusions: null\n'
        'exclusions must be arrays: ["sulfate"] not "sulfate"'
    )


def _extract_slots(message: str, state: DialogState) -> Dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": _build_slot_prompt(
                message=message,
                profile=_profile_summary(state.slots),
            )}],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        result = {k: v for k, v in parsed.items() if v is not None}

        # ── Sanitise exclusions against real catalog values ────────────────────
        # The LLM halluccinates values like "soap", "tear", "water", "no more tears".
        # Only keep values that exist in CATALOG_INDEX["sensitivity_safe"].
        # Also: skip exclusion extraction entirely for baby products — the LLM
        # extracts product marketing language ("no more tears", "soap-free") as
        # exclusions, which is always wrong.
        is_baby = (
            result.get("main_category", "").lower() == "baby"
            or state.slots.get("main_category", "").lower() == "baby"
        )
        if is_baby and "exclusions" in result:
            print(f"[SLOT_EXTRACT] Dropping all exclusions for baby category: {result['exclusions']}")
            del result["exclusions"]
        elif "exclusions" in result:
            valid_bare = {
                tag.lower().removesuffix("_free").replace("_", " ")
                for tag in CATALOG_INDEX.get("sensitivity_safe", [])
            }
            raw_excl = result["exclusions"]
            if isinstance(raw_excl, str):
                raw_excl = [raw_excl]
            clean = []
            for e in raw_excl:
                bare = e.lower().replace("-", "_").replace(" ", "_").removesuffix("_free").replace("_", " ")
                if bare in valid_bare:
                    clean.append(bare.replace(" ", "_"))   # store as "sulfate"
                else:
                    print(f"[SLOT_EXTRACT] Dropped unrecognised exclusion: '{e}'")
            if clean:
                result["exclusions"] = clean
            else:
                del result["exclusions"]   # all were garbage — remove the key

        # ── Sanitise hair_type — LLM sometimes returns "dry hair" not "dry" ────
        if "hair_type" in result:
            raw_ht = str(result["hair_type"]).lower().strip()
            valid_hair_types = set(CATALOG_INDEX.get("hair_types", []))
            # Strip "hair" suffix — "dry hair" → "dry", "wavy hair" → "wavy"
            cleaned_ht = raw_ht.replace(" hair", "").strip()
            if cleaned_ht in valid_hair_types:
                result["hair_type"] = cleaned_ht
            elif raw_ht in valid_hair_types:
                result["hair_type"] = raw_ht
            else:
                print(f"[SLOT_EXTRACT] Dropped unrecognised hair_type: '{raw_ht}'")
                del result["hair_type"]

        # ── Sanitise hair_concern — deduplicate and validate ───────────────────
        if "hair_concern" in result:
            raw_hc = result["hair_concern"]
            if isinstance(raw_hc, str):
                raw_hc = [raw_hc]
            valid_concerns = set(CATALOG_INDEX.get("concerns", []))
            clean_hc = []
            for c in raw_hc:
                c_clean = c.lower().replace(" ", "_")
                # Try both with and without underscore
                if c_clean in valid_concerns:
                    clean_hc.append(c_clean)
                elif c.lower() in valid_concerns:
                    clean_hc.append(c.lower())
                else:
                    # "dry hair" → "dryness" mapping
                    mapped = {"dry hair": "dryness", "frizzy": "frizz",
                              "hair fall": "hair_loss", "thinning": "hair_loss"}.get(c.lower())
                    if mapped and mapped in valid_concerns:
                        clean_hc.append(mapped)
                    else:
                        print(f"[SLOT_EXTRACT] Dropped unrecognised hair_concern: '{c}'")
            if clean_hc:
                result["hair_concern"] = list(dict.fromkeys(clean_hc))  # dedup, preserve order
            else:
                del result["hair_concern"]

        return result

    except Exception as e:
        print(f"[SLOT_EXTRACT] {e}")
        return {}


def _detect_category_from_slots(slots: Dict) -> Optional[str]:
    if slots.get("main_category"):      return slots["main_category"]
    if slots.get("hair_type") or slots.get("hair_concern"):  return "Hair"
    if slots.get("skin_type") or slots.get("skin_concern"):  return "Face"
    if slots.get("baby_section"):                            return "Baby"
    return None


# ═════════════════════════════════════════════════════════════════════════════
# INTENT AGENT
#
# Classifies every user message into an intent.
# Uses a mix of fast keyword rules (no LLM) + LLM call for ambiguous cases.
#
# Outputs one of: collect | section_browse | product_detail | casual | off_topic
# ═════════════════════════════════════════════════════════════════════════════

_CASUAL_PHRASES = {
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "thanks", "thank you", "ok", "okay", "sure", "great", "awesome", "nice",
    "bye", "goodbye", "see you", "haha", "lol", "cool", "wow",
}

_OFF_TOPIC_WORDS = {
    "weather", "football", "cricket", "sports", "news", "movie",
    "music", "recipe", "food", "restaurant", "travel", "hotel",
    "politics", "stock", "bitcoin", "crypto", "game", "coding",
}

# ── All beauty signal words built from CATALOG_INDEX — not hardcoded ──────────
# Covers every concern, type, section keyword, brand, and ingredient in the DB.
_BEAUTY_WORDS: set = (
    set(CATALOG_INDEX.get("concerns", []))
    | set(CATALOG_INDEX.get("skin_types", []))
    | set(CATALOG_INDEX.get("hair_types", []))
    | set(w.lower() for w in CATALOG_INDEX.get("brands", []))
    | set(CATALOG_INDEX.get("section_keywords", {}).keys())
    | {"skin", "hair", "face", "body", "baby", "product", "routine",
       "beauty", "allerg", "sensitiv", "moisturis", "cleanser", "sunscreen",
       "spf", "lotion", "wash", "concern", "dry", "oily", "pimple"}
)

_DETAIL_PHRASES = {
    "tell me more about", "more about", "what is", "tell me about",
    "describe", "ingredients in", "ingredients of", "what are the ingredients",
    "how do i use", "how to use", "is it good for", "who is it for",
    "suitable for", "what does it do", "details about", "info on",
    "information about", "can you explain", "review of", "tell me about the",
}

_PROFILE_WORDS = {
    "my skin", "my hair", "i have", "i want", "i need", "i struggle",
    "my concern", "my problem", "i get", "prone to", "i suffer",
    "my scalp", "my face", "dealing with", "help me with",
    "recommend", "suggest", "looking for a routine",
}

# ── All section keywords + brand names live from catalog ──────────────────────
_SECTION_KEYWORDS: set = set(CATALOG_INDEX.get("section_keywords", {}).keys())
_BRAND_NAMES:      set = {b.lower() for b in CATALOG_INDEX.get("brands", []) if len(b) > 3}

# ── Product type words: section keywords + concern/type signals ───────────────
# These are words that suggest the user wants to SEE products in a category.
# Built from real catalog values so adding products auto-expands this set.
_PRODUCT_TYPE_WORDS: set = (
    _SECTION_KEYWORDS
    | set(CATALOG_INDEX.get("concerns", []))
    | set(CATALOG_INDEX.get("skin_types", []))
    | set(CATALOG_INDEX.get("hair_types", []))
    | {"show me", "do you have", "looking for", "find me", "got any"}
)


@dataclass
class IntentResult:
    intent:  str                      # one of the INTENT_* constants
    section: Optional[str] = None     # resolved section name (Activity 2A)
    product_name: Optional[str] = None  # product name hint (Activity 2B)
    output_pref:  Optional[str] = None  # "routine" | "product_line" (ask_output answer)
    reason:  str = ""                 # for debug logging


def classify_intent(
    message: str,
    state: DialogState,
    new_slots: Dict,
) -> IntentResult:
    """
    Classify user intent. Priority order:
      1. Casual/greeting — fast keyword (no LLM)
      2. Off-topic — fast keyword (no LLM)
      3. Product detail — detail phrase + products shown
      4. Output preference (routine vs product line) — if step=ask_output
      5. Section browse — section keyword detected
      6. Profile collection — profile words or collection step active
      7. LLM fallback — short LLM call for ambiguous cases
    """
    msg   = message.lower().strip()
    slots = state.slots
    step  = slots.get("_step", "")

    # ── 1. Casual ─────────────────────────────────────────────────────────────
    if msg in _CASUAL_PHRASES or (len(msg.split()) <= 3 and msg in _CASUAL_PHRASES):
        return IntentResult(INTENT_CASUAL, reason="casual phrase match")

    # ── 2. Off-topic ──────────────────────────────────────────────────────────
    has_beauty = any(w in msg for w in _BEAUTY_WORDS)
    has_offtopic = any(w in msg for w in _OFF_TOPIC_WORDS)
    if has_offtopic and not has_beauty:
        return IntentResult(INTENT_OFF_TOPIC, reason="off-topic keywords, no beauty words")

    # ── 3. Product detail ─────────────────────────────────────────────────────
    has_shown = bool(slots.get("_shown_products"))
    is_detail_phrase = any(phrase in msg for phrase in _DETAIL_PHRASES)
    if is_detail_phrase and has_shown:
        return IntentResult(INTENT_PRODUCT_DETAIL, reason="detail phrase + products shown")

    # ── 4. Output preference (answer to "routine or product line?") ───────────
    if step == STEP_ASK_OUTPUT:
        pref = _detect_output_preference(msg)
        if pref:
            return IntentResult(INTENT_COLLECT, output_pref=pref,
                                reason=f"output preference: {pref}")
        # If step=ask_output but no clear preference, re-ask
        return IntentResult(INTENT_COLLECT, reason="ask_output: preference unclear")

    # ── 5. Section browse ─────────────────────────────────────────────────────
    # CRITICAL: If the user's category is Baby, the section must come from
    # the baby_section slot — never from generic keyword detection which
    # would map "shampoo" → "Shampoos" (adult) instead of "Baby Bath & Shampoo".
    inferred_cat = (
        new_slots.get("main_category")
        or slots.get("main_category", "")
    ).lower()

    if inferred_cat == "baby":
        baby_section = (
            new_slots.get("baby_section")
            or slots.get("baby_section")
        )
        if baby_section:
            return IntentResult(
                INTENT_SECTION_BROWSE, section=baby_section,
                reason=f"baby category → slot-derived section: {baby_section}"
            )
        # baby category but no specific section yet → keep collecting
        return IntentResult(INTENT_COLLECT, reason="baby category, no section yet")

    section = detect_section(message)
    if section:
        return IntentResult(INTENT_SECTION_BROWSE, section=section,
                            reason=f"section keyword → {section}")

    # Brand name mention → section browse in that brand's range
    if any(brand in msg for brand in _BRAND_NAMES):
        brand_name = next(b for b in _BRAND_NAMES if b in msg)
        return IntentResult(INTENT_SECTION_BROWSE, reason=f"brand mention: {brand_name}")

    # ── 6. Profile collection signals ─────────────────────────────────────────
    # Active collection step: keep collecting
    if step in (STEP_ASK_CONCERN, STEP_ASK_TYPE, STEP_ASK_ALLERGY):
        return IntentResult(INTENT_COLLECT, reason=f"active step: {step}")

    # Profile words detected → Activity 1
    has_profile_word = any(w in msg for w in _PROFILE_WORDS)
    has_new_profile  = bool(new_slots.get("skin_type") or new_slots.get("skin_concern") or
                            new_slots.get("hair_type") or new_slots.get("hair_concern"))
    if has_profile_word or has_new_profile:
        return IntentResult(INTENT_COLLECT, reason="profile words / new profile slots detected")

    # Product type word but NO section and NO profile → could be section browse
    if any(w in msg for w in _PRODUCT_TYPE_WORDS):
        return IntentResult(INTENT_SECTION_BROWSE, reason="product type word without section match")

    # ── 7. LLM fallback for ambiguous messages ─────────────────────────────────
    return _classify_intent_llm(message, state, new_slots)


OUTPUT_EXPLAIN = "explain"   # user wants an explanation of the difference

def _detect_output_preference(msg: str) -> Optional[str]:
    """
    Detect whether the user wants a routine, a product line, or an explanation.

    Called when step == STEP_ASK_OUTPUT.

    Returns: OUTPUT_ROUTINE | OUTPUT_PRODUCT_LINE | OUTPUT_EXPLAIN | None
    """
    m = msg.lower().strip()

    # ── Check for QUESTION / EXPLAIN first ───────────────────────────────────
    # Must come before keyword checks — "what's the difference between a routine
    # and a product line?" contains "routine" but is NOT a preference answer.
    question_triggers = (
        "difference", "what's the difference", "whats the difference",
        "explain", "what do you mean", "what is a routine", "what is a product line",
        "how does", "tell me more", "can you explain", "what does", "versus", " vs ",
    )
    if any(t in m for t in question_triggers):
        return OUTPUT_EXPLAIN

    # ── Routine ───────────────────────────────────────────────────────────────
    routine_triggers = (
        "routine", "step by step", "steps", "build me a routine",
        "my routine", "full routine", "all the steps", "personalized routine",
        "personalised routine",
    )
    if any(t in m for t in routine_triggers):
        return OUTPUT_ROUTINE

    # ── Product line ──────────────────────────────────────────────────────────
    # Original triggers (explicit product_line language)
    product_line_triggers = (
        "product line", "product list", "products only", "just products",
        "show me products", "list of products", "all products",
    )
    # Also catch the ACTUAL button text we generate:
    # "Show me the best products for my main concern."
    # "Show me the best [section] products."
    concern_browse_triggers = (
        "best products", "products for my", "show me the best",
        "for my concern", "for my main concern", "products for my concern",
        "show me products", "for my skin", "for my hair",
    )
    if any(t in m for t in product_line_triggers + concern_browse_triggers):
        return OUTPUT_PRODUCT_LINE

    # Single word answers
    if m in ("routine", "routines"):
        return OUTPUT_ROUTINE
    if m in ("products", "product", "line", "browse", "concern"):
        return OUTPUT_PRODUCT_LINE

    return None


def _classify_intent_llm(message: str, state: DialogState, new_slots: Dict) -> IntentResult:
    """LLM-based fallback for ambiguous messages."""
    # Use full subcategory list from catalog — not a truncated keyword sample
    all_sections    = ", ".join(CATALOG_INDEX.get("subcategories", []))
    all_brands      = ", ".join(CATALOG_INDEX.get("brands", []))
    profile_summary = _profile_summary(state.slots)

    prompt = f"""You are classifying a user message for a beauty chatbot.

Current conversation step: {state.slots.get("_step", "none")}
Current user profile: {profile_summary}
Message: "{message}"

Available product sections: {all_sections}
Available brands: {all_brands}

Classify the intent as ONE of:
- "collect"        — user is providing skin/hair profile info (concern, type, allergy)
- "section_browse" — user wants to see products in a category/section/brand
- "product_detail" — user is asking about a specific named product
- "casual"         — greeting or general chitchat
- "off_topic"      — completely unrelated to beauty/skincare/haircare

Return ONLY valid JSON:
{{"intent": "collect|section_browse|product_detail|casual|off_topic", "section": "exact section name from list or null", "product_name": "product name or null"}}"""

    result = _llm_json(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=80,
    )

    intent = result.get("intent", INTENT_COLLECT)
    if intent not in (INTENT_COLLECT, INTENT_SECTION_BROWSE, INTENT_PRODUCT_DETAIL,
                      INTENT_CASUAL, INTENT_OFF_TOPIC):
        intent = INTENT_COLLECT

    return IntentResult(
        intent=intent,
        section=result.get("section"),
        product_name=result.get("product_name"),
        reason="LLM classification",
    )


# ═════════════════════════════════════════════════════════════════════════════
# COLLECTION AGENT (Activity 1)
#
# Manages the guided profile collection flow.
# Steps: ask_concern → ask_type → ask_allergy → ask_output
#
# ask_output is the NEW final step: "Would you like a full routine
# or would you prefer to browse a product line for your main concern?"
#
# After ask_output → Routine Agent OR Section Agent
# ═════════════════════════════════════════════════════════════════════════════

def _next_collection_step(slots: Dict) -> str:
    """Determine the next step needed based on what's been collected."""
    cat = slots.get("main_category")

    # Baby — only need category + section, no type/concern questions
    if cat == "Baby" and slots.get("baby_section"):
        return STEP_ASK_OUTPUT

    has_concern = bool(slots.get("skin_concern") or slots.get("hair_concern"))
    has_type    = bool(slots.get("skin_type") or slots.get("hair_type"))

    # Body — only needs one of concern or type
    if cat == "Body" and (has_concern or has_type):
        if slots.get("_allergy_asked"):
            return STEP_ASK_OUTPUT
        return STEP_ASK_ALLERGY

    if not has_concern:
        return STEP_ASK_CONCERN
    if not has_type:
        return STEP_ASK_TYPE
    if not slots.get("_allergy_asked"):
        return STEP_ASK_ALLERGY
    return STEP_ASK_OUTPUT


def _build_collection_prompt(state: DialogState, message: str, step: str) -> str:
    slots = state.slots
    cat   = (slots.get("main_category") or "").lower() or None

    concerns_list    = ", ".join(get_valid_values("concerns", cat))         or "dryness, acne, frizz, dullness, sensitivity"
    skin_types_list  = ", ".join(get_valid_values("skin_types"))            or "oily, dry, combination, normal, sensitive"
    hair_types_list  = ", ".join(get_valid_values("hair_types"))            or "straight, wavy, curly, coily"
    sensitivity_list = ", ".join(get_valid_values("sensitivity_safe", cat)) or "sulfate_free, fragrance_free, paraben_free"

    if step == STEP_ASK_CONCERN:
        task = f"""
TASK: Warmly greet the customer and ask for their main skin or hair concern.

STRUCTURE:
1. EMPATHISE — 1 warm personalised greeting
2. EDUCATE   — 1 sentence on why the right product match matters
3. ASK       — "What's your main skin or hair concern?"

BUTTONS (4 full sentences from catalog concerns):
Valid concerns: {concerns_list}
Examples:
  "My skin feels very dry and dehydrated."
  "I have oily skin and frequent breakouts."
  "My hair is frizzy and hard to manage."
  "I want to reduce dark spots and dullness."
"""
    elif step == STEP_ASK_TYPE:
        concern_str = ", ".join(
            slots.get("skin_concern") or slots.get("hair_concern") or ["your concern"]
        )
        is_hair = bool(slots.get("hair_concern") or slots.get("hair_type") or
                       (cat and "hair" in cat))
        if is_hair:
            task = f"""
TASK: Acknowledge their hair concern and ask their hair type.
Concern collected: {concern_str}

STRUCTURE:
1. EMPATHISE — acknowledge their specific concern with warmth
2. EDUCATE   — why hair type changes which ingredients work best
3. ASK       — "What's your hair type?"

BUTTONS (4 full sentences):
Valid hair types: {hair_types_list}
  "My hair is curly and tends to be dry."
  "I have straight hair — fine and flat."
  "My hair is wavy and gets frizzy in humidity."
  "My hair is coily and very tightly coiled."
"""
        else:
            task = f"""
TASK: Acknowledge their skin concern and ask their skin type.
Concern collected: {concern_str}

STRUCTURE:
1. EMPATHISE — acknowledge their specific concern with warmth
2. EDUCATE   — why skin type determines the right ingredients
3. ASK       — "What's your skin type?"

BUTTONS (4 full sentences):
Valid skin types: {skin_types_list}
  "I have oily skin — shiny by midday."
  "My skin is dry and feels tight after washing."
  "I have combination skin — oily T-zone, dry cheeks."
  "My skin is sensitive and reacts to most products."
"""
    elif step == STEP_ASK_ALLERGY:
        concern_str = ", ".join(slots.get("skin_concern") or slots.get("hair_concern") or [])
        type_str    = slots.get("skin_type") or slots.get("hair_type") or ""
        task = f"""
TASK: Ask about ingredient allergies before building the routine.
Profile so far — concern: {concern_str}, type: {type_str}

STRUCTURE:
1. EMPATHISE — acknowledge their profile warmly
2. EDUCATE   — why knowing allergies makes the routine safer
3. ASK       — "Are there any ingredients you're allergic to or want to avoid?"

BUTTONS (4 full sentences):
Valid sensitivity tags: {sensitivity_list}
  "I'm sensitive to fragrances — please avoid them."
  "I prefer sulfate-free products for my hair."
  "I'm allergic to parabens."
  "I don't have any specific allergies or sensitivities."
"""
    elif step == STEP_ASK_OUTPUT:
        profile_str = _profile_summary(slots)
        task = f"""
TASK: Profile collection is complete. Ask whether the user wants a ROUTINE or a PRODUCT LINE.

Profile collected:
{profile_str}

STRUCTURE:
1. CELEBRATE — acknowledge their complete profile with genuine excitement
2. EXPLAIN   — briefly explain the difference:
   - Routine = personalised step-by-step plan (Cleanser → Toner → Serum → Moisturiser etc.)
   - Product Line = browse the best products for their #1 concern in one category
3. ASK       — "Would you like a personalised routine, or would you prefer to browse a product line?"

BUTTONS (exactly 2 + 1 extra):
  "Build me a personalised routine with all the steps."
  "Show me the best products for my main concern."
  "What's the difference between a routine and a product line?"
"""
    else:
        task = "TASK: Warmly greet and ask for their main skin or hair concern."

    return f"""You are BeautyAI — a warm, empathetic beauty sales assistant at Beauty Mart Sri Lanka.
Your goal is to find the perfect products for this customer.

Current profile:
{_profile_summary(slots)}

{task}

TONE: Warm, specific, empathetic. React to what they said FIRST. 2–4 sentences max.
Emojis sparingly: ✨ 💧 🌿 😊 💆 🫶

Return ONLY valid JSON:
{{"message": "your reply", "buttons": ["Button 1", "Button 2", "Button 3", "Button 4"]}}"""


def run_collection_agent(state: DialogState, message: str) -> Dict:
    """Activity 1: guided collection. Determines next step from collected slots."""
    step = _next_collection_step(state.slots)

    # Mark allergy step as asked so we don't loop back
    if step == STEP_ASK_ALLERGY:
        state.slots["_allergy_asked"] = True

    state.slots["_step"] = step

    prompt = _build_collection_prompt(state, message, step)
    messages = [{"role": "system", "content": prompt}]
    for turn in state.conversation_history[-6:]:
        messages.append({"role": "user",      "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})
    messages.append({"role": "user", "content": message})

    result = _llm_json(messages, temperature=0.5, max_tokens=500)

    # ── CRITICAL: ask_output buttons must be EXACT strings ────────────────────
    # The intent dispatcher uses _detect_output_preference() which matches against
    # specific trigger words. If the LLM paraphrases these buttons (e.g. "Show me
    # products" instead of "Show me the best products for my main concern."),
    # the buttons silently fail. We hardcode them here so they can never drift.
    if step == STEP_ASK_OUTPUT:
        buttons = [
            "Build me a personalised routine with all the steps.",
            "Show me the best products for my main concern.",
            "What's the difference between a routine and a product line?",
        ]
    else:
        buttons = result.get("buttons", [])

    return {
        "reply_text":    result.get("message", ""),
        "buttons":       buttons,
        "show_products": False,
        "routine":       [],
        "products":      [],
        "_next_step":    step,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CASUAL AGENT
# Handles greetings and chitchat — always redirects to the beauty flow.
# ═════════════════════════════════════════════════════════════════════════════

def _build_casual_system(is_off_topic: bool = False) -> str:
    """Build casual/off-topic system prompt with live category list from catalog."""
    # Real categories from catalog — capitalised for display
    cats = [c.title() for c in CATALOG_INDEX.get("categories", ["face", "hair", "body", "baby"])]
    cat_str = ", ".join(cats)  # "Face, Hair, Body, Baby"

    if is_off_topic:
        return f"""You are BeautyAI at Beauty Mart Sri Lanka.
The user went off-topic (not about beauty/skincare/haircare).

1. Acknowledge briefly with humour or warmth (1 sentence)
2. Redirect gently: "I specialise in {cat_str} care — happy to help there!"

Return ONLY valid JSON:
{{"message": "your reply", "buttons": ["Let's talk skincare!", "Help me with my hair.", "Show me your product range."]}}"""
    else:
        return f"""You are BeautyAI at Beauty Mart Sri Lanka — a warm beauty sales assistant.
The user just said something casual or said hello.

We carry: {cat_str} care products.

Respond warmly (1–2 sentences) then naturally invite them into the beauty flow.
Suggest they tell you their skin or hair concern to get started.

Return ONLY valid JSON:
{{"message": "your reply", "buttons": ["I want help with my skincare routine.", "My hair needs some serious help!", "Show me what products you carry."]}}"""


def run_casual_agent(state: DialogState, message: str, is_off_topic: bool = False) -> Dict:
    result = _llm_json(
        messages=[
            {"role": "system", "content": _build_casual_system(is_off_topic)},
            {"role": "user",   "content": message},
        ],
        temperature=0.6,
        max_tokens=200,
    )
    return {
        "reply_text":    result.get("message", "Hey! Let's find you the perfect products. 😊"),
        "buttons":       result.get("buttons", [
            "Help me with my skincare.",
            "My hair needs attention!",
            "Show me what you carry.",
        ]),
        "show_products": False,
        "routine":       [],
        "products":      [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# ROUTINE AGENT (Activity 1 → "routine" branch)
#
# Called after full profile is collected AND user chose "routine".
# Builds a personalised step-by-step routine from the catalog.
# ═════════════════════════════════════════════════════════════════════════════

def _build_routine_intro_prompt(profile: str, routine_summary: str, slots: Dict) -> str:
    """Build routine intro prompt with catalog-driven follow-up buttons."""
    cat = (slots.get("main_category") or "face").lower()

    # Real concerns for this category — for personalised follow-up button suggestions
    cat_concerns  = get_valid_values("concerns", cat)
    user_concerns = slots.get("skin_concern") or slots.get("hair_concern") or []

    # Pick a concern the user has NOT already mentioned for the "update" button
    unused_concerns = [c for c in cat_concerns if c not in user_concerns]
    extra_concern   = unused_concerns[0].replace("_", " ") if unused_concerns else "redness"

    # Real sensitivity filters for this category for the "avoid" button
    cat_safety = get_valid_values("sensitivity_safe", cat)
    avoid_tag  = cat_safety[0].replace("_free", "").replace("_", " ") if cat_safety else "fragrance"

    # Price range context
    price_range = CATALOG_INDEX.get("by_category", {}).get(cat, {}).get("price_range", {})
    price_hint  = f"Price range for {cat}: Rs.{price_range['min']:,} – Rs.{price_range['max']:,}" if price_range else ""

    return f"""You are BeautyAI at Beauty Mart Sri Lanka.

User profile:
{profile}
{price_hint}

Routine built:
{routine_summary}

Write a warm intro following this structure:
1. EMPATHISE  — acknowledge their specific concern + skin/hair type by name
2. EDUCATE    — 1 sentence on why a layered routine works better than a single product
3. HANDOFF    — tell them their routine is ready, scroll through each step below 👇

Keep to 3 sentences total. Warm and specific, never generic.

Then EXACTLY 3 follow-up buttons — use the real values from their profile:
  "I want to avoid {avoid_tag} in my routine."
  "Can I get a cheaper option for one of the steps?"
  "I also struggle with {extra_concern} — can you update that?"

Return ONLY valid JSON:
{{"message": "intro", "buttons": ["Button 1", "Button 2", "Button 3"]}}"""


def run_routine_agent(state: DialogState) -> Dict:
    routine  = build_routine(state)
    products = [s["product"] for s in routine if s.get("product")]
    state.products = products

    if not products:
        # Fallback buttons use real category sections from catalog
        cat = (state.slots.get("main_category") or "face").lower()
        first_section = (
            CATALOG_INDEX.get("by_category", {}).get(cat, {}).get("subcategories", ["face care products"])[0]
        )
        return {
            "reply_text": "I couldn't find a perfect match for your profile right now — let me know if you'd like to adjust your preferences.",
            "buttons": [
                "Let me update my skin type.",
                f"Show me all {first_section}.",
                "Show me your best sellers.",
            ],
            "show_products": False,
            "routine": [],
            "products": [],
        }

    routine_summary = "\n".join(
        f"Step {s['step_number']}: {s['step_name']} — "
        f"{s['product'].get('name')} by {s['product'].get('brand')} "
        f"(Rs.{s['product'].get('price'):,})"
        for s in routine if s.get("product")
    )

    result = _llm_json(
        messages=[{"role": "user", "content": _build_routine_intro_prompt(
            profile=_profile_summary(state.slots),
            routine_summary=routine_summary,
            slots=state.slots,
        )}],
        temperature=0.4,
        max_tokens=300,
    )

    # Fallback buttons also use real catalog values
    cat         = (state.slots.get("main_category") or "face").lower()
    cat_safety  = get_valid_values("sensitivity_safe", cat)
    avoid_tag   = cat_safety[0].replace("_free", "").replace("_", " ") if cat_safety else "fragrance"
    return {
        "reply_text":    result.get("message", format_routine_intro(routine, state)),
        "buttons":       result.get("buttons", [
            f"I want to avoid {avoid_tag} in my routine.",
            "Can I get a cheaper option for one of the steps?",
            "Can I update my skin concern?",
        ]),
        "show_products": True,
        "routine":       routine,
        "products":      products,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION AGENT (Activity 1 → "product_line" branch + Activity 2A)
#
# Two modes:
#   Profile-filtered  — called from Activity 1 product_line path
#                       → scores by collected concern + type
#   Plain browse      — called from Activity 2A (user asked for a section)
#                       → returns full section sorted by rating
#
# Always honours profile exclusions (never shows excluded ingredients).
# ═════════════════════════════════════════════════════════════════════════════

def _product_has_excluded_ingredient(product: Dict, exclusions: List[str]) -> bool:
    """
    Return True if the product CONTAINS an ingredient the user wants to avoid.

    Logic:
      The product schema stores what a product IS FREE FROM in two places:
        - New schema : product["exclusions"]          e.g. ["fragrance", "paraben"]
        - Old schema : product["attributes"]["sensitivity_safe"] e.g. ["fragrance_free"]

      We check BOTH. A product is only excluded when it positively lacks the
      "free from" marker AND the exclusion is a real catalog value.

      IMPORTANT: We only block if the product fails ALL of the following:
        1. Product explicitly lists the exclusion in its "free from" set
        2. We never block because a product simply doesn't advertise a badge —
           most products don't list every ingredient they avoid. Only block when
           the product's own ingredient list positively contains the ingredient.

      Simplified safe rule used here:
        - Build the product's "free from" set from both schema paths
        - If the user says "avoid sulfate" and the product has sulfate_free → safe (show it)
        - If the product has NO free_from data at all → show it (benefit of doubt)
        - If the product DOES have free_from data AND the user's exclusion is NOT covered → skip it

      This prevents false positives where well-stocked products with partial
      safety tagging get incorrectly hidden.
    """
    if not exclusions:
        return False

    # Valid catalog exclusion values — only filter on real values
    valid_excl = set(CATALOG_INDEX.get("sensitivity_safe", []))

    attrs = product.get("attributes") or {}

    # Build free_from set from BOTH schema paths, normalised to "sulfate_free" format
    free_from: set = set()

    # Old schema: attributes.sensitivity_safe → already in "sulfate_free" format
    for tag in (attrs.get("sensitivity_safe") or []):
        free_from.add(tag.lower())

    # New schema: product["exclusions"] → bare names like "fragrance", "paraben"
    for tag in (product.get("exclusions") or []):
        bare = tag.lower().replace("-", "_").replace(" ", "_")
        free_from.add(bare)                          # "sulfate"
        free_from.add(f"{bare}_free")                # "sulfate_free"

    # If the product has NO free_from data at all → benefit of the doubt, show it
    if not free_from:
        return False

    for excl in exclusions:
        excl_clean = excl.lower().replace("-", "_").replace(" ", "_")

        # Normalise to bare name (strip _free if already tagged)
        bare_excl = excl_clean.removesuffix("_free")
        free_tag  = f"{bare_excl}_free"

        # Skip exclusions that aren't real catalog values (hallucinated by LLM)
        if free_tag not in valid_excl and bare_excl not in valid_excl:
            print(f"[EXCL_FILTER] Skipping unrecognised exclusion: '{excl}'")
            continue

        # Product has free_from data AND doesn't cover this exclusion → block it
        if free_tag not in free_from and bare_excl not in free_from:
            return True

    return False


def _resolve_section_for_profile(slots: Dict) -> Optional[str]:
    """
    For Activity 1 product_line path: find the best section for this profile.
    Uses CATALOG_INDEX to validate section names — if the catalog doesn't have
    "Serums & Targeted Treatments", this won't point there.
    """
    cat = (slots.get("main_category") or "").lower()
    concern_list = slots.get("skin_concern") or slots.get("hair_concern") or []

    # Live subcategories for this category — used to validate all lookups below
    valid_sections: set = set(
        CATALOG_INDEX.get("by_category", {}).get(cat, {}).get("subcategories", [])
        or CATALOG_INDEX.get("subcategories", [])
    )

    def _valid(section: str) -> Optional[str]:
        """Return section only if it exists in catalog, else None."""
        return section if section in valid_sections else None

    # Category default section — first subcategory in catalog order is most popular
    cat_first_section = (
        CATALOG_INDEX.get("by_category", {}).get(cat, {}).get("subcategories", [None])[0]
    )

    # Concern → most relevant section (validated against live catalog)
    concern_section_map = {
        "acne":              _valid("Cleansers"),
        "dullness":          _valid("Serums & Targeted Treatments"),
        "hyperpigmentation": _valid("Serums & Targeted Treatments"),
        "dark spots":        _valid("Serums & Targeted Treatments"),
        "dark_spots":        _valid("Serums & Targeted Treatments"),
        "dryness":           _valid("Moisturisers") or _valid("Body Creams & Lotions"),
        "dehydration":       _valid("Moisturisers"),
        "wrinkles":          _valid("Night Cream"),
        "aging":             _valid("Night Cream"),
        "sun damage":        _valid("Sun Care"),
        "sun_damage":        _valid("Sun Care"),
        "frizz":             _valid("Conditioners"),
        "hair loss":         _valid("Hair Masks & Deep Treatments"),
        "hair_loss":         _valid("Hair Masks & Deep Treatments"),
        "dandruff":          _valid("Shampoos"),
        "breakage":          _valid("Hair Masks & Deep Treatments"),
        "sensitivity":       _valid("Cleansers") or _valid("Moisturisers"),
        "redness":           _valid("Serums & Targeted Treatments") or _valid("Moisturisers"),
        "large pores":       _valid("Toners"),
        "blackheads":        _valid("Exfoliators"),
        "oiliness":          _valid("Toners") or _valid("Cleansers"),
        "oily scalp":        _valid("Shampoos"),
        "split ends":        _valid("Conditioners"),
    }

    for concern in concern_list:
        mapped = concern_section_map.get(concern)
        if mapped:
            return mapped

    # Baby — use the exact section they told us
    if cat == "baby" and slots.get("baby_section"):
        return slots["baby_section"]

    # Fall back to first subcategory for category (catalog-ordered)
    return cat_first_section


def _build_section_reply_prompt(
    message: str,
    mode: str,
    section: str,
    count: int,
    products_list: str,
    first_product_name: str,
    slots: Dict,
) -> str:
    """
    Build the section agent reply prompt.
    Injects real sensitivity filters from catalog_index for the specific section
    so Button 2 always suggests a filter that ACTUALLY EXISTS in this section.
    """
    # Valid filters for THIS section — e.g. Shampoos → ["sulfate_free", "paraben_free"]
    valid_filters = get_section_valid_filters(section)
    filter_examples = ""
    if valid_filters:
        # Human-readable: "sulfate_free" → "sulfate-free"
        readable = [f.replace("_free", "-free").replace("_", " ") for f in valid_filters[:3]]
        filter_examples = f"Valid filters for {section}: {', '.join(readable)}"
    else:
        # Fallback to global sensitivity tags
        global_tags = get_valid_values("sensitivity_safe")[:3]
        readable = [f.replace("_free", "-free").replace("_", " ") for f in global_tags]
        filter_examples = f"Available sensitivity filters: {', '.join(readable)}"

    # Price context from catalog
    cat = (slots.get("main_category") or "face").lower()
    price_range = CATALOG_INDEX.get("by_category", {}).get(cat, {}).get("price_range", {})
    budget_hint = ""
    if price_range:
        mid = int((price_range["min"] + price_range["max"]) / 2)
        budget_hint = f"Mid-range price for {cat}: Rs.{mid:,}"

    # Concern context for profile_filtered mode
    concerns = slots.get("skin_concern") or slots.get("hair_concern") or []
    concern_hint = f"User concern: {', '.join(concerns)}" if concerns and mode == "profile_filtered" else ""

    return f"""You are BeautyAI at Beauty Mart Sri Lanka.

User message: "{message}"
Mode: {mode}  (profile_filtered = personalised for their concern | plain_browse = they asked to see this section)
{concern_hint}

Showing {section} ({count} products):
{products_list}

Top product: {first_product_name}
{filter_examples}
{budget_hint}

Write a warm 2-sentence reply:
1. Acknowledge their request with enthusiasm — mention their specific concern if mode=profile_filtered
2. Tell them you're showing the {section} range, sorted best-match first

Then EXACTLY 3 buttons:
- Button 1: "Tell me more about the {first_product_name}."
- Button 2: A relevant follow-up using ONE of the valid filters listed above (e.g. "Show me only sulfate-free {section.lower()}.")
- Button 3: "Build me a full routine with this as one of the steps."

Return ONLY valid JSON:
{{"message": "reply", "buttons": ["Button 1", "Button 2", "Button 3"]}}"""


def run_section_agent(
    state: DialogState,
    message: str,
    section: Optional[str] = None,
    mode: str = "plain_browse",
    brand_filter: Optional[str] = None,
) -> Dict:
    """
    Section Agent — show products in a section.

    mode="profile_filtered"  → Activity 1 product_line path (ranked by profile)
    mode="plain_browse"      → Activity 2A (user asked for section directly)
    """
    slots       = state.slots
    exclusions  = slots.get("exclusions") or []
    cat         = (slots.get("main_category") or "").lower()

    # ── SAFETY GUARD: Baby category must only show baby sections ──────────────
    # The intent agent may resolve "shampoo" → "Shampoos" (adult) before seeing
    # the baby_section slot. Correct this here as a final gate.
    if cat == "baby" and section:
        baby_subcats = set(
            CATALOG_INDEX.get("by_category", {}).get("baby", {}).get("subcategories", [])
        )
        if section not in baby_subcats:
            # Override with the slot-derived baby section
            slot_section = slots.get("baby_section")
            print(f"[SECTION] Baby category mismatch — overriding '{section}' → '{slot_section}'")
            section = slot_section

    # Also ensure baby category ALWAYS uses baby_section slot if no section given
    if cat == "baby" and section is None:
        section = slots.get("baby_section")
        print(f"[SECTION] Baby category — using slot-derived section: {section}")

    # ── Resolve which section to show ─────────────────────────────────────────
    if section is None:
        if mode == "profile_filtered":
            section = _resolve_section_for_profile(slots)
        if section is None:
            # LLM extraction as last resort — scoped to known category if available
            known_cat = slots.get("main_category")
            params = _llm_json(
                messages=[{"role": "user", "content": _build_search_extract_prompt(
                    message, category=known_cat
                )}],
                temperature=0.0,
                max_tokens=200,
            )
            section = params.get("section")

    if not section:
        # Could not resolve a section — ask for clarification
        return {
            "reply_text": "I'm not sure which product range you're looking for. Could you tell me more? 😊",
            "buttons": [
                "Show me face care products.",
                "Show me hair care products.",
                "Show me all available sections.",
            ],
            "show_products": False,
            "routine":  [],
            "products": [],
        }

    # ── Fetch and filter products ──────────────────────────────────────────────
    skin_type    = slots.get("skin_type")
    skin_concern = slots.get("skin_concern")
    hair_type    = slots.get("hair_type")
    hair_concern = slots.get("hair_concern")
    has_profile  = skin_type or skin_concern or hair_type or hair_concern

    if mode == "profile_filtered" and has_profile:
        # Use scoring to rank by profile
        profile_excl_tags = [
            (e.lower().replace("-","_").replace(" ","_") + "_free"
             if not e.lower().endswith("_free") else e.lower())
            for e in exclusions
        ]
        search_kwargs = {k: v for k, v in {
            "section":          section,
            "skin_types":       [skin_type]    if skin_type    else None,
            "hair_types":       [hair_type]    if hair_type    else None,
            "concerns":         skin_concern or hair_concern or None,
            "sensitivity_safe": profile_excl_tags or None,
            "brand":            brand_filter,
            "limit":            8,
        }.items() if v is not None}
        print(f"[SECTION] profile-filtered kwargs: {search_kwargs}")
        products_to_show = search_products(**search_kwargs)

        # Fallback to plain section if scoring returns nothing
        if not products_to_show:
            products_to_show = [
                p for p in get_section_products(section)
                if not _product_has_excluded_ingredient(p, exclusions)
            ]
    else:
        # Plain browse — all products in section, exclusions filtered, sorted by rating
        print(f"[SECTION] plain browse: {section}")
        all_products = get_section_products(section)
        if brand_filter:
            all_products = [p for p in all_products
                            if (p.get("brand") or "").lower() == brand_filter.lower()]
        if exclusions:
            products_to_show = [
                p for p in all_products
                if not _product_has_excluded_ingredient(p, exclusions)
            ]
        else:
            products_to_show = all_products

    if not products_to_show:
        return {
            "reply_text":    f"I couldn't find products in the {section} range matching your preferences. Shall I show you the full range?",
            "buttons":       [
                f"Show me all {section}.",
                "Let me update my preferences.",
                "Show me a different section.",
            ],
            "show_products": False,
            "routine":       [],
            "products":      [],
        }

    # ── Generate reply ─────────────────────────────────────────────────────────
    products_list      = "\n".join(
        f"- {p['name']} by {p['brand']}  Rs.{p.get('price', 0):,}"
        for p in products_to_show[:8]
    )
    first_product_name = products_to_show[0]["name"]

    result = _llm_json(
        messages=[{"role": "user", "content": _build_section_reply_prompt(
            message=message,
            mode=mode,
            section=section,
            count=len(products_to_show),
            products_list=products_list,
            first_product_name=first_product_name,
            slots=slots,
        )}],
        temperature=0.5,
        max_tokens=300,
    )

    # Guarantee Button 1 references the first product
    buttons = result.get("buttons", [])
    if not buttons or first_product_name.lower() not in buttons[0].lower():
        b2 = buttons[1] if len(buttons) > 1 else f"Do you have a fragrance-free {section.lower()} option?"
        b3 = buttons[2] if len(buttons) > 2 else "Build me a full routine with this as one of the steps."
        buttons = [f"Tell me more about the {first_product_name}.", b2, b3]

    return {
        "reply_text":    result.get("message", f"Here's our {section} range! ✨"),
        "buttons":       buttons[:3],
        "show_products": True,
        "routine":       [],
        "products":      products_to_show[:8],
    }


def _build_search_extract_prompt(message: str, category: Optional[str] = None) -> str:
    """
    Build extraction prompt for the section agent LLM fallback.
    Uses category-scoped catalog context when category is known —
    gives the LLM a tighter, more accurate set of valid values.
    """
    # Scoped context if category is known, global otherwise
    cat_context = build_search_context(category) if category else build_search_context()

    # Category-specific fields for tighter prompting
    if category:
        cat_lower = category.lower()
        bc = CATALOG_INDEX.get("by_category", {}).get(cat_lower, {})
        valid_sections = ", ".join(bc.get("subcategories", CATALOG_INDEX["subcategories"]))
        valid_brands   = ", ".join(bc.get("brands",        CATALOG_INDEX["brands"]))
        valid_concerns = ", ".join(bc.get("concerns",      CATALOG_INDEX.get("concerns", [])))
        valid_safety   = ", ".join(bc.get("sensitivity_safe", CATALOG_INDEX.get("sensitivity_safe", [])))
        skin_types_str = ", ".join(bc.get("skin_types",    CATALOG_INDEX.get("skin_types", [])))
        hair_types_str = ", ".join(bc.get("hair_types",    CATALOG_INDEX.get("hair_types", [])))
    else:
        valid_sections = ", ".join(CATALOG_INDEX["subcategories"])
        valid_brands   = ", ".join(CATALOG_INDEX["brands"])
        valid_concerns = ", ".join(CATALOG_INDEX.get("concerns", []))
        valid_safety   = ", ".join(CATALOG_INDEX.get("sensitivity_safe", []))
        skin_types_str = ", ".join(CATALOG_INDEX.get("skin_types", []))
        hair_types_str = ", ".join(CATALOG_INDEX.get("hair_types", []))

    pr = CATALOG_INDEX.get("price_range", {})

    return f"""Extract search parameters for a beauty product catalog.

Catalog context:
{cat_context}

Valid sections        : {valid_sections}
Valid brands          : {valid_brands}
Valid concerns        : {valid_concerns}
Valid sensitivity tags: {valid_safety}
Valid skin types      : {skin_types_str}
Valid hair types      : {hair_types_str}
Price range           : Rs.{pr.get('min', 0):,} – Rs.{pr.get('max', 0):,}

Message: "{message}"

Return ONLY valid JSON using values EXACTLY as listed above:
{{
  "main_category"   : "face | hair | body | baby  or null",
  "section"         : "exact section name from list or null",
  "brand"           : "brand name or null",
  "skin_types"      : ["oily","dry",...] or null,
  "hair_types"      : ["curly","wavy",...] or null,
  "concerns"        : ["concern from list",...] or null,
  "sensitivity_safe": ["tag_free from list",...] or null,
  "max_price"       : number or null,
  "query"           : "specific product name or null"
}}

Only set section if clearly implied. Use null otherwise.
sensitivity_safe values MUST end in _free: "sulfate_free" not "sulfate"."""


# ═════════════════════════════════════════════════════════════════════════════
# DETAIL AGENT (Activity 2B)
#
# Triggered when user asks about a specific product by name.
# Looks up from shown products first, then full catalog.
# Returns full product info + complementary products.
# ═════════════════════════════════════════════════════════════════════════════

_DETAIL_EXTRACT_PROMPT = """A user is asking about a specific beauty product.

Message: "{message}"
Currently shown products:
{products_list}

Which product are they asking about?
Return ONLY valid JSON:
{{"product_id": "id or null", "product_name": "name or null"}}

Match by name — partial or approximate matches count.
If they say "the first one" or "the serum" pick the most likely match."""

def _build_detail_reply_prompt(
    name: str, brand: str, price: int, stock_label: str,
    best_for: str, free_from: str, section: str,
) -> str:
    """Build detail reply prompt with real section filters from catalog for Button 2."""
    section_filters = get_section_valid_filters(section)
    if section_filters:
        # Pick first available filter for this section
        readable_filter = section_filters[0].replace("_free", "-free").replace("_", " ")
        alt_button_hint = f'"Show me a {readable_filter} alternative in {section}."'
    else:
        alt_button_hint = '"Show me similar products at a lower price."'

    # Price context
    pr = CATALOG_INDEX.get("price_range", {})
    price_hint = f"Catalog price range: Rs.{pr.get('min',0):,} – Rs.{pr.get('max',0):,}" if pr else ""

    return f"""You are BeautyAI at Beauty Mart Sri Lanka.

Product being described:
Name     : {name}
Brand    : {brand}
Price    : Rs.{price:,} {stock_label}
Best for : {best_for}
Free from: {free_from}
Section  : {section}
{price_hint}

Write ONE warm enthusiastic intro sentence.
The frontend renders full details as a card — don't repeat ingredients or how-to-use.

EXACTLY 3 follow-up buttons:
- Button 1: "I'd like to buy the {name}."
- Button 2: {alt_button_hint}
- Button 3: "Show me my full routine." or "Show me other {section} options."

Return ONLY valid JSON:
{{"message": "one intro sentence", "buttons": ["Button 1", "Button 2", "Button 3"]}}"""


def _find_product_in_context(
    product_id: Optional[str],
    product_name: Optional[str],
    shown_products: List[Dict],
) -> Optional[Dict]:
    if not shown_products:
        return None
    if product_id:
        for p in shown_products:
            if p.get("product_id") == product_id:
                return p
    if product_name:
        name_lower = product_name.lower()
        for p in shown_products:
            if name_lower in p.get("name", "").lower():
                return p
        for p in shown_products:
            if p.get("name", "").lower() in name_lower:
                return p
    return shown_products[0] if shown_products else None


def run_detail_agent(state: DialogState, message: str) -> Dict:
    """Activity 2B: full product detail view."""
    shown = state.slots.get("_shown_products", [])

    products_list = "\n".join(
        f"- product_id: {p.get('product_id')}  name: {p.get('name')}"
        for p in shown[:10]
    ) if shown else "No products currently shown."

    extraction = _llm_json(
        messages=[{"role": "user", "content": _DETAIL_EXTRACT_PROMPT.format(
            message=message,
            products_list=products_list,
        )}],
        temperature=0.0,
        max_tokens=100,
    )

    product = _find_product_in_context(
        extraction.get("product_id"),
        extraction.get("product_name"),
        shown,
    )

    # Fallback: search full catalog by name
    if not product and extraction.get("product_name"):
        name_q = extraction["product_name"].lower()
        for p in PRODUCT_CATALOG:
            if name_q in p.get("name", "").lower():
                product = p
                break

    if not product:
        return {
            "reply_text":    "I'm not sure which product you mean — could you tell me the name? 😊",
            "buttons":       [
                "Tell me about the first product shown.",
                "Show me all products in this section.",
                "Start over and build my routine.",
            ],
            "show_products": False,
            "routine":       [],
            "products":      [],
        }

    attrs       = product.get("attributes", {})
    skin_types  = attrs.get("skin_types",       product.get("skin_types",  []))
    hair_types  = attrs.get("hair_types",        product.get("hair_types",  []))
    concerns    = attrs.get("concerns",          product.get("concerns",    []))
    safety_tags = attrs.get("sensitivity_safe",  product.get("sensitivity_safe", []))
    ingredients = product.get("key_ingredients", attrs.get("key_ingredients", []))

    type_list = skin_types or hair_types or []
    best_for  = ""
    if type_list:
        best_for += ", ".join(t.title() for t in type_list) + " skin/hair"
    if concerns:
        best_for += (" — " if best_for else "") + "targets " + ", ".join(concerns)

    free_from   = ", ".join(tag.replace("_", "-").title() for tag in safety_tags) if safety_tags else "No specific exclusions listed"
    in_stock    = product.get("in_stock", True)
    stock_label = "· In Stock ✅" if in_stock else "· Out of Stock ❌"
    section     = product.get("_section", "products")

    result = _llm_json(
        messages=[{"role": "user", "content": _build_detail_reply_prompt(
            name=product.get("name", ""),
            brand=product.get("brand", ""),
            price=product.get("price", 0),
            stock_label=stock_label,
            best_for=best_for or "All skin types",
            free_from=free_from,
            section=section,
        )}],
        temperature=0.4,
        max_tokens=250,
    )

    buttons = result.get("buttons", [])
    if len(buttons) < 3:
        # Fallback buttons also use real catalog data
        section_filters = get_section_valid_filters(section)
        alt_label = (
            f"Show me a {section_filters[0].replace('_free','').replace('_',' ')}-free alternative."
            if section_filters else "Show me similar products at a lower price."
        )
        buttons = [
            f"I'd like to buy the {product.get('name')}.",
            alt_label,
            f"Show me other {section} options.",
        ]

    # Complementary products from same section
    complementary = sorted(
        [p for p in PRODUCT_CATALOG
         if p.get("_section") == section
         and p.get("product_id") != product.get("product_id")
         and p.get("in_stock", True)],
        key=lambda p: -((p.get("ranking_signals") or {}).get("rating", 0)),
    )[:4]

    product_detail = {
        "product_id":      product.get("product_id"),
        "name":            product.get("name"),
        "brand":           product.get("brand"),
        "image_url":       product.get("image_url", ""),
        "price":           product.get("price", 0),
        "description":     product.get("description", ""),
        "how_to_use":      product.get("how_to_use", ""),
        "key_ingredients": ingredients,
        "attributes": {
            "skin_types":  skin_types,
            "hair_types":  hair_types,
            "concerns":    concerns,
            "texture":     attrs.get("texture", product.get("texture", "")),
            "free_from":   safety_tags,
        },
        "ranking_signals": {
            "rating":         (product.get("ranking_signals") or {}).get("rating", product.get("average_rating", 0)),
            "review_count":   (product.get("ranking_signals") or {}).get("review_count", product.get("review_count", 0)),
            "purchase_count": (product.get("ranking_signals") or {}).get("purchase_count", product.get("purchase_count", 0)),
        },
        "in_stock": in_stock,
    }

    return {
        "reply_text":             result.get("message", f"Here's everything about the {product.get('name')}! ✨"),
        "buttons":                buttons[:3],
        "show_products":          False,
        "show_product_detail":    True,
        "product_detail":         product_detail,
        "complementary_products": complementary,
        "routine":                [],
        "products":               [],
    }


def _run_explain_agent(state: DialogState) -> Dict:
    """
    Called when user asks 'what's the difference between a routine and a product line?'
    at the STEP_ASK_OUTPUT step.
    Explains both options clearly, then re-asks for their preference.
    Does NOT advance the step — still waits for a real preference answer.
    """
    cat = (state.slots.get("main_category") or "face").lower()
    # Get a real routine step example from catalog for this category
    cat_steps = CATALOG_INDEX.get("by_category", {}).get(cat, {}).get("subcategories", [])
    step_example = " → ".join(cat_steps[:3]) + "…" if cat_steps else "Cleanser → Toner → Serum…"

    # Get a real section example from catalog for the concern
    concerns = state.slots.get("skin_concern") or state.slots.get("hair_concern") or []
    section_example = _resolve_section_for_profile(state.slots) or (cat_steps[0] if cat_steps else "Serums")

    result = _llm_json(
        messages=[{"role": "user", "content": f"""You are BeautyAI at Beauty Mart Sri Lanka.

The user asked what the difference is between a ROUTINE and a PRODUCT LINE.

Explain clearly in 3–4 sentences:
- A ROUTINE = a personalised step-by-step plan built for their skin/hair type.
  Example for {cat}: {step_example}
  Each step has one recommended product, ordered so they work together.

- A PRODUCT LINE = browsing all products in ONE category they care about most.
  Example for concern "{', '.join(concerns) or cat}": showing all {section_example} products.
  Great if they just want to pick one product, not commit to a full routine.

End with: "Which would you like to try?"

Then EXACTLY 2 buttons (their actual choices):
  "Build me a personalised routine — all the steps."
  "Show me the best {section_example} for my {concerns[0] if concerns else cat}."

Return ONLY valid JSON:
{{"message": "explanation", "buttons": ["Button 1", "Button 2"]}}"""}],
        temperature=0.4,
        max_tokens=350,
    )

    return {
        "reply_text":    result.get("message", "A routine gives you a full step-by-step plan. A product line shows the best single category for your concern. Which would you prefer?"),
        # Hardcoded — must match trigger words in _detect_output_preference exactly
        "buttons":       [
            "Build me a personalised routine with all the steps.",
            "Show me the best products for my main concern.",
        ],
        "show_products": False,
        "routine":       [],
        "products":      [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def process_message(state: DialogState, user_message: str) -> Dict:
    """
    Single pipeline for every message:

      ① Slot extraction     — pull profile values from the message
      ② Intent Agent        — classify intent (collect / section_browse / product_detail / casual)
      ③ Dispatch            — route to correct agent
      ④ Post-processing     — update _shown_products, save turn, format buttons
    """

    # ── ① Slot extraction ─────────────────────────────────────────────────────
    prev_category = state.slots.get("main_category")
    new_slots     = _extract_slots(user_message, state)
    if new_slots:
        state.slots = _merge_slots(state.slots, new_slots)
        print(f"[SLOTS] {new_slots}")

    # Auto-detect category from slot signals if still missing
    if not state.slots.get("main_category"):
        cat = _detect_category_from_slots(state.slots)
        if cat:
            state.slots["main_category"] = cat

    # Track category changes for detecting category switches
    if prev_category and prev_category != state.slots.get("main_category"):
        state.slots["_prev_category"] = prev_category
        # Reset collection when user switches category
        state.slots["_step"] = ""
        state.slots.pop("_allergy_asked", None)
        print(f"[PIPELINE] Category switched {prev_category} → {state.slots['main_category']} — resetting collection")
    elif not state.slots.get("_prev_category"):
        state.slots["_prev_category"] = state.slots.get("main_category")

    # ── ② Intent Agent ────────────────────────────────────────────────────────
    intent_result = classify_intent(user_message, state, new_slots)
    print(f"[INTENT] {intent_result.intent}"
          + (f" section={intent_result.section}" if intent_result.section else "")
          + f" ← {intent_result.reason}")

    # ── ③ Dispatch ────────────────────────────────────────────────────────────
    step = state.slots.get("_step", "")

    if intent_result.intent == INTENT_CASUAL:
        agent_result = run_casual_agent(state, user_message, is_off_topic=False)

    elif intent_result.intent == INTENT_OFF_TOPIC:
        agent_result = run_casual_agent(state, user_message, is_off_topic=True)

    elif intent_result.intent == INTENT_PRODUCT_DETAIL:
        agent_result = run_detail_agent(state, user_message)

    elif intent_result.intent == INTENT_SECTION_BROWSE:
        # Activity 2A — direct section browse
        # Detect brand name from message for brand filtering
        msg_lower  = user_message.lower()
        brand_hit  = next((b for b in _BRAND_NAMES if b in msg_lower and len(b) > 3), None)
        brand_name = brand_hit.title() if brand_hit else None

        agent_result = run_section_agent(
            state,
            user_message,
            section=intent_result.section,
            mode="plain_browse",
            brand_filter=brand_name,
        )

    elif intent_result.intent == INTENT_COLLECT:
        # Activity 1 — guided collection flow

        output_pref = intent_result.output_pref

        # Handle output preference answer (routine vs product line)
        if output_pref == OUTPUT_ROUTINE or (step == STEP_ASK_OUTPUT and output_pref == OUTPUT_ROUTINE):
            state.slots["_step"] = STEP_DONE
            agent_result = run_routine_agent(state)

        elif output_pref == OUTPUT_PRODUCT_LINE or (step == STEP_ASK_OUTPUT and output_pref == OUTPUT_PRODUCT_LINE):
            state.slots["_step"] = STEP_DONE
            agent_result = run_section_agent(
                state,
                user_message,
                section=None,
                mode="profile_filtered",
            )

        elif output_pref == OUTPUT_EXPLAIN or (step == STEP_ASK_OUTPUT and output_pref == OUTPUT_EXPLAIN):
            # User asked "what's the difference?" — explain and re-ask
            agent_result = _run_explain_agent(state)

        else:
            # Still collecting — determine next step
            next_step = _next_collection_step(state.slots)

            if next_step == STEP_ASK_OUTPUT:
                # Profile complete — ask for preference
                agent_result = run_collection_agent(state, user_message)

            elif next_step == STEP_DONE:
                # Should not happen but handle gracefully — go to ask_output
                state.slots["_step"] = STEP_ASK_OUTPUT
                agent_result = run_collection_agent(state, user_message)

            else:
                # Still need more profile info
                agent_result = run_collection_agent(state, user_message)

    else:
        # Unknown intent — fallback to collection
        agent_result = run_collection_agent(state, user_message)

    # ── ④ Post-processing ─────────────────────────────────────────────────────

    # Remember shown products for Detail Agent
    if agent_result.get("products"):
        state.slots["_shown_products"] = agent_result["products"]
    elif agent_result.get("routine"):
        state.slots["_shown_products"] = [
            s["product"] for s in agent_result["routine"] if s.get("product")
        ]

    # Save conversation turn
    state.conversation_history.append(
        ConversationTurn(user=user_message, assistant=agent_result.get("reply_text", ""))
    )

    # Format suggested buttons
    buttons = [
        {"label": b, "payload": {"slot": "_text", "value": b}}
        for b in agent_result.get("buttons", [])[:4]
    ]

    return {
        "reply_text":             agent_result.get("reply_text", ""),
        "suggested_options":      buttons,
        "current_node":           "conversation",
        "routine":                agent_result.get("routine", []),
        "products":               agent_result.get("products", []) if agent_result.get("show_products") else [],
        "show_product_detail":    agent_result.get("show_product_detail", False),
        "product_detail":         agent_result.get("product_detail", None),
        "complementary_products": agent_result.get("complementary_products", []),
    }