"""
conversational_engine.py — BeautyAI multi-agent sales engine

Key behaviours:
  - React to the user FIRST (empathy/acknowledgement), then help
  - Search products by any combination of fields (brand, texture, ingredient, concern, etc.)
  - Buttons are generated from REAL catalog values — never made up
  - NO unsolicited routines — only show a routine if the user explicitly asks for one
  - Show individual products or filtered sets when user asks for a specific item
  - Routine builder is only called when user says "show me a routine" or equivalent
"""

import json
import time
from typing import Optional, List, Dict, Any
from groq import Groq
from config import GROQ_API_KEY,MODEL
from models import DialogState, ConversationTurn
from products import (
    build_routine, format_routine_intro, get_step_alternatives,
    search_products, CATALOG_VALUES, PRODUCT_CATALOG
)

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# CATALOG SNAPSHOT — injected into prompts so LLM knows real values
# ─────────────────────────────────────────────────────────────────────────────

CATALOG_SNAPSHOT = f"""
Available product categories: {', '.join(CATALOG_VALUES['categories'])}
Available sections: {', '.join(CATALOG_VALUES['sections'])}
Available textures: {', '.join(CATALOG_VALUES['textures'])}
Available sensitivity tags: {', '.join(CATALOG_VALUES['sensitivity_tags'])}
Available brands: {', '.join(CATALOG_VALUES['brands'])}
Key ingredients in catalog: {', '.join(CATALOG_VALUES['ingredients'][:30])}
Price range: Rs.420 – Rs.1,950
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _has_enough_to_recommend(state: DialogState) -> bool:
    s   = state.slots
    cat = s.get("main_category")
    if not cat:
        return False
    if cat in ("Face", "Body"):
        return bool(s.get("skin_type") or s.get("skin_concern"))
    if cat == "Hair":
        return bool(s.get("hair_type") or s.get("hair_concern"))
    if cat == "Baby":
        return bool(s.get("baby_section"))
    return False


def _questions_asked(state: DialogState) -> int:
    return len(state.conversation_history)


def _summarize_profile(state: DialogState) -> str:
    s = state.slots
    if not s:
        return "No profile yet."
    lines = []
    if s.get("main_category"):  lines.append(f"Category: {s['main_category']}")
    if s.get("skin_type"):      lines.append(f"Skin type: {s['skin_type']}")
    if s.get("skin_concern"):
        c = s["skin_concern"]
        lines.append(f"Skin concerns: {', '.join(c) if isinstance(c, list) else c}")
    if s.get("hair_type"):      lines.append(f"Hair type: {s['hair_type']}")
    if s.get("hair_concern"):
        c = s["hair_concern"]
        lines.append(f"Hair concerns: {', '.join(c) if isinstance(c, list) else c}")
    if s.get("baby_section"):   lines.append(f"Baby section: {s['baby_section']}")
    if s.get("age_range"):      lines.append(f"Age: {s['age_range']}")
    if s.get("sensitivity"):    lines.append(f"Sensitivity: {s['sensitivity']}")
    if s.get("goal"):           lines.append(f"Goal: {s['goal']}")
    if s.get("exclusions"):     lines.append(f"Exclusions: {s['exclusions']}")
    return "\n".join(lines) if lines else "No profile yet."


def _summarize_products(state: DialogState) -> str:
    if not state.products:
        return "No products shown yet."
    lines = []
    for p in state.products[:6]:
        section = p.get("categories", {}).get("section", "")
        lines.append(f"- [{section}] {p.get('name')} by {p.get('brand')} — Rs.{p.get('price')}")
    return "\n".join(lines)


def _format_history_short(history) -> str:
    if not history:
        return "none"
    return "\n".join(f"U: {t.user[:60]} | B: {t.assistant[:60]}" for t in history)


def _merge_slots(existing: Dict, new_slots: Dict) -> Dict:
    merged = dict(existing)
    for key, value in new_slots.items():
        if value is None:
            continue
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = list(dict.fromkeys(merged[key] + value))
        else:
            merged[key] = value
    return merged


def _extract_exclusions(message: str) -> List[str]:
    keywords = ["no ", "without ", "exclude ", "don't want ", "not want ", "avoid ", "allergic to "]
    msg_lower = message.lower()
    result = []
    for kw in keywords:
        if kw in msg_lower:
            idx    = msg_lower.index(kw) + len(kw)
            phrase = message[idx:idx+30].split(",")[0].split(".")[0].strip()
            if phrase:
                result.append(phrase)
    return result


def _call_llm_json(messages: List[Dict], temperature: float = 0.4, max_tokens: int = 600) -> Dict:
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
        except json.JSONDecodeError as e:
            print(f"[LLM] JSON error attempt {attempt+1}: {e}")
        except Exception as e:
            print(f"[LLM] Error attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
    return {"message": "Sorry, could you repeat that? 😊", "slots": {}, "buttons": []}


# ─────────────────────────────────────────────────────────────────────────────
# SLOT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

QUICK_SLOT_PROMPT = """Extract beauty product profile slots from this message.
Current profile: {profile}
Message: "{message}"

Return ONLY valid JSON (null if not mentioned):
{{
  "main_category": "Face|Hair|Body|Baby or null",
  "skin_type": "oily|dry|combination|normal|sensitive or null",
  "skin_concern": ["acne","wrinkles","dryness","hyperpigmentation","sensitivity"] or null,
  "hair_type": "straight|wavy|curly|coily or null",
  "hair_concern": ["dandruff","hair fall","frizz","dryness","oily scalp"] or null,
  "baby_section": "Baby Bath & Shampoo|Baby Lotions & creams|Baby Milk Powder or null",
  "section": "exact section name or null"
}}

Rules:
- "milk powder", "baby milk", "baby powder" → main_category: "Baby", baby_section: "Baby Milk Powder"
- "baby shampoo", "baby bath", "baby wash" → main_category: "Baby", baby_section: "Baby Bath & Shampoo"
- "baby lotion", "baby cream", "baby moisturiser" → main_category: "Baby", baby_section: "Baby Lotions & creams"
- "baby" alone (no specific section) → main_category: "Baby"
- "shampoo", "hair shampoo" → main_category: "Hair", section: "Shampoos"
- "conditioner", "hair conditioner" → main_category: "Hair", section: "Conditioners"
- "hair mask", "hair treatment", "hair deep treatment" → main_category: "Hair", section: "Hair Masks & Deep Treatments"
- "cleanser", "face wash", "foaming wash" → main_category: "Face", section: "Cleansers"
- "toner" → main_category: "Face", section: "Toners"
- "serum" → main_category: "Face", section: "Serums & Targeted Treatments"
- "eye cream", "eye care" → main_category: "Face", section: "Eye Care"
- "moisturiser", "moisturizer", "day cream" → main_category: "Face", section: "Moisturisers"
- "night cream" → main_category: "Face", section: "Night Cream"
- "sunscreen", "spf", "sun protection" → main_category: "Face", section: "Sun Care"
- "exfoliator", "exfoliant", "scrub" → main_category: "Face", section: "Exfoliators"
- "body lotion", "body cream", "body wash", "body moisturiser" → main_category: "Body", section: "Body Creams & Lotions"
- "hand cream", "foot cream", "hand and foot" → main_category: "Body", section: "Hand and Foot Care"
- "hair products", "hair care" (generic, no specific product) → main_category: "Hair", section: null
- "face products", "skincare" (generic) → main_category: "Face", section: null
- "dry hair" → hair_concern: ["dryness"], NOT hair_type
- concerns MUST be arrays
- Only extract what is clearly stated"""


def _quick_slot_extract(message: str, state: DialogState) -> Dict:
    profile = _summarize_profile(state)
    prompt  = QUICK_SLOT_PROMPT.format(profile=profile, message=message)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        return {k: v for k, v in parsed.items() if v is not None}
    except Exception as e:
        print(f"[SLOT_EXTRACT] Error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """Classify this message for a beauty product sales chatbot.

User profile: {profile}
Products shown: {has_products}
Bot turns so far: {turns}
Recent history: {history}
Message: "{message}"

Catalog info:
{catalog}

Actions — pick EXACTLY ONE:
- "casual"          : greeting, thanks, small talk, who are you
- "off_topic"       : completely unrelated to beauty (weather, sports, etc.)
- "product_detail"  : user asks for details/description/ingredients/how-to-use about a SPECIFIC product
                      Examples: "tell me more about the SPF lotion", "what are the ingredients?",
                      "how do I use this?", "tell me about the DermaSoft cleanser",
                      "what does this product do?", "describe this product"
- "search"          : user asks for a product type, brand, ingredient, texture, concern, or price
                      Examples: "show me a gel cleanser", "I want fragrance-free products",
                      "recommend a moisturiser for dry skin", "I want something under Rs.1000"
- "qa"              : general question not about a specific product (e.g. "what is niacinamide?")
- "rebuild"         : wants to exclude something from shown products ("no retinol", "allergic to X")
- "refine_step"     : wants more options for a specific shown step ("show me other cleansers")
- "refine_price"    : wants cheaper or pricier version of shown products
- "new_category"    : switching category (hair→face, face→baby)
- "collect"         : need more info before we can do anything (missing category entirely)

IMPORTANT:
- "product_detail" takes priority over "qa" when user asks about a SPECIFIC product by name/reference
- "search" is the DEFAULT for any product-related request
- Only use "collect" if you truly cannot do anything without more info

Reply with ONLY the action word."""


def route_message(state: DialogState, message: str) -> str:
    msg_lower = message.lower().strip()

    # ── CONCERN COLLECT FLOW LOCK ─────────────────────────────────────────────
    # If the concern collect flow is active, ALWAYS stay in it until complete.
    # This prevents the router from hijacking button answers mid-flow.
    if state.slots.get("_concern_step") and state.slots["_concern_step"] != "done":
        return "concern_collect"

    # Fast local checks
    casual_phrases = {"hi", "hello", "hey", "thanks", "thank you", "good morning",
                      "good evening", "good afternoon", "ok", "okay", "great", "nice",
                      "who are you", "who r u", "what are you", "sup", "yo"}
    if msg_lower in casual_phrases or len(msg_lower) < 4:
        return "casual"

    # Catalog overview — "Shop by Category" pill
    if any(kw in msg_lower for kw in [
        "what categories", "what do you carry", "show me categories",
        "what sections", "shop by category", "what products do you have", "what can you help"
    ]):
        return "catalog_overview"

    # Concern collect flow — "Shop by Concern" pill
    if any(kw in msg_lower for kw in [
        "what concerns do you cover", "shop by concern", "shop by skin", "shop by hair concern"
    ]):
        return "concern_collect"

    # Best sellers — all categories OR specific category
    if any(kw in msg_lower for kw in ["best selling", "best sellers", "most popular", "top rated", "top products"]):
        # Detect if a specific category is mentioned
        for cat_kw, cat_val in [("face", "Face"), ("hair", "Hair"), ("body", "Body"), ("baby", "Baby")]:
            if cat_kw in msg_lower:
                state.slots["_filter_category"] = cat_val
                break
        else:
            state.slots.pop("_filter_category", None)  # clear — show all
        return "best_sellers"

    # Budget picks — all categories OR specific category
    if any(kw in msg_lower for kw in ["under rs.1000", "under rs 1000", "budget picks", "budget picks"]) or \
       ("budget" in msg_lower and "pick" in msg_lower):
        for cat_kw, cat_val in [("face", "Face"), ("hair", "Hair"), ("body", "Body"), ("baby", "Baby")]:
            if cat_kw in msg_lower:
                state.slots["_filter_category"] = cat_val
                break
        else:
            state.slots.pop("_filter_category", None)
        return "budget"

    # Refine actions (only if products already shown)
    detail_keywords = ["tell me more", "more about", "describe this", "describe the",
                       "what are the ingredients", "ingredients in", "how do i use",
                       "how to use", "what does this", "what does the", "details about",
                       "info about", "information about", "explain this", "explain the",
                       "what is in", "what's in this", "tell me about the"]
    if any(kw in msg_lower for kw in detail_keywords):
        return "product_detail"

    # Refine actions (only if products already shown)
    if state.products:
        refine_step_kw = ["other cleanser", "more serum", "different moistur", "other option",
                          "show me more", "other toner", "other mask", "swap"]
        refine_price_kw = ["cheaper", "more affordable", "budget option", "less expensive",
                           "pricier", "premium option", "luxury option", "higher end"]
        if any(kw in msg_lower for kw in refine_step_kw):
            return "refine_step"
        if any(kw in msg_lower for kw in refine_price_kw):
            return "refine_price"

    # Exclusion/rebuild
    if state.products and any(kw in msg_lower for kw in
                               ["no retinol", "allergic", "without ", "exclude ", "avoid "]):
        return "rebuild"

    # Category switch
    category_switch = ["instead", "actually i want", "switch to", "what about"]
    if state.slots.get("main_category") and any(kw in msg_lower for kw in category_switch):
        return "new_category"

    # FORCE SEARCH if user has given a category or any product hint
    # Don't keep asking questions — show products
    product_keywords = [
        "shampoo", "cleanser", "toner", "serum", "moisturiser", "moisturizer", "sunscreen",
        "spf", "mask", "conditioner", "wash", "cream", "lotion", "gel", "foam", "exfoliat",
        "eye cream", "product", "something", "show me", "what do you have", "do you have",
        "ingredient", "niacinamide", "vitamin c", "hyaluronic", "retinol", "fragrance",
        "oil-free", "paraben", "sulfate", "affordable", "cheap", "budget", "premium",
        "luxury", "under rs", "brand", "recommend", "suggest", "find me", "looking for",
        "want", "need", "buy", "purchase", "hair care", "skincare", "skin care",
        "face care", "body care", "daily", "everyday", "for my", "for oily", "for dry",
        "for sensitive", "for combination", "for normal", "for curly", "for frizzy"
    ]
    if any(kw in msg_lower for kw in product_keywords):
        return "search"

    # If category is already known → search, don't collect more
    if state.slots.get("main_category"):
        return "search"

    # After 2+ turns with no category → still collect (need at minimum a category)
    if _questions_asked(state) >= 2 and not state.slots.get("main_category"):
        return "collect"

    # LLM routing for ambiguous cases
    profile      = _summarize_profile(state)
    has_products = "yes" if state.products else "no"
    history      = _format_history_short(state.conversation_history[-3:])
    turns        = len(state.conversation_history)

    prompt = ROUTER_PROMPT.format(
        profile=profile,
        has_products=has_products,
        turns=turns,
        history=history,
        message=message,
        catalog=CATALOG_SNAPSHOT
    )

    valid = {"casual", "off_topic", "qa", "search", "rebuild", "product_detail",
             "refine_step", "refine_price", "new_category", "collect",
             "catalog_overview", "concern_collect", "best_sellers", "budget"}

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            action = resp.choices[0].message.content.strip().lower().strip('"').strip("'")
            if action in valid:
                print(f"[ROUTER] → {action}")
                return action
        except Exception as e:
            print(f"[ROUTER] Error: {e}")
            time.sleep(0.3)

    return "search" if state.slots.get("main_category") else "collect"


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1: SEARCH AGENT
# Handles: search — finds products matching user's request
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_EXTRACT_PROMPT = """Extract search parameters from the user's message for a beauty product catalog.

Catalog available:
{catalog}

User profile so far: {profile}
Message: "{message}"

Return ONLY valid JSON:
{{
  "query": "free text product name search or null",
  "main_category": "Face|Hair|Body|Baby or null",
  "section": "MUST be one of the exact section names listed in catalog or null",
  "brand": "brand name or null",
  "skin_types": ["oily skin", "dry skin", etc.] or null,
  "hair_types": ["curly", "dry hair", etc.] or null,
  "concerns": ["acne", "dryness", "frizz", etc.] or null,
  "texture": "gel|cream|liquid|serum|foam|powder or null",
  "sensitivity_safe": ["fragrance_free", "oil_free", "paraben_free", etc.] or null,
  "contains_irritants": false or null,
  "key_ingredients": ["Niacinamide", "Vitamin C", etc.] or null,
  "min_price": number or null,
  "max_price": number or null
}}

CRITICAL Rules:
- "milk powder", "baby milk", "baby powder" → main_category: "Baby", section: "Baby Milk Powder"
- "baby shampoo", "baby bath", "baby wash" → main_category: "Baby", section: "Baby Bath & Shampoo"
- "baby lotion", "baby cream" → main_category: "Baby", section: "Baby Lotions & creams"
- "baby" alone → main_category: "Baby", section: null
- "shampoo" → main_category: "Hair", section: "Shampoos"
- "conditioner" → main_category: "Hair", section: "Conditioners"
- "hair mask" → main_category: "Hair", section: "Hair Masks & Deep Treatments"
- "gel cleanser" → section: "Cleansers", texture: "gel"
- "fragrance-free" → sensitivity_safe: ["fragrance_free"]
- "under Rs.1000" or "budget" → max_price: 1000
- "affordable" → max_price: 1200
- "premium" or "luxury" → min_price: 1600
- "for oily skin" → skin_types: ["oily skin"]
- "dry and frizzy hair" → hair_types: ["dry hair"], concerns: ["frizz"]
- section MUST exactly match one of: {sections}
- If unsure about section, set it to null — never guess a section name"""

SEARCH_RESPONSE_PROMPT = """You are BeautyAI — a warm, enthusiastic beauty sales assistant at Beauty Mart Sri Lanka.

User profile: {profile}
Products found: {products_found}
User's message: "{message}"
Action: search

━━━ YOUR JOB ━━━
1. FIRST react to the user's message with empathy or enthusiasm (1 sentence)
   - If they mentioned a skin/hair concern: show understanding ("Dry skin can be so uncomfortable!")
   - If they asked for a specific product: show excitement ("Great choice!")
   - If they mentioned a problem: sympathise first
2. THEN give a short description of what you found (1-2 sentences)
3. Suggest 3 follow-up buttons based on REAL catalog values

Catalog info (use these for buttons — don't make up values):
{catalog}

━━━ BUTTON RULES ━━━
- Must be full sentences a shopper would actually say
- Buttons must be DIRECTLY about the products just shown — reference specific product names, brands, or properties
- Examples of GOOD buttons after showing shampoos:
  * "Tell me more about the Garnier Honey Water Shampoo"
  * "Show me something for damaged hair too"
  * "Do you have a matching conditioner for this?"
- Examples of GOOD buttons after showing conditioners:
  * "Tell me more about the Deep Repair Hair Mask"
  * "Do you have a fragrance-free option?"
  * "Show me something cheaper"
- NEVER generate generic buttons like "Let's find you a new favorite product" or "What brings you to our store"
- NEVER repeat buttons from previous turns
- NEVER use made-up ingredients or brands
- At least one button should ask about a specific product shown by name

━━━ TONE ━━━
- Warm, friendly, sales-focused
- Short replies — don't over-explain
- Use: "we have", "I love this one for you", "this is a great pick for..."
- Emojis sparingly: ✨ 💧 🌿 😊

Return ONLY valid JSON:
{{
  "empathy": "One warm reaction to their message",
  "message": "Short description of what you found + any tips",
  "slots": {{
    "main_category": null, "skin_type": null, "skin_concern": null,
    "hair_type": null, "hair_concern": null, "baby_section": null,
    "age_range": null, "sensitivity": null, "goal": null
  }},
  "buttons": ["Full sentence 1", "Full sentence 2", "Full sentence 3"]
}}"""


def run_search_agent(state: DialogState, message: str, section_override: str = None) -> Dict:
    profile = _summarize_profile(state)
    sections_list = ", ".join(CATALOG_VALUES["sections"])

    # Step 1: Extract search parameters from message
    extract_resp = _call_llm_json(
        messages=[{"role": "user", "content": SEARCH_EXTRACT_PROMPT.format(
            catalog=CATALOG_SNAPSHOT,
            profile=profile,
            message=message,
            sections=sections_list,
        )}],
        temperature=0.0,
        max_tokens=300,
    )

    slots = state.slots

    # Validate section — prefer section_override (from slot extractor), then LLM extraction
    requested_section = section_override or extract_resp.get("section")
    if requested_section and requested_section not in CATALOG_VALUES["sections"]:
        print(f"[SEARCH] Invalid section '{requested_section}' — ignoring")
        requested_section = None

    search_params = {
        "query":              extract_resp.get("query"),
        "main_category":      extract_resp.get("main_category") or slots.get("main_category"),
        "section":            requested_section,
        "brand":              extract_resp.get("brand"),
        "skin_types":         extract_resp.get("skin_types") or (
            [f"{slots['skin_type']} skin"] if slots.get("skin_type") else None
        ),
        "hair_types":         extract_resp.get("hair_types") or (
            [slots["hair_type"]] if slots.get("hair_type") else None
        ),
        "concerns":           extract_resp.get("concerns") or (
            slots.get("skin_concern") or slots.get("hair_concern")
        ),
        "texture":            extract_resp.get("texture"),
        "sensitivity_safe":   extract_resp.get("sensitivity_safe"),
        "contains_irritants": extract_resp.get("contains_irritants"),
        "key_ingredients":    extract_resp.get("key_ingredients"),
        "min_price":          extract_resp.get("min_price"),
        "max_price":          extract_resp.get("max_price"),
        "limit":              5,
    }

    print(f"[SEARCH] Params: { {k:v for k,v in search_params.items() if v is not None} }")

    # Step 2: Search — try full params first, then progressively broaden
    found_products = search_products(**{k: v for k, v in search_params.items() if v is not None})

    if not found_products:
        # Drop section constraint and retry
        search_params["section"] = None
        found_products = search_products(**{k: v for k, v in search_params.items() if v is not None})

    if not found_products:
        # Drop price constraints and retry
        search_params["min_price"] = None
        search_params["max_price"] = None
        found_products = search_products(**{k: v for k, v in search_params.items() if v is not None})

    if not found_products and search_params.get("main_category"):
        # Last resort: just show everything in category
        found_products = search_products(main_category=search_params["main_category"], limit=5)

    # Step 3: Detect if user asked for something we don't carry
    no_carry_msg = None
    msg_lower = message.lower()
    if not found_products:
        no_carry_msg = "we don't carry that specific product type yet"

    products_summary = "\n".join([
        f"- {p['name']} by {p['brand']} | Rs.{p['price']} | "
        f"Section: {p['categories'].get('section','N/A')} | "
        f"Texture: {p['attributes'].get('texture', 'N/A')} | "
        f"For: {', '.join(p['attributes'].get('skin_types', p['attributes'].get('hair_types', [])))}"
        for p in found_products
    ]) if found_products else "No products found."

    # Step 4: Generate empathetic response
    no_carry_context = f"\nIMPORTANT: {no_carry_msg}. Be honest about this but pivot warmly to what we DO have." if no_carry_msg else ""

    result = _call_llm_json(
        messages=[{"role": "user", "content": SEARCH_RESPONSE_PROMPT.format(
            profile=profile,
            products_found=products_summary,
            message=message,
            catalog=CATALOG_SNAPSHOT,
        ) + no_carry_context}],
        temperature=0.5,
        max_tokens=400,
    )

    empathy  = result.get("empathy", "")
    main_msg = result.get("message", "")
    full_reply = f"{empathy}\n\n{main_msg}".strip() if empathy else main_msg

    return {
        "reply_text":    full_reply,
        "new_slots":     result.get("slots", {}),
        "buttons":       result.get("buttons", []),
        "show_products": bool(found_products),
        "routine":       [],
        "products":      found_products,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2: CHAT AGENT
# Handles: casual, qa, collect, off_topic, new_category
# ─────────────────────────────────────────────────────────────────────────────

CHAT_SYSTEM = """You are BeautyAI — a friendly, upbeat sales assistant at Beauty Mart, Sri Lanka's go-to beauty store.
Your job is to help customers find products they'll love. You are NOT a medical consultant.

User profile: {profile}
Products shown: {products_shown}
Mode: {mode}

Catalog info:
{catalog}

━━━ TONE ━━━
• Warm, enthusiastic, sales-focused
• Short replies — don't over-explain
• ALWAYS react to what the user said FIRST before asking anything
  - Concern mentioned → sympathise ("Oily skin in Sri Lanka's heat — totally understand! 😅")
  - Preference mentioned → validate ("Love that you know what you want!")
  - Question asked → acknowledge before answering
• Use "we have", "our store carries", "I'd love to show you"
• Emojis sparingly: ✨ 💧 🌿 😊 💆

━━━ MODE RULES ━━━

"casual": Short warm reply (1-2 sentences). Reference profile if possible. 3 natural next-step buttons that help them discover products — e.g. "I'm looking for a moisturiser for dry skin", "Show me your best sellers for hair", "I need something for oily skin". NEVER generate generic buttons like "Let's find you a new favorite" or "What brings you to our store".

"off_topic": Briefly acknowledge, pivot to beauty. 3 beauty-related buttons.

"qa": Answer question about shown product/ingredient. Concise. 3 relevant follow-up buttons.
  Current products: {products_shown}

"collect": React to what they said, THEN ask the ONE most important missing piece.
  → Need category first, then EITHER skin type OR concern
  → Ask ONE question only. Buttons = real example answers from catalog values below.
  → NEVER ask about price, sulfate-free etc before showing products

"new_category": Enthusiastically acknowledge the switch. Ask ONE question about new category.

━━━ BUTTON RULES ━━━
- Always full sentences
- Use REAL values from the catalog when suggesting options
- Available concerns: Dryness, dehydration, acne, Hyperpigmentation, Premature aging, sensitive skin, frizz, dandruff
- Available skin types: oily skin, dry skin, combination skin, normal skin, sensitive skin
- Available hair types: curly, straight, wavy, dry hair, combination hair

━━━ OUTPUT — valid JSON only ━━━
{{
  "message": "Your reply (react first, then help)",
  "slots": {{
    "main_category": null, "skin_type": null, "skin_concern": null,
    "hair_type": null, "hair_concern": null, "baby_section": null,
    "age_range": null, "sensitivity": null, "goal": null
  }},
  "buttons": ["Full sentence 1", "Full sentence 2", "Full sentence 3"]
}}

Slots: only fill what you KNOW. null = don't know. Concerns = arrays always."""


def run_chat_agent(state: DialogState, message: str, mode: str) -> Dict:
    profile        = _summarize_profile(state)
    products_shown = _summarize_products(state)

    system = CHAT_SYSTEM.format(
        profile=profile,
        products_shown=products_shown,
        mode=mode,
        catalog=CATALOG_SNAPSHOT,
    )

    messages = [{"role": "system", "content": system}]
    for turn in state.conversation_history[-8:]:
        messages.append({"role": "user",      "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})
    messages.append({"role": "user", "content": message})

    result = _call_llm_json(messages, temperature=0.5, max_tokens=500)

    return {
        "reply_text":    result.get("message", ""),
        "new_slots":     result.get("slots", {}),
        "buttons":       result.get("buttons", []),
        "show_products": False,
        "routine":       [],
        "products":      [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3: ROUTINE AGENT (only called when user explicitly asks for a routine)
# ─────────────────────────────────────────────────────────────────────────────

ROUTINE_INTRO_SYSTEM = """You are BeautyAI. User explicitly asked for a personalised routine.

Profile: {profile}

Write a SHORT (1-2 sentence) enthusiastic intro for their routine.
Then 3 follow-up buttons a shopper would click AFTER seeing a routine.

Good buttons:
  "Tell me more about the serum you recommended."
  "Can you suggest a more affordable option for the moisturiser?"
  "What order should I apply these products in?"

Return ONLY valid JSON:
{{
  "message": "Short enthusiastic intro",
  "buttons": ["Button 1", "Button 2", "Button 3"]
}}"""


def run_routine_agent(state: DialogState, exclusions: List[str] = None) -> Dict:
    exclusions = exclusions or []
    if exclusions:
        existing = state.slots.get("exclusions", [])
        state.slots["exclusions"] = list(set(existing + exclusions))

    # Check we have enough to build a routine
    if not _has_enough_to_recommend(state):
        return run_chat_agent(state, "I want a routine", mode="collect")

    routine  = build_routine(state)
    products = [s["product"] for s in routine if s.get("product")]
    state.products = products

    if not routine:
        return {
            "reply_text":    "I couldn't find a perfect match in our catalog for your preferences. Would you like to adjust?",
            "new_slots":     {},
            "buttons":       [
                "Yes, let me try different preferences.",
                "Show me all available face products.",
                "What categories do you carry?"
            ],
            "show_products": False,
            "routine":       [],
            "products":      [],
        }

    profile = _summarize_profile(state)
    result  = _call_llm_json(
        messages=[
            {"role": "system", "content": ROUTINE_INTRO_SYSTEM.format(profile=profile)},
            {"role": "user",   "content": "Generate intro and follow-up buttons."}
        ],
        temperature=0.4,
        max_tokens=250,
    )

    return {
        "reply_text":    result.get("message", format_routine_intro(routine, state)),
        "new_slots":     {},
        "buttons":       result.get("buttons", [
            "Tell me more about the serum you recommended.",
            "Can you suggest a more affordable option?",
            "What order should I apply these products in?"
        ]),
        "show_products": True,
        "routine":       routine,
        "products":      products,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4: REFINE AGENT
# ─────────────────────────────────────────────────────────────────────────────

REFINE_SYSTEM = """You are BeautyAI. User wants to refine their existing recommendations.

Profile: {profile}
Current products: {products}
Request: "{request}"

Return ONLY valid JSON:
{{
  "target_step": "step name or null",
  "price_direction": "lower|higher or null",
  "message": "Warm 1-sentence acknowledgement"
}}"""


def run_refine_agent(state: DialogState, message: str) -> Dict:
    profile  = _summarize_profile(state)
    products = _summarize_products(state)

    result          = _call_llm_json(
        messages=[{"role": "system", "content": REFINE_SYSTEM.format(
            profile=profile, products=products, request=message)},
            {"role": "user", "content": message}],
        temperature=0.2,
        max_tokens=150,
    )
    target_step     = result.get("target_step")
    price_direction = result.get("price_direction")
    ack             = result.get("message", "Let me find some alternatives! ✨")

    alternatives = get_step_alternatives(
        state=state,
        step_name=target_step,
        price_direction=price_direction,
        exclude_ids={p["product_id"] for p in state.products}
    )

    if not alternatives:
        return {
            "reply_text":    ack + " Unfortunately I couldn't find other matches — would you like to adjust your preferences?",
            "new_slots":     {},
            "buttons":       [
                "Yes, let me change my preferences.",
                "Show me the original recommendations.",
                "What other products do you carry?"
            ],
            "show_products": False,
            "routine":       [],
            "products":      [],
        }

    mini_routine = [{
        "step_number": i + 1,
        "step_name":   target_step or "Alternative",
        "purpose":     "Alternative option based on your request",
        "product":     p
    } for i, p in enumerate(alternatives)]

    return {
        "reply_text":    ack,
        "new_slots":     {},
        "buttons":       [
            "Tell me more about the first option.",
            "Can you suggest something even more budget-friendly?",
            "What are the key differences between these options?"
        ],
        "show_products": True,
        "routine":       mini_routine,
        "products":      alternatives,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5: PRODUCT DETAIL AGENT
# Triggered when user asks "tell me more about X", "what are the ingredients", etc.
# Pulls structured data directly from the product JSON — no hallucination possible
# ─────────────────────────────────────────────────────────────────────────────

DETAIL_SYSTEM = """You are BeautyAI — a warm, enthusiastic beauty sales assistant at Beauty Mart Sri Lanka.

The user wants details about a specific product. You have the FULL product data below.
Write a rich, conversational description that sells the product through benefits and emotion — NOT a catalog card.
Do NOT invent anything not in the data.

Product data:
{product_json}

━━━ OUTPUT FORMAT ━━━
Follow this EXACT structure. Use markdown exactly as shown:

Hey! ✨ I think you'd really love this one:

**[Product Name]** by [Brand]
[price]

[2-3 sentences: what makes this product exciting RIGHT NOW for this customer. Lead with the emotional benefit — "If you've been struggling with X, this is exactly what you need." Draw from the description field. Make it feel personal, not generic.]

🌟 **What makes it special for you:**
• [Benefit 1 — customer outcome, not feature. E.g. "Visibly evens skin tone — many customers see results in 3–4 weeks"]
• [Benefit 2 — outcome-focused]
• [Benefit 3 — outcome-focused]
• [Texture/formula benefit — e.g. "Lightweight gel texture — absorbs instantly, never feels heavy or greasy"]
[Add sensitivity_safe tags as benefits: e.g. "Fragrance-free — no risk of irritation for reactive skin"]

🧴 **Best match for:**
[skin types listed naturally, e.g. "Normal ↔ Combination skin" or "Dry → Sensitive skin"]
Main concerns: [list the top 3-4 concerns from the data, comma-separated, keep short]

🌿 **The hero ingredients people talk about most:**
• [Ingredient 1] → [what it does — 1 punchy phrase, e.g. "brightens + boosts glow"]
• [Ingredient 2] → [what it does]
• [Ingredient 3] → [what it does]

**How most people use it:**
→ [Step/tip 1 — paraphrase how_to_use naturally, e.g. "Morning and evening on clean skin"]
→ [Step/tip 2]
→ [Step/tip 3 if applicable]
[Add any important usage note, e.g. "Always follow with sunscreen if using actives!"]

💬 **Real customer love** (from [review_count]+ reviews averaging [rating] ⭐)
[One warm sentence about what customers consistently praise — infer from rating/purchase_count/description]

━━━ TONE RULES ━━━
- Conversational, warm, benefit-first — like a knowledgeable friend recommending something
- Lead with WHY this is good FOR THE USER, not just what it contains
- Avoid dry catalog language like "This product contains..." or "Suitable for..."
- Keep bullets SHORT and punchy — max 10 words each
- Use → arrows for how-to steps (feels more natural than numbered lists)
- Do NOT add sections if the data is empty (e.g. no hair_types → skip hair section)
- Do NOT make up ingredients, claims, or review content
- if contains_irritants is false → mention "Irritant-free formula" as a trust signal

━━━ BUTTONS ━━━
Generate exactly 3 follow-up buttons. NO buy/cart buttons.
- One asking about complementary products that pair well with this
- One about a specific concern this product addresses
- One about seeing an alternative (different price range or texture)

All buttons must be full natural sentences a real shopper would say.

Return ONLY valid JSON:
{{
  "message": "Your full product description using markdown exactly as structured above",
  "buttons": ["Full sentence 1", "Full sentence 2", "Full sentence 3"]
}}"""


def _find_product_in_catalog(name_or_id: str, shown_products: List[Dict]) -> Optional[Dict]:
    """Find a product by name or ID. Check shown products first, then full catalog."""
    name_lower = name_or_id.lower().strip()

    # Check shown products first (most likely what user is asking about)
    for p in shown_products:
        if (name_lower in p.get("name", "").lower() or
                name_lower in p.get("product_id", "").lower() or
                name_lower in p.get("brand", "").lower()):
            return p

    # Fall back to full catalog
    for p in PRODUCT_CATALOG:
        if (name_lower in p.get("name", "").lower() or
                p.get("product_id", "").lower() == name_lower or
                name_lower in p.get("brand", "").lower()):
            return p

    return None


def _extract_product_name_from_message(message: str, shown_products: List[Dict]) -> Optional[str]:
    """
    Extract which product the user is asking about.
    First check if any shown product name/brand appears in the message.
    """
    msg_lower = message.lower()

    # Check shown products first
    for p in shown_products:
        name_lower = p.get("name", "").lower()
        brand_lower = p.get("brand", "").lower()
        # Match on significant words (skip short words)
        name_words = [w for w in name_lower.split() if len(w) > 3]
        if any(w in msg_lower for w in name_words):
            return p.get("name")
        if brand_lower and brand_lower in msg_lower:
            return p.get("name")

    # If only one product is shown and user says "this product", "it", "that one" etc
    vague_refs = ["this product", "this one", "it ", "that product", "that one",
                  "this lotion", "this cream", "this serum", "the product"]
    if shown_products and any(ref in msg_lower for ref in vague_refs):
        return shown_products[0].get("name")

    return None


def run_product_detail_agent(state: DialogState, message: str) -> Dict:
    """
    Generates a rich product description using the product's JSON data directly.
    No hallucination — everything comes from the catalog.
    """
    # Find which product the user is asking about
    product_name = _extract_product_name_from_message(message, state.products)
    product = None

    if product_name:
        product = _find_product_in_catalog(product_name, state.products)

    # If we couldn't identify the product, fall back to the most recently shown one
    if not product and state.products:
        product = state.products[0]

    if not product:
        return run_chat_agent(state, message, mode="qa")

    # Build clean product JSON for the LLM (include all useful fields)
    product_data = {
        "name":          product.get("name"),
        "brand":         product.get("brand"),
        "price":         f"Rs.{product.get('price')} {product.get('currency', '')}",
        "in_stock":      product.get("in_stock"),
        "description":   product.get("description"),
        "how_to_use":    product.get("how_to_use"),
        "key_ingredients": product.get("key_ingredients", []),
        "for_skin_types":  product["attributes"].get("skin_types", []),
        "for_hair_types":  product["attributes"].get("hair_types", []),
        "concerns_it_addresses": product["attributes"].get("concerns", []),
        "texture":         product["attributes"].get("texture"),
        "sensitivity_safe_tags": product["attributes"].get("sensitivity_safe", []),
        "contains_irritants":    product["attributes"].get("contains_irritants"),
        "rating":          product["ranking_signals"].get("rating"),
        "review_count":    product["ranking_signals"].get("review_count"),
        "purchase_count":  product["ranking_signals"].get("purchase_count"),
        "section":         product["categories"].get("section"),
    }

    result = _call_llm_json(
        messages=[
            {"role": "system", "content": DETAIL_SYSTEM.format(
                product_json=json.dumps(product_data, indent=2)
            )},
            {"role": "user", "content": message},
        ],
        temperature=0.4,
        max_tokens=1000,
    )

    return {
        "reply_text":    result.get("message", ""),
        "new_slots":     {},
        "buttons":       result.get("buttons", [
            "I'd like to add this to my cart.",
            "Show me products that work well with this.",
            "Can you suggest a more affordable option?",
        ]),
        "show_products": True,   # Keep the product card visible
        "routine":       [],
        "products":      [product],  # Show the product card alongside description
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESS
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# AGENT: GROUPED SEARCH (Best Sellers / Budget Picks)
# Returns product_groups: [{category, products[]}] — 3 products per category
# Frontend renders each group with a heading + carousel
# ─────────────────────────────────────────────────────────────────────────────

def run_grouped_search_agent(state: DialogState, message: str, mode: str) -> Dict:
    """
    mode = "best_sellers" | "budget"

    If state.slots["_filter_category"] is set → show only that category (flat carousel).
    Otherwise → show all 4 categories grouped.
    """
    ALL_CATEGORIES = ["Face", "Hair", "Body", "Baby"]
    filter_cat = state.slots.pop("_filter_category", None)  # consume and clear

    CAT_EMOJI = {"Face": "✨", "Hair": "💆", "Body": "🧴", "Baby": "🍼"}

    # ── SINGLE CATEGORY ───────────────────────────────────────────────────────
    if filter_cat:
        if mode == "budget":
            products = search_products(main_category=filter_cat, max_price=1000, limit=6)
        else:
            products = search_products(main_category=filter_cat, limit=6)

        emoji = CAT_EMOJI.get(filter_cat, "🛍️")
        label = "budget picks" if mode == "budget" else "best sellers"

        if not products:
            return {
                "reply_text":     f"Sorry, we don't have any {label} in {filter_cat} right now. Try another category!",
                "new_slots":      {},
                "buttons":        [
                    f"Show me {label} for Face.",
                    f"Show me {label} for Hair.",
                    f"Show me {label} for Body.",
                ],
                "show_products":  False,
                "product_groups": [],
                "routine":        [],
                "products":       [],
            }

        price_note = " under Rs.1000" if mode == "budget" else ""
        reply = f"Here are our top {filter_cat.lower()} {label}{price_note}! {emoji}"

        # Suggest the other 3 categories as follow-up buttons
        other_cats = [c for c in ALL_CATEGORIES if c != filter_cat]
        buttons = [f"Show me {label} for {c}." for c in other_cats[:3]]

        return {
            "reply_text":     reply,
            "new_slots":      {},
            "buttons":        buttons,
            "show_products":  True,
            "product_groups": [],               # flat — no group headers needed
            "routine":        [],
            "products":       products,
        }

    # ── ALL CATEGORIES GROUPED ────────────────────────────────────────────────
    groups = []
    for cat in ALL_CATEGORIES:
        if mode == "budget":
            products = search_products(main_category=cat, max_price=1000, limit=3)
        else:
            products = search_products(main_category=cat, limit=3)
        if products:
            groups.append({"category": cat, "products": products})

    all_products = [p for g in groups for p in g["products"]]

    if mode == "budget":
        reply = "Here are our best budget-friendly picks under Rs.1000, sorted by category! ✨ Great quality doesn't have to break the bank."
        buttons = [
            "Show me budget picks just for face care.",
            "Show me budget picks just for hair care.",
            "Do you have anything even cheaper?",
        ]
    else:
        reply = "Here are our top-rated products across every category! 🌟 These are what our customers love most."
        buttons = [
            "Show me only the best sellers for face care.",
            "Show me only the best sellers for hair care.",
            "Show me only the best sellers for body care.",
        ]

    return {
        "reply_text":     reply,
        "new_slots":      {},
        "buttons":        buttons,
        "show_products":  True,
        "product_groups": groups,
        "routine":        [],
        "products":       all_products,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT: CONCERN COLLECT FLOW (Shop by Concern pill)
# Step 1: Ask category (Face / Hair / Body / Baby)
# Step 2: Ask their main concern for that category
# Step 3: Ask skin/hair type
# Then hand off to search
# ─────────────────────────────────────────────────────────────────────────────

CONCERN_COLLECT_PROMPTS = {
    # ── Step 1: Category ──────────────────────────────────────────────────────
    "ask_category": {
        "message": "I'd love to help you find exactly what you need! 😊\n\nFirst, which area are you shopping for?",
        "buttons": [
            "I'm shopping for face care.",
            "I need something for my hair.",
            "I'm looking for body care products.",
            # Baby intentionally excluded — Baby flow goes direct to search after category
        ],
    },

    # ── Step 2: Concern per category ─────────────────────────────────────────
    "ask_concern_face": {
        "message": "Great — face care it is! ✨\n\nWhat's your main skin concern right now?",
        "buttons": [
            "My skin is very dry and dehydrated.",
            "I have oily skin and frequent breakouts.",
            "I want to reduce dark spots and uneven skin tone.",
        ],
    },
    "ask_concern_hair": {
        "message": "Love it! 💆 Hair care coming right up.\n\nWhat's your main hair concern?",
        "buttons": [
            "My hair is dry, frizzy and damaged.",
            "I have an oily scalp and greasy roots.",
            "I want to reduce hair fall and strengthen my hair.",
        ],
    },
    "ask_concern_body": {
        "message": "Perfect — let's find you the best body care! 🧴\n\nWhat's your main body care concern?",
        "buttons": [
            "My skin is very dry and needs deep moisture.",
            "I have sensitive skin that gets irritated easily.",
            "I want sun protection for my body.",
        ],
    },

    # ── Step 3: Type per category ─────────────────────────────────────────────
    "ask_type_face": {
        "message": "Almost there! One last thing — what's your skin type?",
        "buttons": [
            "I have oily skin.",
            "My skin is dry.",
            "I have combination or sensitive skin.",
        ],
    },
    "ask_type_hair": {
        "message": "One more thing — what's your hair type?",
        "buttons": [
            "My hair is curly or coily.",
            "I have straight hair.",
            "My hair is wavy or slightly wavy.",
        ],
    },
}

# Slot keywords to extract from button answers
_CATEGORY_MAP = {
    "face": "Face", "facial": "Face",
    "hair": "Hair",
    "body": "Body",
    "baby": "Baby",
}
_CONCERN_MAP = {
    "dry":          {"skin_concern": ["Dryness", "dehydration"]},
    "dehydrat":     {"skin_concern": ["Dryness", "dehydration"]},
    "oily":         {"skin_concern": ["acne"], "skin_type": "oily"},
    "breakout":     {"skin_concern": ["acne"]},
    "acne":         {"skin_concern": ["acne"]},
    "dark spot":    {"skin_concern": ["Hyperpigmentation"]},
    "pigment":      {"skin_concern": ["Hyperpigmentation"]},
    "uneven":       {"skin_concern": ["Hyperpigmentation"]},
    "frizz":        {"hair_concern": ["frizz", "dryness"]},
    "damaged":      {"hair_concern": ["dryness", "frizz"]},
    "greasy":       {"hair_concern": ["greasy roots"]},
    "scalp":        {"hair_concern": ["greasy roots"]},
    "hair fall":    {"hair_concern": ["hair fall"]},
    "strengthen":   {"hair_concern": ["hair fall"]},
    "sensitive":    {"skin_concern": ["sensitive skin"], "skin_type": "sensitive"},
    "sun":          {"skin_concern": ["Sunburn", "sun damage"]},
    "moisture":     {"skin_concern": ["Dryness", "dehydration"]},
}
_TYPE_MAP = {
    "oily":        {"skin_type": "oily"},
    "dry":         {"skin_type": "dry"},
    "combination": {"skin_type": "combination"},
    "sensitive":   {"skin_type": "sensitive"},
    "curly":       {"hair_type": "curly"},
    "coily":       {"hair_type": "coily"},
    "straight":    {"hair_type": "straight"},
    "wavy":        {"hair_type": "wavy"},
}


def _extract_slots_from_concern_answer(message: str, step: str) -> Dict:
    """Extract slots from user's button answer during concern collect flow."""
    msg_lower = message.lower()
    extracted = {}

    if step == "ask_category":
        for kw, cat in _CATEGORY_MAP.items():
            if kw in msg_lower:
                extracted["main_category"] = cat
                break

    elif step.startswith("ask_concern"):
        for kw, slots in _CONCERN_MAP.items():
            if kw in msg_lower:
                extracted.update(slots)

    elif step.startswith("ask_type"):
        for kw, slots in _TYPE_MAP.items():
            if kw in msg_lower:
                extracted.update(slots)
                break

    return extracted


def _empty_collect_response(prompt_key: str, next_step: str) -> Dict:
    """Build a no-product collect response for a given prompt."""
    p = CONCERN_COLLECT_PROMPTS[prompt_key]
    return {
        "reply_text":     p["message"],
        "new_slots":      {"_concern_step": next_step},
        "buttons":        p["buttons"],
        "show_products":  False,
        "product_groups": [],
        "routine":        [],
        "products":       [],
    }


def run_concern_collect_agent(state: DialogState, message: str) -> Dict:
    """
    Strict 3-step guided flow using _concern_step flag in state.slots.

    _concern_step values:
      "ask_category"     → waiting for user to pick category
      "ask_concern_face" → waiting for face concern
      "ask_concern_hair" → waiting for hair concern
      "ask_concern_body" → waiting for body concern
      "ask_type_face"    → waiting for skin type
      "ask_type_hair"    → waiting for hair type
      "done"             → flow complete, search runs
      (absent)           → first time, start from step 1
    """
    slots     = state.slots
    step      = slots.get("_concern_step", "")   # current step
    msg_lower = message.lower()

    # ── ENTRY: First time (no step yet) ───────────────────────────────────────
    if not step:
        return _empty_collect_response("ask_category", next_step="ask_category")

    # ── Process answer for current step, then advance ─────────────────────────
    extracted = _extract_slots_from_concern_answer(message, step)
    # Merge extracted slots into state immediately so next step sees them
    if extracted:
        state.slots = _merge_slots(state.slots, extracted)

    slots = state.slots   # re-read after merge
    cat   = slots.get("main_category")

    # ── STEP: ask_category → advance to ask_concern ───────────────────────────
    if step == "ask_category":
        if not cat:
            # Couldn't extract category — ask again
            return _empty_collect_response("ask_category", next_step="ask_category")

        if cat == "Baby":
            # Baby skips concern+type steps — search immediately
            state.slots["_concern_step"] = "done"
            return run_search_agent(state, message)

        concern_key = f"ask_concern_{cat.lower()}"
        return _empty_collect_response(concern_key, next_step=concern_key)

    # ── STEP: ask_concern_* → advance to ask_type or search ──────────────────
    if step.startswith("ask_concern"):
        has_concern = bool(
            slots.get("skin_concern") or slots.get("hair_concern")
        )
        if not has_concern:
            # Couldn't extract concern — ask again
            concern_key = f"ask_concern_{cat.lower()}" if cat else "ask_category"
            return _empty_collect_response(concern_key, next_step=concern_key)

        if cat == "Body":
            # Body skips type step — search now
            state.slots["_concern_step"] = "done"
            return run_search_agent(state, message)

        type_key = f"ask_type_{cat.lower()}"   # ask_type_face or ask_type_hair
        return _empty_collect_response(type_key, next_step=type_key)

    # ── STEP: ask_type_* → search ─────────────────────────────────────────────
    if step.startswith("ask_type"):
        has_type = bool(slots.get("skin_type") or slots.get("hair_type"))
        if not has_type:
            # Couldn't extract type — ask again
            type_key = f"ask_type_{cat.lower()}" if cat else "ask_category"
            return _empty_collect_response(type_key, next_step=type_key)

        state.slots["_concern_step"] = "done"
        return run_search_agent(state, message)

    # ── DONE: flow complete ────────────────────────────────────────────────────
    if step == "done":
        return run_search_agent(state, message)

    # Fallback
    return _empty_collect_response("ask_category", next_step="ask_category")


# ─────────────────────────────────────────────────────────────────────────────
# Triggered by "Shop by Category" and "Shop by Concern" pills
# Gives a warm structured overview of what we carry + actionable buttons
# ─────────────────────────────────────────────────────────────────────────────

def run_catalog_overview_agent(state: DialogState, message: str) -> Dict:
    """
    Responds to "what do you carry" / "shop by concern" type queries.
    Determines whether the user wants categories or concerns and responds accordingly.
    """
    msg_lower = message.lower()
    is_concern_query = any(kw in msg_lower for kw in [
        "concern", "problem", "issue", "condition", "treat", "help with"
    ])

    # Build real data from catalog
    cats = sorted(CATALOG_VALUES["categories"])  # Face, Hair, Body, Baby
    top_concerns = [
        "Dryness & dehydration", "Acne & oily skin", "Hyperpigmentation & dark spots",
        "Premature aging & fine lines", "Sensitive skin & eczema",
        "Frizz & damaged hair", "Dandruff & oily scalp", "Hair fall & breakage"
    ]
    sections_by_cat = {}
    for p in PRODUCT_CATALOG:
        cat = p["categories"].get("main_category")
        sec = p["categories"].get("section")
        if cat and sec:
            sections_by_cat.setdefault(cat, set()).add(sec)

    catalog_detail = "\n".join([
        f"- {cat}: {', '.join(sorted(secs))}"
        for cat, secs in sorted(sections_by_cat.items())
    ])
    concerns_list = "\n".join([f"- {c}" for c in top_concerns])

    if is_concern_query:
        prompt = f"""You are BeautyAI, a friendly beauty sales assistant at Beauty Mart Sri Lanka.
The user wants to shop by skin or hair concern.

Our catalog covers these concerns:
{concerns_list}

Our categories: {', '.join(cats)}

Write a warm, enthusiastic response (3-4 sentences max) that:
1. Lists the main concerns we cover in a natural conversational way
2. Invites them to pick one so you can show matching products

Then generate exactly 3 buttons — each one is a specific concern the user might pick, e.g.:
"I want products for dry and dehydrated skin."
"Show me something for frizzy, damaged hair."
"I need help with dark spots and hyperpigmentation."

Return ONLY valid JSON:
{{
  "message": "Your warm response listing concerns",
  "buttons": ["Concern button 1", "Concern button 2", "Concern button 3"]
}}"""
    else:
        prompt = f"""You are BeautyAI, a friendly beauty sales assistant at Beauty Mart Sri Lanka.
The user wants to know what categories and products we carry.

Our catalog:
{catalog_detail}

Write a warm, enthusiastic response (3-4 sentences max) that:
1. Briefly introduces our 4 main categories: Face, Hair, Body, Baby
2. Mentions a few highlights from each
3. Invites them to pick a category to explore

Then generate exactly 3 buttons — each one picks a specific category to browse:
"Show me your face care products."
"I want to explore hair care."
"What do you have for body care?"

Return ONLY valid JSON:
{{
  "message": "Your warm overview of what we carry",
  "buttons": ["Category button 1", "Category button 2", "Category button 3"]
}}"""

    result = _call_llm_json(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=400,
    )

    return {
        "reply_text":    result.get("message", "We carry Face, Hair, Body and Baby products — just tell me what you're looking for!"),
        "new_slots":     {},
        "buttons":       result.get("buttons", [
            "Show me your face care products.",
            "I want to explore hair care.",
            "Show me products for dry skin.",
        ]),
        "show_products": False,
        "routine":       [],
        "products":      [],
    }


async def process_message(state: DialogState, user_message: str) -> Dict:
    """Main entry: extract slots → route → dispatch agent → merge state → respond."""

    # Always extract slots first
    slots_from_message = _quick_slot_extract(user_message, state)

    # ── CATEGORY SWITCH DETECTION ─────────────────────────────────────────────
    # Skip category switch during concern collect flow — the flow handles category internally
    new_cat = slots_from_message.get("main_category")
    old_cat = state.slots.get("main_category")
    in_concern_flow = state.slots.get("_concern_step") and state.slots["_concern_step"] != "done"

    if new_cat and old_cat and new_cat != old_cat and not in_concern_flow:
        print(f"[CATEGORY SWITCH] {old_cat} → {new_cat} — clearing stale slots & products")
        state.slots = {"main_category": new_cat}
        state.products = []
        slots_from_message = {k: v for k, v in slots_from_message.items()}
    elif slots_from_message:
        state.slots = _merge_slots(state.slots, slots_from_message)

    if slots_from_message:
        print(f"[SLOTS] Extracted: {slots_from_message}")

    # Route
    action = route_message(state, user_message)

    # OVERRIDE: If we've had 2+ collect turns and have any useful slot → force search
    collect_turns = sum(1 for t in state.conversation_history if t.assistant and
                        any(q in t.assistant for q in ["?", "What's", "What is", "Which", "How"]))
    if action == "collect" and collect_turns >= 2 and (
        state.slots.get("main_category") or state.slots.get("skin_type") or
        state.slots.get("hair_type") or state.slots.get("skin_concern") or
        state.slots.get("hair_concern")
    ):
        print(f"[OVERRIDE] Too many collect turns — forcing search")
        action = "search"

    print(f"[ACTION] {action}")

    # ── Clear stale products before any new search ────────────────────────────
    if action == "search":
        state.products = []
        # Save section extracted THIS message before clearing, pass it into search
        # Section is per-message only — don't carry it into future searches
        _current_section = state.slots.pop("section", None)
    else:
        _current_section = None

    # Dispatch
    if action == "catalog_overview":
        agent_result = run_catalog_overview_agent(state, user_message)

    elif action == "concern_collect":
        agent_result = run_concern_collect_agent(state, user_message)

    elif action == "best_sellers":
        agent_result = run_grouped_search_agent(state, user_message, mode="best_sellers")

    elif action == "budget":
        agent_result = run_grouped_search_agent(state, user_message, mode="budget")

    elif action == "product_detail":
        agent_result = run_product_detail_agent(state, user_message)

    elif action == "search":
        agent_result = run_search_agent(state, user_message, section_override=_current_section)

    elif action == "rebuild":
        exclusions = _extract_exclusions(user_message)
        state.products = []
        if exclusions:
            existing = state.slots.get("exclusions", [])
            state.slots["exclusions"] = list(set(existing + exclusions))
        agent_result = run_search_agent(state, user_message)

    elif action in ("refine_step", "refine_price"):
        agent_result = run_refine_agent(state, user_message)

    elif action in ("casual", "off_topic", "qa", "collect", "new_category"):
        agent_result = run_chat_agent(state, user_message, mode=action)

    else:
        agent_result = run_chat_agent(state, user_message, mode="collect")

    # Merge new slots
    new_slots = agent_result.get("new_slots", {})
    if new_slots:
        state.slots = _merge_slots(state.slots, new_slots)

    # Update products in state
    if agent_result.get("show_products") and agent_result.get("products"):
        state.products = agent_result["products"]

    # Save turn
    state.conversation_history.append(
        ConversationTurn(user=user_message, assistant=agent_result.get("reply_text", ""))
    )

    # Format buttons
    buttons = [
        {"label": b, "payload": {"slot": "_text", "value": b}}
        for b in agent_result.get("buttons", [])[:3]
    ]

    return {
        "reply_text":          agent_result.get("reply_text", ""),
        "suggested_options":   buttons,
        "current_node":        "conversation",
        "routine":             agent_result.get("routine", []),
        "products":            agent_result.get("products", []) if agent_result.get("show_products") else [],
        "product_groups":      agent_result.get("product_groups", []),
        "trigger_recommender": False,
        "recommender_context": None,
    }