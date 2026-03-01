"""
products.py — Product search + routine builder for BeautyAI
"""
import json
from typing import List, Dict, Any, Optional, Set
from models import DialogState

PRODUCT_CATALOG: List[Dict[str, Any]] = []
try:
    with open("products_db.json", "r", encoding="utf-8") as f:
        PRODUCT_CATALOG = json.load(f)
    print(f"[PRODUCTS] Loaded {len(PRODUCT_CATALOG)} products")
except FileNotFoundError:
    print("[PRODUCTS] ERROR: products_db.json not found!")
except json.JSONDecodeError as e:
    print(f"[PRODUCTS] ERROR: Invalid JSON → {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CATALOG METADATA — dynamic values extracted from real products
# Used by the engine to generate accurate follow-up buttons
# ─────────────────────────────────────────────────────────────────────────────

def get_catalog_values() -> Dict[str, Any]:
    """Extract all unique queryable values from the catalog."""
    textures: Set[str]         = set()
    sensitivity_tags: Set[str] = set()
    all_concerns: Set[str]     = set()
    sections: Set[str]         = set()
    brands: Set[str]           = set()
    ingredients: Set[str]      = set()
    categories: Set[str]       = set()

    for p in PRODUCT_CATALOG:
        attrs = p.get("attributes", {})
        cats  = p.get("categories", {})

        if attrs.get("texture"):
            textures.add(attrs["texture"])
        for tag in attrs.get("sensitivity_safe", []):
            sensitivity_tags.add(tag)
        for c in attrs.get("concerns", []):
            # Normalize — only add short, clean concern terms
            c_clean = c.strip()
            if len(c_clean) < 50:
                all_concerns.add(c_clean)
        if cats.get("section"):
            sections.add(cats["section"])
        if cats.get("main_category"):
            categories.add(cats["main_category"])
        if p.get("brand"):
            brands.add(p["brand"])
        for ing in p.get("key_ingredients", []):
            ingredients.add(ing)

    return {
        "textures":        sorted(textures),
        "sensitivity_tags": sorted(sensitivity_tags),
        "concerns":        sorted(all_concerns),
        "sections":        sorted(sections),
        "categories":      sorted(categories),
        "brands":          sorted(brands),
        "ingredients":     sorted(ingredients),
    }

CATALOG_VALUES = get_catalog_values()


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-FIELD PRODUCT SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_products(
    query: Optional[str]             = None,      # free-text name search
    main_category: Optional[str]     = None,      # Face | Hair | Body | Baby
    section: Optional[str]           = None,      # Cleansers | Toners | etc.
    brand: Optional[str]             = None,      # brand name (partial)
    skin_types: Optional[List[str]]  = None,      # e.g. ["oily skin", "dry skin"]
    hair_types: Optional[List[str]]  = None,      # e.g. ["curly", "dry hair"]
    concerns: Optional[List[str]]    = None,      # e.g. ["acne", "dryness"]
    texture: Optional[str]           = None,      # gel | cream | liquid | serum | foam | powder
    sensitivity_safe: Optional[List[str]] = None, # e.g. ["fragrance_free", "oil_free"]
    contains_irritants: Optional[bool]   = None,  # True | False
    key_ingredients: Optional[List[str]] = None,  # e.g. ["Niacinamide", "Vitamin C"]
    min_price: Optional[float]       = None,
    max_price: Optional[float]       = None,
    in_stock_only: bool              = True,
    limit: int                       = 6,
) -> List[Dict]:
    """
    Search the product catalog across all fields.
    Returns scored, ranked list of matching products.
    """
    results = []

    for product in PRODUCT_CATALOG:
        if in_stock_only and not product.get("in_stock", True):
            continue

        attrs = product.get("attributes", {})
        cats  = product.get("categories", {})
        score = 0.0
        matched = True

        # ── Hard filters (must match if specified) ──

        # Main category
        if main_category:
            if cats.get("main_category", "").lower() != main_category.lower():
                continue

        # Section
        if section:
            if section.lower() not in cats.get("section", "").lower():
                continue

        # Brand
        if brand:
            if brand.lower() not in product.get("brand", "").lower():
                continue

        # Price range
        price = product.get("price", 0)
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue

        # contains_irritants hard filter
        if contains_irritants is not None:
            if attrs.get("contains_irritants") != contains_irritants:
                continue

        # ── Soft scoring (partial matches boost score) ──

        # Name/description free-text
        if query:
            q_lower = query.lower()
            name_lower = product.get("name", "").lower()
            desc_lower = product.get("description", "").lower()
            brand_lower = product.get("brand", "").lower()
            if q_lower in name_lower:
                score += 80
            elif any(word in name_lower for word in q_lower.split() if len(word) > 2):
                score += 40
            if q_lower in desc_lower:
                score += 20
            if q_lower in brand_lower:
                score += 30

        # Skin types
        if skin_types:
            p_skin = [s.lower() for s in attrs.get("skin_types", [])]
            for st in skin_types:
                if any(st.lower() in ps for ps in p_skin):
                    score += 60

        # Hair types
        if hair_types:
            p_hair = [h.lower() for h in attrs.get("hair_types", [])]
            for ht in hair_types:
                if any(ht.lower() in ph for ph in p_hair):
                    score += 60

        # Concerns (partial match)
        if concerns:
            p_concerns = [c.lower() for c in attrs.get("concerns", [])]
            for concern in concerns:
                if any(concern.lower() in pc for pc in p_concerns):
                    score += 35

        # Texture
        if texture:
            if attrs.get("texture", "").lower() == texture.lower():
                score += 50

        # Sensitivity safe tags
        if sensitivity_safe:
            p_safe = [s.lower() for s in attrs.get("sensitivity_safe", [])]
            for tag in sensitivity_safe:
                if tag.lower() in p_safe:
                    score += 30

        # Key ingredients
        if key_ingredients:
            p_ings = [i.lower() for i in product.get("key_ingredients", [])]
            for ing in key_ingredients:
                if any(ing.lower() in pi for pi in p_ings):
                    score += 40

        # If NO filters given at all, include everything with base ranking score
        has_any_filter = any([
            query, main_category, section, brand, skin_types, hair_types,
            concerns, texture, sensitivity_safe, key_ingredients,
            min_price, max_price
        ])
        if not has_any_filter:
            score = 1.0

        # Ranking signals tiebreaker
        signals = product.get("ranking_signals", {})
        score += (signals.get("rating") or 0) * 5
        score += (signals.get("trending_score") or 0) / 10

        # Only include if it scored at all (or no filters)
        if score > 0 or not has_any_filter:
            results.append((score, product))

    results.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in results[:limit]]


# ─────────────────────────────────────────────────────────────────────────────
# ROUTINE BUILDER (unchanged — used only when user explicitly requests routine)
# ─────────────────────────────────────────────────────────────────────────────

FACE_ROUTINE_STEPS = [
    {"step": "Cleanser",       "sections": ["Cleansers"],                    "purpose": "Removes impurities, oil, and makeup without stripping the skin"},
    {"step": "Toner",          "sections": ["Toners"],                       "purpose": "Balances skin pH and preps skin to absorb the next steps better"},
    {"step": "Exfoliator",     "sections": ["Exfoliators"],                  "purpose": "Gently removes dead skin cells for smoother, brighter skin (2–3x per week)"},
    {"step": "Serum",          "sections": ["Serums & Targeted Treatments"], "purpose": "Delivers concentrated actives targeting your specific skin concerns"},
    {"step": "Eye Care",       "sections": ["Eye Care"],                     "purpose": "Treats the delicate under-eye area — puffiness, dark circles, fine lines"},
    {"step": "Moisturiser",    "sections": ["Moisturisers", "Day Creams"],   "purpose": "Locks in hydration and keeps your skin barrier strong all day"},
    {"step": "Night Cream",    "sections": ["Night Cream"],                  "purpose": "Supports overnight repair and deep nourishment while you sleep"},
    {"step": "Sun Protection", "sections": ["Sun Care"],                     "purpose": "Shields skin from UV damage — prevents premature aging and dark spots"},
]

BODY_ROUTINE_STEPS = [
    {"step": "Body Wash",        "sections": ["Body Creams & Lotions"],  "purpose": "Cleanses and nourishes skin during your shower"},
    {"step": "Body Moisturiser", "sections": ["Body Creams & Lotions"],  "purpose": "Seals in moisture and keeps skin soft all day"},
    {"step": "Hand & Foot Care", "sections": ["Hand and Foot Care"],     "purpose": "Targets dryness and roughness on hands, heels, and feet"},
]

HAIR_ROUTINE_STEPS = [
    {"step": "Conditioner",    "sections": ["Conditioners"],                "purpose": "Softens, detangles and nourishes hair after every wash"},
    {"step": "Hair Treatment", "sections": ["Hair Masks & Deep Treatments"],"purpose": "Intensively repairs and restores hair — use 1–2x per week"},
]

BABY_ROUTINE_STEPS = [
    {"step": "Baby Cleanser",    "sections": ["Baby Bath & Shampoo", "Baby Milk Powder"], "purpose": "Gently cleanses baby's delicate skin and hair without irritation"},
    {"step": "Baby Moisturiser", "sections": ["Baby Lotions & creams"],                   "purpose": "Keeps baby's skin soft and protected after bath time"},
]

ROUTINE_MAP = {
    "Face": FACE_ROUTINE_STEPS,
    "Body": BODY_ROUTINE_STEPS,
    "Hair": HAIR_ROUTINE_STEPS,
    "Baby": BABY_ROUTINE_STEPS,
}


def _score_product(product: Dict, state: DialogState) -> float:
    attrs   = product.get("attributes", {})
    signals = product.get("ranking_signals", {})
    slots   = state.slots
    score   = 0.0

    skin_type     = slots.get("skin_type", "")
    skin_concerns = slots.get("skin_concern", []) or []
    hair_type     = slots.get("hair_type", "")
    hair_concerns = slots.get("hair_concern", []) or []
    sensitivity   = slots.get("sensitivity", "")

    if isinstance(skin_concerns, str): skin_concerns = [skin_concerns]
    if isinstance(hair_concerns, str): hair_concerns = [hair_concerns]

    cat_skin = [s.lower() for s in attrs.get("skin_types", [])]
    cat_hair = [h.lower() for h in attrs.get("hair_types", [])]
    cat_conc = [c.lower() for c in attrs.get("concerns", [])]
    cat_safe = [s.lower() for s in attrs.get("sensitivity_safe", [])]

    if skin_type and any(skin_type.lower() in t for t in cat_skin):  score += 60
    for c in skin_concerns:
        if any(c.lower() in x for x in cat_conc): score += 25
    if hair_type and any(hair_type.lower() in t for t in cat_hair):  score += 60
    for c in hair_concerns:
        if any(c.lower() in x for x in cat_conc): score += 25
    if sensitivity and any(sensitivity.lower() in s for s in cat_safe): score += 15

    score += (signals.get("rating") or 0) * 5
    score += (signals.get("trending_score") or 0) / 10
    return score


def _best_for_step(step_def: Dict, main_cat: str, baby_section: Optional[str],
                   state: DialogState, used_ids: set) -> Optional[Dict]:
    candidates = []
    for product in PRODUCT_CATALOG:
        if not product.get("in_stock", True): continue
        if product["product_id"] in used_ids: continue
        cats  = product.get("categories", {})
        if cats.get("main_category") != main_cat: continue
        if cats.get("section", "") not in step_def["sections"]: continue
        score = _score_product(product, state)
        candidates.append((score, product))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (
        x[0],
        x[1]["ranking_signals"].get("rating") or 0,
        x[1]["ranking_signals"].get("trending_score") or 0,
    ), reverse=True)

    best = candidates[0][1]
    used_ids.add(best["product_id"])
    return best


def build_routine(state: DialogState) -> List[Dict]:
    main_cat     = state.slots.get("main_category")
    baby_section = state.slots.get("baby_section")
    if not main_cat or main_cat not in ROUTINE_MAP:
        return []

    used_ids: set = set()
    routine  = []
    step_num = 1

    for step_def in ROUTINE_MAP[main_cat]:
        product = _best_for_step(step_def, main_cat, baby_section, state, used_ids)
        if product:
            routine.append({
                "step_number": step_num,
                "step_name":   step_def["step"],
                "purpose":     step_def["purpose"],
                "product":     product,
            })
            step_num += 1

    return routine


def get_recommendations(state: DialogState) -> List[Dict]:
    return [s["product"] for s in build_routine(state) if s.get("product")]


def format_routine_intro(routine: List[Dict], state: DialogState) -> str:
    if not routine:
        return "I wasn't able to find matching products right now. Let me know if you'd like to adjust your preferences! 🔍"

    slots     = state.slots
    skin_type = slots.get("skin_type", "")
    concerns  = slots.get("skin_concern") or slots.get("hair_concern") or []
    cat       = slots.get("main_category", "")
    baby_sec  = slots.get("baby_section", "")

    if isinstance(concerns, str): concerns = [concerns]

    if skin_type and concerns:
        profile = f"your **{skin_type}** skin with **{', '.join(concerns)}** concerns"
    elif skin_type:
        profile = f"your **{skin_type}** skin"
    elif concerns:
        profile = f"your **{', '.join(concerns)}** concerns"
    elif baby_sec:
        profile = f"**{baby_sec}**"
    else:
        profile = f"**{cat}** care"

    return (
        f"Here's a routine I built for {profile}. ✨\n\n"
        "Feel free to ask about any product, request a swap, or tell me your budget!"
    )


def get_step_alternatives(
    state: DialogState,
    step_name: Optional[str],
    price_direction: Optional[str],
    exclude_ids: set,
) -> List[Dict]:
    main_cat = state.slots.get("main_category")
    if not main_cat or main_cat not in ROUTINE_MAP:
        return []

    target_sections = []
    if step_name:
        for step_def in ROUTINE_MAP[main_cat]:
            if step_name.lower() in step_def["step"].lower():
                target_sections = step_def["sections"]
                break
    if not target_sections:
        target_sections = [s for step in ROUTINE_MAP[main_cat] for s in step["sections"]]

    current_prices = [p.get("price", 0) for p in state.products]
    avg_price = sum(current_prices) / len(current_prices) if current_prices else 1500

    candidates = []
    for product in PRODUCT_CATALOG:
        if not product.get("in_stock", True): continue
        if product["product_id"] in exclude_ids: continue
        cats = product.get("categories", {})
        if cats.get("main_category") != main_cat: continue
        if cats.get("section") not in target_sections: continue

        price = product.get("price", 0)
        if price_direction == "lower" and price >= avg_price: continue
        if price_direction == "higher" and price <= avg_price: continue

        score = _score_product(product, state)
        candidates.append((score, product))

    candidates.sort(key=lambda x: (x[0], x[1]["ranking_signals"].get("rating") or 0), reverse=True)
    return [p for _, p in candidates[:3]]