"""
products.py — BeautyAI product catalog, routine builder, and search

Responsibilities:
  - Load products_db.json at startup
  - Expose PRODUCT_CATALOG, CATALOG_VALUES
  - build_routine(state)  → scored routine for a user profile
  - search_products(...)  → filtered + scored product search
  - get_section_products(section) → ALL products in a section, sorted by rating
"""

import json
import os
from typing import List, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CATALOG
# ─────────────────────────────────────────────────────────────────────────────

def _find_catalog() -> str:
    """Search common locations for products_db.json."""
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "products_db.json"),           # same folder as products.py
        os.path.join(base, "data", "products_db.json"),   # data/ subfolder
        os.path.join(base, "..", "products_db.json"),     # one level up
        "products_db.json",                               # cwd
        "data/products_db.json",                          # cwd/data/
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]  # fallback — will raise FileNotFoundError with clear message


def _normalize_product(p: Dict) -> Dict:
    """
    Normalize a product dict to a consistent internal format.

    Supports two schemas:
      Schema A (new): p["categories"]["main_category"], p["categories"]["section"]
      Schema B (old/flat): p["category"] (lowercase), p["subcategory"]

    After normalization, always use:
      p["_category"] → e.g. "Face"
      p["_section"]  → e.g. "Cleansers"
    """
    if "_category" in p:
        return p  # already normalized

    cats = p.get("categories", {})

    # Schema A
    if cats.get("main_category"):
        cat     = cats["main_category"]
        section = cats.get("section", "")
    # Schema B (flat)
    else:
        raw_cat = p.get("category", p.get("main_category", ""))
        cat     = raw_cat.capitalize() if raw_cat else ""
        section = p.get("subcategory", p.get("section", ""))

    p["_category"] = cat
    p["_section"]  = section
    return p


def _load_catalog() -> List[Dict]:
    path = _find_catalog()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data if isinstance(data, list) else data.get("products", data.get("items", []))
        products = [_normalize_product(p) for p in raw]
        print(f"[CATALOG] Loaded {len(products)} products from {path}")
        return products
    except FileNotFoundError:
        print(f"[CATALOG] WARNING: products_db.json not found in any expected location")
        print(f"[CATALOG] Searched: same dir as products.py, data/, parent dir, cwd")
        return []
    except Exception as e:
        print(f"[CATALOG] ERROR loading {path}: {e}")
        return []


PRODUCT_CATALOG: List[Dict] = _load_catalog()


def _build_catalog_values() -> Dict:
    categories = sorted({p["_category"] for p in PRODUCT_CATALOG if p.get("_category")})
    sections   = sorted({p["_section"]  for p in PRODUCT_CATALOG if p.get("_section")})
    brands     = sorted({p.get("brand", "") for p in PRODUCT_CATALOG if p.get("brand")})
    return {"categories": categories, "sections": sections, "brands": brands}


CATALOG_VALUES: Dict = _build_catalog_values()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTINE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ROUTINE_CONFIG = {
    "face": [
        {"step": 1, "routine_step": "cleanser",        "subcategories": ["Cleansers"]},
        {"step": 2, "routine_step": "makeup_remover",  "subcategories": ["Makeup Removers"]},
        {"step": 3, "routine_step": "toner",           "subcategories": ["Toners"]},
        {"step": 4, "routine_step": "exfoliator",      "subcategories": ["Exfoliators"]},
        {"step": 5, "routine_step": "treatment_serum", "subcategories": ["Serums & Targeted Treatments"]},
        {"step": 6, "routine_step": "eye_care",        "subcategories": ["Eye Care"]},
        {"step": 7, "routine_step": "moisturiser",     "subcategories": ["Moisturisers", "Day Creams"]},
        {"step": 8, "routine_step": "night_cream",     "subcategories": ["Night Cream"]},
        {"step": 9, "routine_step": "sun_protection",  "subcategories": ["Sun Care", "Sunscreens"]},
    ],
    "hair": [
        {"step": 1, "routine_step": "shampoo",       "subcategories": ["Shampoos"]},
        {"step": 2, "routine_step": "pre_treatment",  "subcategories": ["Shampoos"]},
        {"step": 3, "routine_step": "conditioner",   "subcategories": ["Conditioners"]},
        {"step": 4, "routine_step": "hair_mask",     "subcategories": ["Hair Masks & Deep Treatments"]},
        {"step": 5, "routine_step": "hair_serum",    "subcategories": ["Hair Masks & Deep Treatments"]},
    ],
    "body": [
        {"step": 1, "routine_step": "body_wash",        "subcategories": ["Body Washes & Cleansers"]},
        {"step": 2, "routine_step": "body_moisturiser", "subcategories": ["Body Creams & Lotions"], "texture_filter": ["cream", "lotion", "liquid"]},
        {"step": 3, "routine_step": "hand_foot_care",   "subcategories": ["Hand and Foot Care"]},
        {"step": 4, "routine_step": "deodorant",        "subcategories": ["Deodorants & Antiperspirants"]},
        {"step": 5, "routine_step": "body_sunscreen",   "subcategories": ["Sunscreens"]},
    ],
}

# Human-readable step names
_STEP_NAMES = {
    "cleanser":        "Cleanser",
    "makeup_remover":  "Makeup Remover",
    "toner":           "Toner",
    "exfoliator":      "Exfoliator",
    "treatment_serum": "Serum",
    "eye_care":        "Eye Care",
    "moisturiser":     "Moisturiser",
    "night_cream":     "Night Cream",
    "sun_protection":  "Sun Protection",
    "shampoo":         "Shampoo",
    "pre_treatment":   "Pre-Treatment",
    "conditioner":     "Conditioner",
    "hair_mask":       "Hair Mask",
    "hair_serum":      "Hair Serum",
    "body_wash":       "Body Wash",
    "body_moisturiser":"Body Moisturiser",
    "hand_foot_care":  "Hand & Foot Care",
    "deodorant":       "Deodorant",
    "body_sunscreen":  "Body Sunscreen",
}

# Step purpose descriptions shown in the routine card
_STEP_PURPOSES = {
    "cleanser":        "Removes dirt, oil, and impurities without stripping the skin.",
    "makeup_remover":  "Gently dissolves makeup and sunscreen before cleansing.",
    "toner":           "Rebalances skin pH and preps skin to absorb serums better.",
    "exfoliator":      "Removes dead skin cells 1–3× a week for a brighter complexion.",
    "treatment_serum": "Targets your specific concern with active ingredients.",
    "eye_care":        "Hydrates and protects the delicate skin around your eyes.",
    "moisturiser":     "Locks in hydration and keeps your skin barrier healthy.",
    "night_cream":     "Deep repair and renewal while you sleep.",
    "sun_protection":  "Protects against UV damage — essential every single morning.",
    "shampoo":         "Cleanses the scalp and removes product buildup.",
    "pre_treatment":   "Applied before shampooing to protect and strengthen hair.",
    "conditioner":     "Smooths the hair cuticle and adds moisture after shampooing.",
    "hair_mask":       "Intense weekly treatment for deep repair and nourishment.",
    "hair_serum":      "Finishing treatment to add shine and tame frizz.",
    "body_wash":       "Cleanses the skin while maintaining moisture balance.",
    "body_moisturiser":"Seals in moisture right after your shower.",
    "hand_foot_care":  "Targeted hydration for hands and feet.",
    "deodorant":       "Keeps you fresh and confident all day.",
    "body_sunscreen":  "Protects exposed skin from UV rays on sunny days.",
}


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT SCORER
# Scores a product against a user profile — higher = better match
# ─────────────────────────────────────────────────────────────────────────────

def _score_product(product: Dict, slots: Dict) -> float:
    score = 0.0
    attrs = product.get("attributes", {})

    skin_type    = slots.get("skin_type", "")
    skin_concern = slots.get("skin_concern") or []
    hair_type    = slots.get("hair_type", "")
    hair_concern = slots.get("hair_concern") or []
    exclusions   = slots.get("exclusions") or []

    if isinstance(skin_concern, str): skin_concern = [skin_concern]
    if isinstance(hair_concern, str): hair_concern = [hair_concern]
    if isinstance(exclusions,   str): exclusions   = [exclusions]

    # Skin type match
    skin_types_attr = attrs.get("skin_types") or product.get("skin_types") or []
    if skin_type and skin_type in skin_types_attr:
        score += 60

    # Skin concern match
    concerns_attr = attrs.get("concerns") or product.get("concerns") or []
    for c in skin_concern:
        if c in concerns_attr:
            score += 25

    # Hair type match
    hair_types_attr = attrs.get("hair_types") or product.get("hair_types") or []
    if hair_type and hair_type in hair_types_attr:
        score += 60

    # Hair concern match
    for c in hair_concern:
        if c in concerns_attr:
            score += 25

    # Sensitivity / exclusion safety
    sensitivity_safe = attrs.get("sensitivity_safe") or product.get("sensitivity_safe") or []
    for tag in sensitivity_safe:
        score += 15

    # Hard penalty: product contains an excluded ingredient
    product_exclusions = attrs.get("exclusions") or product.get("exclusions") or []
    for ex in exclusions:
        if ex in product_exclusions:
            score -= 999  # effectively disqualifies

    # Ranking signals as tiebreaker
    signals = product.get("ranking_signals") or {}
    rating         = signals.get("rating", product.get("average_rating", 0)) or 0
    trending_score = signals.get("trending_score", product.get("trending_score", 0)) or 0
    score += rating * 5 + trending_score / 10

    return score


# ─────────────────────────────────────────────────────────────────────────────
# SECTION HELPER — used by search agent for fresh section browsing
# Returns ALL in-stock products in a section, sorted by rating descending
# No profile filtering — this is a clean catalog browse
# ─────────────────────────────────────────────────────────────────────────────

def get_section_products(section: str) -> List[Dict]:
    """
    Return all in-stock products in a section, sorted by rating.
    This is intentionally profile-agnostic — used when user asks for a
    section directly (e.g. "show me all toners") with no filtering.
    """
    products = [
        p for p in PRODUCT_CATALOG
        if p.get("_section") == section
        and p.get("in_stock", True)
    ]
    products.sort(key=lambda p: -(
        (p.get("ranking_signals") or {}).get("rating", p.get("average_rating", 0)) or 0
    ))
    return products


# ─────────────────────────────────────────────────────────────────────────────
# ROUTINE BUILDER
# Builds a scored, profile-matched routine for the user
# ─────────────────────────────────────────────────────────────────────────────

def build_routine(state) -> List[Dict]:
    """
    Build a personalised routine based on state.slots.

    Returns a list of step dicts:
    {
        step_number : int,
        routine_step: str,   # internal key e.g. "cleanser"
        step_name   : str,   # display name e.g. "Cleanser"
        purpose     : str,   # one-line description
        product     : Dict,  # best-matched product for this step
    }

    Steps with no matching product are skipped.
    """
    slots = state.slots if hasattr(state, "slots") else state
    category = (slots.get("main_category") or "").lower()

    if category not in ROUTINE_CONFIG:
        # Try to infer from slots
        if slots.get("hair_type") or slots.get("hair_concern"):
            category = "hair"
        elif slots.get("skin_type") or slots.get("skin_concern"):
            category = "face"
        else:
            return []

    config = ROUTINE_CONFIG[category]
    routine = []

    # Track which product IDs we've already used so no duplicate products in routine
    used_ids = set()

    for step_config in config:
        step_num     = step_config["step"]
        routine_step = step_config["routine_step"]
        subcategories = step_config["subcategories"]
        texture_filter = step_config.get("texture_filter")

        # Gather candidates from all subcategories for this step
        candidates = []
        for section in subcategories:
            for p in PRODUCT_CATALOG:
                if p.get("product_id") in used_ids:
                    continue
                if not p.get("in_stock", True):
                    continue
                p_section = p.get("_section", "")
                if p_section != section:
                    continue
                if texture_filter:
                    p_texture = (p.get("attributes") or {}).get("texture", p.get("texture", ""))
                    if p_texture not in texture_filter:
                        continue
                candidates.append(p)

        if not candidates:
            continue

        # Score all candidates against user profile
        scored = sorted(candidates, key=lambda p: _score_product(p, slots), reverse=True)
        best = scored[0]

        # Skip this step if the product is disqualified (score < -900 means excluded ingredient)
        if _score_product(best, slots) < -900:
            continue

        used_ids.add(best.get("product_id"))
        routine.append({
            "step_number":  step_num,
            "routine_step": routine_step,
            "step_name":    _STEP_NAMES.get(routine_step, routine_step.replace("_", " ").title()),
            "purpose":      _STEP_PURPOSES.get(routine_step, ""),
            "product":      best,
        })

    return routine


def format_routine_intro(routine: List[Dict], state) -> str:
    """Generate a plain-text intro string for the routine (used as fallback)."""
    slots = state.slots if hasattr(state, "slots") else state
    concern = slots.get("skin_concern") or slots.get("hair_concern") or []
    if isinstance(concern, list):
        concern = ", ".join(concern)
    skin_type = slots.get("skin_type") or slots.get("hair_type") or ""
    steps = len(routine)
    return (
        f"Here's your personalised {steps}-step routine"
        + (f" for {skin_type} skin" if skin_type else "")
        + (f" targeting {concern}" if concern else "")
        + "! Scroll through each step below. ✨"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH — profile-aware product search (used when no specific section)
# ─────────────────────────────────────────────────────────────────────────────

def search_products(
    query:            Optional[str]       = None,
    main_category:    Optional[str]       = None,
    section:          Optional[str]       = None,
    brand:            Optional[str]       = None,
    skin_types:       Optional[List[str]] = None,
    hair_types:       Optional[List[str]] = None,
    concerns:         Optional[List[str]] = None,
    texture:          Optional[str]       = None,
    sensitivity_safe: Optional[List[str]] = None,
    key_ingredients:  Optional[List[str]] = None,
    max_price:        Optional[float]     = None,
    limit:            int                 = 6,
) -> List[Dict]:
    """
    Filter + score products against search parameters.
    Returns up to `limit` products sorted by score descending.
    """
    results = []

    for p in PRODUCT_CATALOG:
        if not p.get("in_stock", True):
            continue

        attrs = p.get("attributes", {})

        # Hard filters
        if main_category and p.get("_category", "").lower() != main_category.lower():
            continue
        if section and p.get("_section") != section:
            continue
        if brand and brand.lower() not in p.get("brand", "").lower():
            continue
        if max_price is not None and p.get("price", 999999) > max_price:
            continue
        if texture:
            p_texture = attrs.get("texture", p.get("texture", ""))
            if p_texture != texture:
                continue
        if sensitivity_safe:
            p_safe = attrs.get("sensitivity_safe", p.get("sensitivity_safe", []))
            if not all(s in p_safe for s in sensitivity_safe):
                continue
        if key_ingredients:
            p_ings = [i.lower() for i in (p.get("key_ingredients") or attrs.get("key_ingredients") or [])]
            if not any(k.lower() in p_ings for k in key_ingredients):
                continue

        # Build a score for this product
        score = 0.0

        # Skin type match
        p_skin_types = attrs.get("skin_types", p.get("skin_types", []))
        if skin_types:
            for st in skin_types:
                clean = st.replace(" skin", "").strip()
                if clean in p_skin_types:
                    score += 60

        # Hair type match
        p_hair_types = attrs.get("hair_types", p.get("hair_types", []))
        if hair_types:
            for ht in hair_types:
                clean = ht.replace(" hair", "").strip()
                if clean in p_hair_types:
                    score += 60

        # Concern match
        p_concerns = attrs.get("concerns", p.get("concerns", []))
        if concerns:
            for c in concerns:
                if c in p_concerns:
                    score += 25

        # Query: re-rank (not hard filter)
        if query:
            ql = query.lower()
            name_lower = p.get("name", "").lower()
            desc_lower = p.get("description", "").lower()
            if ql in name_lower:
                score += 50
            elif ql in desc_lower:
                score += 20

        # Ranking signals tiebreaker
        signals        = p.get("ranking_signals") or {}
        rating         = signals.get("rating", p.get("average_rating", 0)) or 0
        trending_score = signals.get("trending_score", p.get("trending_score", 0)) or 0
        score += rating * 5 + trending_score / 10

        results.append((score, p))

    results.sort(key=lambda x: -x[0])
    return [p for _, p in results[:limit]]