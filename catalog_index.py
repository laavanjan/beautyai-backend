"""
catalog_index.py — Catalog Intelligence Agent

Runs ONCE on startup. Reads products_db.json and builds a live index
of every valid value for every filterable property.

All other agents (search, chat, routine) import from here instead of
using hardcoded strings in prompts or keyword maps.

Usage:
    from catalog_index import CATALOG_INDEX, build_search_context

CATALOG_INDEX structure:
{
  "categories":        ["face", "hair", "body", "baby"],
  "subcategories":     ["Cleansers", "Shampoos", ...],
  "brands":            ["CeraVe", "Garnier", ...],
  "skin_types":        ["oily", "dry", "combination", "normal", "sensitive"],
  "hair_types":        ["curly", "wavy", "straight", "coily", "damaged", "fine", "thick"],
  "concerns":          ["acne", "dryness", "frizz", ...],
  "sensitivity_safe":  ["sulfate_free", "fragrance_free", "paraben_free", ...],
  "textures":          ["gel", "cream", "foam", "serum", ...],
  "key_ingredients":   ["Niacinamide", "Vitamin C", ...],
  "price_range":       {"min": 420, "max": 8073},
  "routine_steps":     ["cleanser", "toner", "moisturiser", ...],

  # Per-category breakdowns — used for targeted prompting
  "by_category": {
    "face": {
      "subcategories": ["Cleansers", "Moisturisers", ...],
      "skin_types":    ["oily", "dry", ...],
      "concerns":      ["acne", "dryness", ...],
      "sensitivity_safe": ["fragrance_free", "oil_free", ...],
      "price_range":   {"min": 800, "max": 5000},
      "brands":        ["CeraVe", "Neutrogena", ...],
      "count":         20,
    },
    "hair": { ... },
    ...
  },

  # Maps section name → list of sensitivity_safe tags present in that section
  # Used by search agent to know which filters are valid for a query
  "section_filters": {
    "Shampoos": ["sulfate_free", "paraben_free"],
    "Cleansers": ["fragrance_free", "oil_free"],
    ...
  },

  # Keyword → subcategory map — built from actual catalog subcategory names
  # Replaces the hardcoded _SECTION_KEYWORD_MAP in conversational_engine.py
  "section_keywords": {
    "cleanser": "Cleansers",
    "shampoo":  "Shampoos",
    ...
  },

  "total_products": 40,
  "in_stock_count": 38,
}
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CATALOG LOADER — same multi-path search as products.py
# ─────────────────────────────────────────────────────────────────────────────

def _find_catalog() -> List[Dict]:
    here = Path(__file__).parent
    candidates = [
        here / "products_db.json",
        here / "data" / "products_db.json",
        here.parent / "products_db.json",
        Path.cwd() / "products_db.json",
        Path.cwd() / "data" / "products_db.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            products = data if isinstance(data, list) else data.get("products", [])
            print(f"[CATALOG INDEX] Loaded {len(products)} products from {path}")
            return products
    raise FileNotFoundError(
        f"products_db.json not found. Tried: {[str(c) for c in candidates]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION KEYWORD GENERATOR
# Derives keyword → subcategory map from actual catalog subcategory names.
# No hardcoding — if you add "Serums" to the catalog, it auto-appears here.
# ─────────────────────────────────────────────────────────────────────────────

def _build_section_keywords(subcategories: List[str]) -> Dict[str, str]:
    """
    Auto-generates keyword → subcategory mappings from real subcategory names.

    Strategy:
    1. Use the full subcategory name lowercased (e.g. "cleansers" → "Cleansers")
    2. Use each word in the subcategory name if it's meaningful (len > 3)
    3. Add known short aliases (e.g. "spf" → "Sun Care", "bb" → "BB & CC Creams")

    Result: a dict sorted by keyword length (longest first) so longer,
    more specific keywords win over shorter ones during matching.
    """
    kw_map: Dict[str, str] = {}

    # Static aliases for subcategory names that don't match common user language
    STATIC_ALIASES = {
        "spf":             "Sun Care",
        "sunscreen":       "Sun Care",
        "sun block":       "Sun Care",
        "sunblock":        "Sun Care",
        "exfoliant":       "Exfoliators",
        "scrub":           "Exfoliators",
        "peeling":         "Exfoliators",
        "eye cream":       "Eye Care",
        "eye gel":         "Eye Care",
        "under eye":       "Eye Care",
        "micellar":        "Makeup Removers",
        "cleansing wipe":  "Makeup Removers",
        "makeup remover":  "Makeup Removers",
        "bb cream":        "BB & CC Creams",
        "cc cream":        "BB & CC Creams",
        "hair oil":        "Hair Oils & Serums",
        "hair serum":      "Hair Oils & Serums",
        "argan oil":       "Hair Oils & Serums",
        "leave in":        "Leave-In Conditioners & Creams",
        "leave-in":        "Leave-In Conditioners & Creams",
        "pre shampoo":     "Pre-Shampoo / Rinse-Out Treatments",
        "pre-shampoo":     "Pre-Shampoo / Rinse-Out Treatments",
        "rinse out":       "Pre-Shampoo / Rinse-Out Treatments",
        "styling gel":     "Styling Gels & Creams",
        "hair gel":        "Styling Gels & Creams",
        "body wash":       "Body Washes & Cleansers",
        "shower gel":      "Body Washes & Cleansers",
        "body cleanser":   "Body Washes & Cleansers",
        "body lotion":     "Body Creams & Lotions",
        "body cream":      "Body Creams & Lotions",
        "body moisturiser":"Body Creams & Lotions",
        "body moisturizer":"Body Creams & Lotions",
        "hand cream":      "Hand and Foot Care",
        "foot cream":      "Hand and Foot Care",
        "hand lotion":     "Hand and Foot Care",
        "deodorant":       "Deodorants & Antiperspirants",
        "antiperspirant":  "Deodorants & Antiperspirants",
        "roll on":         "Deodorants & Antiperspirants",
        "baby wash":       "Baby Bath & Shampoo",
        "baby shampoo":    "Baby Bath & Shampoo",
        "baby bath":       "Baby Bath & Shampoo",
        "baby lotion":     "Baby Lotions & Creams",
        "baby cream":      "Baby Lotions & Creams",
        "baby milk":       "Baby Milk Powder",
        "milk powder":     "Baby Milk Powder",
        "formula":         "Baby Milk Powder",
        # ── Explicit word-level overrides ────────────────────────────────────
        # These MUST be here because the auto-deriver could pull the wrong
        # section from a subcategory whose name contains these words.
        # e.g. "Baby Bath & Shampoo" contains "shampoo" → must not win.
        # e.g. "Night Cream" contains "cream" → must not override "Body Creams & Lotions".
        "shampoo":         "Shampoos",
        "serum":           "Serums & Targeted Treatments",
        "toner":           "Toners",
        "exfoliator":      "Exfoliators",
        "moisturizer":     "Moisturisers",   # American spelling
        "face wash":       "Cleansers",
    }

    # Auto-derive from real subcategory names first (lower priority)
    # Static aliases are added AFTER so they override any auto-derived conflicts
    SKIP_WORDS = {"and", "the", "for", "with", "&", "creams", "treatments", "bath"}
    for subcat in subcategories:
        # Full lowercase name
        full_lower = subcat.lower()
        kw_map[full_lower] = subcat

        # Singular form (strip trailing 's')
        if full_lower.endswith("s") and len(full_lower) > 4:
            singular = full_lower[:-1]
            if singular not in kw_map:
                kw_map[singular] = subcat

        # Individual meaningful words
        # IMPORTANT: Baby subcategories only match when "baby" is explicit.
        # Skip word-level derivation for baby sections to prevent "shampoo"
        # alone from matching "Baby Bath & Shampoo".
        if full_lower.startswith("baby"):
            continue
        words = re.split(r"[\s&/\-]+", full_lower)
        for word in words:
            if len(word) > 3 and word not in SKIP_WORDS and word not in kw_map:
                # Only add if this word appears in exactly ONE subcategory name.
                # Words like "cream" appear in Night Cream, Day Creams, Body Creams
                # — too ambiguous to use as a standalone keyword.
                occurrences = sum(1 for s in subcategories if word in s.lower())
                if occurrences == 1:
                    kw_map[word] = subcat

    # Add static aliases LAST — they override any auto-derived conflicts.
    # e.g. "baby shampoo" must map to "Baby Bath & Shampoo", not "Shampoos"
    for alias, section in STATIC_ALIASES.items():
        if section in subcategories:
            kw_map[alias] = section

    return kw_map


# ─────────────────────────────────────────────────────────────────────────────
# FILTER QUALIFIER DETECTOR — built from catalog values, not hardcoded
# ─────────────────────────────────────────────────────────────────────────────

def _build_filter_qualifiers(index: Dict) -> List[str]:
    """
    Build a list of phrases that indicate the user wants FILTERED results
    (not a full section browse). Derived entirely from catalog values.

    Any message matching one of these → skip section mode → use search_products()
    """
    qualifiers = set()

    # All sensitivity_safe tags (with both formats: underscore and hyphen)
    for tag in index.get("sensitivity_safe", []):
        qualifiers.add(tag.replace("_", "-"))   # "sulfate-free"
        qualifiers.add(tag.replace("_", " "))   # "sulfate free"

    # All concerns
    for concern in index.get("concerns", []):
        qualifiers.add(f"for {concern}")
        qualifiers.add(concern)

    # All skin/hair types
    for t in index.get("skin_types", []):
        qualifiers.add(f"for {t}")
        qualifiers.add(f"for {t} skin")
        qualifiers.add(f"{t} skin")
    for t in index.get("hair_types", []):
        qualifiers.add(f"for {t}")
        qualifiers.add(f"for {t} hair")
        qualifiers.add(f"{t} hair")

    # All brands (user asking for a specific brand = filtered search)
    for brand in index.get("brands", []):
        qualifiers.add(brand.lower())

    # All key ingredients
    for ing in index.get("key_ingredients", []):
        qualifiers.add(f"with {ing.lower()}")
        qualifiers.add(ing.lower())

    # Price qualifiers
    qualifiers.update({
        "under rs", "below rs", "under rs.", "below rs.",
        "budget", "affordable", "cheap", "inexpensive",
        "under 500", "under 1000", "under 2000", "under 3000",
    })

    # General filter words
    qualifiers.update({
        "natural", "organic", "gentle", "hypoallergenic",
        "best for", "good for", "recommended for", "suitable for",
    })

    return sorted(qualifiers)


# ─────────────────────────────────────────────────────────────────────────────
# INDEX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_catalog_index(products: List[Dict]) -> Dict:
    """
    Scan every product and extract all unique values per property.
    Returns the full CATALOG_INDEX dict.
    """

    # Global accumulators
    categories       = set()
    subcategories    = set()
    brands           = set()
    skin_types       = set()
    hair_types       = set()
    concerns         = set()
    sensitivity_safe = set()
    textures         = set()
    key_ingredients  = set()
    routine_steps    = set()
    prices           = []

    # Per-category accumulators
    by_category: Dict[str, Dict[str, Any]] = {}

    # Section → valid filters (what sensitivity tags actually exist per section)
    section_filters: Dict[str, set] = {}

    total = 0
    in_stock = 0

    for p in products:
        total += 1
        if p.get("in_stock", True):
            in_stock += 1

        # Normalise category/subcategory — support flat and nested schemas
        cat = (
            p.get("category") or
            (p.get("categories") or {}).get("main_category", "")
        ).strip().lower()

        subcat = (
            p.get("subcategory") or
            (p.get("categories") or {}).get("section", "")
        ).strip()

        attrs = p.get("attributes") or {}

        p_skin_types       = attrs.get("skin_types")       or p.get("skin_types")       or []
        p_hair_types       = attrs.get("hair_types")       or p.get("hair_types")       or []
        p_concerns         = attrs.get("concerns")         or p.get("concerns")         or []
        p_sensitivity_safe = attrs.get("sensitivity_safe") or p.get("sensitivity_safe") or []
        p_texture          = attrs.get("texture")          or p.get("texture")
        p_key_ingredients  = attrs.get("key_ingredients")  or p.get("key_ingredients")  or []
        p_routine_step     = p.get("routine_step", "")
        p_brand            = (p.get("brand") or "").strip()
        p_price            = p.get("price")

        # Global sets
        if cat:              categories.add(cat)
        if subcat:           subcategories.add(subcat)
        if p_brand:          brands.add(p_brand)
        if p_texture:        textures.add(p_texture)
        if p_routine_step:   routine_steps.add(p_routine_step)
        if p_price:          prices.append(p_price)

        skin_types.update(p_skin_types)
        hair_types.update(p_hair_types)
        concerns.update(p_concerns)
        sensitivity_safe.update(p_sensitivity_safe)
        key_ingredients.update(p_key_ingredients)

        # Section → valid filters
        if subcat:
            if subcat not in section_filters:
                section_filters[subcat] = set()
            section_filters[subcat].update(p_sensitivity_safe)

        # Per-category breakdown
        if cat not in by_category:
            by_category[cat] = {
                "subcategories":    set(),
                "skin_types":       set(),
                "hair_types":       set(),
                "concerns":         set(),
                "sensitivity_safe": set(),
                "textures":         set(),
                "brands":           set(),
                "key_ingredients":  set(),
                "prices":           [],
                "count":            0,
            }
        bc = by_category[cat]
        bc["count"] += 1
        if subcat:          bc["subcategories"].add(subcat)
        if p_brand:         bc["brands"].add(p_brand)
        if p_texture:       bc["textures"].add(p_texture)
        bc["skin_types"].update(p_skin_types)
        bc["hair_types"].update(p_hair_types)
        bc["concerns"].update(p_concerns)
        bc["sensitivity_safe"].update(p_sensitivity_safe)
        bc["key_ingredients"].update(p_key_ingredients)
        if p_price:         bc["prices"].append(p_price)

    # Convert sets → sorted lists, sets → dicts for price ranges
    def _finalise_category(bc: Dict) -> Dict:
        result = {}
        for key, val in bc.items():
            if isinstance(val, set):
                result[key] = sorted(val)
            elif key == "prices" and val:
                result["price_range"] = {"min": min(val), "max": max(val)}
            elif key != "prices":
                result[key] = val
        return result

    index = {
        "categories":        sorted(categories),
        "subcategories":     sorted(subcategories),
        "brands":            sorted(brands),
        "skin_types":        sorted(skin_types),
        "hair_types":        sorted(hair_types),
        "concerns":          sorted(concerns),
        "sensitivity_safe":  sorted(sensitivity_safe),
        "textures":          sorted(textures),
        "key_ingredients":   sorted(key_ingredients),
        "routine_steps":     sorted(routine_steps),
        "price_range":       {"min": min(prices), "max": max(prices)} if prices else {},
        "by_category":       {cat: _finalise_category(bc) for cat, bc in by_category.items()},
        "section_filters":   {sec: sorted(tags) for sec, tags in section_filters.items()},
        "total_products":    total,
        "in_stock_count":    in_stock,
    }

    # Build dynamic keyword map and filter qualifiers FROM the index
    index["section_keywords"]   = _build_section_keywords(index["subcategories"])
    index["filter_qualifiers"]  = _build_filter_qualifiers(index)

    return index


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT HELPERS — called by agents to inject live catalog context
# ─────────────────────────────────────────────────────────────────────────────

def build_search_context(category: Optional[str] = None) -> str:
    """
    Returns a compact catalog context string for injection into LLM prompts.
    Scoped to category if provided, otherwise global.

    Example output (Hair):
        Subcategories : Shampoos, Conditioners, Hair Masks & Deep Treatments
        Hair types    : curly, wavy, straight, coily, damaged
        Concerns      : frizz, dryness, dandruff, breakage, hair loss
        Sensitivity   : sulfate_free, paraben_free, silicone_free
        Brands        : Garnier, L'Oréal, SheaMoisture
        Price range   : Rs.420 – Rs.3,200
    """
    if category and category.lower() in CATALOG_INDEX.get("by_category", {}):
        bc = CATALOG_INDEX["by_category"][category.lower()]
        pr = bc.get("price_range", {})
        lines = [
            f"Subcategories : {', '.join(bc.get('subcategories', []))}",
        ]
        if bc.get("skin_types"):
            lines.append(f"Skin types    : {', '.join(bc['skin_types'])}")
        if bc.get("hair_types"):
            lines.append(f"Hair types    : {', '.join(bc['hair_types'])}")
        if bc.get("concerns"):
            lines.append(f"Concerns      : {', '.join(bc['concerns'])}")
        if bc.get("sensitivity_safe"):
            lines.append(f"Sensitivity   : {', '.join(bc['sensitivity_safe'])}")
        if bc.get("brands"):
            lines.append(f"Brands        : {', '.join(bc['brands'])}")
        if pr:
            lines.append(f"Price range   : Rs.{pr['min']:,} – Rs.{pr['max']:,}")
        return "\n".join(lines)
    else:
        pr = CATALOG_INDEX.get("price_range", {})
        return "\n".join([
            f"Categories    : {', '.join(CATALOG_INDEX.get('categories', []))}",
            f"Subcategories : {', '.join(CATALOG_INDEX.get('subcategories', []))}",
            f"Concerns      : {', '.join(CATALOG_INDEX.get('concerns', []))}",
            f"Skin types    : {', '.join(CATALOG_INDEX.get('skin_types', []))}",
            f"Hair types    : {', '.join(CATALOG_INDEX.get('hair_types', []))}",
            f"Sensitivity   : {', '.join(CATALOG_INDEX.get('sensitivity_safe', []))}",
            f"Brands        : {', '.join(CATALOG_INDEX.get('brands', []))}",
            f"Price range   : Rs.{pr.get('min', 0):,} – Rs.{pr.get('max', 0):,}",
        ])


def get_valid_values(field: str, category: Optional[str] = None) -> List[str]:
    """
    Returns all valid values for a given field from the live catalog.
    Optionally scoped to a category.

    Examples:
        get_valid_values("concerns", "hair")
        → ["breakage", "dandruff", "dryness", "frizz", "hair_loss"]

        get_valid_values("sensitivity_safe")
        → ["fragrance_free", "oil_free", "paraben_free", "sulfate_free"]
    """
    if category:
        cat_lower = category.lower()
        bc = CATALOG_INDEX.get("by_category", {}).get(cat_lower, {})
        return bc.get(field, [])
    return CATALOG_INDEX.get(field, [])


def has_filter_qualifier(message: str) -> bool:
    """
    Returns True if the message contains a filter qualifier — meaning the
    user wants a filtered subset, not a full section browse.
    Used by the search agent to decide between section mode and search mode.
    """
    msg = message.lower()
    return any(q in msg for q in CATALOG_INDEX.get("filter_qualifiers", []))


def detect_section(message: str) -> Optional[str]:
    """
    Detect which catalog section the user is asking about.
    Returns None if filter qualifiers are present (those need search_products).
    Longer keywords checked first to prevent partial matches.
    """
    if has_filter_qualifier(message):
        return None

    msg = message.lower()
    kw_map = CATALOG_INDEX.get("section_keywords", {})
    for kw in sorted(kw_map.keys(), key=len, reverse=True):
        if kw in msg:
            return kw_map[kw]
    return None


def get_section_valid_filters(section: str) -> List[str]:
    """
    Returns which sensitivity_safe filters actually exist for a given section.
    Used to validate user filter requests and build smart prompt context.

    Example:
        get_section_valid_filters("Shampoos") → ["paraben_free", "sulfate_free"]
        get_section_valid_filters("Cleansers") → ["fragrance_free", "oil_free"]
    """
    return CATALOG_INDEX.get("section_filters", {}).get(section, [])


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON — built once on import, shared across all agents
# ─────────────────────────────────────────────────────────────────────────────

_products = _find_catalog()
CATALOG_INDEX: Dict = build_catalog_index(_products)

def _print_index() -> None:
    """Pretty-print the full CATALOG_INDEX so you can verify what was extracted."""
    I = CATALOG_INDEX
    pr = I.get("price_range", {})

    SEP  = "─" * 60
    SEP2 = "━" * 60

    print(f"\n{SEP2}")
    print(f"  CATALOG INDEX — Beauty Mart")
    print(f"{SEP2}")
    print(f"  Total products : {I['total_products']}  (in stock: {I['in_stock_count']})")
    print(f"  Price range    : Rs.{pr.get('min', 0):,} – Rs.{pr.get('max', 0):,}")
    print()

    print(f"  {'GLOBAL VALUES':}")
    print(f"  {SEP}")
    _pf("Categories",       I.get("categories", []))
    _pf("Subcategories",    I.get("subcategories", []))
    _pf("Brands",           I.get("brands", []))
    _pf("Skin types",       I.get("skin_types", []))
    _pf("Hair types",       I.get("hair_types", []))
    _pf("Concerns",         I.get("concerns", []))
    _pf("Sensitivity tags", I.get("sensitivity_safe", []))
    _pf("Textures",         I.get("textures", []))
    _pf("Key ingredients",  I.get("key_ingredients", []))
    _pf("Routine steps",    I.get("routine_steps", []))
    print()

    print(f"  PER-CATEGORY BREAKDOWN")
    print(f"  {SEP}")
    for cat, bc in I.get("by_category", {}).items():
        cpr = bc.get("price_range", {})
        print(f"  [{cat.upper()}]  {bc['count']} products  |  "
              f"Rs.{cpr.get('min', 0):,} – Rs.{cpr.get('max', 0):,}")
        _pf("  Subcategories",    bc.get("subcategories", []), indent=4)
        _pf("  Brands",           bc.get("brands", []),        indent=4)
        _pf("  Skin types",       bc.get("skin_types", []),    indent=4)
        _pf("  Hair types",       bc.get("hair_types", []),    indent=4)
        _pf("  Concerns",         bc.get("concerns", []),      indent=4)
        _pf("  Sensitivity",      bc.get("sensitivity_safe",[]),indent=4)
        _pf("  Textures",         bc.get("textures", []),      indent=4)
        print()

    print(f"  SECTION → VALID FILTERS  (what sensitivity tags exist per section)")
    print(f"  {SEP}")
    for section, tags in I.get("section_filters", {}).items():
        print(f"    {section:<40} {tags}")
    print()

    print(f"  SECTION KEYWORDS  ({len(I.get('section_keywords', {}))} keywords → section name)")
    print(f"  {SEP}")
    # Group by target section for readability
    grouped: Dict[str, list] = {}
    for kw, sec in sorted(I.get("section_keywords", {}).items(), key=lambda x: x[1]):
        grouped.setdefault(sec, []).append(f'"{kw}"')
    for sec, kws in grouped.items():
        print(f"    {sec:<40} ← {', '.join(kws)}")
    print()

    print(f"  FILTER QUALIFIERS  ({len(I.get('filter_qualifiers', []))} phrases that force filtered search)")
    print(f"  {SEP}")
    # Print in rows of 4
    quals = I.get("filter_qualifiers", [])
    for i in range(0, len(quals), 4):
        row = quals[i:i+4]
        print("    " + "   ".join(f"{q:<22}" for q in row))
    print()

    print(f"{SEP2}\n")


def _pf(label: str, values: list, indent: int = 2) -> None:
    """Print a labeled list, wrapping long lines."""
    if not values:
        return
    prefix = " " * indent
    print(f"{prefix}{label:<20}: {', '.join(str(v) for v in values)}")


_print_index()