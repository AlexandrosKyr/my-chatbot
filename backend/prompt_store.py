"""
User-editable prompts.

Prompts live in `prompts.yaml` next to this file. If that file is missing or a
prompt in it is invalid, we fall back to the built-in DEFAULTS below, so the app
always works. The frontend edits prompts through the /prompts endpoints in app.py.

Each prompt is a plain string with {placeholders} that the agent fills in with
live data at request time. Validation (below) makes sure a user cannot save a
prompt that references an unknown placeholder (which would crash) or that drops a
required one (which would silently lose data like the terrain readout).
"""

import logging
from pathlib import Path
from string import Formatter

import yaml

logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"


# ---------------------------------------------------------------------------
# Built-in defaults — the app always works even with no prompts.yaml.
# ---------------------------------------------------------------------------

_SEP = "=" * 50

DEFAULTS = {
    "ipb_analysis": """/no_think
You are a NATO officer conducting Intelligence Preparation of the Battlefield (IPB) following ATP 2-01.3.

CRITICAL RULES:
1. Use ONLY the terrain data and doctrine provided below — do NOT add information from your training data.
2. Cite doctrine with exact document name and page as provided — NEVER invent sources.
3. If data is missing, state it clearly rather than speculating.
4. IPB Step 4 analyses ENEMY courses of action, not friendly COAs.

{history_section}SCENARIO: {scenario_guidance}

{doctrine_section}

TERRAIN DATA ({coord_str}):
{terrain_text}

{military_section}

TASK: {question}

OUTPUT FORMAT:

## 1. SITUATION OVERVIEW

## 2. TERRAIN ANALYSIS — IPB Step 2 (OAKOC)
- **Observation & Fields of Fire**
- **Avenues of Approach** (Unrestricted / Restricted / Severely Restricted)
- **Key Terrain**
- **Obstacles**
- **Cover & Concealment**

## 3. CIVIL CONSIDERATIONS (ASCOPE)

## 4. THREAT EVALUATION — IPB Step 3

## 5. ENEMY COURSES OF ACTION — IPB Step 4
**Most Probable COA (MPCOA):**
**Most Dangerous COA (MDCOA):**

## 6. NAMED AREAS OF INTEREST

## 7. RECOMMENDATIONS

Begin analysis. Use only the data provided above.""",

    "scenario_defensive": "DEFENSIVE OPERATIONS: Identify defensible terrain, engagement areas, obstacle integration, key terrain to retain, and enemy avenues of approach.",
    "scenario_offensive": "OFFENSIVE OPERATIONS: Identify friendly avenues of approach, key terrain objectives, obstacles to breach/bypass, cover and concealment for movement.",
    "scenario_stability": "STABILITY OPERATIONS: Focus on civil considerations (ASCOPE), population centers, sensitive sites, critical infrastructure.",
    "scenario_reconnaissance": "RECONNAISSANCE: Identify observation positions, named areas of interest (NAIs), screen lines, information collection priorities.",
    "scenario_general": "COMPREHENSIVE IPB: Full OAKOC terrain analysis, threat evaluation, course of action development.",

    "followup": """You are a NATO Tactical Terrain Analysis Assistant.

{location_context}""" + _SEP + """
CONVERSATION HISTORY
""" + _SEP + """
{history_text}
""" + _SEP + """
CURRENT QUESTION
""" + _SEP + """
{question}

Answer based on the previous analysis. If new coordinates are needed, ask for them.""",
}


# ---------------------------------------------------------------------------
# Spec: label + which placeholders each prompt may use / must keep.
# Order here is the order the frontend shows them.
# ---------------------------------------------------------------------------

SPEC = {
    "ipb_analysis": {
        "label": "Main IPB analysis",
        "description": "The main prompt used whenever coordinates are given.",
        "allowed": {"history_section", "scenario_guidance", "doctrine_section",
                    "coord_str", "terrain_text", "military_section", "question"},
        # Retrieved / mined data (terrain, doctrine, military) plus the user's
        # question are locked in — the user may reword everything around them but
        # cannot drop the data itself.
        "required": {"terrain_text", "doctrine_section", "military_section", "question"},
    },
    "scenario_defensive": {
        "label": "Scenario: Defensive", "description": "Guidance injected for defensive queries.",
        "allowed": set(), "required": set(),
    },
    "scenario_offensive": {
        "label": "Scenario: Offensive", "description": "Guidance injected for offensive queries.",
        "allowed": set(), "required": set(),
    },
    "scenario_stability": {
        "label": "Scenario: Stability", "description": "Guidance injected for stability queries.",
        "allowed": set(), "required": set(),
    },
    "scenario_reconnaissance": {
        "label": "Scenario: Reconnaissance", "description": "Guidance injected for recon queries.",
        "allowed": set(), "required": set(),
    },
    "scenario_general": {
        "label": "Scenario: General", "description": "Guidance used when no scenario is detected.",
        "allowed": set(), "required": set(),
    },
    "followup": {
        "label": "Follow-up question",
        "description": "Used for follow-up questions about a previous analysis.",
        "allowed": {"location_context", "history_text", "question"},
        "required": {"question"},
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _placeholders_in(template: str) -> set:
    """Return the set of {name} placeholders used in a template string."""
    return {field for _, field, _, _ in Formatter().parse(template) if field}


def validate(key: str, template: str) -> list:
    """Return a list of human-readable errors ([] means valid)."""
    if key not in SPEC:
        return [f"Unknown prompt '{key}'."]
    if not isinstance(template, str) or not template.strip():
        return ["Prompt cannot be empty."]

    spec = SPEC[key]
    used = _placeholders_in(template)
    errors = []

    unknown = used - spec["allowed"]
    if unknown:
        errors.append("Not allowed here: " + ", ".join("{%s}" % u for u in sorted(unknown)))

    missing = spec["required"] - used
    if missing:
        errors.append("Must keep: " + ", ".join("{%s}" % m for m in sorted(missing)))

    return errors


# ---------------------------------------------------------------------------
# Load / get / save
# ---------------------------------------------------------------------------

_cache = None


def load() -> dict:
    """Load prompts from disk, falling back to defaults for anything missing/invalid."""
    global _cache
    prompts = dict(DEFAULTS)

    if PROMPTS_PATH.exists():
        try:
            data = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8")) or {}
            for key, value in data.items():
                if key not in DEFAULTS:
                    continue
                if validate(key, value):
                    logger.warning("Ignoring invalid prompt '%s' from prompts.yaml; using default.", key)
                else:
                    prompts[key] = value
        except Exception as e:
            logger.error("Could not read prompts.yaml (%s); using defaults.", e)

    _cache = prompts
    return prompts


def get(key: str) -> str:
    """Get the current template for a prompt (cached)."""
    if _cache is None:
        load()
    return _cache.get(key, DEFAULTS.get(key, ""))


def save(new_prompts: dict) -> dict:
    """
    Validate and persist edited prompts. Returns {"ok": bool, "errors": {key: [...]}}.
    On any validation error, nothing is written.
    """
    errors = {}
    for key, value in new_prompts.items():
        if key not in DEFAULTS:
            continue
        errs = validate(key, value)
        if errs:
            errors[key] = errs

    if errors:
        return {"ok": False, "errors": errors}

    merged = load()
    for key, value in new_prompts.items():
        if key in DEFAULTS:
            merged[key] = value

    try:
        PROMPTS_PATH.write_text(
            yaml.safe_dump(merged, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error("Could not write prompts.yaml: %s", e)
        return {"ok": False, "errors": {"_file": [f"Could not save file: {e}"]}}

    load()  # refresh cache so the running agent picks up changes immediately
    return {"ok": True, "errors": {}}


def api_payload() -> dict:
    """Everything the frontend editor needs."""
    load()
    items = []
    for key, spec in SPEC.items():
        items.append({
            "key": key,
            "label": spec["label"],
            "description": spec["description"],
            "required": sorted(spec["required"]),
            "allowed": sorted(spec["allowed"]),
            "value": _cache.get(key, DEFAULTS[key]),
            "default": DEFAULTS[key],
        })
    return {"prompts": items}
