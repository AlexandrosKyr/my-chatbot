import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from tools.terrain import fetch_terrain_data, get_last_result, _parse_radius_from_text
from tools.doctrine import retrieve_doctrine
from tools.military import lookup_military_power
from tools.coordinates import (
    parse_coordinates,
    convert_coordinates,
    encode_geohash,
    decode_geohash,
    encode_plus_code,
    decode_plus_code,
    encode_maidenhead,
    decode_maidenhead,
    convert_to_mils,
    convert_from_mils,
)
from tools.geometry import calculate_distance, calculate_bearing, analyze_aspect
from tools.files import parse_kml_file, parse_kml_string
import prompt_store

logger = logging.getLogger(__name__)

# Only genuinely optional tools go here — things the LLM may or may not need.
OPTIONAL_TOOLS = [
    # Coordinate conversion & datums
    convert_coordinates,
    encode_geohash,
    decode_geohash,
    encode_plus_code,
    decode_plus_code,
    encode_maidenhead,
    decode_maidenhead,
    convert_to_mils,
    convert_from_mils,
    # Geometry
    calculate_distance,
    calculate_bearing,
    analyze_aspect,
    # File parsing
    parse_kml_file,
    parse_kml_string,
]

MAX_TOOL_ITERATIONS = 4


class TacticalAgent:
    """
    Mandatory pipeline for every query that contains coordinates:
      1. Parse coordinates       (Python)
      2. Fetch terrain data      (Python)
      3. Retrieve doctrine       (Python)
      4. Look up military power  (Python)
      5. Single LLM call to synthesise the IPB analysis

    For queries without coordinates:
      - Follow-up on previous analysis, or
      - Ask the user for coordinates, or
      - Handle utility requests (coordinate conversion etc.) with optional tools
    """

    def __init__(self, llm):
        self.llm = llm
        self.llm_with_tools = llm.bind_tools(OPTIONAL_TOOLS)

        # Stored after each run for frontend display and follow-up reuse.
        self.last_terrain_summary = None
        self.last_terrain_data = None
        self.last_coords = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, question: str, conversation_history: list = None) -> tuple:
        """
        Returns (response_text, mode, data_availability).
        mode: 'terrain_analysis' | 'followup' | 'awaiting_coordinates' | 'utility'
        """
        coords = parse_coordinates(question)

        if coords:
            return self._run_mandatory_pipeline(question, coords, conversation_history)

        if conversation_history and self.last_terrain_data:
            return self._run_followup(question, conversation_history)

        if self._is_utility_request(question):
            return self._run_utility(question)

        return self._ask_for_coordinates(question), "awaiting_coordinates", _empty_availability()

    # ------------------------------------------------------------------
    # Mandatory pipeline — always runs when coordinates are present
    # ------------------------------------------------------------------

    def _run_mandatory_pipeline(self, question: str, coords: dict, conversation_history: list) -> tuple:
        radius_km = _parse_radius_from_text(question)

        # Step 1 — terrain
        terrain_text = fetch_terrain_data(coords["lat"], coords["lon"], radius_km)
        last = get_last_result()
        terrain_data = last["terrain_data"] or {}

        # Step 2 — doctrine
        doctrine_text = retrieve_doctrine(question, terrain_data)

        # Step 3 — military power (country derived from terrain reverse-geocode)
        country_code = terrain_data.get("address", {}).get("country_code", "")
        military_text = lookup_military_power(country_code)

        # Store for follow-up reuse and frontend display.
        self.last_terrain_data = terrain_data
        self.last_terrain_summary = last["summary"]
        self.last_coords = last["coords"]

        # Step 4 — single LLM synthesis call
        scenario_type = _detect_scenario_type(question)
        prompt = _build_ipb_prompt(
            question=question,
            terrain_text=terrain_text,
            doctrine_text=doctrine_text,
            military_text=military_text,
            coords=coords,
            scenario_type=scenario_type,
            conversation_history=conversation_history,
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])

        data_availability = {
            "coordinates_found": True,
            "terrain_data": "available" if terrain_data else "unavailable",
            "osm_data": "available" if terrain_data.get("osm_data_available", True) else "unavailable",
            "elevation_data": "available" if terrain_data.get("elevation") else "unavailable",
            "doctrine_documents": "available" if doctrine_text else "unavailable",
            "message": "" if doctrine_text else "No doctrine documents loaded",
        }

        return response.content, "terrain_analysis", data_availability

    # ------------------------------------------------------------------
    # Follow-up — no new coordinates, continuing previous analysis
    # ------------------------------------------------------------------

    def _run_followup(self, question: str, conversation_history: list) -> tuple:
        prompt = _build_followup_prompt(question, conversation_history, self.last_terrain_summary)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content, "followup", _empty_availability()

    # ------------------------------------------------------------------
    # Utility — coordinate conversion etc., LLM with optional tools
    # ------------------------------------------------------------------

    def _run_utility(self, question: str) -> tuple:
        messages = [
            SystemMessage(content="You are a military coordinate and mapping assistant. Use the available tools when needed."),
            HumanMessage(content=question),
        ]
        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                return response.content, "utility", _empty_availability()
            tool_map = {t.name: t for t in OPTIONAL_TOOLS}
            for call in response.tool_calls:
                tool_fn = tool_map.get(call["name"])
                result = str(tool_fn.invoke(call["args"])) if tool_fn else f"Unknown tool: {call['name']}"
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        return response.content, "utility", _empty_availability()

    # ------------------------------------------------------------------
    # No coordinates — ask the user
    # ------------------------------------------------------------------

    def _ask_for_coordinates(self, question: str) -> str:
        prompt = f"""You are a NATO Terrain Analysis Assistant.

The user sent a message but no coordinates were found.

User message: {question}

Respond by:
1. Acknowledging their request briefly.
2. Explaining that you need coordinates to run terrain analysis.
3. Asking them to provide coordinates in any format (decimal, MGRS, DMS, UTM).
4. Give a short example: e.g. "Analyze 54.6872, 25.2797 for defensive positions"

Keep it concise."""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _is_utility_request(self, question: str) -> bool:
        keywords = [
            # Coordinate conversion
            "convert", "mgrs", "utm", "ed50", "wgs84", "etrs", "εγσα",
            "datum", "coordinate system", "grid reference",
            "geohash", "plus code", "open location code",
            # Geometry
            "distance", "bearing", "azimuth", "how far", "how long",
            "aspect", "slope direction", "which way",
            # Files
            "kml", "kmz", "waypoint", "import",
        ]
        return any(kw in question.lower() for kw in keywords)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _detect_scenario_type(question: str) -> str:
    q = question.lower()
    checks = [
        ("reconnaissance", ["reconnaissance", "recon", "screen", "surveil", "scout", "isr"]),
        ("stability",      ["stability", "coin", "counterinsurgency", "peacekeeping", "humanitarian"]),
        ("defensive",      ["defend", "defense", "defensive", "hold", "delay", "retrograde", "blocking"]),
        ("offensive",      ["attack", "assault", "offensive", "seize", "capture", "advance", "breach"]),
    ]
    for scenario, keywords in checks:
        if any(kw in q for kw in keywords):
            logger.info(f"Scenario: {scenario}")
            return scenario
    return "general"


def _get_scenario_guidance(scenario_type: str) -> str:
    key = f"scenario_{scenario_type}"
    if key not in prompt_store.DEFAULTS:
        key = "scenario_general"
    return prompt_store.get(key)


def _build_ipb_prompt(question: str, terrain_text: str, doctrine_text: str,
                      military_text: str, coords: dict, scenario_type: str,
                      conversation_history: list = None) -> str:

    coord_str = f"{coords['lat']:.6f}°N, {coords['lon']:.6f}°E"
    scenario_guidance = _get_scenario_guidance(scenario_type)

    history_section = ""
    if conversation_history:
        history_section = "PRIOR CONTEXT:\n"
        for msg in conversation_history[-4:]:
            role = "U" if msg.get("role") == "user" else "A"
            history_section += f"{role}: {msg.get('text', '')[:300]}\n"
        history_section += "\n"

    doctrine_section = (
        f"RETRIEVED DOCTRINE (cite ONLY from this):\n{doctrine_text}"
        if doctrine_text else
        "NO DOCTRINE DOCUMENTS LOADED — base analysis on standard IPB methodology."
    )

    military_section = (
        f"MILITARY POWER DATA:\n{military_text}"
        if military_text else
        "MILITARY POWER DATA: Not available for this location."
    )

    fields = dict(
        history_section=history_section,
        scenario_guidance=scenario_guidance,
        doctrine_section=doctrine_section,
        coord_str=coord_str,
        terrain_text=terrain_text,
        military_section=military_section,
        question=question,
    )
    template = prompt_store.get("ipb_analysis")
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError) as e:
        logger.error("Bad ipb_analysis prompt (%s); using built-in default.", e)
        return prompt_store.DEFAULTS["ipb_analysis"].format(**fields)


def _build_followup_prompt(question: str, conversation_history: list,
                           last_terrain_summary: dict = None) -> str:
    history_text = ""
    for msg in conversation_history[-8:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_text += f"\n{role}: {msg.get('text', '')[:800]}\n"

    location_context = ""
    if last_terrain_summary:
        place = last_terrain_summary.get("location", "")
        coords = last_terrain_summary.get("coordinates", {})
        if place and coords:
            location_context = f"Previous analysis: {place} ({coords.get('lat')}, {coords.get('lon')})\n\n"

    fields = dict(
        location_context=location_context,
        history_text=history_text,
        question=question,
    )
    template = prompt_store.get("followup")
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError) as e:
        logger.error("Bad followup prompt (%s); using built-in default.", e)
        return prompt_store.DEFAULTS["followup"].format(**fields)


def _empty_availability() -> dict:
    return {
        "coordinates_found": False,
        "terrain_data": "unavailable",
        "osm_data": "unavailable",
        "elevation_data": "unavailable",
        "doctrine_documents": "unavailable",
        "message": "",
    }
