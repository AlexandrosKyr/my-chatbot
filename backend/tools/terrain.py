import json
import logging
import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terrain_data_fetcher import TerrainDataFetcher

logger = logging.getLogger(__name__)

_fetcher: TerrainDataFetcher = None

# Stores the last fetch result so app.py can read the summary for the frontend.
_last_result = {"terrain_data": None, "coords": None, "summary": None}


def initialize(fetcher: TerrainDataFetcher) -> None:
    """Set the TerrainDataFetcher instance. Call once at startup."""
    global _fetcher
    _fetcher = fetcher


def get_last_result() -> dict:
    """Return the terrain data and summary from the most recent fetch."""
    return _last_result


def _parse_radius_from_text(text: str) -> float:
    """Extract radius in km from user message, defaulting to 5km."""
    patterns = [
        r'(\d+)\s*km\s*radius',
        r'radius\s*(?:of\s*)?(\d+)\s*km',
        r'within\s*(\d+)\s*km',
        r'(\d+)\s*kilometer',
        r'for\s*(\d+)\s*km',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return float(max(1, min(50, int(match.group(1)))))
    return 5.0


def fetch_terrain_data(lat: float, lon: float, radius_km: float = 5.0) -> str:
    """Fetch terrain intelligence and return a formatted report string.
    Also stores raw data in _last_result for the agent to read."""
    if _fetcher is None:
        return "Terrain fetcher not initialized."

    try:
        data = _fetcher.fetch_terrain_data(lat, lon, radius_km=radius_km)
        coords = {"lat": lat, "lon": lon}

        _last_result["terrain_data"] = data
        _last_result["coords"] = coords
        _last_result["summary"] = _create_terrain_summary(data, coords, radius_km)

        return _format_terrain_intel(data, coords, radius_km)

    except Exception as e:
        logger.error(f"Terrain fetch failed: {e}")
        return f"Terrain data unavailable: {e}"


# ---------------------------------------------------------------------------
# Formatting helpers — logic preserved exactly from services.py
# ---------------------------------------------------------------------------

def _create_terrain_summary(terrain_data: dict, coords: dict, radius_km: float) -> dict:
    """Build the structured summary dict sent to the frontend."""
    roads = terrain_data.get("roads", [])
    road_type_counts = dict(Counter(r["type"] for r in roads))
    named_roads = sorted(set(
        r["name"] for r in roads
        if r.get("name") and r["name"] not in ["Unnamed road", ""]
    ))[:10]

    waterway_summary = terrain_data.get("waterway_summary", {})
    movement = terrain_data.get("movement_times", {})
    slope = terrain_data.get("slope_analysis", {})

    return {
        "coordinates": {"lat": round(coords["lat"], 6), "lon": round(coords["lon"], 6)},
        "location": terrain_data.get("place_name", "Unknown location"),
        "radius_km": radius_km,
        "elevation": terrain_data.get("elevation"),
        "terrain": {
            "avg_slope_percent": slope.get("average_slope_percent", 0),
            "max_slope_percent": slope.get("max_slope_percent", 0),
            "mobility": slope.get("mobility", "unknown"),
        },
        "infrastructure": {
            "roads": {
                "total_segments": len(roads),
                "by_type": road_type_counts,
                "named_roads": named_roads,
            },
            "waterways": {
                "total_segments": waterway_summary.get("total_segments", len(terrain_data.get("waterways", []))),
                "segments_by_type": waterway_summary.get("segments_by_type", {}),
                "distinct_named": waterway_summary.get("distinct_named", {}),
                "total_distinct_named": waterway_summary.get("total_distinct_named", 0),
            },
            "buildings": len(terrain_data.get("buildings", [])),
            "forests": terrain_data.get("forest_summary", {"total_segments": len(terrain_data.get("forests", []))}),
            "crossings": terrain_data.get("crossing_summary", {"total_segments": len(terrain_data.get("crossings", []))}),
            "railways": terrain_data.get("railway_summary", {"total_segments": len(terrain_data.get("railways", []))}),
        },
        "tactical": {
            "power_lines": len(terrain_data.get("power_lines", [])),
            "cell_towers": len(terrain_data.get("cell_towers", [])),
            "fuel_stations": len(terrain_data.get("fuel_stations", [])),
            "medical_facilities": terrain_data.get("medical_summary", {}).get(
                "total_distinct_named", len(terrain_data.get("medical_facilities", []))
            ),
            "medical_summary": terrain_data.get("medical_summary", {}),
            "schools": terrain_data.get("school_summary", {}).get(
                "total_distinct_named", len(terrain_data.get("schools", []))
            ),
            "school_summary": terrain_data.get("school_summary", {}),
            "helipads": len(terrain_data.get("helipads", [])),
        },
        "movement": {
            "summary": movement.get("summary", "N/A"),
            "unit_estimates": movement.get("unit_estimates", {}),
        },
        "weather": terrain_data.get("weather", {}).get("weekly_summary", {}),
    }


def _format_terrain_intel(terrain_data: dict, coords: dict, radius_km: float) -> str:
    """Format terrain data as a tactical intelligence report for the LLM."""
    parts = []
    osm_available = terrain_data.get("osm_data_available", True)
    place_name = terrain_data.get("place_name", "Unknown")

    parts.append("=" * 60)
    parts.append("TERRAIN INTELLIGENCE REPORT")
    parts.append(f"Analysis Radius: {radius_km}km")
    parts.append("=" * 60)
    parts.append(f"\nLOCATION: {place_name}")
    parts.append(f"COORDINATES: {coords['lat']:.6f}°N, {coords['lon']:.6f}°E")

    if not osm_available:
        parts += [
            "",
            "INFRASTRUCTURE DATA UNAVAILABLE (API timeout)",
            f"NOTE: Use world knowledge of '{place_name}' to infer terrain characteristics.",
            "",
        ]

    elevation = terrain_data.get("elevation")
    if elevation:
        parts.append(f"ELEVATION: {elevation}m ASL")

    slope_data = terrain_data.get("slope_analysis", {})
    if slope_data:
        avg = slope_data.get("average_slope_percent", 0)
        mx = slope_data.get("max_slope_percent", 0)
        mob = slope_data.get("mobility", {})
        mob_text = mob.get("assessment", "unknown") if isinstance(mob, dict) else str(mob)
        parts.append(f"\nSLOPE ANALYSIS:")
        parts.append(f"  Average: {avg}% | Maximum: {mx}%")
        parts.append(f"  Mobility Assessment: {mob_text.upper()}")
        for direction, info in slope_data.get("direction_slopes", {}).items():
            parts.append(f"    {direction.upper()}: {info.get('slope_percent', 0)}% ({info.get('direction', 'flat')})")

    los_data = terrain_data.get("line_of_sight", {})
    if los_data:
        parts.append(f"\nOBSERVATION & FIELDS OF FIRE:")
        parts.append(f"  Dominant Position: {'YES' if los_data.get('is_high_ground') else 'NO'}")
        parts.append(f"  Visibility: {los_data.get('overall_visibility', 'unknown')}")
        if los_data.get("obstructed_directions"):
            parts.append(f"  Obstructed Directions: {', '.join(los_data['obstructed_directions'])}")

    crossings = terrain_data.get("crossings", [])
    crossing_summary = terrain_data.get("crossing_summary", {})
    if crossings:
        distinct = crossing_summary.get("distinct_named", {})
        total = crossing_summary.get("total_distinct_named", 0)
        parts.append(f"\nCROSSING POINTS ({total} distinct, {len(crossings)} segments):")
        for ctype in ["bridge", "ford", "tunnel", "dam"]:
            info = distinct.get(ctype, {})
            count = info.get("count", 0)
            names = info.get("names", [])
            if count:
                parts.append(f"  {ctype.title()}s ({count}): {', '.join(names[:5])}")
            elif crossing_summary.get("segments_by_type", {}).get(ctype, 0):
                parts.append(f"  {ctype.title()}s: {crossing_summary['segments_by_type'][ctype]} segments (unnamed)")

    roads = terrain_data.get("roads", [])
    if roads:
        road_type_counts = Counter(r["type"] for r in roads)
        parts.append(f"\nAVENUES OF APPROACH ({len(roads)} road segments in {radius_km}km radius):")
        for road_type, count in road_type_counts.most_common():
            parts.append(f"  {road_type.replace('_', ' ').title():20s}: {count:3d} segments")
        named_roads = sorted(set(
            r["name"] for r in roads
            if r.get("name") and r["name"] not in ["Unnamed road", ""]
        ))
        if named_roads:
            parts.append(f"\n  Named Routes ({len(named_roads)} unique):")
            for name in named_roads[:8]:
                parts.append(f"    - {name}")

    waterways = terrain_data.get("waterways", [])
    waterway_summary = terrain_data.get("waterway_summary", {})
    railways = terrain_data.get("railways", [])
    if waterways or railways:
        parts.append(f"\nOBSTACLES:")
        if waterways:
            distinct = waterway_summary.get("distinct_named", {})
            wparts = []
            for wtype in ["river", "stream", "canal"]:
                info = distinct.get(wtype, {})
                if info.get("count", 0):
                    entry = f"{info['count']} {wtype}s"
                    if info.get("names"):
                        entry += f" ({', '.join(info['names'][:5])})"
                    wparts.append(entry)
            if wparts:
                parts.append(f"  Water: {', '.join(wparts)}")
            else:
                seg = waterway_summary.get("segments_by_type", {})
                parts.append(f"  Water: {len(waterways)} segments ({', '.join(f'{v} {k}' for k, v in seg.items())})")
        if railways:
            railway_summary = terrain_data.get("railway_summary", {})
            named_lines = railway_summary.get("distinct_named", [])
            if named_lines:
                parts.append(f"  Railways: {len(named_lines)} lines ({', '.join(named_lines[:5])}), {railway_summary.get('total_segments', len(railways))} segments")
            else:
                parts.append(f"  Railways: {len(railways)} segments (linear obstacles)")

    forests = terrain_data.get("forests", [])
    forest_summary = terrain_data.get("forest_summary", {})
    buildings = terrain_data.get("buildings", [])
    parts.append(f"\nCOVER & CONCEALMENT:")
    if osm_available:
        named_forests = forest_summary.get("distinct_named", [])
        unnamed_segs = forest_summary.get("unnamed_segments", len(forests))
        if named_forests:
            parts.append(f"  Forest Areas: {len(named_forests)} named ({', '.join(named_forests[:5])}), {unnamed_segs} unnamed segments")
        else:
            parts.append(f"  Forest Areas: {forest_summary.get('total_segments', len(forests))} segments (none named)")
        parts.append(f"  Buildings/Structures: {len(buildings)}")
        if len(buildings) > 200:
            parts.append("  URBAN TERRAIN - expect channelized movement")
    else:
        parts.append(f"  DATA UNAVAILABLE - Infer from location: '{place_name}'")

    power_lines = terrain_data.get("power_lines", [])
    cell_towers = terrain_data.get("cell_towers", [])
    fuel_stations = terrain_data.get("fuel_stations", [])
    medical = terrain_data.get("medical_facilities", [])
    medical_summary = terrain_data.get("medical_summary", {})
    schools = terrain_data.get("schools", [])
    school_summary = terrain_data.get("school_summary", {})
    helipads = terrain_data.get("helipads", [])

    if any([power_lines, cell_towers, fuel_stations, medical, schools, helipads]):
        parts.append(f"\nTACTICAL INFRASTRUCTURE:")
        if power_lines:
            parts.append(f"  Power Lines: {len(power_lines)} (AVIATION HAZARD)")
        if cell_towers:
            parts.append(f"  Comm Towers: {len(cell_towers)}")
        if fuel_stations:
            parts.append(f"  Fuel Points: {len(fuel_stations)} (resupply potential)")
        if helipads:
            parts.append(f"  Helipads: {len(helipads)} (confirmed LZ)")
        if medical or schools:
            parts.append("  SENSITIVE SITES (ROE):")
            for mtype in ["hospital", "clinic"]:
                info = medical_summary.get("distinct_named", {}).get(mtype, {})
                if info.get("count", 0):
                    parts.append(f"    {mtype.title()}s ({info['count']}): {', '.join(info['names'][:8])}")
            unnamed_med = medical_summary.get("unnamed_segments", 0)
            if unnamed_med:
                parts.append(f"    + {unnamed_med} unnamed medical facilities")
            for stype in ["school", "university"]:
                info = school_summary.get("distinct_named", {}).get(stype, {})
                if info.get("count", 0):
                    parts.append(f"    {stype.title()}s ({info['count']}): {', '.join(info['names'][:8])}")
            unnamed_sch = school_summary.get("unnamed_segments", 0)
            if unnamed_sch:
                parts.append(f"    + {unnamed_sch} unnamed schools")

    movement = terrain_data.get("movement_times", {})
    if movement:
        parts.append(f"\nMOVEMENT TIME ESTIMATES ({radius_km}km radius):")
        parts.append(f"  Assessment: {movement.get('summary', 'N/A')}")
        for unit_type, est in movement.get("unit_estimates", {}).items():
            parts.append(f"  {est.get('description', unit_type)}: {int(est.get('time_to_radius_minutes', 0))} min")

    weather = terrain_data.get("weather", {})
    if weather and weather.get("weekly_summary"):
        s = weather["weekly_summary"]
        parts.append(f"\nWEATHER (Past 7 Days):")
        if s.get("avg_temp_c") is not None:
            parts.append(f"  Temperature: avg {s['avg_temp_c']}°C (range: {s.get('avg_temp_min_c', 'N/A')}°C to {s.get('avg_temp_max_c', 'N/A')}°C)")
        precip = s.get("total_precipitation_mm", 0)
        rain = s.get("total_rain_mm", 0)
        snow = s.get("total_snow_cm", 0)
        rainy_days = s.get("rainy_days", 0)
        snowy_days = s.get("snowy_days", 0)
        if precip > 0 or rain > 0 or snow > 0:
            parts.append(f"  Precipitation: {precip}mm total ({rainy_days} rainy days)")
            if snow > 0:
                parts.append(f"  Snowfall: {snow}cm ({snowy_days} snowy days)")
        else:
            parts.append("  Precipitation: Dry conditions")
        if s.get("avg_wind_speed_max_kmh") is not None:
            parts.append(f"  Wind: avg max {s['avg_wind_speed_max_kmh']} km/h, gusts up to {s.get('max_wind_gust_kmh', 'N/A')} km/h")
        if s.get("avg_sunshine_hours") is not None:
            parts.append(f"  Sunshine: avg {s['avg_sunshine_hours']} hours/day")
        conditions = s.get("predominant_conditions", [])
        if conditions:
            parts.append(f"  Conditions: {', '.join(c.replace('_', ' ') for c in conditions)}")
        parts.append("  TACTICAL IMPACT:")
        if precip > 20 or rainy_days >= 3:
            parts.append("    - Wet conditions: reduced off-road mobility, potential flooding")
        if snow > 5:
            parts.append("    - Snow cover: affects concealment, tracked vehicle advantage")
        if s.get("max_wind_gust_kmh", 0) and s["max_wind_gust_kmh"] > 50:
            parts.append("    - High winds: affects aviation ops, smoke deployment")
        if s.get("avg_sunshine_hours", 0) and s["avg_sunshine_hours"] < 4:
            parts.append("    - Low visibility conditions: reduced observation range")
        if precip == 0 and s.get("avg_sunshine_hours", 0) and s["avg_sunshine_hours"] > 8:
            parts.append("    - Clear/dry conditions: good visibility, firm ground")

    # Military power data is injected by get_military_power tool — not duplicated here.

    parts.append("\n" + "=" * 60)
    return "\n".join(parts)
