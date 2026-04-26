"""
Geometric calculations: distance, bearing, and aspect analysis.

All functions use pyproj.Geod for geodesic accuracy on the WGS84 ellipsoid.
Aspect analysis works from the directional slope data returned by the terrain fetcher.

Optional LLM-callable tools:
  calculate_distance() — geodesic distance between two points
  calculate_bearing()  — forward bearing from point A to point B
  analyze_aspect()     — which direction a slope faces, from terrain data
"""

import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float, unit: str = "km") -> str:
    """Calculate the geodesic distance between two WGS84 points.

    unit: 'km' (default), 'm', or 'nm' (nautical miles)

    Example: calculate_distance(37.9838, 23.7275, 40.6401, 22.9444)
    """
    try:
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        _, _, distance_m = geod.inv(lon1, lat1, lon2, lat2)

        unit = unit.lower()
        if unit == "km":
            return f"{distance_m / 1000:.3f} km"
        elif unit == "m":
            return f"{distance_m:.1f} m"
        elif unit == "nm":
            return f"{distance_m / 1852:.3f} nm"
        else:
            return f"Unknown unit '{unit}'. Use 'km', 'm', or 'nm'."

    except ImportError:
        return "pyproj not installed. Run: uv sync"
    except Exception as e:
        return f"Distance calculation error: {e}"


@tool
def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Calculate the forward bearing (azimuth) from point A to point B on WGS84.

    Returns bearing in degrees from north (0-360) and the back-azimuth.

    Example: calculate_bearing(37.9838, 23.7275, 40.6401, 22.9444)
    """
    try:
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        fwd_az, back_az, _ = geod.inv(lon1, lat1, lon2, lat2)

        # Normalise to 0–360
        fwd_az  = fwd_az  % 360
        back_az = back_az % 360

        return (
            f"Forward bearing:  {fwd_az:.2f}°  ({_degrees_to_cardinal(fwd_az)})\n"
            f"Back bearing:     {back_az:.2f}°  ({_degrees_to_cardinal(back_az)})"
        )

    except ImportError:
        return "pyproj not installed. Run: uv sync"
    except Exception as e:
        return f"Bearing calculation error: {e}"


@tool
def analyze_aspect(lat: float, lon: float) -> str:
    """Determine the aspect (downhill-facing direction) of terrain at a location.

    Fetches terrain slope data and identifies which direction the slope faces.
    Useful for assessing solar exposure, drainage, and defensive positioning.

    Example: analyze_aspect(37.9838, 23.7275)
    """
    try:
        # Import here to avoid circular imports — terrain tool is a sibling module.
        from tools.terrain import fetch_terrain_data, get_last_result

        fetch_terrain_data(lat, lon, radius_km=1.0)
        terrain_data = get_last_result().get("terrain_data") or {}
        slope_data = terrain_data.get("slope_analysis", {})
        direction_slopes = slope_data.get("direction_slopes", {})

        if not direction_slopes:
            return "No slope data available for this location."

        return _compute_aspect(direction_slopes)

    except Exception as e:
        return f"Aspect analysis error: {e}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_aspect(direction_slopes: dict) -> str:
    """Find the dominant downhill direction from directional slope data."""
    downhill = {
        direction: info["slope_percent"]
        for direction, info in direction_slopes.items()
        if info.get("direction") == "downhill" and info.get("slope_percent", 0) > 0
    }

    if not downhill:
        avg = sum(i.get("slope_percent", 0) for i in direction_slopes.values()) / max(len(direction_slopes), 1)
        if avg < 1.0:
            return "Flat terrain — no dominant aspect (slope < 1%)."
        # All directions uphill means we're on a peak
        return "Summit or ridge — terrain rises in all sampled directions."

    aspect_dir = max(downhill, key=lambda d: downhill[d])
    slope_pct  = downhill[aspect_dir]

    # Tactical interpretation
    if slope_pct < 5:
        tactical = "Gentle slope — minimal impact on movement or observation."
    elif slope_pct < 15:
        tactical = "Moderate slope — restricts wheeled vehicles off-road."
    elif slope_pct < 30:
        tactical = "Steep slope — restricts all but tracked vehicles."
    else:
        tactical = "Very steep slope — effectively impassable to vehicles."

    lines = [
        f"Aspect:   {aspect_dir} ({slope_pct:.1f}% grade downhill)",
        f"Tactical: {tactical}",
        "",
        "All directional slopes:",
    ]
    for direction, info in direction_slopes.items():
        marker = " ◀ dominant" if direction == aspect_dir else ""
        lines.append(f"  {direction:3s}: {info.get('slope_percent', 0):5.1f}%  {info.get('direction', '')}{marker}")

    return "\n".join(lines)


def _degrees_to_cardinal(degrees: float) -> str:
    """Convert a bearing in degrees to a cardinal/intercardinal label."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]
