"""
Coordinate parsing and conversion tools.

Mandatory pipeline:
  parse_coordinates() — called by Python, not the LLM

Optional LLM-callable tools:
  convert_coordinates() — coordinate system / datum transforms
  encode_geohash()      — WGS84 → Geohash
  decode_geohash()      — Geohash → WGS84
  encode_plus_code()    — WGS84 → Plus Code (Open Location Code)
  decode_plus_code()    — Plus Code → WGS84
"""

import logging
import sys
import os

from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordinate_parser import CoordinateParser

logger = logging.getLogger(__name__)

_parser = CoordinateParser()

# Maps human-friendly system names to EPSG codes understood by pyproj.
# Add new systems here as needed.
COORDINATE_SYSTEMS = {
    "WGS84":    "EPSG:4326",
    "ED50":     "EPSG:4230",
    "ED78":     "EPSG:4231",
    "ETRS89":   "EPSG:4258",
    "EGSA87":   "EPSG:2100",   # Greek Grid (projected, metres)
    "ΕΓΣΑ87":   "EPSG:2100",
    "NAD83":    "EPSG:4269",
    "NAD27":    "EPSG:4267",
    "OSGB36":   "EPSG:4277",
    "GRS80":    "EPSG:4019",
    "TOKYO":    "EPSG:4301",
    "SAD69":    "EPSG:4618",
    "SIRGAS":   "EPSG:4674",
    "SK42":     "EPSG:4284",
    "ITRF":     "EPSG:7912",   # ITRF2014
}


# ---------------------------------------------------------------------------
# Mandatory pipeline function (called by Python, not the LLM)
# ---------------------------------------------------------------------------

def parse_coordinates(text: str) -> dict | None:
    """Extract coordinates from text. Returns {'lat': ..., 'lon': ...} or None."""
    result = _parser.parse(text)
    if result:
        logger.info(f"Parsed coordinates: {result}")
    return result


# ---------------------------------------------------------------------------
# Optional LLM-callable tools
# ---------------------------------------------------------------------------

@tool
def convert_coordinates(value: str, from_system: str, to_system: str) -> str:
    """Convert coordinates between systems or datums.

    Supported systems: WGS84, ED50, ED78, ETRS89, EGSA87, NAD83, NAD27,
                       OSGB36, GRS80, TOKYO, SAD69, SIRGAS, SK42, ITRF,
                       MGRS, UTM (auto-zone)

    Examples:
      convert_coordinates("37.9838, 23.7275", "WGS84", "MGRS")
      convert_coordinates("34VCJ5488432445", "MGRS", "WGS84")
      convert_coordinates("37.9838, 23.7275", "WGS84", "EGSA87")
      convert_coordinates("37.9838, 23.7275", "WGS84", "UTM")
    """
    from_upper = from_system.upper().replace(" ", "")
    to_upper   = to_system.upper().replace(" ", "")

    try:
        # --- MGRS conversions ---
        if from_upper == "MGRS" or to_upper == "MGRS":
            return _convert_mgrs(value, from_upper, to_upper)

        # --- UTM auto-zone ---
        if to_upper == "UTM":
            return _to_utm(value, from_upper)

        # --- Datum / projected CRS transform via pyproj ---
        return _transform_crs(value, from_upper, to_upper)

    except Exception as e:
        return f"Conversion error: {e}"


@tool
def encode_geohash(lat: float, lon: float, precision: int = 9) -> str:
    """Encode WGS84 coordinates to a Geohash string.
    Precision controls length (1=coarse, 12=very fine). Default 9 (~2.4m).
    Example: encode_geohash(37.9838, 23.7275)"""
    try:
        import pygeohash as gh
        result = gh.encode(lat, lon, precision=precision)
        return f"Geohash: {result}  (precision={precision}, ~{_geohash_accuracy(precision)})"
    except ImportError:
        return "pygeohash not installed. Run: uv add pygeohash"


@tool
def decode_geohash(geohash: str) -> str:
    """Decode a Geohash string to WGS84 lat/lon with error bounds.
    Example: decode_geohash('swbb1estv')"""
    try:
        import pygeohash as gh
        lat, lon, lat_err, lon_err = gh.decode_exactly(geohash)
        return (
            f"Lat: {lat:.7f}  Lon: {lon:.7f}\n"
            f"Error bounds: ±{lat_err:.7f}° lat, ±{lon_err:.7f}° lon"
        )
    except ImportError:
        return "pygeohash not installed. Run: uv add pygeohash"
    except Exception as e:
        return f"Decode error: {e}"


@tool
def encode_plus_code(lat: float, lon: float, code_length: int = 10) -> str:
    """Encode WGS84 coordinates to a Plus Code (Open Location Code).
    code_length 10 gives ~14x14m accuracy. 11 gives ~3x3m.
    Example: encode_plus_code(37.9838, 23.7275)"""
    try:
        import openlocationcode as olc
        code = olc.encode(lat, lon, codeLength=code_length)
        return f"Plus Code: {code}"
    except ImportError:
        return "openlocationcode not installed. Run: uv add openlocationcode"


@tool
def decode_plus_code(plus_code: str) -> str:
    """Decode a Plus Code (Open Location Code) to WGS84 lat/lon bounds.
    Example: decode_plus_code('8G95WMJ4+QQ')"""
    try:
        import openlocationcode as olc
        if not olc.isValid(plus_code):
            return f"'{plus_code}' is not a valid Plus Code."
        area = olc.decode(plus_code)
        return (
            f"Center: {area.latitudeCenter:.7f}, {area.longitudeCenter:.7f}\n"
            f"Bounds: {area.latitudeLo:.7f}–{area.latitudeHi:.7f} lat, "
            f"{area.longitudeLo:.7f}–{area.longitudeHi:.7f} lon"
        )
    except ImportError:
        return "openlocationcode not installed. Run: uv add openlocationcode"
    except Exception as e:
        return f"Decode error: {e}"


# ---------------------------------------------------------------------------
# Internal helpers — not exposed as tools
# ---------------------------------------------------------------------------

def _parse_latlon(value: str) -> tuple[float, float]:
    """Parse 'lat, lon' or 'lat lon' string into floats."""
    parts = value.replace(",", " ").split()
    if len(parts) < 2:
        raise ValueError(f"Expected 'lat lon', got: '{value}'")
    return float(parts[0]), float(parts[1])


def _convert_mgrs(value: str, from_upper: str, to_upper: str) -> str:
    import mgrs as mgrs_lib
    m = mgrs_lib.MGRS()
    if from_upper == "MGRS":
        lat, lon = m.toLatLon(value.strip().encode())
        if to_upper in ("WGS84", "DECIMAL", "LATLON"):
            return f"{lat:.7f}, {lon:.7f}"
        # Chain into another transform
        return _transform_crs(f"{lat}, {lon}", "WGS84", to_upper)
    else:
        lat, lon = _parse_latlon(value)
        # Convert to WGS84 first if needed
        if from_upper not in ("WGS84", "DECIMAL", "LATLON"):
            wgs84 = _transform_crs(value, from_upper, "WGS84")
            lat, lon = _parse_latlon(wgs84)
        return m.toMGRS(lat, lon).decode()


def _to_utm(value: str, from_upper: str) -> str:
    from pyproj import Transformer, CRS
    lat, lon = _parse_latlon(value)
    if from_upper not in ("WGS84", "DECIMAL", "LATLON"):
        wgs84 = _transform_crs(value, from_upper, "WGS84")
        lat, lon = _parse_latlon(wgs84)
    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    utm_crs = CRS.from_dict({"proj": "utm", "zone": zone, "south": hemisphere == "south"})
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return f"Zone {zone}{('N' if lat >= 0 else 'S')}  Easting: {easting:.1f}  Northing: {northing:.1f}"


def _transform_crs(value: str, from_upper: str, to_upper: str) -> str:
    from pyproj import Transformer
    src = COORDINATE_SYSTEMS.get(from_upper)
    dst = COORDINATE_SYSTEMS.get(to_upper)
    if not src:
        return f"Unknown source system '{from_upper}'. Supported: {', '.join(COORDINATE_SYSTEMS)}"
    if not dst:
        return f"Unknown target system '{to_upper}'. Supported: {', '.join(COORDINATE_SYSTEMS)}"
    lat, lon = _parse_latlon(value)
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return f"{y:.7f}, {x:.7f}"


def _geohash_accuracy(precision: int) -> str:
    # Approximate cell sizes per precision level
    sizes = {1: "±2500km", 2: "±630km", 3: "±78km", 4: "±20km", 5: "±2.4km",
             6: "±610m", 7: "±76m", 8: "±19m", 9: "±2.4m", 10: "±60cm",
             11: "±7.4cm", 12: "±1.9cm"}
    return sizes.get(precision, "unknown")
