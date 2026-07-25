"""
Coordinate parsing and conversion tools.

Mandatory pipeline:
  parse_coordinates()        — called by Python, not the LLM

Optional LLM-callable tools:
  convert_coordinates()      — coordinate system / datum transforms
  encode_geohash()           — WGS84 → Geohash
  decode_geohash()           — Geohash → WGS84
  encode_plus_code()         — WGS84 → Plus Code (Open Location Code)
  decode_plus_code()         — Plus Code → WGS84
  encode_maidenhead()        — WGS84 → Maidenhead (QTH) locator
  decode_maidenhead()        — Maidenhead locator → WGS84
  convert_to_mils()          — decimal degrees → NATO mils (6400 in a circle)
  encode_microdegrees()      — WGS84 → integer microdegrees (degrees × 10⁶)
  decode_microdegrees()      — integer microdegrees → WGS84
  encode_dms_milliseconds()  — WGS84 → Degrees, Minutes, arc-Milliseconds
  encode_gars()              — WGS84 → GARS 30-minute cell + 15-minute quadrant
  decode_gars()              — GARS code → WGS84 centre point
  encode_georef()            — WGS84 → GEOREF (World Geographic Reference System)
  decode_georef()            — GEOREF code → WGS84 SW-corner of cell
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
# To add a new system: find its EPSG code at epsg.io, add one line here.
COORDINATE_SYSTEMS = {
    # Global
    "WGS84":        "EPSG:4326",
    "GRS80":        "EPSG:4019",

    # European
    "ED50":         "EPSG:4230",
    "ED78":         "EPSG:4231",
    "ETRS89":       "EPSG:4258",
    "EGSA87":       "EPSG:2100",    # Greek Grid (projected, metres)
    "ΕΓΣΑ87":       "EPSG:2100",
    "OSGB36":       "EPSG:4277",    # UK Ordnance Survey
    "CH1903":       "EPSG:21781",   # Swiss LV03
    "LV03":         "EPSG:21781",
    "CH1903+":      "EPSG:2056",    # Swiss LV95
    "LV95":         "EPSG:2056",
    "RT90":         "EPSG:3021",    # Swedish RT90 2.5 gon V

    # Russian / Soviet
    "SK42":         "EPSG:4284",    # Pulkovo 1942
    "SK-42":        "EPSG:4284",
    "SK95":         "EPSG:4200",    # Pulkovo 1995
    "SK-95":        "EPSG:4200",
    "PZ90":         "EPSG:4740",    # GLONASS reference frame
    "PZ-90":        "EPSG:4740",

    # North American
    "NAD83":        "EPSG:4269",
    "NAD27":        "EPSG:4267",

    # South American
    "SAD69":        "EPSG:4618",
    "SIRGAS":       "EPSG:4674",

    # Asian / Pacific
    "TOKYO":        "EPSG:4301",

    # Middle East / Africa
    "EGYPT1907":    "EPSG:4229",    # Egypt 1907 geographic
    "EGD1907":      "EPSG:4229",
    "ETM1975":      "EPSG:22992",   # Egyptian Transverse Mercator 1975 (Red Belt)
    "ELD79":        "EPSG:4159",    # Libyan datum (ELD79 — closest to LGD 1954)
    "LGD1954":      "EPSG:4159",
    "BALKAN1970":   "EPSG:3844",    # Pulkovo 1942(58) / Stereo70 — Balkan system

    # ITRF frames
    "ITRF":         "EPSG:7912",    # ITRF2014 (default)
    "ITRF2014":     "EPSG:7912",
    "ITRF2008":     "EPSG:7911",
    "ITRF2005":     "EPSG:7910",    # geographic 3D (4896 is geocentric XYZ)
    "ITRF96":       "EPSG:8995",

    # EASE-Grid / Polar Stereographic (remote sensing / polar ops)
    "EASEGRID2":    "EPSG:6933",    # EASE-Grid 2.0 global cylindrical equal-area
    "EASEGRID2N":   "EPSG:6931",    # EASE-Grid 2.0 North (Lambert azimuthal)
    "EASEGRID2S":   "EPSG:6932",    # EASE-Grid 2.0 South (Lambert azimuthal)
    "POLARNORTH":   "EPSG:3413",    # NSIDC Polar Stereographic North
    "POLARSOUTH":   "EPSG:3031",    # Antarctic Polar Stereographic South
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
        from openlocationcode import openlocationcode as olc
        code = olc.encode(lat, lon, codeLength=code_length)
        return f"Plus Code: {code}"
    except ImportError:
        return "openlocationcode not installed. Run: uv add openlocationcode"


@tool
def decode_plus_code(plus_code: str) -> str:
    """Decode a Plus Code (Open Location Code) to WGS84 lat/lon bounds.
    Example: decode_plus_code('8G95WMJ4+QQ')"""
    try:
        from openlocationcode import openlocationcode as olc
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
        # toLatLon accepts str or bytes depending on mgrs version — try both
        try:
            lat, lon = m.toLatLon(value.strip())
        except TypeError:
            lat, lon = m.toLatLon(value.strip().encode())
        if to_upper in ("WGS84", "DECIMAL", "LATLON"):
            return f"{lat:.7f}, {lon:.7f}"
        return _transform_crs(f"{lat}, {lon}", "WGS84", to_upper)
    else:
        lat, lon = _parse_latlon(value)
        if from_upper not in ("WGS84", "DECIMAL", "LATLON"):
            wgs84 = _transform_crs(value, from_upper, "WGS84")
            lat, lon = _parse_latlon(wgs84)
        result = m.toMGRS(lat, lon)
        # toMGRS returns str in mgrs>=1.5, bytes in older versions
        return result if isinstance(result, str) else result.decode()


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
    sizes = {1: "±2500km", 2: "±630km", 3: "±78km", 4: "±20km", 5: "±2.4km",
             6: "±610m", 7: "±76m", 8: "±19m", 9: "±2.4m", 10: "±60cm",
             11: "±7.4cm", 12: "±1.9cm"}
    return sizes.get(precision, "unknown")


# ---------------------------------------------------------------------------
# Microdegrees (degrees × 10⁶)
# ---------------------------------------------------------------------------

@tool
def encode_microdegrees(lat: float, lon: float) -> str:
    """Encode WGS84 decimal degrees to integer microdegrees (degrees × 10⁶).
    Example: encode_microdegrees(37.9838, 23.7275) → 37983800, 23727500
    """
    lat_ud = round(lat * 1_000_000)
    lon_ud = round(lon * 1_000_000)
    return f"{lat_ud}, {lon_ud}  (microdegrees)"


@tool
def decode_microdegrees(lat_ud: int, lon_ud: int) -> str:
    """Decode integer microdegrees (degrees × 10⁶) to WGS84 decimal degrees.
    Example: decode_microdegrees(37983800, 23727500)
    """
    lat = lat_ud / 1_000_000
    lon = lon_ud / 1_000_000
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return f"Out-of-range values: lat={lat}, lon={lon}"
    return f"{lat:.7f}, {lon:.7f}"


# ---------------------------------------------------------------------------
# DMS + arc-Milliseconds
# ---------------------------------------------------------------------------

@tool
def encode_dms_milliseconds(lat: float, lon: float) -> str:
    """Format WGS84 decimal degrees as Degrees, Minutes, and arc-Milliseconds.

    Arc-milliseconds are thousandths of an arc-second (1 arc-sec = 1000 ms).
    Example: encode_dms_milliseconds(37.9838, 23.7275)
             → 37° 59' 1680ms N, 23° 43' 39900ms E
    """
    def _fmt(deg: float, is_lat: bool) -> str:
        direction = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
        deg = abs(deg)
        d = int(deg)
        m = int((deg - d) * 60)
        s_total = (deg - d - m / 60) * 3600
        ms = round(s_total * 1000)
        return f"{d}° {m}' {ms}ms {direction}"

    return f"{_fmt(lat, True)}, {_fmt(lon, False)}"


# ---------------------------------------------------------------------------
# GARS (Global Area Reference System)
# ---------------------------------------------------------------------------

# 24 letters used in GARS latitude bands — I and O are excluded
_GARS_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


@tool
def encode_gars(lat: float, lon: float) -> str:
    """Encode WGS84 coordinates to a GARS (Global Area Reference System) code.

    Returns the 5-character 30-minute cell identifier plus the 15-minute
    quadrant digit (1=SW, 2=SE, 3=NW, 4=NE).

    Example: encode_gars(37.9838, 23.7275)  →  GARS: 408LR3
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "Invalid coordinates."
    # Clamp poles/antimeridian to stay within valid band indices
    lat = min(lat,  89.9999)
    lon = min(lon, 179.9999)

    lon_band    = int((lon + 180) / 0.5) + 1          # 1–720
    lat_band_n  = int((lat + 90)  / 0.5)              # 0–359

    first_letter  = _GARS_LETTERS[lat_band_n // 24]
    second_letter = _GARS_LETTERS[lat_band_n % 24]
    cell = f"{lon_band:03d}{first_letter}{second_letter}"

    # 15-minute quadrant within the 30-minute cell
    lon_offset = (lon + 180) - (lon_band - 1) * 0.5
    lat_offset = (lat + 90)  - lat_band_n * 0.5
    col = 1 if lon_offset >= 0.25 else 0   # 0=west half, 1=east half
    row = 1 if lat_offset >= 0.25 else 0   # 0=south half, 1=north half
    quadrant = 1 + col + row * 2           # 1=SW,2=SE,3=NW,4=NE

    return f"GARS: {cell}{quadrant}  (30-min cell: {cell}, 15-min quadrant: {quadrant})"


@tool
def decode_gars(gars_code: str) -> str:
    """Decode a GARS code to WGS84 lat/lon (centre of the cell or quadrant).

    Accepts 5-character (30-min cell) or 6-character (with 15-min quadrant) codes.
    Quadrant digits: 1=SW, 2=SE, 3=NW, 4=NE.

    Example: decode_gars('408LR3')
    """
    gars_code = gars_code.strip().upper()
    if len(gars_code) < 5:
        return "GARS code must be at least 5 characters (e.g. '408LR' or '408LR3')."

    try:
        lon_band   = int(gars_code[:3])
        first_idx  = _GARS_LETTERS.index(gars_code[3])
        second_idx = _GARS_LETTERS.index(gars_code[4])
    except (ValueError, IndexError):
        return f"Invalid GARS code '{gars_code}'."

    if not (1 <= lon_band <= 720):
        return f"Longitude band {lon_band} out of range (001–720)."

    lat_band_n = first_idx * 24 + second_idx
    lon_sw = (lon_band - 1) * 0.5 - 180.0
    lat_sw = lat_band_n * 0.5 - 90.0

    if len(gars_code) >= 6:
        try:
            quadrant = int(gars_code[5])
        except ValueError:
            return f"Invalid quadrant character '{gars_code[5]}'. Must be 1–4."
        if quadrant not in (1, 2, 3, 4):
            return f"Quadrant {quadrant} out of range (1–4)."
        col = (quadrant - 1) % 2   # 0=west, 1=east
        row = (quadrant - 1) // 2  # 0=south, 1=north
        lon_c = lon_sw + col * 0.25 + 0.125
        lat_c = lat_sw + row * 0.25 + 0.125
    else:
        lon_c = lon_sw + 0.25
        lat_c = lat_sw + 0.25

    return f"Lat: {lat_c:.7f}  Lon: {lon_c:.7f}"


# ---------------------------------------------------------------------------
# GEOREF (World Geographic Reference System)
# ---------------------------------------------------------------------------

# NGA/ICAO spec letter sets — I and O always excluded
_GEOREF_LON_ZONES = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # 24 letters — 15° lon zones from 180°W
_GEOREF_LAT_BANDS = "ABCDEFGHJKLM"              # 12 letters — 15° lat bands from 90°S
_GEOREF_1DEG      = "ABCDEFGHJKLMNPQ"           # 15 letters — 1° sub-cells (lon & lat)


@tool
def encode_georef(lat: float, lon: float, precision: int = 2) -> str:
    """Encode WGS84 coordinates to a GEOREF (World Geographic Reference System) code.

    precision controls the digit suffix appended after the 4-letter quadrangle code:
      1 → 4 chars  (no digits,  ±0.5°,    ~55 km)
      2 → 8 chars  (2+2 digits, ±0.5',    ~1 km,  default)
      3 → 10 chars (3+3 digits, ±0.05',   ~100 m)
      4 → 12 chars (4+4 digits, ±0.005',  ~10 m)

    Digits are always written easting (longitude) first, then northing (latitude).

    Example: encode_georef(37.9838, 23.7275)  →  GEOREF: PJJH4359
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "Invalid coordinates."
    if precision not in (1, 2, 3, 4):
        return "precision must be 1, 2, 3, or 4."

    lat = min(lat, 89.9999)
    lon = min(lon, 179.9999)

    lon_zone_idx = int((lon + 180) / 15)
    lat_band_idx = int((lat + 90) / 15)
    lon_zone_sw  = lon_zone_idx * 15 - 180
    lat_band_sw  = lat_band_idx * 15 - 90

    lon_1deg_idx = int(lon - lon_zone_sw)
    lat_1deg_idx = int(lat - lat_band_sw)

    letters = (
        _GEOREF_LON_ZONES[lon_zone_idx]
        + _GEOREF_LAT_BANDS[lat_band_idx]
        + _GEOREF_1DEG[lon_1deg_idx]
        + _GEOREF_1DEG[lat_1deg_idx]
    )

    if precision == 1:
        return f"GEOREF: {letters}"

    # Minutes within the 1° cell (range 0–59.999...)
    lon_min = (lon - (lon_zone_sw + lon_1deg_idx)) * 60
    lat_min = (lat - (lat_band_sw + lat_1deg_idx)) * 60

    # precision 2 → 1× (00–59), 3 → 10× (000–599), 4 → 100× (0000–5999)
    scale = 10 ** (precision - 2)
    lon_int = int(lon_min * scale)
    lat_int = int(lat_min * scale)

    fmt = f"0{precision}d"
    return f"GEOREF: {letters}{format(lon_int, fmt)}{format(lat_int, fmt)}"


@tool
def decode_georef(georef_code: str) -> str:
    """Decode a GEOREF code to WGS84 lat/lon (SW corner of the cell).

    Accepts 4-char (1°), 8-char (1'), 10-char (0.1'), or 12-char (0.01') codes.

    Example: decode_georef('PJJH4359')
    """
    code = georef_code.strip().upper()
    if len(code) < 4:
        return "GEOREF code must be at least 4 characters."

    try:
        lon_zone_idx = _GEOREF_LON_ZONES.index(code[0])
        lat_band_idx = _GEOREF_LAT_BANDS.index(code[1])
        lon_1deg_idx = _GEOREF_1DEG.index(code[2])
        lat_1deg_idx = _GEOREF_1DEG.index(code[3])
    except ValueError as e:
        return f"Invalid GEOREF letter in '{georef_code}': {e}"

    lon = lon_zone_idx * 15 - 180 + lon_1deg_idx
    lat = lat_band_idx * 15 - 90  + lat_1deg_idx

    digit_part = code[4:]
    if digit_part:
        if len(digit_part) not in (4, 6, 8):
            return f"Digit portion must be 4, 6, or 8 digits, got {len(digit_part)}."
        n = len(digit_part) // 2
        try:
            lon_int = int(digit_part[:n])
            lat_int = int(digit_part[n:])
        except ValueError:
            return f"Non-numeric digits in '{georef_code}'."
        # n=2 → ÷60, n=3 → ÷600, n=4 → ÷6000
        divisor = 60 * (10 ** (n - 2))
        lon += lon_int / divisor
        lat += lat_int / divisor

    return f"Lat: {lat:.7f}  Lon: {lon:.7f}  (SW corner of cell)"


# ---------------------------------------------------------------------------
# Maidenhead (QTH) locator
# ---------------------------------------------------------------------------

_MAIDENHEAD_UPPER = "ABCDEFGHIJKLMNOPQR"
_MAIDENHEAD_LOWER = "abcdefghijklmnopqrstuvwx"


@tool
def encode_maidenhead(lat: float, lon: float, precision: int = 6) -> str:
    """Encode WGS84 coordinates to a Maidenhead (QTH) locator grid square.
    precision: 4 (field+square, ~100x50km), 6 (subsquare, ~5x2.5km, default),
               8 (extended, ~300x150m).
    Example: encode_maidenhead(51.4778, -0.0015)  →  IO91wm
    """
    if not (4 <= precision <= 8 and precision % 2 == 0):
        return "precision must be 4, 6, or 8."
    try:
        result = _maidenhead_encode(lat, lon, precision)
        return f"Maidenhead: {result}"
    except Exception as e:
        return f"Encoding error: {e}"


@tool
def decode_maidenhead(locator: str) -> str:
    """Decode a Maidenhead (QTH) locator to WGS84 lat/lon (centre of grid square).
    Example: decode_maidenhead('IO91wm')
    """
    try:
        lat, lon = _maidenhead_decode(locator.strip())
        return f"Centre: {lat:.5f}, {lon:.5f}"
    except Exception as e:
        return f"Decoding error: {e}"


def _maidenhead_encode(lat: float, lon: float, precision: int = 6) -> str:
    lon += 180.0
    lat += 90.0
    result = ""
    result += _MAIDENHEAD_UPPER[int(lon / 20)]
    result += _MAIDENHEAD_UPPER[int(lat / 10)]
    lon = lon % 20
    lat = lat % 10
    result += str(int(lon / 2))
    result += str(int(lat))
    if precision >= 6:
        lon = (lon % 2) * 12
        lat = (lat % 1) * 24
        result += _MAIDENHEAD_LOWER[int(lon)]
        result += _MAIDENHEAD_LOWER[int(lat)]
    if precision >= 8:
        lon = (lon % 1) * 10
        lat = (lat % 1) * 10
        result += str(int(lon))
        result += str(int(lat))
    return result


def _maidenhead_decode(locator: str) -> tuple[float, float]:
    if len(locator) < 4:
        raise ValueError("Locator must be at least 4 characters.")
    locator = locator.upper()
    lon = (_MAIDENHEAD_UPPER.index(locator[0])) * 20 - 180
    lat = (_MAIDENHEAD_UPPER.index(locator[1])) * 10 - 90
    lon += int(locator[2]) * 2
    lat += int(locator[3])
    if len(locator) >= 6:
        lon += (_MAIDENHEAD_LOWER.index(locator[4].lower())) / 12
        lat += (_MAIDENHEAD_LOWER.index(locator[5].lower())) / 24
        # Centre of subsquare
        lon += 1 / 24
        lat += 1 / 48
    else:
        lon += 1.0   # centre of square
        lat += 0.5
    return lat, lon


# ---------------------------------------------------------------------------
# NATO military mils
# ---------------------------------------------------------------------------

_MILS_PER_DEGREE = 6400 / 360   # 17.7778 NATO mils per degree


@tool
def convert_to_mils(degrees: float) -> str:
    """Convert an angular value in decimal degrees to NATO mils (6400 mils = 360°).
    Used for artillery bearings and map angles.
    Example: convert_to_mils(45.0)  →  800.00 mils
    """
    mils = degrees * _MILS_PER_DEGREE
    return f"{mils:.2f} mils  ({degrees:.6f}°)"


@tool
def convert_from_mils(mils: float) -> str:
    """Convert NATO mils to decimal degrees.
    Example: convert_from_mils(800)  →  45.0°
    """
    degrees = mils / _MILS_PER_DEGREE
    return f"{degrees:.6f}°  ({mils:.2f} mils)"
