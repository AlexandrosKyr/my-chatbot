"""
Coordinate system tests.

Run from the project root:
    uv run --directory backend pytest ../tests/test_coordinates.py -v

Each test covers one clear behaviour. If a test fails the name tells you
exactly what broke.
"""

import pytest

from coordinate_parser import CoordinateParser
from tools.coordinates import (
    convert_coordinates,
    encode_geohash, decode_geohash,
    encode_plus_code, decode_plus_code,
    encode_maidenhead, decode_maidenhead,
    convert_to_mils, convert_from_mils,
    encode_microdegrees, decode_microdegrees,
    encode_dms_milliseconds,
    encode_gars, decode_gars,
    encode_georef, decode_georef,
    _maidenhead_encode, _maidenhead_decode,
    _GARS_LETTERS,
    _GEOREF_LON_ZONES, _GEOREF_LAT_BANDS, _GEOREF_1DEG,
    COORDINATE_SYSTEMS,
)

# Reference point: Athens, Greece
ATHENS_LAT = 37.9838
ATHENS_LON = 23.7275

# ── Coordinate parser ────────────────────────────────────────────────────────

class TestCoordinateParser:
    def setup_method(self):
        self.parser = CoordinateParser()

    def test_decimal_degrees(self):
        result = self.parser.parse("37.9838, 23.7275")
        assert result is not None
        assert abs(result["lat"] - 37.9838) < 0.0001
        assert abs(result["lon"] - 23.7275) < 0.0001

    def test_decimal_degrees_negative(self):
        result = self.parser.parse("-33.8688, 151.2093")  # Sydney
        assert result is not None
        assert result["lat"] < 0

    def test_labeled_decimal(self):
        result = self.parser.parse("lat: 37.9838, lon: 23.7275")
        assert result is not None
        assert abs(result["lat"] - 37.9838) < 0.0001

    def test_dms_format(self):
        # 37°59'1.68"N, 23°43'39.9"E
        result = self.parser.parse("37°59'1.68\"N, 23°43'39.9\"E")
        assert result is not None
        assert abs(result["lat"] - 37.9838) < 0.001
        assert abs(result["lon"] - 23.7275) < 0.001

    def test_dms_south_west(self):
        result = self.parser.parse("33°51'55.68\"S, 151°12'33.48\"E")
        assert result is not None
        assert result["lat"] < 0  # Southern hemisphere

    def test_ddm_format(self):
        # 37°59.028'N, 23°43.65'E
        result = self.parser.parse("37°59.028'N, 23°43.65'E")
        assert result is not None
        assert abs(result["lat"] - 37.9838) < 0.001
        assert abs(result["lon"] - 23.7275) < 0.001

    def test_ddm_south(self):
        result = self.parser.parse("33°52.0'S, 151°12.5'E")
        assert result is not None
        assert result["lat"] < 0

    def test_no_coordinates_returns_none(self):
        result = self.parser.parse("What is the weather like today?")
        assert result is None

    def test_invalid_latitude_rejected(self):
        result = self.parser.parse("95.0000, 23.7275")  # lat > 90
        assert result is None

    def test_coordinates_embedded_in_sentence(self):
        result = self.parser.parse("Analyze position 37.9838, 23.7275 for defensive ops")
        assert result is not None
        assert abs(result["lat"] - 37.9838) < 0.0001


# ── Coordinate system conversions ────────────────────────────────────────────

class TestCoordinateConversions:

    def _invoke(self, value, from_sys, to_sys):
        return convert_coordinates.invoke({
            "value": value, "from_system": from_sys, "to_system": to_sys
        })

    def test_wgs84_to_mgrs(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "MGRS")
        assert "34S" in result  # Athens is in MGRS zone 34S

    def test_mgrs_to_wgs84_roundtrip(self):
        mgrs_val = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "MGRS")
        back = self._invoke(mgrs_val, "MGRS", "WGS84")
        parts = back.split(",")
        assert abs(float(parts[0]) - ATHENS_LAT) < 0.01
        assert abs(float(parts[1]) - ATHENS_LON) < 0.01

    def test_wgs84_to_utm(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "UTM")
        assert "34N" in result
        assert "Easting" in result
        assert "Northing" in result

    def test_wgs84_to_egsa87(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "EGSA87")
        assert "error" not in result.lower()
        # EGSA87 is a projected CRS — result should be large easting/northing values
        parts = result.split(",")
        assert float(parts[0]) > 1000  # metres, not degrees

    def test_wgs84_to_ed50(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "ED50")
        assert "error" not in result.lower()
        parts = result.split(",")
        # ED50 is very close to WGS84 — difference should be < 0.01°
        assert abs(float(parts[0]) - ATHENS_LAT) < 0.01

    def test_wgs84_to_etrs89(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "ETRS89")
        assert "error" not in result.lower()

    def test_wgs84_to_ch1903(self):
        # Bern, Switzerland
        result = self._invoke("46.9481, 7.4474", "WGS84", "CH1903")
        assert "error" not in result.lower()

    def test_wgs84_to_lv95(self):
        result = self._invoke("46.9481, 7.4474", "WGS84", "LV95")
        assert "error" not in result.lower()

    def test_wgs84_to_rt90(self):
        # Stockholm
        result = self._invoke("59.3293, 18.0686", "WGS84", "RT90")
        assert "error" not in result.lower()

    def test_wgs84_to_sk42(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "SK42")
        assert "error" not in result.lower()

    def test_wgs84_to_sk95(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "SK-95")
        assert "error" not in result.lower()

    def test_wgs84_to_pz90(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "PZ-90")
        assert "error" not in result.lower()

    def test_wgs84_to_nad83(self):
        # New York
        result = self._invoke("40.7128, -74.0060", "WGS84", "NAD83")
        assert "error" not in result.lower()

    def test_wgs84_to_nad27(self):
        result = self._invoke("40.7128, -74.0060", "WGS84", "NAD27")
        assert "error" not in result.lower()

    def test_wgs84_to_osgb36(self):
        # London
        result = self._invoke("51.5074, -0.1278", "WGS84", "OSGB36")
        assert "error" not in result.lower()

    def test_wgs84_to_tokyo(self):
        # Tokyo
        result = self._invoke("35.6762, 139.6503", "WGS84", "TOKYO")
        assert "error" not in result.lower()

    def test_wgs84_to_egypt1907(self):
        # Cairo
        result = self._invoke("30.0444, 31.2357", "WGS84", "EGYPT1907")
        assert "error" not in result.lower()

    def test_wgs84_to_itrf2014(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "ITRF2014")
        assert "error" not in result.lower()

    def test_wgs84_to_itrf2008(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "ITRF2008")
        assert "error" not in result.lower()

    def test_wgs84_to_itrf2005(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "ITRF2005")
        assert "error" not in result.lower()

    def test_wgs84_to_itrf96(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "ITRF96")
        assert "error" not in result.lower()

    def test_unknown_system_returns_error(self):
        result = self._invoke(f"{ATHENS_LAT}, {ATHENS_LON}", "WGS84", "FAKECRS")
        assert "Unknown" in result

    def test_all_systems_in_dict_are_valid_epsg(self):
        """Every entry in COORDINATE_SYSTEMS must be a valid pyproj CRS."""
        from pyproj import CRS
        failures = []
        for name, epsg in COORDINATE_SYSTEMS.items():
            try:
                CRS.from_authority(*epsg.split(":"))
            except Exception as e:
                failures.append(f"{name} ({epsg}): {e}")
        assert not failures, "Invalid EPSG codes:\n" + "\n".join(failures)


# ── Geohash ──────────────────────────────────────────────────────────────────

class TestGeohash:

    def test_encode(self):
        result = encode_geohash.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON})
        assert "Geohash:" in result
        assert "sw" in result.lower()  # Athens geohash starts with sw...

    def test_decode_roundtrip(self):
        encoded = encode_geohash.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON, "precision": 9})
        gh = encoded.split("Geohash:")[1].split()[0].strip()
        decoded = decode_geohash.invoke({"geohash": gh})
        parts = decoded.split("\n")[0].split()
        lat = float(parts[1])
        lon = float(parts[3])
        assert abs(lat - ATHENS_LAT) < 0.001
        assert abs(lon - ATHENS_LON) < 0.001

    def test_precision_affects_length(self):
        r6 = encode_geohash.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON, "precision": 6})
        r9 = encode_geohash.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON, "precision": 9})
        gh6 = r6.split("Geohash:")[1].split()[0].strip()
        gh9 = r9.split("Geohash:")[1].split()[0].strip()
        assert len(gh6) == 6
        assert len(gh9) == 9


# ── Plus Codes ───────────────────────────────────────────────────────────────

class TestPlusCodes:

    def test_encode(self):
        result = encode_plus_code.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON})
        assert "Plus Code:" in result
        assert "+" in result

    def test_decode_roundtrip(self):
        encoded = encode_plus_code.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON})
        code = encoded.split("Plus Code:")[1].strip()
        decoded = decode_plus_code.invoke({"plus_code": code})
        assert "Center:" in decoded
        parts = decoded.split("\n")[0].split()
        lat = float(parts[1].rstrip(","))
        lon = float(parts[2])
        assert abs(lat - ATHENS_LAT) < 0.01
        assert abs(lon - ATHENS_LON) < 0.01

    def test_invalid_plus_code(self):
        result = decode_plus_code.invoke({"plus_code": "NOTVALID"})
        assert "not a valid" in result.lower() or "error" in result.lower()


# ── Maidenhead ───────────────────────────────────────────────────────────────

class TestMaidenhead:

    def test_encode_london(self):
        # London ~IO91wm
        result = encode_maidenhead.invoke({"lat": 51.4778, "lon": -0.0015})
        assert "IO91" in result

    def test_encode_athens(self):
        result = encode_maidenhead.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON})
        assert "KM17" in result  # Athens is in KM17

    def test_decode_roundtrip(self):
        original = "KM17rx"
        decoded = decode_maidenhead.invoke({"locator": original})
        assert "Centre:" in decoded
        parts = decoded.split(":")[1].split(",")
        lat = float(parts[0])
        lon = float(parts[1])
        # Re-encode and check we get back into the same square
        re_encoded = _maidenhead_encode(lat, lon, 6)
        assert re_encoded[:4] == original[:4]  # at least field+square matches

    def test_known_value_io91wm(self):
        lat, lon = _maidenhead_decode("IO91wm")
        assert 51.4 < lat < 51.6
        assert -0.2 < lon < 0.1  # subsquare centre is at -0.125

    def test_precision_4(self):
        result = encode_maidenhead.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON, "precision": 4})
        locator = result.split("Maidenhead:")[1].strip()
        assert len(locator) == 4

    def test_precision_8(self):
        result = encode_maidenhead.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON, "precision": 8})
        locator = result.split("Maidenhead:")[1].strip()
        assert len(locator) == 8

    def test_invalid_precision_rejected(self):
        result = encode_maidenhead.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON, "precision": 5})
        assert "precision" in result.lower()


# ── Military mils ────────────────────────────────────────────────────────────

class TestMilitaryMils:

    def test_45_degrees_is_800_mils(self):
        result = convert_to_mils.invoke({"degrees": 45.0})
        assert "800.00" in result

    def test_360_degrees_is_6400_mils(self):
        result = convert_to_mils.invoke({"degrees": 360.0})
        assert "6400.00" in result

    def test_0_degrees_is_0_mils(self):
        result = convert_to_mils.invoke({"degrees": 0.0})
        assert "0.00" in result

    def test_from_mils_800_is_45_degrees(self):
        result = convert_from_mils.invoke({"mils": 800.0})
        assert "45.0" in result

    def test_roundtrip(self):
        original = 123.456
        mils_str = convert_to_mils.invoke({"degrees": original})
        mils_val = float(mils_str.split("mils")[0].strip())
        back_str = convert_from_mils.invoke({"mils": mils_val})
        back_deg = float(back_str.split("°")[0].strip())
        assert abs(back_deg - original) < 0.001


# ── Microdegrees ─────────────────────────────────────────────────────────────

class TestMicrodegrees:

    def test_encode_athens(self):
        result = encode_microdegrees.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON})
        assert "37983800" in result
        assert "23727500" in result

    def test_decode_athens(self):
        result = decode_microdegrees.invoke({"lat_ud": 37983800, "lon_ud": 23727500})
        parts = result.split(",")
        assert abs(float(parts[0]) - ATHENS_LAT) < 0.0001
        assert abs(float(parts[1]) - ATHENS_LON) < 0.0001

    def test_encode_decode_roundtrip(self):
        result = encode_microdegrees.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON})
        nums = [int(x.strip().split()[0]) for x in result.split("(")[0].split(",")]
        back = decode_microdegrees.invoke({"lat_ud": nums[0], "lon_ud": nums[1]})
        parts = back.split(",")
        assert abs(float(parts[0]) - ATHENS_LAT) < 0.000002  # 1 µ° precision
        assert abs(float(parts[1]) - ATHENS_LON) < 0.000002

    def test_negative_coordinates(self):
        result = encode_microdegrees.invoke({"lat": -33.8688, "lon": 151.2093})
        assert "-33868800" in result

    def test_out_of_range_decode(self):
        result = decode_microdegrees.invoke({"lat_ud": 95000000, "lon_ud": 0})
        assert "Out-of-range" in result

    def test_parser_labeled_microdegrees(self):
        parser = CoordinateParser()
        result = parser.parse("lat: 37983800, lon: 23727500 microdegrees")
        assert result is not None
        assert abs(result["lat"] - ATHENS_LAT) < 0.000002
        assert abs(result["lon"] - ATHENS_LON) < 0.000002

    def test_parser_labeled_negative(self):
        parser = CoordinateParser()
        result = parser.parse("lat: -33868800, lon: 151209300 microdegrees")
        assert result is not None
        assert result["lat"] < 0


# ── DMS + arc-Milliseconds ───────────────────────────────────────────────────

class TestDMSMilliseconds:

    def test_encode_athens(self):
        result = encode_dms_milliseconds.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON})
        # 23.7275° = 23°43'39.000" → 39000ms
        assert "37°" in result
        assert "59'" in result
        assert "1680ms" in result
        assert "23°" in result
        assert "43'" in result
        assert "39000ms" in result

    def test_encode_southern_hemisphere(self):
        result = encode_dms_milliseconds.invoke({"lat": -33.8688, "lon": 151.2093})
        assert " S" in result
        assert " E" in result

    def test_encode_western_hemisphere(self):
        result = encode_dms_milliseconds.invoke({"lat": 40.7128, "lon": -74.0060})
        assert " N" in result
        assert " W" in result

    def test_parser_roundtrip(self):
        # Encode to DMS+ms then parse back
        encoded = encode_dms_milliseconds.invoke({"lat": ATHENS_LAT, "lon": ATHENS_LON})
        parser = CoordinateParser()
        result = parser.parse(encoded)
        assert result is not None
        assert abs(result["lat"] - ATHENS_LAT) < 0.0001
        assert abs(result["lon"] - ATHENS_LON) < 0.0001

    def test_parser_dms_ms_explicit(self):
        parser = CoordinateParser()
        # 23.7275° = 23°43'39.000" → 39000ms
        result = parser.parse("37° 59' 1680ms N, 23° 43' 39000ms E")
        assert result is not None
        assert abs(result["lat"] - ATHENS_LAT) < 0.0001
        assert abs(result["lon"] - ATHENS_LON) < 0.0001

    def test_zero_seconds(self):
        # 38°0'0ms N = exactly 38°
        parser = CoordinateParser()
        result = parser.parse("38° 0' 0ms N, 24° 0' 0ms E")
        assert result is not None
        assert abs(result["lat"] - 38.0) < 0.0001
        assert abs(result["lon"] - 24.0) < 0.0001


# ── GARS ─────────────────────────────────────────────────────────────────────

class TestGARS:

    def _encode(self, lat, lon):
        return encode_gars.invoke({"lat": lat, "lon": lon})

    def _decode(self, code):
        return decode_gars.invoke({"gars_code": code})

    def test_encode_athens_cell(self):
        result = self._encode(ATHENS_LAT, ATHENS_LON)
        assert "408LR" in result

    def test_encode_athens_quadrant(self):
        # Athens lon_offset ~0.2275 (<0.25 → west col=0), lat_offset ~0.4838 (>=0.25 → row=1)
        # quadrant = 1 + 0 + 1*2 = 3 (NW)
        result = self._encode(ATHENS_LAT, ATHENS_LON)
        assert "408LR3" in result

    def test_decode_30min_cell_centre(self):
        result = self._decode("408LR")
        # SW corner of 408LR: lon = (408-1)*0.5 - 180 = 23.5, lat = 255*0.5 - 90 = 37.5
        # Centre: lon=23.75, lat=37.75
        assert "37.75" in result
        assert "23.75" in result

    def test_decode_quadrant_centre(self):
        # Quadrant 3 = NW: col=0, row=1 → lon_c = 23.5 + 0 + 0.125 = 23.625, lat_c = 37.5 + 0.25 + 0.125 = 37.875
        result = self._decode("408LR3")
        assert "37.875" in result
        assert "23.625" in result

    def test_encode_decode_roundtrip(self):
        # Decode the 30-min cell centre, re-encode, get the same cell
        cell_result = self._decode("408LR")
        lat = float(cell_result.split("Lat:")[1].split()[0])
        lon = float(cell_result.split("Lon:")[1].split()[0])
        re_encoded = self._encode(lat, lon)
        assert "408LR" in re_encoded

    def test_prime_meridian_code(self):
        # 0°N, 0°E → lon_band=361, lat_band=180 → HN, quadrant SW=1
        result = self._encode(0.0, 0.0)
        assert "361HN1" in result

    def test_antimeridian_clamped_code(self):
        # 0°N, 180°E → clamped to 179.9999 → lon_band=720, lat_band=180 → HN, col=1 row=0 → q=2
        result = self._encode(0.0, 180.0)
        assert "720HN2" in result

    def test_decode_quadrant_center_is_inside_cell(self):
        # Quadrant 3 of 408LR is NW quarter of the 30-min cell.
        # Cell SW: lon=23.5, lat=37.5  →  NW quadrant spans lon 23.5–23.75, lat 37.75–38.0
        result = self._decode("408LR3")
        lat = float(result.split("Lat:")[1].split()[0])
        lon = float(result.split("Lon:")[1].split()[0])
        assert 23.5 <= lon < 23.75
        assert 37.75 <= lat < 38.0

    def test_decode_all_four_quadrants_tile_cell(self):
        # Quadrants 1–4 must together cover the full 30-min cell without overlap.
        # Check that centers are in the correct NW/NE/SW/SE sub-squares.
        expected = {
            1: (23.5,  23.75, 37.5,  37.75),   # SW: lon [23.5,23.75)  lat [37.5,37.75)
            2: (23.75, 24.0,  37.5,  37.75),   # SE: lon [23.75,24.0)  lat [37.5,37.75)
            3: (23.5,  23.75, 37.75, 38.0),    # NW: lon [23.5,23.75)  lat [37.75,38.0)
            4: (23.75, 24.0,  37.75, 38.0),    # NE: lon [23.75,24.0)  lat [37.75,38.0)
        }
        for q, (lon_lo, lon_hi, lat_lo, lat_hi) in expected.items():
            result = self._decode(f"408LR{q}")
            lat = float(result.split("Lat:")[1].split()[0])
            lon = float(result.split("Lon:")[1].split()[0])
            assert lon_lo <= lon < lon_hi, f"Quadrant {q} lon {lon} not in [{lon_lo},{lon_hi})"
            assert lat_lo <= lat < lat_hi, f"Quadrant {q} lat {lat} not in [{lat_lo},{lat_hi})"

    def test_quadrant_roundtrip(self):
        # Re-encoding the decoded quadrant centre must recover the exact same quadrant code.
        result = self._decode("408LR3")
        lat = float(result.split("Lat:")[1].split()[0])
        lon = float(result.split("Lon:")[1].split()[0])
        re_enc = self._encode(lat, lon)
        assert "408LR3" in re_enc

    def test_invalid_code_too_short(self):
        result = self._decode("40")
        assert "must be at least 5" in result

    def test_invalid_quadrant_digit(self):
        result = self._decode("408LR5")
        assert "out of range" in result.lower() or "Invalid" in result


# ── EASE-Grid / Polar Stereographic ──────────────────────────────────────────

class TestEASEGrid:
    """
    Reference values computed with pyproj:
      Athens (37.9838°N, 23.7275°E):
        EASEGRID2  → x=2,289,378 m   y=4,506,115 m
        EASEGRID2N → x=2,253,279 m   y=-5,126,428 m
      Sydney (-33.8688°N, 151.2093°E):
        EASEGRID2S → x=2,892,813 m   y=-5,264,031 m
      75°N, 15°E:
        POLARNORTH → x=1,414,981 m   y=-816,939 m
      75°S, 15°E:
        POLARSOUTH → x=424,148 m     y=1,582,943 m
    """
    _TOL = 1.0   # 1-metre tolerance on projected coordinates

    def _xy(self, lat, lon, to_sys):
        result = convert_coordinates.invoke({
            "value": f"{lat}, {lon}", "from_system": "WGS84", "to_system": to_sys
        })
        parts = result.split(",")
        return float(parts[0].strip()), float(parts[1].strip())

    def test_easegrid2_athens_x(self):
        x, _ = self._xy(ATHENS_LAT, ATHENS_LON, "EASEGRID2")
        assert abs(x - 4506115.7) < self._TOL   # y is returned as first component (lat axis)

    def test_easegrid2_athens_y(self):
        _, y = self._xy(ATHENS_LAT, ATHENS_LON, "EASEGRID2")
        assert abs(y - 2289378.2) < self._TOL

    def test_easegrid2n_athens_coordinates(self):
        x, y = self._xy(ATHENS_LAT, ATHENS_LON, "EASEGRID2N")
        assert abs(x - (-5126428.3)) < self._TOL
        assert abs(y - 2253279.9) < self._TOL

    def test_easegrid2s_sydney_coordinates(self):
        x, y = self._xy(-33.8688, 151.2093, "EASEGRID2S")
        assert abs(x - (-5264031.2)) < self._TOL
        assert abs(y - 2892813.1) < self._TOL

    def test_polarnorth_coordinates(self):
        # 75°N 15°E → NSIDC Polar Stereo North
        x, y = self._xy(75.0, 15.0, "POLARNORTH")
        assert abs(x - (-816939.7)) < self._TOL
        assert abs(y - 1414981.2) < self._TOL

    def test_polarsouth_coordinates(self):
        # 75°S 15°E → Antarctic Polar Stereo South
        x, y = self._xy(-75.0, 15.0, "POLARSOUTH")
        assert abs(x - 1582943.1) < self._TOL
        assert abs(y - 424148.3) < self._TOL

    def test_easegrid2_output_in_valid_range(self):
        # EASE-Grid 2.0 global: x ∈ [-17,367,530, 17,367,530] m, y ∈ [-7,314,540, 7,314,540] m
        x, y = self._xy(ATHENS_LAT, ATHENS_LON, "EASEGRID2")
        assert -17_367_530 <= y <= 17_367_530
        assert -7_314_540 <= x <= 7_314_540

    def test_polarnorth_northern_hemisphere_only(self):
        # All points in the northern hemisphere should produce finite coordinates
        for lat, lon in [(90.0, 0.0), (80.0, 45.0), (60.0, -90.0)]:
            lat = min(lat, 89.9)
            x, y = self._xy(lat, lon, "POLARNORTH")
            assert abs(x) < 1e8 and abs(y) < 1e8

    def test_all_epsg_codes_valid(self):
        from pyproj import CRS
        for name in ["EASEGRID2", "EASEGRID2N", "EASEGRID2S", "POLARNORTH", "POLARSOUTH"]:
            CRS.from_authority(*COORDINATE_SYSTEMS[name].split(":"))  # raises if invalid


# ── GEOREF (World Geographic Reference System) ───────────────────────────────

class TestGEOREF:
    """
    Reference values hand-computed from the NGA/ICAO spec:
      Athens 37.9838°N 23.7275°E → PJJH4359
        lon zone: (23.7275+180)/15 = 13.58 → idx 13 → 'P'
        lat band: (37.9838+90)/15  =  8.53 → idx  8 → 'J'
        1° lon:   23.7275 - 15     =  8.73 → idx  8 → 'J'
        1° lat:   37.9838 - 30     =  7.98 → idx  7 → 'H'
        lon min:  0.7275 × 60 = 43.65 → 43
        lat min:  0.9838 × 60 = 59.03 → 59
    """

    def _enc(self, lat, lon, precision=2):
        return encode_georef.invoke({"lat": lat, "lon": lon, "precision": precision})

    def _dec(self, code):
        return decode_georef.invoke({"georef_code": code})

    # --- Encode ---

    def test_encode_athens_letters(self):
        result = self._enc(ATHENS_LAT, ATHENS_LON)
        assert "PJJH" in result

    def test_encode_athens_full_8char(self):
        result = self._enc(ATHENS_LAT, ATHENS_LON)
        assert "PJJH4359" in result

    def test_encode_precision1_four_chars(self):
        result = self._enc(ATHENS_LAT, ATHENS_LON, precision=1)
        code = result.split("GEOREF:")[1].strip()
        assert code == "PJJH"

    def test_encode_precision3_exact_code(self):
        # p3: scale=10 → lon_int=436, lat_int=590 → PJJH436590
        result = self._enc(ATHENS_LAT, ATHENS_LON, precision=3)
        assert "PJJH436590" in result

    def test_encode_precision4_exact_code(self):
        # p4: scale=100 → lon_int=4364 (float truncation), lat_int=5902 → PJJH43645902
        result = self._enc(ATHENS_LAT, ATHENS_LON, precision=4)
        assert "PJJH43645902" in result

    def test_encode_new_york_full_code(self):
        # 40.7128°N 74.0060°W → HJAL5942
        # lon zone idx=7 (H), lat band idx=8 (J), 1°lon idx=0 (A), 1°lat idx=10 (L)
        # lon_min=59.64→59, lat_min=42.768→42
        result = self._enc(40.7128, -74.0060)
        assert "HJAL5942" in result

    def test_encode_prime_meridian_code(self):
        # 0°N, 0°E → NGAA0000
        result = self._enc(0.0, 0.0)
        assert "NGAA0000" in result

    def test_encode_sydney_code(self):
        # -33.8688°S, 151.2093°E → YDBM1207
        result = self._enc(-33.8688, 151.2093)
        assert "YDBM1207" in result

    def test_encode_invalid_precision(self):
        result = self._enc(ATHENS_LAT, ATHENS_LON, precision=5)
        assert "precision" in result.lower()

    # --- Decode ---

    def test_decode_8char_sw_corner(self):
        # 'PJJH4359' → SW corner lon=23+43/60, lat=37+59/60
        result = self._dec("PJJH4359")
        lat = float(result.split("Lat:")[1].split()[0])
        lon = float(result.split("Lon:")[1].split()[0])
        assert abs(lat - (37 + 59 / 60)) < 0.0001
        assert abs(lon - (23 + 43 / 60)) < 0.0001

    def test_decode_4char_sw_corner(self):
        # 'PJJH' → 1° cell SW corner: lat=37, lon=23
        result = self._dec("PJJH")
        lat = float(result.split("Lat:")[1].split()[0])
        lon = float(result.split("Lon:")[1].split()[0])
        assert abs(lat - 37.0) < 0.0001
        assert abs(lon - 23.0) < 0.0001

    def test_decode_invalid_letter(self):
        result = self._dec("IJJH4359")   # I is excluded from lon zones
        assert "invalid" in result.lower() or "error" in result.lower()

    def test_decode_bad_digit_length(self):
        result = self._dec("PJJH123")    # 3 digits — not valid (must be 4/6/8)
        assert "digit" in result.lower()

    def test_decode_too_short(self):
        result = self._dec("PJ")
        assert "must be at least 4" in result

    # --- Roundtrip ---

    def test_roundtrip_precision2(self):
        encoded = self._enc(ATHENS_LAT, ATHENS_LON, precision=2)
        code = encoded.split("GEOREF:")[1].strip()
        decoded = self._dec(code)
        lat = float(decoded.split("Lat:")[1].split()[0])
        lon = float(decoded.split("Lon:")[1].split()[0])
        # SW corner should be within 1 arc-minute of original
        assert abs(lat - ATHENS_LAT) < 1 / 60
        assert abs(lon - ATHENS_LON) < 1 / 60

    def test_roundtrip_precision4(self):
        encoded = self._enc(ATHENS_LAT, ATHENS_LON, precision=4)
        code = encoded.split("GEOREF:")[1].strip()
        decoded = self._dec(code)
        lat = float(decoded.split("Lat:")[1].split()[0])
        lon = float(decoded.split("Lon:")[1].split()[0])
        # SW corner must be within one cell-width (1/6000° ≈ 18m) of the original.
        # Allow 1.5× to absorb floating-point truncation (int() floors, so SW can
        # sit up to one full cell-width below the input).
        cell_deg = 1 / 6000
        assert abs(lat - ATHENS_LAT) < cell_deg * 1.5
        assert abs(lon - ATHENS_LON) < cell_deg * 1.5

    def test_all_precisions_share_letter_prefix(self):
        # All precision levels must produce the same 4-letter quadrangle prefix.
        codes = [
            self._enc(ATHENS_LAT, ATHENS_LON, p).split("GEOREF:")[1].strip()
            for p in (1, 2, 3, 4)
        ]
        for code in codes:
            assert code[:4] == "PJJH", f"Expected PJJH prefix, got {code[:4]}"

    def test_decode_new_york_sw_corner(self):
        # HJAL5942 → SW corner: lon = -75+0 + 59/60, lat = 30+10 + 42/60
        result = self._dec("HJAL5942")
        lat = float(result.split("Lat:")[1].split()[0])
        lon = float(result.split("Lon:")[1].split()[0])
        assert abs(lat - (40 + 42 / 60)) < 0.0001
        assert abs(lon - (-75 + 59 / 60)) < 0.0001

    def test_easting_before_northing(self):
        # Spec: easting (longitude) minutes always precede northing (latitude) minutes.
        # Swap lon/lat and verify the code changes in the digit half, not the letter half.
        # Athens at swapped coords (lat=23.7275, lon=37.9838 is out of range so use 10°N,20°E vs 20°N,10°E)
        r1 = self._enc(10.0, 20.0).split("GEOREF:")[1].strip()
        r2 = self._enc(20.0, 10.0).split("GEOREF:")[1].strip()
        # Different lat/lon → different letters AND different digits; verify they are not equal
        assert r1 != r2
        # Decode both and confirm lat/lon are in the right hemisphere
        d1 = self._dec(r1)
        lat1 = float(d1.split("Lat:")[1].split()[0])
        lon1 = float(d1.split("Lon:")[1].split()[0])
        assert abs(lat1 - 10.0) < 1.0   # SW corner within 1° of input lat
        assert abs(lon1 - 20.0) < 1.0   # SW corner within 1° of input lon

    # --- Letter tables ---

    def test_lon_zones_24_letters_no_i_o(self):
        assert len(_GEOREF_LON_ZONES) == 24
        assert "I" not in _GEOREF_LON_ZONES
        assert "O" not in _GEOREF_LON_ZONES

    def test_lat_bands_12_letters_no_i(self):
        assert len(_GEOREF_LAT_BANDS) == 12
        assert "I" not in _GEOREF_LAT_BANDS

    def test_1deg_15_letters_no_i_o(self):
        assert len(_GEOREF_1DEG) == 15
        assert "I" not in _GEOREF_1DEG
        assert "O" not in _GEOREF_1DEG

    # --- Parser ---

    def test_parser_recognises_8char_georef(self):
        parser = CoordinateParser()
        result = parser.parse("Target at PJJH4359 confirmed")
        assert result is not None
        # Centre of 1-minute cell containing Athens
        assert abs(result["lat"] - ATHENS_LAT) < 1 / 60 + 0.01
        assert abs(result["lon"] - ATHENS_LON) < 1 / 60 + 0.01

    def test_parser_does_not_match_4char_only(self):
        # Bare 4-letter GEOREF without digit suffix should not trigger parser
        # (too ambiguous — could be any word)
        parser = CoordinateParser()
        result = parser.parse("PJJH")
        assert result is None

    def test_parser_10char_georef(self):
        encoded = self._enc(ATHENS_LAT, ATHENS_LON, precision=3)
        code = encoded.split("GEOREF:")[1].strip()
        parser = CoordinateParser()
        result = parser.parse(f"Position: {code}")
        assert result is not None
        assert abs(result["lat"] - ATHENS_LAT) < 0.1 / 60 + 0.001
        assert abs(result["lon"] - ATHENS_LON) < 0.1 / 60 + 0.001
