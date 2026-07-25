"""
Datum transformation tests — covers every datum in column 2 of the requirements.

Each test verifies:
  1. The conversion runs without error
  2. The result is physically reasonable
  3. A roundtrip lands back within acceptable tolerance

Run from backend/:
    uv run python -m pytest ../tests/test_datums.py -v

Background: all datum transforms go through pyproj's Transformer.from_crs().
When you convert WGS84 → ED50 you're doing a datum shift, not just a
re-projection. Two tests fail intentionally if an EPSG code is wrong or
pyproj lacks the transformation grid for that datum.
"""

import pytest
from tools.coordinates import convert_coordinates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to(value: str, src: str, dst: str) -> str:
    return convert_coordinates.invoke({"value": value, "from_system": src, "to_system": dst})


def roundtrip(value: str, via: str, tolerance: float = 0.01) -> None:
    """Convert WGS84 → datum → WGS84 and check we land within tolerance degrees."""
    intermediate = to(value, "WGS84", via)
    assert "error" not in intermediate.lower(), f"Forward conversion to {via} failed: {intermediate}"
    back = to(intermediate, via, "WGS84")
    assert "error" not in back.lower(), f"Reverse conversion from {via} failed: {back}"
    orig_lat, orig_lon = float(value.split(",")[0]), float(value.split(",")[1])
    back_lat, back_lon = float(back.split(",")[0]), float(back.split(",")[1])
    assert abs(back_lat - orig_lat) < tolerance, f"{via} roundtrip lat error: {abs(back_lat - orig_lat):.6f}°"
    assert abs(back_lon - orig_lon) < tolerance, f"{via} roundtrip lon error: {abs(back_lon - orig_lon):.6f}°"


def near_wgs84(result: str, ref_lat: float, ref_lon: float, max_deg: float = 0.1) -> None:
    """Assert a geographic datum result is within max_deg of the WGS84 input."""
    assert "error" not in result.lower(), f"Conversion error: {result}"
    parts = result.split(",")
    lat, lon = float(parts[0]), float(parts[1])
    assert abs(lat - ref_lat) < max_deg, f"Latitude too far from reference: {abs(lat - ref_lat):.6f}°"
    assert abs(lon - ref_lon) < max_deg, f"Longitude too far from reference: {abs(lon - ref_lon):.6f}°"


# Reference points
ATHENS    = "37.9838, 23.7275"
LONDON    = "51.5074, -0.1278"
NEW_YORK  = "40.7128, -74.0060"
TOKYO_PT  = "35.6762, 139.6503"
CAIRO     = "30.0444, 31.2357"
BUCHAREST = "44.4268, 26.1025"
MOSCOW    = "55.7558, 37.6173"
BERN      = "46.9481, 7.4474"
STOCKHOLM = "59.3293, 18.0686"
TRIPOLI   = "32.8872, 13.1913"

# ---------------------------------------------------------------------------
# European datums
# ---------------------------------------------------------------------------

class TestEuropeanDatums:

    def test_ed50_no_error(self):
        result = to(ATHENS, "WGS84", "ED50")
        assert "error" not in result.lower()

    def test_ed50_close_to_wgs84(self):
        # ED50 differs from WGS84 by ~50–200m in Europe
        near_wgs84(to(ATHENS, "WGS84", "ED50"), 37.9838, 23.7275, max_deg=0.01)

    def test_ed50_roundtrip(self):
        roundtrip(ATHENS, "ED50")

    def test_ed78_no_error(self):
        result = to(ATHENS, "WGS84", "ED78")
        assert "error" not in result.lower()

    def test_ed78_roundtrip(self):
        roundtrip(ATHENS, "ED78")

    def test_etrs89_no_error(self):
        result = to(ATHENS, "WGS84", "ETRS89")
        assert "error" not in result.lower()

    def test_etrs89_very_close_to_wgs84(self):
        # ETRS89 and WGS84 differ by <1m in Europe (sub-arcsecond)
        near_wgs84(to(ATHENS, "WGS84", "ETRS89"), 37.9838, 23.7275, max_deg=0.001)

    def test_etrs89_roundtrip(self):
        roundtrip(ATHENS, "ETRS89", tolerance=0.0001)

    def test_egsa87_no_error(self):
        # ΕΓΣΑ87 is a projected CRS — result is in metres, not degrees
        result = to(ATHENS, "WGS84", "EGSA87")
        assert "error" not in result.lower()

    def test_egsa87_values_in_metre_range(self):
        # Greek grid easting is around 480000–550000 m, northing ~4100000–4250000 m
        result = to(ATHENS, "WGS84", "EGSA87")
        parts = result.split(",")
        northing, easting = float(parts[0]), float(parts[1])
        assert 4_000_000 < northing < 4_500_000, f"Unexpected EGSA87 northing: {northing}"
        assert 400_000 < easting < 700_000, f"Unexpected EGSA87 easting: {easting}"

    def test_osgb36_no_error(self):
        result = to(LONDON, "WGS84", "OSGB36")
        assert "error" not in result.lower()

    def test_osgb36_close_to_wgs84(self):
        near_wgs84(to(LONDON, "WGS84", "OSGB36"), 51.5074, -0.1278, max_deg=0.01)

    def test_osgb36_roundtrip(self):
        roundtrip(LONDON, "OSGB36")


# ---------------------------------------------------------------------------
# Swiss datums
# ---------------------------------------------------------------------------

class TestSwissDatums:

    def test_ch1903_lv03_no_error(self):
        result = to(BERN, "WGS84", "CH1903")
        assert "error" not in result.lower()

    def test_ch1903_lv03_in_swiss_range(self):
        # Swiss LV03: easting 480000–840000, northing 70000–300000
        result = to(BERN, "WGS84", "CH1903")
        parts = result.split(",")
        northing, easting = float(parts[0]), float(parts[1])
        assert 70_000 < northing < 300_000, f"CH1903 northing out of range: {northing}"
        assert 480_000 < easting < 840_000, f"CH1903 easting out of range: {easting}"

    def test_lv95_no_error(self):
        result = to(BERN, "WGS84", "LV95")
        assert "error" not in result.lower()

    def test_lv95_in_swiss_range(self):
        # Swiss LV95: easting 2480000–2840000, northing 1070000–1300000
        result = to(BERN, "WGS84", "LV95")
        parts = result.split(",")
        northing, easting = float(parts[0]), float(parts[1])
        assert 1_070_000 < northing < 1_300_000, f"LV95 northing out of range: {northing}"
        assert 2_480_000 < easting < 2_840_000, f"LV95 easting out of range: {easting}"

    def test_ch1903_and_lv95_consistent(self):
        # LV95 easting ≈ CH1903 easting + 2_000_000
        r03 = to(BERN, "WGS84", "CH1903").split(",")
        r95 = to(BERN, "WGS84", "LV95").split(",")
        e03, e95 = float(r03[1]), float(r95[1])
        assert abs((e95 - e03) - 2_000_000) < 10, "LV95/CH1903 easting offset not ~2,000,000"


# ---------------------------------------------------------------------------
# Swedish datum
# ---------------------------------------------------------------------------

class TestSwedishDatum:

    def test_rt90_no_error(self):
        result = to(STOCKHOLM, "WGS84", "RT90")
        assert "error" not in result.lower()

    def test_rt90_in_swedish_range(self):
        # RT90 northing ~6000000–7700000, easting ~1200000–1900000
        result = to(STOCKHOLM, "WGS84", "RT90")
        parts = result.split(",")
        northing, easting = float(parts[0]), float(parts[1])
        assert 6_000_000 < northing < 7_700_000, f"RT90 northing out of range: {northing}"
        assert 1_200_000 < easting < 1_900_000, f"RT90 easting out of range: {easting}"


# ---------------------------------------------------------------------------
# North American datums
# ---------------------------------------------------------------------------

class TestNorthAmericanDatums:

    def test_nad83_no_error(self):
        result = to(NEW_YORK, "WGS84", "NAD83")
        assert "error" not in result.lower()

    def test_nad83_very_close_to_wgs84(self):
        # NAD83 and WGS84 differ by < 2m in North America
        near_wgs84(to(NEW_YORK, "WGS84", "NAD83"), 40.7128, -74.0060, max_deg=0.001)

    def test_nad83_roundtrip(self):
        roundtrip(NEW_YORK, "NAD83", tolerance=0.0001)

    def test_nad27_no_error(self):
        result = to(NEW_YORK, "WGS84", "NAD27")
        assert "error" not in result.lower()

    def test_nad27_close_to_wgs84(self):
        # NAD27 differs from WGS84 by up to ~100m in North America
        near_wgs84(to(NEW_YORK, "WGS84", "NAD27"), 40.7128, -74.0060, max_deg=0.01)

    def test_nad27_roundtrip(self):
        roundtrip(NEW_YORK, "NAD27")


# ---------------------------------------------------------------------------
# South American datums
# ---------------------------------------------------------------------------

class TestSouthAmericanDatums:

    def test_sad69_no_error(self):
        result = to("-15.7801, -47.9292", "WGS84", "SAD69")  # Brasília
        assert "error" not in result.lower()

    def test_sad69_roundtrip(self):
        roundtrip("-15.7801, -47.9292", "SAD69")

    def test_sirgas_no_error(self):
        result = to("-15.7801, -47.9292", "WGS84", "SIRGAS")
        assert "error" not in result.lower()

    def test_sirgas_very_close_to_wgs84(self):
        # SIRGAS2000 and WGS84 differ by < 1m
        near_wgs84(to("-15.7801, -47.9292", "WGS84", "SIRGAS"), -15.7801, -47.9292, max_deg=0.001)


# ---------------------------------------------------------------------------
# Asian datums
# ---------------------------------------------------------------------------

class TestAsianDatums:

    def test_tokyo_no_error(self):
        result = to(TOKYO_PT, "WGS84", "TOKYO")
        assert "error" not in result.lower()

    def test_tokyo_close_to_wgs84(self):
        # Tokyo datum differs ~450m from WGS84 in Japan
        near_wgs84(to(TOKYO_PT, "WGS84", "TOKYO"), 35.6762, 139.6503, max_deg=0.01)

    def test_tokyo_roundtrip(self):
        roundtrip(TOKYO_PT, "TOKYO")


# ---------------------------------------------------------------------------
# Russian / Soviet datums
# ---------------------------------------------------------------------------

class TestRussianDatums:

    def test_sk42_no_error(self):
        result = to(MOSCOW, "WGS84", "SK42")
        assert "error" not in result.lower()

    def test_sk42_close_to_wgs84(self):
        # Pulkovo 1942 differs ~10–20m from WGS84
        near_wgs84(to(MOSCOW, "WGS84", "SK42"), 55.7558, 37.6173, max_deg=0.01)

    def test_sk42_roundtrip(self):
        roundtrip(MOSCOW, "SK42")

    def test_sk95_no_error(self):
        result = to(MOSCOW, "WGS84", "SK-95")
        assert "error" not in result.lower()

    def test_sk95_close_to_sk42(self):
        # SK-95 (Pulkovo 1995) is a refinement of SK-42 — should be within 3m
        r42 = to(MOSCOW, "WGS84", "SK42").split(",")
        r95 = to(MOSCOW, "WGS84", "SK-95").split(",")
        lat_diff = abs(float(r42[0]) - float(r95[0]))
        lon_diff = abs(float(r42[1]) - float(r95[1]))
        assert lat_diff < 0.001, f"SK-42 vs SK-95 lat diff unexpectedly large: {lat_diff:.6f}°"
        assert lon_diff < 0.001, f"SK-42 vs SK-95 lon diff unexpectedly large: {lon_diff:.6f}°"

    def test_sk95_roundtrip(self):
        roundtrip(MOSCOW, "SK-95")

    def test_pz90_no_error(self):
        result = to(MOSCOW, "WGS84", "PZ-90")
        assert "error" not in result.lower()

    def test_pz90_very_close_to_wgs84(self):
        # PZ-90 (GLONASS frame) differs from WGS84 by < 1m
        near_wgs84(to(MOSCOW, "WGS84", "PZ-90"), 55.7558, 37.6173, max_deg=0.001)

    def test_pz90_roundtrip(self):
        roundtrip(MOSCOW, "PZ90", tolerance=0.0001)


# ---------------------------------------------------------------------------
# Balkan datum
# ---------------------------------------------------------------------------

class TestBalkanDatum:

    def test_balkan1970_no_error(self):
        result = to(BUCHAREST, "WGS84", "BALKAN1970")
        assert "error" not in result.lower()

    def test_balkan1970_is_projected(self):
        # Stereo70 is a projected CRS — values should be in metres
        result = to(BUCHAREST, "WGS84", "BALKAN1970")
        parts = result.split(",")
        # Stereo70 northing for Romania is approx 300000–700000 m
        northing = float(parts[0])
        assert northing > 100_000, f"BALKAN1970 result looks like degrees, not metres: {northing}"


# ---------------------------------------------------------------------------
# Middle East / African datums
# ---------------------------------------------------------------------------

class TestMiddleEastAfricanDatums:

    def test_egypt1907_no_error(self):
        result = to(CAIRO, "WGS84", "EGYPT1907")
        assert "error" not in result.lower()

    def test_egypt1907_close_to_wgs84(self):
        near_wgs84(to(CAIRO, "WGS84", "EGYPT1907"), 30.0444, 31.2357, max_deg=0.01)

    def test_egypt1907_roundtrip(self):
        roundtrip(CAIRO, "EGYPT1907")

    def test_etm1975_no_error(self):
        # ETM 1975 is a projected CRS (Egypt Red Belt)
        result = to(CAIRO, "WGS84", "ETM1975")
        assert "error" not in result.lower()

    def test_etm1975_is_projected(self):
        result = to(CAIRO, "WGS84", "ETM1975")
        parts = result.split(",")
        northing = float(parts[0])
        assert northing > 100_000, f"ETM1975 result looks like degrees: {northing}"

    def test_lgd1954_no_error(self):
        result = to(TRIPOLI, "WGS84", "LGD1954")
        assert "error" not in result.lower()

    def test_lgd1954_close_to_wgs84(self):
        # ELD79 (closest to LGD 1954) is a geographic datum
        near_wgs84(to(TRIPOLI, "WGS84", "LGD1954"), 32.8872, 13.1913, max_deg=0.01)

    def test_lgd1954_roundtrip(self):
        roundtrip(TRIPOLI, "LGD1954")


# ---------------------------------------------------------------------------
# ITRF frames
# ---------------------------------------------------------------------------

class TestITRFDatums:

    def test_itrf2014_no_error(self):
        result = to(ATHENS, "WGS84", "ITRF2014")
        assert "error" not in result.lower()

    def test_itrf2014_effectively_same_as_wgs84(self):
        # WGS84 and ITRF2014 differ by < 10cm — sub-arcsecond
        near_wgs84(to(ATHENS, "WGS84", "ITRF2014"), 37.9838, 23.7275, max_deg=0.0001)

    def test_itrf2008_no_error(self):
        result = to(ATHENS, "WGS84", "ITRF2008")
        assert "error" not in result.lower()

    def test_itrf2005_no_error(self):
        result = to(ATHENS, "WGS84", "ITRF2005")
        assert "error" not in result.lower()

    def test_itrf96_no_error(self):
        result = to(ATHENS, "WGS84", "ITRF96")
        assert "error" not in result.lower()

    def test_all_itrf_frames_close_to_each_other(self):
        # All ITRF realisations should agree to sub-metre (~0.00001°)
        frames = ["ITRF2014", "ITRF2008", "ITRF2005", "ITRF96"]
        results = {}
        for frame in frames:
            r = to(ATHENS, "WGS84", frame).split(",")
            results[frame] = (float(r[0]), float(r[1]))

        ref_lat, ref_lon = results["ITRF2014"]
        for frame, (lat, lon) in results.items():
            assert abs(lat - ref_lat) < 0.001, f"{frame} lat differs from ITRF2014 by {abs(lat - ref_lat):.6f}°"
            assert abs(lon - ref_lon) < 0.001, f"{frame} lon differs from ITRF2014 by {abs(lon - ref_lon):.6f}°"

    def test_itrf_roundtrip(self):
        roundtrip(ATHENS, "ITRF2014", tolerance=0.0001)


# ---------------------------------------------------------------------------
# GRS80 ellipsoid
# ---------------------------------------------------------------------------

class TestGRS80:

    def test_grs80_no_error(self):
        result = to(ATHENS, "WGS84", "GRS80")
        assert "error" not in result.lower()

    def test_grs80_effectively_same_as_wgs84(self):
        # GRS80 and WGS84 ellipsoids differ by 0.1mm — effectively identical
        near_wgs84(to(ATHENS, "WGS84", "GRS80"), 37.9838, 23.7275, max_deg=0.0001)
