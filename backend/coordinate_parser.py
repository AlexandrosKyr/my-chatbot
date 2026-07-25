#!/usr/bin/env python3
"""
Coordinate Parser — extract geographic coordinates from natural language text.

Supported input formats:
  - Decimal degrees:          37.9838, 23.7275
  - Labeled decimal:          lat: 37.9838, lon: 23.7275
  - DMS:                      37°59'1.68"N, 23°43'39.9"E
  - DDM (degrees dec. min):   37°59.028'N, 23°43.665'E
  - DMS + milliseconds:       37°59'1680ms N, 23°43'39900ms E
  - Microdegrees (labeled):   lat: 37983800, lon: 23727500 (microdegrees)
  - GEOREF (8/10/12-char):    PJJH4359
  - MGRS:                     34SGH1234567890
  - UTM:                      34 S 123456 4567890
"""

import re
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CoordinateParser:
    """Parse geographic coordinates from natural language text."""

    def __init__(self):
        self.patterns = {
            # 37.9838, 23.7275
            "decimal": r"(-?\d{1,3}\.\d{4,10})\s*[,\s]\s*(-?\d{1,3}\.\d{4,10})",

            # lat: 37.9838, lon: 23.7275
            "decimal_labeled": (
                r"(?:lat|latitude)[:\s]*(-?\d{1,3}\.\d{4,10})"
                r"\s*[,\s]?\s*(?:lon|long|longitude)[:\s]*(-?\d{1,3}\.\d{4,10})"
            ),

            # 37°59'1.68"N, 23°43'39.9"E
            "dms": (
                r"(\d{1,3})[°]\s*(\d{1,2})[\'′]\s*(\d{1,2}(?:\.\d+)?)[\"″]?\s*([NSns])"
                r"\s*[,\s]?\s*"
                r"(\d{1,3})[°]\s*(\d{1,2})[\'′]\s*(\d{1,2}(?:\.\d+)?)[\"″]?\s*([EWew])"
            ),

            # 37°59.028'N, 23°43.665'E  (degrees + decimal minutes)
            "ddm": (
                r"(\d{1,3})[°]\s*(\d{1,2}(?:\.\d+)?)[\'′]\s*([NSns])"
                r"\s*[,\s]?\s*"
                r"(\d{1,3})[°]\s*(\d{1,2}(?:\.\d+)?)[\'′]\s*([EWew])"
            ),

            # 37°59'1680ms N, 23°43'39900ms E  (arc-milliseconds: 1 arc-sec = 1000 ms)
            "dms_ms": (
                r"(\d{1,3})[°]\s*(\d{1,2})[\'′]\s*(\d{1,7})\s*ms\s*([NSns])"
                r"\s*[,\s]?\s*"
                r"(\d{1,3})[°]\s*(\d{1,2})[\'′]\s*(\d{1,7})\s*ms\s*([EWew])"
            ),

            # PJJH4359 / PJJH436590 / PJJH43365928  (8, 10, or 12 chars)
            # Only match with digit suffix — bare 4-letter codes are too ambiguous.
            "georef": (
                r"\b([ABCDEFGHJKLMNPQRSTUVWXYZ][ABCDEFGHJKLM][ABCDEFGHJKLMNPQ]{2})"
                r"(\d{8}|\d{6}|\d{4})\b"
            ),

            # lat: 37983800, lon: 23727500 (microdegrees)
            "microdegrees_labeled": (
                r"(?:lat|latitude)[:\s]*(-?\d{5,9})\s*(?:ud|µ°|microdeg(?:rees?)?)?"
                r"\s*[,\s]?\s*(?:lon|long|longitude)[:\s]*(-?\d{5,9})\s*(?:ud|µ°|microdeg(?:rees?)?)?"
            ),

            # 34SGH1234567890
            "mgrs": r"\b\d{1,2}[A-Z][A-Z]{2}\d{10}\b",

            # 34 S 123456 4567890
            "utm": r"(\d{1,2})\s*([A-Z])\s*(\d{6})\s*(\d{7})",
        }

    def parse(self, text: str) -> Optional[Dict[str, float]]:
        """Return {'lat': float, 'lon': float} or None if no coordinates found."""
        text = text.strip()

        for method in [
            self._parse_decimal_labeled,  # labeled before plain decimal
            self._parse_microdegrees_labeled,
            self._parse_georef,
            self._parse_dms_ms,
            self._parse_decimal,
            self._parse_dms,
            self._parse_ddm,
        ]:
            result = method(text)
            if result:
                logger.info(f"Parsed coordinates via {method.__name__}: {result}")
                return result

        logger.info("No coordinates found in text")
        return None

    # ------------------------------------------------------------------

    def _parse_decimal(self, text: str) -> Optional[Dict[str, float]]:
        match = re.search(self.patterns["decimal"], text)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            if self._valid(lat, lon):
                return {"lat": lat, "lon": lon}
        return None

    def _parse_decimal_labeled(self, text: str) -> Optional[Dict[str, float]]:
        match = re.search(self.patterns["decimal_labeled"], text, re.IGNORECASE)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            if self._valid(lat, lon):
                return {"lat": lat, "lon": lon}
        return None

    def _parse_dms(self, text: str) -> Optional[Dict[str, float]]:
        """Degrees, Minutes, Seconds — 37°59'1.68"N 23°43'39.9"E"""
        match = re.search(self.patterns["dms"], text)
        if match:
            ld, lm, ls, ld_dir, od, om, os_, od_dir = match.groups()
            lat = float(ld) + float(lm) / 60 + float(ls) / 3600
            lon = float(od) + float(om) / 60 + float(os_) / 3600
            if ld_dir.upper() == "S":
                lat = -lat
            if od_dir.upper() == "W":
                lon = -lon
            if self._valid(lat, lon):
                return {"lat": lat, "lon": lon}
        return None

    def _parse_ddm(self, text: str) -> Optional[Dict[str, float]]:
        """Degrees Decimal Minutes — 37°59.028'N 23°43.665'E"""
        match = re.search(self.patterns["ddm"], text)
        if match:
            ld, lm, ld_dir, od, om, od_dir = match.groups()
            lat = float(ld) + float(lm) / 60
            lon = float(od) + float(om) / 60
            if ld_dir.upper() == "S":
                lat = -lat
            if od_dir.upper() == "W":
                lon = -lon
            if self._valid(lat, lon):
                return {"lat": lat, "lon": lon}
        return None

    def _parse_georef(self, text: str) -> Optional[Dict[str, float]]:
        """GEOREF with digit suffix — PJJH4359, PJJH436590, PJJH43365928"""
        _LON_ZONES = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        _LAT_BANDS = "ABCDEFGHJKLM"
        _ONE_DEG   = "ABCDEFGHJKLMNPQ"

        match = re.search(self.patterns["georef"], text)
        if not match:
            return None
        letters, digits = match.group(1), match.group(2)
        try:
            lon_zone_idx = _LON_ZONES.index(letters[0])
            lat_band_idx = _LAT_BANDS.index(letters[1])
            lon_1deg_idx = _ONE_DEG.index(letters[2])
            lat_1deg_idx = _ONE_DEG.index(letters[3])
        except ValueError:
            return None

        lon = lon_zone_idx * 15 - 180 + lon_1deg_idx
        lat = lat_band_idx * 15 - 90  + lat_1deg_idx

        n = len(digits) // 2
        lon_int = int(digits[:n])
        lat_int = int(digits[n:])
        divisor = 60 * (10 ** (n - 2))
        lon += lon_int / divisor
        lat += lat_int / divisor

        # Return centre of cell, not SW corner
        cell_size_deg = 1 / (60 * (10 ** (n - 2)))
        lon += cell_size_deg / 2
        lat += cell_size_deg / 2

        if self._valid(lat, lon):
            return {"lat": lat, "lon": lon}
        return None

    def _parse_dms_ms(self, text: str) -> Optional[Dict[str, float]]:
        """Degrees, Minutes, arc-Milliseconds — 37°59'1680ms N 23°43'39900ms E"""
        match = re.search(self.patterns["dms_ms"], text, re.IGNORECASE)
        if match:
            ld, lm, lms, ld_dir, od, om, oms, od_dir = match.groups()
            lat = float(ld) + float(lm) / 60 + float(lms) / 3_600_000
            lon = float(od) + float(om) / 60 + float(oms) / 3_600_000
            if ld_dir.upper() == "S":
                lat = -lat
            if od_dir.upper() == "W":
                lon = -lon
            if self._valid(lat, lon):
                return {"lat": lat, "lon": lon}
        return None

    def _parse_microdegrees_labeled(self, text: str) -> Optional[Dict[str, float]]:
        """Labeled microdegrees — lat: 37983800, lon: 23727500 (microdegrees)"""
        match = re.search(self.patterns["microdegrees_labeled"], text, re.IGNORECASE)
        if match:
            lat = int(match.group(1)) / 1_000_000
            lon = int(match.group(2)) / 1_000_000
            if self._valid(lat, lon):
                return {"lat": lat, "lon": lon}
        return None

    def _valid(self, lat: float, lon: float) -> bool:
        if not (-90 <= lat <= 90):
            logger.warning(f"Invalid latitude: {lat}")
            return False
        if not (-180 <= lon <= 180):
            logger.warning(f"Invalid longitude: {lon}")
            return False
        return True

    def format_coordinates(self, lat: float, lon: float) -> str:
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        return f"{abs(lat):.6f}°{lat_dir}, {abs(lon):.6f}°{lon_dir}"


def extract_coordinates(text: str) -> Optional[Dict[str, float]]:
    """Convenience wrapper around CoordinateParser.parse()."""
    return CoordinateParser().parse(text)
