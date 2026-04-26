"""
File parsing tools for user-supplied geospatial files.

Uses only Python stdlib (xml.etree.ElementTree, zipfile) — no extra dependencies.

Optional LLM-callable tools:
  parse_kml_file()    — extract waypoints/features from a KML or KMZ file path
  parse_kml_string()  — extract waypoints/features from a KML string (pasted content)
"""

import logging
import os
import xml.etree.ElementTree as ET
import zipfile

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# KML namespace
_KML_NS = "http://www.opengis.net/kml/2.2"


@tool
def parse_kml_file(filepath: str) -> str:
    """Parse a KML or KMZ file and return all placemarks (waypoints/features).

    Extracts name, description, and coordinates for each feature.
    KMZ files (zipped KML) are handled automatically.

    Example: parse_kml_file('/path/to/waypoints.kmz')
    """
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"

    try:
        if filepath.lower().endswith(".kmz"):
            kml_text = _extract_kml_from_kmz(filepath)
        else:
            with open(filepath, encoding="utf-8") as f:
                kml_text = f.read()

        return _parse_kml_text(kml_text)

    except Exception as e:
        return f"Failed to parse file: {e}"


@tool
def parse_kml_string(kml_content: str) -> str:
    """Parse a KML string and return all placemarks (waypoints/features).

    Use this when the user pastes KML content directly into the chat.

    Example: parse_kml_string('<kml>...</kml>')
    """
    try:
        return _parse_kml_text(kml_content)
    except Exception as e:
        return f"Failed to parse KML: {e}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_kml_from_kmz(filepath: str) -> str:
    """Unzip a KMZ and return the content of the first .kml file inside."""
    with zipfile.ZipFile(filepath, "r") as z:
        kml_files = [name for name in z.namelist() if name.lower().endswith(".kml")]
        if not kml_files:
            raise ValueError("No .kml file found inside the .kmz archive.")
        with z.open(kml_files[0]) as f:
            return f.read().decode("utf-8")


def _parse_kml_text(kml_text: str) -> str:
    """Parse KML XML and extract all Placemark elements."""
    # Handle both namespaced and non-namespaced KML
    try:
        root = ET.fromstring(kml_text)
    except ET.ParseError as e:
        return f"Invalid XML: {e}"

    # Detect namespace
    ns = _KML_NS if root.tag.startswith(f"{{{_KML_NS}}}") else ""
    tag = lambda name: f"{{{_KML_NS}}}{name}" if ns else name

    placemarks = root.iter(tag("Placemark"))
    results = []

    for pm in placemarks:
        name        = _text(pm, tag("name")) or "Unnamed"
        description = _text(pm, tag("description")) or ""
        coords      = _extract_coordinates(pm, tag)

        entry = [f"Name: {name}"]
        if description:
            entry.append(f"Description: {description[:200]}")
        if coords:
            for c in coords:
                entry.append(f"Coordinates: {c}")
        results.append("\n".join(entry))

    if not results:
        return "No placemarks found in this KML."

    return f"Found {len(results)} placemark(s):\n\n" + "\n\n---\n\n".join(results)


def _text(element: ET.Element, tag: str) -> str | None:
    """Return stripped text of a child element, or None."""
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None


def _extract_coordinates(placemark: ET.Element, tag) -> list[str]:
    """Extract all coordinate strings from a Placemark."""
    coords = []
    for coord_el in placemark.iter(tag("coordinates")):
        if coord_el.text:
            # KML coords are "lon,lat,alt" — convert to "lat, lon [alt]"
            for point in coord_el.text.strip().split():
                parts = point.split(",")
                if len(parts) >= 2:
                    lon, lat = parts[0], parts[1]
                    alt = f"  alt: {parts[2]}m" if len(parts) >= 3 and parts[2] != "0" else ""
                    coords.append(f"{float(lat):.7f}, {float(lon):.7f}{alt}")
    return coords
