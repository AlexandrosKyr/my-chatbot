import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_military_db: dict = {}


def initialize(db_path: Path) -> None:
    """Load the military power JSON database. Call once at startup."""
    global _military_db
    try:
        with open(db_path, encoding="utf-8") as f:
            _military_db = json.load(f)
        logger.info(f"Military power DB loaded: {len(_military_db)} countries")
    except FileNotFoundError:
        logger.warning(f"Military power DB not found at {db_path}")


def lookup_military_power(country_code: str) -> str:
    """Return military power data for a country as a formatted string, or empty string."""
    if not country_code:
        return ""
    data = _military_db.get(country_code.upper())
    if data:
        return json.dumps(data, indent=2)
    return ""
