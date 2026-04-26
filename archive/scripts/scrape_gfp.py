#!/usr/bin/env python3
"""
Global Firepower Scraper
Scrapes military strength data from globalfirepower.com and saves as
a structured JSON keyed by ISO alpha-2 country code (matching the
country_code field returned by terrain_data_fetcher.py's reverse geocoder).

Output: db/military_power_data.json

HTML structure (verified against live site):
  - All data lives in .specsGenContainers divs
  - Equipment format: "Label: Stock: X,XXX Readiness: Y,YYY*"
  - Simple count format: "Label: N"
  - Manpower format: "Label N (XX.X%)" or "Label: N"
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pycountry
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.globalfirepower.com"
LISTING_URL = f"{BASE_URL}/countries-listing.php"
DETAIL_URL_TPL = f"{BASE_URL}/country-military-strength-detail.php?country_id={{slug}}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

REQUEST_DELAY = 2.5  # seconds between country requests (be respectful)

OUTPUT_PATH = Path(__file__).parent.parent.parent / "db" / "military_power_data.json"
PROMPT_DB_PATH = Path(__file__).parent.parent.parent / "db" / "military_power_prompt.json"

# ---------------------------------------------------------------------------
# Manual ISO alpha-2 overrides for slugs/names pycountry can't resolve
# ---------------------------------------------------------------------------

SLUG_TO_ALPHA2: dict[str, str] = {
    "russia": "RU",
    "south-korea": "KR",
    "north-korea": "KP",
    "iran": "IR",
    "taiwan": "TW",
    "bolivia": "BO",
    "venezuela": "VE",
    "vietnam": "VN",
    "laos": "LA",
    "syria": "SY",
    "tanzania": "TZ",
    "moldova": "MD",
    "congo": "CG",
    "democratic-republic-of-the-congo": "CD",
    "ivory-coast": "CI",
    "cote-divoire": "CI",
    "south-sudan": "SS",
    "trinidad-and-tobago": "TT",
    "united-arab-emirates": "AE",
    "uae": "AE",
    "saudi-arabia": "SA",
    "united-kingdom": "GB",
    "czech-republic": "CZ",
    "czechia": "CZ",
    "slovakia": "SK",
    "brunei": "BN",
    "myanmar": "MM",
    "burma": "MM",
    "timor-leste": "TL",
    "east-timor": "TL",
    "cabo-verde": "CV",
    "cape-verde": "CV",
    "eswatini": "SZ",
    "swaziland": "SZ",
    "north-macedonia": "MK",
    "macedonia": "MK",
    "kyrgyzstan": "KG",
    "turkmenistan": "TM",
    "uzbekistan": "UZ",
    "tajikistan": "TJ",
    "azerbaijan": "AZ",
    "nigeria": "NG",
}

NAME_TO_ALPHA2: dict[str, str] = {
    "russia": "RU",
    "south korea": "KR",
    "north korea": "KP",
    "iran": "IR",
    "taiwan": "TW",
    "bolivia": "BO",
    "venezuela": "VE",
    "vietnam": "VN",
    "laos": "LA",
    "syria": "SY",
    "tanzania": "TZ",
    "moldova": "MD",
    "congo": "CG",
    "ivory coast": "CI",
    "cote d'ivoire": "CI",
    "dr congo": "CD",
    "democratic republic of the congo": "CD",
    "south sudan": "SS",
    "trinidad and tobago": "TT",
    "united arab emirates": "AE",
    "saudi arabia": "SA",
    "united kingdom": "GB",
    "czech republic": "CZ",
    "czechia": "CZ",
    "brunei": "BN",
    "myanmar": "MM",
    "burma": "MM",
    "east timor": "TL",
    "cape verde": "CV",
    "cabo verde": "CV",
    "eswatini": "SZ",
    "swaziland": "SZ",
    "north macedonia": "MK",
}


# ---------------------------------------------------------------------------
# ISO alpha-2 resolution
# ---------------------------------------------------------------------------

def resolve_alpha2(slug: str, display_name: str, short_code: str) -> str | None:
    """
    Resolve a GFP country to an ISO alpha-2 code.

    Resolution order:
      1. Slug override dict
      2. pycountry alpha-3 lookup (shortFormName from listing, e.g. "USA")
      3. pycountry exact name lookup
      4. pycountry common_name lookup
      5. pycountry fuzzy lookup
      6. Display name override dict
    """
    # 1. Slug override
    if slug in SLUG_TO_ALPHA2:
        return SLUG_TO_ALPHA2[slug]

    # 2. Alpha-3 lookup (most reliable when available)
    if short_code and len(short_code) == 3:
        country = pycountry.countries.get(alpha_3=short_code.upper())
        if country:
            return country.alpha_2

    # 3. Exact name
    country = pycountry.countries.get(name=display_name)
    if country:
        return country.alpha_2

    # 4. Common name
    for c in pycountry.countries:
        if hasattr(c, "common_name") and c.common_name.lower() == display_name.lower():
            return c.alpha_2

    # 5. Fuzzy
    try:
        results = pycountry.countries.search_fuzzy(display_name)
        if results:
            return results[0].alpha_2
    except LookupError:
        pass

    # 6. Display name override
    key = display_name.lower().strip()
    if key in NAME_TO_ALPHA2:
        return NAME_TO_ALPHA2[key]

    return None


# ---------------------------------------------------------------------------
# Number parsing helpers
# ---------------------------------------------------------------------------

def _first_int(text: str) -> int | None:
    """Extract the first integer from a string (ignores commas and %)."""
    cleaned = text.replace(",", "")
    match = re.search(r"\d+", cleaned)
    return int(match.group()) if match else None


def _parse_stock_readiness(text: str) -> dict:
    """
    Parse equipment count text.

    Handles formats seen on GFP:
      "Tanks: Stock: 4,666 Readiness: 3,500*"  → {stock: 4666, readiness: 3500}
      "Aircraft Carriers: 11"                  → {stock: 11, readiness: None}
      "Total Assets: 465"                       → {stock: 465, readiness: None}
    """
    # Stock: X Readiness: Y pattern
    stock_match = re.search(r"[Ss]tock:\s*([\d,]+)", text)
    ready_match = re.search(r"[Rr]eadiness:\s*([\d,]+)", text)
    if stock_match:
        stock = int(stock_match.group(1).replace(",", ""))
        readiness = int(ready_match.group(1).replace(",", "")) if ready_match else None
        return {"stock": stock, "readiness": readiness}

    # Plain number after colon: "Label: N"
    colon_match = re.search(r":\s*([\d,]+)", text)
    if colon_match:
        return {"stock": int(colon_match.group(1).replace(",", "")), "readiness": None}

    # Fallback: first number in string
    n = _first_int(text)
    return {"stock": n, "readiness": None}


def _parse_manpower(text: str) -> int | None:
    """
    Parse manpower text.

    Handles:
      "Total Population: 341,963,408"
      "Available Manpower 150,463,900 (44.0%)"
      "Active Personnel 1,333,030 (0.4%)"
    """
    # Strip everything after a "(" to avoid grabbing percentages
    text = re.sub(r"\(.*", "", text)
    return _first_int(text)


# ---------------------------------------------------------------------------
# Container text lookup
# ---------------------------------------------------------------------------

def _find_gen(containers: list, keywords: list[str]) -> dict:
    """
    Find the first .specsGenContainers element whose text contains any keyword.
    Returns parsed stock/readiness dict.
    """
    kw_lower = [k.lower() for k in keywords]
    for el in containers:
        text = el.get_text(" ", strip=True)
        text_lower = text.lower()
        if any(kw in text_lower for kw in kw_lower):
            return _parse_stock_readiness(text)
    return {"stock": None, "readiness": None}


def _find_manpower(containers: list, keywords: list[str]) -> int | None:
    """Find manpower count for given keywords."""
    kw_lower = [k.lower() for k in keywords]
    for el in containers:
        text = el.get_text(" ", strip=True)
        text_lower = text.lower()
        if any(kw in text_lower for kw in kw_lower):
            return _parse_manpower(text)
    return None


# ---------------------------------------------------------------------------
# Country listing
# ---------------------------------------------------------------------------

def fetch_country_list(session: requests.Session) -> list[dict]:
    """
    Scrape the GFP country listing page.

    The <a> tag wraps each .recordsetContainer, so we walk up to the parent
    to find the href.

    Returns list of dicts: {slug, display_name, short_code, rank, power_index}
    """
    logger.info("Fetching country list from %s ...", LISTING_URL)
    resp = session.get(LISTING_URL, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    countries = []

    for container in soup.select(".recordsetContainer"):
        try:
            # The <a href> is the parent of the container div
            parent = container.parent
            href = parent.get("href", "") if parent else ""
            slug_match = re.search(r"country_id=([^&\s]+)", href)
            if not slug_match:
                continue
            slug = slug_match.group(1)

            name_el = container.select_one(".longFormName")
            display_name = name_el.get_text(strip=True) if name_el else slug.replace("-", " ").title()

            short_el = container.select_one(".shortFormName")
            short_code = short_el.get_text(strip=True) if short_el else ""

            rank_el = container.select_one(".rankNumContainer")
            rank = _first_int(rank_el.get_text(strip=True)) if rank_el else None

            pwr_el = container.select_one(".pwrIndxContainer")
            pwr_text = pwr_el.get_text(strip=True) if pwr_el else ""
            pwr_match = re.search(r"(\d+\.\d+)", pwr_text)
            power_index = float(pwr_match.group(1)) if pwr_match else None

            countries.append({
                "slug": slug,
                "display_name": display_name,
                "short_code": short_code,
                "rank": rank,
                "power_index": power_index,
            })

        except Exception as e:
            logger.warning("Error parsing listing container: %s", e)
            continue

    logger.info("Found %d countries in listing", len(countries))
    return countries


# ---------------------------------------------------------------------------
# Country detail page
# ---------------------------------------------------------------------------

def fetch_country_detail(session: requests.Session, slug: str) -> dict | None:
    """
    Fetch and parse the detail page for a country.
    All data lives in .specsGenContainers — no specsFullContainers needed.
    """
    url = DETAIL_URL_TPL.format(slug=slug)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("  HTTP error fetching %s: %s", slug, e)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove rank boxes so they don't leak numbers into text
    for rank_box in soup.select(".specsRankBox"):
        rank_box.decompose()

    containers = soup.select(".specsGenContainers")

    # ---- Manpower ----
    manpower = {
        "total_population":       _find_manpower(containers, ["total population"]),
        "available_manpower":     _find_manpower(containers, ["available manpower"]),
        "fit_for_service":        _find_manpower(containers, ["fit-for-service"]),
        "reaching_military_age":  _find_manpower(containers, ["reaching mil age", "reaching military age"]),
        "total_military":         _find_manpower(containers, ["tot mil. personnel"]),
        "active_personnel":       _find_manpower(containers, ["active personnel"]),
        "reserve_personnel":      _find_manpower(containers, ["reserve personnel"]),
        "paramilitary":           _find_manpower(containers, ["paramilitary"]),
    }

    # ---- Land forces ----
    land = {
        "tanks":                  _find_gen(containers, ["tanks:"]),
        "armored_vehicles":       _find_gen(containers, ["vehicles:"]),
        "self_propelled_artillery": _find_gen(containers, ["self-propelled artillery"]),
        "towed_artillery":        _find_gen(containers, ["towed artillery"]),
        "rocket_projectors":      _find_gen(containers, ["mlrs", "rocket artillery"]),
    }

    # ---- Air power ----
    air = {
        "total_aircraft":         _find_gen(containers, ["aircraft total:"]),
        "fighters":               _find_gen(containers, ["fighters:"]),
        "attack_aircraft":        _find_gen(containers, ["attack types:"]),
        "transport_aircraft":     _find_gen(containers, ["transports (fixed-wing)", "transports:"]),
        "trainer_aircraft":       _find_gen(containers, ["trainers:"]),
        "total_helicopters":      _find_gen(containers, ["helicopters:"]),
        "attack_helicopters":     _find_gen(containers, ["attack helicopters:"]),
    }

    # ---- Naval forces ----
    naval = {
        "total_assets":           _find_gen(containers, ["total assets:"]),
        "aircraft_carriers":      _find_gen(containers, ["aircraft carriers:"]),
        "helicopter_carriers":    _find_gen(containers, ["helicopter carriers:"]),
        "destroyers":             _find_gen(containers, ["destroyers:"]),
        "frigates":               _find_gen(containers, ["frigates:"]),
        "corvettes":              _find_gen(containers, ["corvettes:"]),
        "submarines":             _find_gen(containers, ["submarines:"]),
        "patrol_vessels":         _find_gen(containers, ["patrol vessels:"]),
        "mine_warfare":           _find_gen(containers, ["mine warfare:"]),
    }

    return {
        "manpower": manpower,
        "land": land,
        "air": air,
        "naval": naval,
    }


# ---------------------------------------------------------------------------
# Prompt DB builder
# ---------------------------------------------------------------------------

def build_prompt_db(raw_path: Path, prompt_path: Path) -> None:
    """
    Read the full scraped JSON and write a cleaned-up version optimised for
    prompt injection.

    What changes:
    - Strips: rank, power_index, slug, scraped_at, readiness values
    - Flattens equipment dicts {"stock": N, "readiness": N} → just the integer N
    - Keeps: name, manpower (active, reserve, total_population),
             land (5 fields), air (7 fields), naval (8 fields)
    """
    logger.info("Building prompt DB from %s ...", raw_path)

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    prompt_db = {}

    for alpha2, entry in raw["countries"].items():
        m = entry.get("manpower", {})
        land = entry.get("land", {})
        air = entry.get("air", {})
        naval = entry.get("naval", {})

        # Helper: pull the stock integer out of {"stock": N, "readiness": N}
        def s(d):
            return d.get("stock") if isinstance(d, dict) else d

        prompt_db[alpha2] = {
            "name": entry["display_name"],
            "manpower": {
                "active_personnel":  m.get("active_personnel"),
                "reserve_personnel": m.get("reserve_personnel"),
                "total_population":  m.get("total_population"),
            },
            "land": {
                "tanks":                    s(land.get("tanks", {})),
                "armored_vehicles":         s(land.get("armored_vehicles", {})),
                "self_propelled_artillery": s(land.get("self_propelled_artillery", {})),
                "towed_artillery":          s(land.get("towed_artillery", {})),
                "rocket_projectors":        s(land.get("rocket_projectors", {})),
            },
            "air": {
                "total_aircraft":      s(air.get("total_aircraft", {})),
                "fighters":            s(air.get("fighters", {})),
                "attack_aircraft":     s(air.get("attack_aircraft", {})),
                "transport_aircraft":  s(air.get("transport_aircraft", {})),
                "total_helicopters":   s(air.get("total_helicopters", {})),
                "attack_helicopters":  s(air.get("attack_helicopters", {})),
            },
            "naval": {
                "total_assets":        s(naval.get("total_assets", {})),
                "aircraft_carriers":   s(naval.get("aircraft_carriers", {})),
                "helicopter_carriers": s(naval.get("helicopter_carriers", {})),
                "destroyers":          s(naval.get("destroyers", {})),
                "frigates":            s(naval.get("frigates", {})),
                "corvettes":           s(naval.get("corvettes", {})),
                "submarines":          s(naval.get("submarines", {})),
                "patrol_vessels":      s(naval.get("patrol_vessels", {})),
            },
        }

    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompt_db, f, indent=2, ensure_ascii=False)

    logger.info("Prompt DB saved: %d countries → %s", len(prompt_db), prompt_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    # --- Step 1: Get country list ---
    countries_raw = fetch_country_list(session)

    scrape_date = datetime.now(timezone.utc).isoformat()
    result = {
        "scraper_meta": {
            "source": BASE_URL,
            "scraped_at": scrape_date,
            "total_countries_found": len(countries_raw),
            "total_scraped": 0,
            "total_failed": 0,
            "failed_countries": [],
        },
        "countries": {},
    }

    success_count = 0
    failed_count = 0

    # --- Step 2: Scrape each country ---
    for i, entry in enumerate(countries_raw, start=1):
        slug = entry["slug"]
        display_name = entry["display_name"]
        short_code = entry["short_code"]
        rank = entry["rank"]
        power_index = entry["power_index"]

        alpha2 = resolve_alpha2(slug, display_name, short_code)
        if not alpha2:
            logger.warning(
                "[%d/%d] Cannot resolve ISO alpha-2 for '%s' (slug: %s, short: %s) — skipping",
                i, len(countries_raw), display_name, slug, short_code,
            )
            failed_count += 1
            result["scraper_meta"]["failed_countries"].append({
                "slug": slug,
                "display_name": display_name,
                "short_code": short_code,
                "reason": "could not resolve ISO alpha-2",
            })
            continue

        logger.info("[%d/%d] %s (%s) → %s", i, len(countries_raw), display_name, short_code, alpha2)

        detail = fetch_country_detail(session, slug)
        if detail is None:
            failed_count += 1
            result["scraper_meta"]["failed_countries"].append({
                "slug": slug,
                "display_name": display_name,
                "alpha2": alpha2,
                "reason": "HTTP error on detail page",
            })
            time.sleep(REQUEST_DELAY)
            continue

        result["countries"][alpha2] = {
            "display_name": display_name,
            "slug": slug,
            "rank": rank,
            "power_index": power_index,
            "manpower": detail["manpower"],
            "land": detail["land"],
            "air": detail["air"],
            "naval": detail["naval"],
            "scraped_at": scrape_date,
        }

        success_count += 1
        time.sleep(REQUEST_DELAY)

    # --- Step 3: Finalize and save ---
    result["scraper_meta"]["total_scraped"] = success_count
    result["scraper_meta"]["total_failed"] = failed_count

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # --- Step 4: Build the cleaned prompt DB ---
    build_prompt_db(OUTPUT_PATH, PROMPT_DB_PATH)

    print(f"\n{'='*50}")
    print(f"Scrape complete!")
    print(f"  Successfully scraped : {success_count}")
    print(f"  Failed               : {failed_count}")
    print(f"  Raw data             : {OUTPUT_PATH}")
    print(f"  Prompt DB            : {PROMPT_DB_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
