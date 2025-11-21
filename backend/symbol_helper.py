"""
NATO Military Symbology Helper Module

Provides symbol recognition support and doctrine query enhancement
using the military-symbol library for NATO APP-6(E) standards.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import military-symbol library
try:
    from military_symbol import MilitarySymbol
    SYMBOL_LIB_AVAILABLE = True
    logger.info("Military symbol library loaded successfully")
except ImportError:
    SYMBOL_LIB_AVAILABLE = False
    logger.warning("Military symbol library not available - using text-only mode")


class SymbolReference:
    """Helper class for NATO symbol recognition and doctrine matching"""

    def __init__(self):
        self.symbol_cache = {}

        # Common unit types mapped to NATO terminology
        # This helps LLaVA recognize symbols and query relevant doctrine
        self.unit_types = {
            # Ground Forces - Infantry
            "infantry_section": "Friendly infantry section",
            "infantry_platoon": "Friendly infantry platoon",
            "infantry_company": "Friendly infantry company",
            "infantry_battalion": "Friendly infantry battalion",

            # Ground Forces - Armor
            "armor_platoon": "Friendly armor platoon",
            "armor_company": "Friendly armor company",
            "armor_battalion": "Friendly armor battalion",

            # Ground Forces - Artillery
            "artillery_battery": "Friendly artillery battery",
            "artillery_battalion": "Friendly artillery battalion",
            "mortar_section": "Friendly mortar section",

            # Ground Forces - Reconnaissance
            "recon_team": "Friendly reconnaissance team",
            "recon_platoon": "Friendly reconnaissance platoon",

            # Ground Forces - Support
            "engineer_platoon": "Friendly engineer platoon",
            "medical_company": "Friendly medical company",
            "logistics_company": "Friendly logistics company",

            # Headquarters
            "company_hq": "Friendly company headquarters",
            "battalion_hq": "Friendly battalion headquarters",
            "brigade_hq": "Friendly brigade headquarters",

            # Enemy equivalents
            "enemy_infantry_platoon": "Enemy infantry platoon",
            "enemy_armor_company": "Enemy armor company",
            "enemy_artillery_battery": "Enemy artillery battery",
        }

        # Doctrine keywords for each unit type
        self.doctrine_keywords = {
            "infantry": ["infantry tactics", "dismounted operations", "close combat", "urban warfare"],
            "armor": ["armor tactics", "armored operations", "tank deployment", "mounted maneuver"],
            "artillery": ["fire support", "indirect fire", "artillery coordination", "fire support planning"],
            "reconnaissance": ["reconnaissance operations", "surveillance", "screen", "guard"],
            "engineer": ["engineer operations", "obstacle breach", "mobility", "counter-mobility"],
            "headquarters": ["command and control", "C2", "command post operations"],
        }

    def get_symbol_reference_guide(self) -> str:
        """Generate a comprehensive symbol reference guide for LLaVA"""

        guide = """
NATO APP-6(E) SYMBOL RECOGNITION QUICK REFERENCE:

═══════════════════════════════════════════════════════════════════
AFFILIATION (Frame Shape & Color):
═══════════════════════════════════════════════════════════════════
• FRIENDLY (Blue): Rectangle frame
• ENEMY/HOSTILE (Red): Diamond frame
• NEUTRAL (Green): Square frame
• UNKNOWN (Yellow): Clover/quatrefoil frame

═══════════════════════════════════════════════════════════════════
ECHELON INDICATORS (Above symbol):
═══════════════════════════════════════════════════════════════════
• Team/Crew: ○ (circle)
• Squad: ● (filled circle)
• Section: ●● (two circles)
• Platoon: ●●● (three circles) or "II"
• Company: I (single bar)
• Battalion: II (two bars)
• Brigade: III (three bars) or "X"
• Division: XX

═══════════════════════════════════════════════════════════════════
COMMON UNIT TYPE ICONS (Center of frame):
═══════════════════════════════════════════════════════════════════
Infantry: Crossed rifles (X)
Armor/Tank: Oval or circle with cross
Artillery: Circle with dot in center
Reconnaissance: Diagonal slash
Engineer: Castle/battlement symbol
Medical: Red cross or crescent
Logistics: Four dots in square pattern
Headquarters: Flag or pennant
Aviation: Propeller or wings
Air Defense: Missile pointing up

═══════════════════════════════════════════════════════════════════
MOBILITY INDICATORS (Below center):
═══════════════════════════════════════════════════════════════════
Wheeled: ○ (circle)
Tracked: ⊙ (circle with dot)
Airborne: ↑ (arrow up)
Air Assault: ⌃ (chevron)
Amphibious: Wave symbol

═══════════════════════════════════════════════════════════════════
KEY TACTICAL GRAPHICS:
═══════════════════════════════════════════════════════════════════
Objective: Circle or triangle with "OBJ" + name
Phase Line: Dashed line perpendicular to axis
Axis of Advance: Arrow with one line
Direction of Attack: Solid arrow
Assembly Area: Rectangle or circle
Engagement Area: Shaded polygon
Boundary: Alternating dashes and dots
"""
        return guide

    def extract_unit_categories(self, llava_analysis: str) -> List[str]:
        """Extract unit categories from LLaVA's analysis for doctrine queries"""

        categories = set()
        analysis_lower = llava_analysis.lower()

        # Check for unit types in the analysis text
        if any(term in analysis_lower for term in ["infantry", "infantr", "riflemen", "dismounted"]):
            categories.add("infantry")

        if any(term in analysis_lower for term in ["armor", "tank", "armored", "armour"]):
            categories.add("armor")

        if any(term in analysis_lower for term in ["artillery", "fire support", "howitzer", "guns", "mortar"]):
            categories.add("artillery")

        if any(term in analysis_lower for term in ["reconnaissance", "recon", "scout", "surveillance"]):
            categories.add("reconnaissance")

        if any(term in analysis_lower for term in ["engineer", "breach", "obstacle"]):
            categories.add("engineer")

        if any(term in analysis_lower for term in ["headquarters", "command post", "hq", "c2"]):
            categories.add("headquarters")

        return list(categories)

    def generate_enhanced_doctrine_query(self, scenario: str, unit_categories: List[str]) -> str:
        """Create doctrine query based on identified unit types"""

        # Base query
        query_parts = [f"How does NATO doctrine address {scenario}?"]

        # Add unit-specific queries
        for category in unit_categories:
            if category in self.doctrine_keywords:
                keywords = " ".join(self.doctrine_keywords[category])
                query_parts.append(f"What are the doctrinal principles for {keywords}?")

        # Add general tactical principles
        query_parts.append(
            "What are the tactical principles for terrain analysis, force deployment, "
            "offensive and defensive operations, intelligence preparation of the battlefield, "
            "and military decision-making processes?"
        )

        return " ".join(query_parts)

    def get_common_symbols_description(self) -> str:
        """Get descriptions of common NATO symbols for context"""

        descriptions = []

        # Sample a few key symbols for LLaVA context
        key_symbols = [
            ("infantry_platoon", "infantry platoon (3 soldiers icon, platoon echelon)"),
            ("armor_company", "armor company (tank icon, company echelon bar)"),
            ("artillery_battery", "artillery battery (circle with dot, battery size)"),
            ("recon_team", "reconnaissance team (diagonal slash, small unit)"),
            ("battalion_hq", "battalion headquarters (flag symbol, II bars for battalion)"),
        ]

        for symbol_key, description in key_symbols:
            descriptions.append(f"- {description}")

        return "\n".join(descriptions)


# Global instance
symbol_helper = SymbolReference()


def get_symbol_helper() -> SymbolReference:
    """Get the global symbol helper instance"""
    return symbol_helper
