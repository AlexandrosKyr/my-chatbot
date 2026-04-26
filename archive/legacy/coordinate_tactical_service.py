#!/usr/bin/env python3
"""
DEPRECATED: This service is deprecated. Use RAGService in services.py instead.

RAGService now handles all coordinate-based tactical analysis including:
- Coordinate parsing from user prompts
- Terrain data fetching (elevation, slope, LOS, infrastructure)
- Terrain-enhanced doctrine retrieval
- Scenario detection and IPB prompt generation
- Doctrine citation and tactical analysis

This file is kept for reference only. Will be removed in a future version.

Original Description:
Coordinate-Based Tactical Analysis Service
Uses real geographic data from APIs instead of vision models.
User provides coordinates in their prompt, system fetches terrain data
and applies NATO doctrine for tactical analysis.
"""

import logging
from typing import Dict, Optional
from coordinate_parser import CoordinateParser, extract_coordinates
from terrain_data_fetcher import TerrainDataFetcher
from utils import hybrid_search

logger = logging.getLogger(__name__)


class CoordinateTacticalService:
    """Tactical analysis based on user-provided coordinates"""

    def __init__(self, llm, vectorstore, google_api_key: Optional[str] = None):
        """
        Initialize coordinate-based tactical service

        Args:
            llm: Ollama LLM instance
            vectorstore: ChromaDB vectorstore for doctrine
            google_api_key: Google Maps API key (optional)
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self.coord_parser = CoordinateParser()
        self.terrain_fetcher = TerrainDataFetcher(google_api_key)

    def analyze_from_prompt(self, user_prompt: str, scenario: str = "Tactical analysis") -> Dict:
        """
        Extract coordinates from user prompt and perform tactical analysis

        Args:
            user_prompt: User's message containing coordinates
            scenario: Tactical scenario context

        Returns:
            Dict with analysis results or error
        """
        logger.info(f"Processing tactical query: {user_prompt[:100]}...")

        # 1. Extract coordinates from user text
        coords = self.coord_parser.parse(user_prompt)
        if not coords:
            return {
                'success': False,
                'error': 'No coordinates found in your message. Please provide coordinates in format: "40.7128, -74.0060" or "40°42\'51"N, 74°00\'21"W"'
            }

        lat, lon = coords['lat'], coords['lon']
        logger.info(f"Coordinates extracted: {lat}, {lon}")

        # 2. Determine analysis radius from prompt (default 5km)
        radius_km = self._extract_radius(user_prompt)

        # 3. Fetch real terrain data from APIs
        try:
            terrain_data = self.terrain_fetcher.fetch_terrain_data(lat, lon, radius_km)
        except Exception as e:
            logger.error(f"Terrain fetch failed: {e}")
            return {
                'success': False,
                'error': f'Failed to fetch terrain data: {str(e)}'
            }

        # 4. Retrieve doctrine for tactical analysis
        doctrine_context = self._retrieve_doctrine(user_prompt, terrain_data)

        # 5. Generate tactical analysis using LLM + doctrine + terrain data
        analysis = self._generate_tactical_analysis(
            user_prompt=user_prompt,
            scenario=scenario,
            coords=coords,
            terrain_data=terrain_data,
            doctrine_context=doctrine_context
        )

        return {
            'success': True,
            'coordinates': coords,
            'terrain_data': terrain_data,
            'analysis': analysis,
            'scenario': scenario
        }

    def _extract_radius(self, text: str) -> float:
        """Extract analysis radius from user text (default 5km)"""
        import re
        # Look for patterns like "5km", "10 km", "3 kilometers"
        match = re.search(r'(\d+)\s*(km|kilometers?|miles?)', text, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()
            if 'mile' in unit:
                value = value * 1.609  # Convert miles to km
            # Limit to reasonable range
            return min(max(value, 1), 20)
        return 5  # Default 5km radius

    def _retrieve_doctrine(self, query: str, terrain_data: Dict) -> str:
        """
        Retrieve relevant NATO doctrine based on query and terrain

        Args:
            query: User query
            terrain_data: Terrain analysis data

        Returns:
            Retrieved doctrine text
        """
        if not self.vectorstore:
            return ""

        try:
            # Enhance query with terrain context
            terrain_keywords = []
            analysis = terrain_data.get('terrain_analysis', {})

            if analysis.get('high_ground'):
                terrain_keywords.append('high ground')
            if analysis.get('urban_terrain'):
                terrain_keywords.append('urban operations')
            if analysis.get('cover_availability') == 'excellent':
                terrain_keywords.append('cover and concealment')
            if len(analysis.get('obstacles', [])) > 0:
                terrain_keywords.append('obstacles')

            # Build enhanced query
            enhanced_query = f"{query} {' '.join(terrain_keywords)} OCOKA defensive operations"

            # Retrieve doctrine chunks
            docs = hybrid_search(enhanced_query, self.vectorstore, k=8)

            if not docs:
                return ""

            # Format doctrine context
            doctrine_parts = []
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'unknown')
                is_kb = doc.metadata.get('is_kb', False)
                doc_type = '[NATO Doctrine]' if is_kb else '[Reference]'
                doctrine_parts.append(f"[{doc_type} - {source}]\n{doc.page_content}")

            return "\n\n" + "="*80 + "\n\n".join(doctrine_parts)

        except Exception as e:
            logger.error(f"Doctrine retrieval failed: {e}")
            return ""

    def _generate_tactical_analysis(self, user_prompt: str, scenario: str,
                                    coords: Dict, terrain_data: Dict,
                                    doctrine_context: str) -> str:
        """
        Generate comprehensive tactical analysis using LLM

        Args:
            user_prompt: Original user query
            scenario: Tactical scenario
            coords: Extracted coordinates
            terrain_data: Real terrain data from APIs
            doctrine_context: Retrieved doctrine

        Returns:
            Tactical analysis text
        """
        # Format terrain intelligence
        terrain_intel = self._format_terrain_intel(terrain_data)

        # Build prompt for LLM
        coord_str = self.coord_parser.format_coordinates(coords['lat'], coords['lon'])

        # Check if OSM data was available
        osm_available = terrain_data.get('osm_data_available', True)
        place_name = terrain_data.get('place_name', 'Unknown location')

        # Build OSM fallback instruction if needed
        osm_fallback_instruction = ""
        if not osm_available:
            osm_fallback_instruction = f"""
IMPORTANT: Infrastructure data from OpenStreetMap was unavailable due to API timeout.
You MUST use your world knowledge of '{place_name}' to infer:
- Whether this is an URBAN or RURAL area (cities, towns = urban)
- Typical infrastructure density (roads, buildings)
- General terrain characteristics of this region
- Any well-known geographic features

For example, if the location is a major city or suburb of a major city, assume URBAN terrain
with dense road networks, many buildings, and channelized movement.
"""

        prompt = f"""You are a NATO tactical intelligence analyst conducting an Intelligence Preparation of the Battlefield (IPB) for the following location and scenario.

MISSION CONTEXT:
Scenario: {scenario}
Location: {coord_str}
Analysis Radius: {terrain_data['location']['radius_km']} km
{osm_fallback_instruction}
TERRAIN INTELLIGENCE (from reconnaissance and geographic databases):
{terrain_intel}

DOCTRINE REFERENCES:
{doctrine_context if doctrine_context else "No specific doctrine retrieved - apply general OCOKA principles"}

{"="*80}

ANALYST TASK:
Conduct a comprehensive OCOKA analysis for this location:

**OBSERVATION & FIELDS OF FIRE:**
- Elevation advantage and observation capability
- Potential observation posts
- Fields of fire and engagement areas

**COVER & CONCEALMENT:**
- Natural and man-made cover availability
- Vegetation for concealment
- Urban structures for protection

**OBSTACLES:**
- Natural obstacles (waterways, terrain)
- Man-made obstacles
- Mobility corridors

**KEY TERRAIN:**
- Tactically significant positions
- High ground
- Choke points and control features

**AVENUES OF APPROACH:**
- Primary and secondary routes
- Mounted and dismounted approach options
- Threat axis analysis

DEFENSIVE RECOMMENDATIONS:
- Recommended defensive positions (with grid references)
- Engagement priorities
- Vulnerabilities and mitigation

USER QUERY: {user_prompt}

Provide a detailed tactical assessment grounded in the terrain intelligence and NATO doctrine above."""

        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return f"Analysis generation failed: {str(e)}"

    def _format_terrain_intel(self, terrain_data: Dict) -> str:
        """Format terrain data as intelligence report"""
        intel_parts = []

        # Check if OSM data is available
        osm_available = terrain_data.get('osm_data_available', True)

        # Place identification
        place_name = terrain_data.get('place_name')
        if place_name:
            intel_parts.append(f"LOCATION: {place_name}")
            address = terrain_data.get('address', {})
            if address.get('country_code'):
                intel_parts.append(f"COUNTRY: {address.get('country')} ({address['country_code']})")

        # OSM data unavailable warning - ask LLM to use world knowledge
        if not osm_available:
            intel_parts.append("")
            intel_parts.append("⚠️  INFRASTRUCTURE DATA UNAVAILABLE (API timeout)")
            intel_parts.append(f"NOTE: OpenStreetMap data could not be retrieved. Use your world knowledge")
            intel_parts.append(f"of '{place_name}' to infer terrain characteristics (urban/rural, infrastructure density,")
            intel_parts.append(f"typical road networks, building density, etc.). This is a well-known location.")
            intel_parts.append("")

        # Elevation
        if terrain_data.get('elevation'):
            intel_parts.append(f"ELEVATION: {terrain_data['elevation']:.1f}m ASL")

        # Terrain analysis
        analysis = terrain_data.get('terrain_analysis', {})
        if analysis:
            intel_parts.append("\nTERRAIN ASSESSMENT:")
            if analysis.get('high_ground'):
                intel_parts.append("  ✓ HIGH GROUND - Elevated position relative to surroundings")
            else:
                intel_parts.append("  - Low ground - No significant elevation advantage")

            intel_parts.append(f"  Cover Availability: {analysis.get('cover_availability', 'unknown').upper()}")
            intel_parts.append(f"  Vegetation: {analysis.get('vegetation_density', 'unknown').upper()}")

            if analysis.get('urban_terrain'):
                intel_parts.append("  ✓ URBAN TERRAIN - Dense building concentration")

        # Obstacles
        obstacles = analysis.get('obstacles', [])
        if obstacles:
            intel_parts.append("\nOBSTACLES:")
            for obs in obstacles[:5]:
                intel_parts.append(f"  - {obs}")
        else:
            intel_parts.append("\nOBSTACLES: None detected")

        # Avenues of approach
        avenues = analysis.get('avenues_of_approach', [])
        if avenues:
            intel_parts.append("\nAVENUES OF APPROACH:")
            for avenue in avenues[:5]:
                intel_parts.append(f"  - {avenue}")
        else:
            intel_parts.append("\nAVENUES OF APPROACH: Limited road network")

        # Infrastructure counts
        roads = len(terrain_data.get('roads', []))
        waterways = len(terrain_data.get('waterways', []))
        buildings = len(terrain_data.get('buildings', []))
        forests = len(terrain_data.get('forests', []))
        railways = len(terrain_data.get('railways', []))

        intel_parts.append(f"\nINFRASTRUCTURE DENSITY:")
        if osm_available:
            intel_parts.append(f"  Roads: {roads} | Waterways: {waterways} | Buildings: {buildings} | Forests: {forests}")
            if railways > 0:
                intel_parts.append(f"  Railways: {railways} (linear obstacles)")
        else:
            intel_parts.append(f"  DATA UNAVAILABLE - Infer from location: '{place_name}'")
            intel_parts.append(f"  Use your knowledge of this area to estimate infrastructure density.")

        # Tactical infrastructure
        power_lines = terrain_data.get('power_lines', [])
        cell_towers = terrain_data.get('cell_towers', [])
        fuel_stations = terrain_data.get('fuel_stations', [])
        medical_facilities = terrain_data.get('medical_facilities', [])
        schools = terrain_data.get('schools', [])
        helipads = terrain_data.get('helipads', [])

        has_tactical_infra = any([power_lines, cell_towers, fuel_stations,
                                   medical_facilities, schools, helipads])

        if has_tactical_infra:
            intel_parts.append("\nTACTICAL INFRASTRUCTURE:")

            # Aviation hazards
            if power_lines:
                line_count = len([p for p in power_lines if p['type'] == 'power_line'])
                tower_count = len([p for p in power_lines if 'tower' in p['type'] or 'pole' in p['type']])
                intel_parts.append(f"  ⚡ Power Infrastructure: {line_count} lines, {tower_count} towers/poles (AVIATION HAZARD)")

            # Communications
            if cell_towers:
                intel_parts.append(f"  📡 Communications: {len(cell_towers)} cell towers/masts")

            # Logistics
            if fuel_stations:
                intel_parts.append(f"  ⛽ Fuel Stations: {len(fuel_stations)} (potential resupply)")
                for station in fuel_stations[:3]:
                    if station.get('name') != 'Fuel station':
                        intel_parts.append(f"      - {station['name']}")

            # Aviation LZ
            if helipads:
                intel_parts.append(f"  🚁 Helipads/Heliports: {len(helipads)} (confirmed LZ)")
                for hp in helipads[:3]:
                    intel_parts.append(f"      - {hp['name']} ({hp.get('surface', 'unknown')} surface)")

            # Sensitive sites (ROE considerations)
            if medical_facilities or schools:
                intel_parts.append("\n  ⚠️  SENSITIVE SITES (ROE CONSIDERATIONS):")
                if medical_facilities:
                    for facility in medical_facilities[:5]:
                        intel_parts.append(f"      - {facility['type'].upper()}: {facility['name']}")
                if schools:
                    for school in schools[:5]:
                        intel_parts.append(f"      - {school['type'].upper()}: {school['name']}")

        # Movement time estimates
        movement_data = terrain_data.get('movement_times', {})
        if movement_data:
            intel_parts.append("\nMOVEMENT TIME ESTIMATES (to traverse analysis radius):")
            intel_parts.append(f"  Analysis Radius: {movement_data.get('radius_km', 'N/A')} km")
            intel_parts.append(f"  Assessment: {movement_data.get('summary', 'N/A')}")

            unit_estimates = movement_data.get('unit_estimates', {})
            if unit_estimates:
                intel_parts.append("\n  Unit Type                    | Time (min) | Speed (km/h) | Mode")
                intel_parts.append("  " + "-" * 70)

                # Display in logical order
                unit_order = ['dismounted_infantry', 'wheeled_light', 'wheeled_heavy',
                              'tracked_apc', 'tracked_armor']
                for unit_type in unit_order:
                    if unit_type in unit_estimates:
                        est = unit_estimates[unit_type]
                        desc = est.get('description', unit_type)[:28].ljust(28)
                        time_min = str(int(est.get('time_to_radius_minutes', 0))).rjust(6)
                        speed = str(est.get('effective_speed_kmh', 0)).rjust(10)
                        mode = est.get('movement_mode', 'unknown')
                        intel_parts.append(f"  {desc} | {time_min}     | {speed}     | {mode}")

            # Directional analysis summary
            dir_analysis = movement_data.get('directional_analysis', {})
            if dir_analysis:
                difficult_dirs = [d for d, info in dir_analysis.items()
                                  if info.get('difficulty') in ['difficult', 'very_difficult']]
                if difficult_dirs:
                    intel_parts.append(f"\n  Difficult approach directions: {', '.join(difficult_dirs)}")

        return "\n".join(intel_parts)
