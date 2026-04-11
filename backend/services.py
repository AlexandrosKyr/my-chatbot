import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from utils import (
    extract_text_with_ocr,
    has_meaningful_text,
    clean_extracted_text,
    create_hierarchical_chunks,
    similarity_search,
    resolve_parents,
    preprocess_query,
    annotate_chunks_with_pages,
    update_parent_pages
)
from config import Config
from coordinate_parser import CoordinateParser
from terrain_data_fetcher import TerrainDataFetcher

logger = logging.getLogger(__name__)


class DocumentService:
    """Handle document upload, OCR, and indexing"""
    
    def __init__(self, vectorstore, parent_store=None):
        self.vectorstore = vectorstore
        self.parent_store = parent_store
        self.raw_documents = []
    
    def upload_and_index(self, filepath, filename, is_kb=False):
        """Upload file, extract text, chunk, and index"""
        try:
            logger.info(f"Processing {'KB ' if is_kb else ''}document: {filename}")
            
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise ValueError("File is empty")
            
            logger.info(f"File size: {file_size / 1024:.2f} KB")
            
            raw_text = ""
            page_offsets = []  # list of (char_offset, page_number)
            if filepath.lower().endswith('.pdf'):
                loader = PyPDFLoader(filepath)
                documents = loader.load()

                if not has_meaningful_text(documents):
                    ocr_text, _ = extract_text_with_ocr(filepath)
                    raw_text = ocr_text
                    # Build page offsets from OCR PAGE markers
                    import re
                    for m in re.finditer(r'\n={50}\nPAGE (\d+)\n={50}\n', raw_text):
                        page_offsets.append((m.start(), int(m.group(1))))
                else:
                    # Build page offsets while concatenating
                    parts = []
                    offset = 0
                    for doc in documents:
                        page_num = doc.metadata.get('page', 0) + 1  # PyPDFLoader is 0-indexed
                        page_offsets.append((offset, page_num))
                        parts.append(doc.page_content)
                        offset += len(doc.page_content) + 2  # +2 for "\n\n" separator
                    raw_text = "\n\n".join(parts)
            else:
                ocr_text, _ = extract_text_with_ocr(filepath)
                raw_text = ocr_text

            # Clean extracted text before chunking (removes TOC, blank pages, headers)
            raw_text = clean_extracted_text(raw_text)

            if len(raw_text.strip()) < Config.OCR_MIN_CHARS:
                raise ValueError(f"Text too short (< {Config.OCR_MIN_CHARS} chars)")
            
            doc_entry = {
                "filename": filename,
                "content": raw_text,
                "timestamp": datetime.now().isoformat(),
                "is_kb": is_kb
            }
            self.raw_documents.append(doc_entry)
            
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized")
            if self.parent_store is None:
                raise ValueError("Parent store not initialized")

            # Hierarchical chunking: children go to ChromaDB, parents to SQLite
            chunks = create_hierarchical_chunks(raw_text, filename, self.parent_store)

            if not chunks:
                raise ValueError("Failed to create chunks")

            # Annotate chunks and parents with page numbers
            if page_offsets:
                annotate_chunks_with_pages(chunks, raw_text, page_offsets)
                update_parent_pages(self.parent_store, raw_text, page_offsets, filename)

            for chunk in chunks:
                chunk.metadata['is_kb'] = is_kb

            self.vectorstore.add_documents(chunks)
            
            logger.info(f"Successfully indexed {filename}: {len(chunks)} chunks")
            
            return {
                "success": True,
                "chunks": len(chunks),
                "text_length": len(raw_text),
                "file_size_kb": round(file_size / 1024, 2)
            }
        
        except Exception as e:
            logger.error(f"Upload error: {e}")
            raise
    
    def delete_all(self):
        """Delete all documents and reset"""
        try:
            if os.path.exists(Config.CHROMA_DB_PATH):
                shutil.rmtree(Config.CHROMA_DB_PATH)
                os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
            
            if os.path.exists(Config.UPLOAD_FOLDER):
                for filename in os.listdir(Config.UPLOAD_FOLDER):
                    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                    except Exception as e:
                        logger.error(f"Error deleting {filename}: {e}")
            
            if os.path.exists(Config.KB_FOLDER):
                for filename in os.listdir(Config.KB_FOLDER):
                    filepath = os.path.join(Config.KB_FOLDER, filename)
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                    except Exception as e:
                        logger.error(f"Error deleting {filename}: {e}")
            
            if self.parent_store:
                self.parent_store.clear()

            docs_deleted = len(self.raw_documents)
            self.raw_documents = []

            logger.info("All documents deleted")
            return {"success": True, "documents_deleted": docs_deleted}
        
        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise


class RAGService:
    """Handle RAG (Retrieval-Augmented Generation) chat with terrain-aware analysis"""

    def __init__(self, llm, vectorstore, raw_documents, parent_store=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.raw_documents = raw_documents
        self.parent_store = parent_store
        self.coordinate_parser = CoordinateParser()
        self.terrain_fetcher = TerrainDataFetcher()

        # Load military power prompt DB (keyed by ISO alpha-2, e.g. "US", "RU")
        _db_path = Path(__file__).parent.parent / "db" / "military_power_prompt.json"
        try:
            with open(_db_path, encoding="utf-8") as f:
                self._military_db = json.load(f)
            logger.info("Military power DB loaded: %d countries", len(self._military_db))
        except FileNotFoundError:
            self._military_db = {}
            logger.warning("Military power DB not found at %s — skipping military intel", _db_path)

        self.last_terrain_summary = None  # Store for API response
        # Store full terrain context for follow-up reuse
        self.last_terrain_intel = None
        self.last_terrain_data = None
        self.last_coords = None

    def _parse_radius_from_text(self, text):
        """Extract radius specification from user message (default 5km)"""
        import re

        # Patterns like "10km radius", "radius of 10km", "10 km", "within 15km"
        patterns = [
            r'(\d+)\s*km\s*radius',
            r'radius\s*(?:of\s*)?(\d+)\s*km',
            r'within\s*(\d+)\s*km',
            r'(\d+)\s*kilometer',
            r'for\s*(\d+)\s*km',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                radius = int(match.group(1))
                # Clamp to reasonable range (1-50km)
                radius = max(1, min(50, radius))
                logger.info(f"User-specified radius detected: {radius}km")
                return radius

        return 5  # Default radius

    def _fetch_and_format_terrain(self, question):
        """Check for coordinates and fetch terrain data if found"""
        coords = self.coordinate_parser.parse(question)
        if not coords:
            # Don't clear stored data - keep it for follow-up questions
            return None, None, None

        logger.info(f"Coordinates detected: {coords['lat']}, {coords['lon']}")

        # Parse radius from user message
        radius_km = self._parse_radius_from_text(question)

        try:
            terrain_data = self.terrain_fetcher.fetch_terrain_data(
                coords['lat'], coords['lon'], radius_km=radius_km
            )
            terrain_intel = self._format_terrain_intel(terrain_data, coords, radius_km)

            # Create summary for frontend display
            self.last_terrain_summary = self._create_terrain_summary(terrain_data, coords, radius_km)

            # Store full context for follow-up reuse
            self.last_terrain_intel = terrain_intel
            self.last_terrain_data = terrain_data
            self.last_coords = coords

            return terrain_intel, coords, terrain_data
        except Exception as e:
            logger.error(f"Terrain fetch failed: {e}")
            self.last_terrain_summary = None
            return None, coords, None

    def _create_terrain_summary(self, terrain_data, coords, radius_km):
        """Create a concise terrain summary for frontend display"""
        from collections import Counter

        roads = terrain_data.get('roads', [])

        # Dynamically group roads by their actual types from the data
        road_type_counts = dict(Counter(r['type'] for r in roads))

        # Get named roads (filter out unnamed segments)
        named_roads = list(set([
            r['name'] for r in roads
            if r.get('name') and r['name'] not in ['Unnamed road', '']
        ]))

        waterways = terrain_data.get('waterways', [])
        waterway_summary = terrain_data.get('waterway_summary', {})

        movement = terrain_data.get('movement_times', {})
        slope = terrain_data.get('slope_analysis', {})

        summary = {
            'coordinates': {
                'lat': round(coords['lat'], 6),
                'lon': round(coords['lon'], 6)
            },
            'location': terrain_data.get('place_name', 'Unknown location'),
            'radius_km': radius_km,
            'elevation': terrain_data.get('elevation'),
            'terrain': {
                'avg_slope_percent': slope.get('average_slope_percent', 0),
                'max_slope_percent': slope.get('max_slope_percent', 0),
                'mobility': slope.get('mobility', 'unknown')
            },
            'infrastructure': {
                'roads': {
                    'total_segments': len(roads),
                    'by_type': road_type_counts,  # Dynamic breakdown by actual types
                    'named_roads': sorted(named_roads)[:10]  # Top 10 named roads
                },
                'waterways': {
                    'total_segments': waterway_summary.get('total_segments', len(waterways)),
                    'segments_by_type': waterway_summary.get('segments_by_type', {}),
                    'distinct_named': waterway_summary.get('distinct_named', {}),
                    'total_distinct_named': waterway_summary.get('total_distinct_named', 0),
                },
                'buildings': len(terrain_data.get('buildings', [])),
                'forests': terrain_data.get('forest_summary', {
                    'total_segments': len(terrain_data.get('forests', [])),
                }),
                'crossings': terrain_data.get('crossing_summary', {
                    'total_segments': len(terrain_data.get('crossings', [])),
                }),
                'railways': terrain_data.get('railway_summary', {
                    'total_segments': len(terrain_data.get('railways', [])),
                })
            },
            'tactical': {
                'power_lines': len(terrain_data.get('power_lines', [])),
                'cell_towers': len(terrain_data.get('cell_towers', [])),
                'fuel_stations': len(terrain_data.get('fuel_stations', [])),
                'medical_facilities': terrain_data.get('medical_summary', {}).get('total_distinct_named', len(terrain_data.get('medical_facilities', []))),
                'medical_summary': terrain_data.get('medical_summary', {}),
                'schools': terrain_data.get('school_summary', {}).get('total_distinct_named', len(terrain_data.get('schools', []))),
                'school_summary': terrain_data.get('school_summary', {}),
                'helipads': len(terrain_data.get('helipads', []))
            },
            'movement': {
                'summary': movement.get('summary', 'N/A'),
                'unit_estimates': movement.get('unit_estimates', {})  # Pass all unit data
            },
            'weather': terrain_data.get('weather', {}).get('weekly_summary', {})
        }

        return summary

    def _format_terrain_intel(self, terrain_data, coords, radius_km=5):
        """Format terrain data as tactical intelligence for the LLM prompt"""
        intel_parts = []

        # Check if OSM data is available
        osm_available = terrain_data.get('osm_data_available', True)

        intel_parts.append("=" * 60)
        intel_parts.append("TERRAIN INTELLIGENCE REPORT")
        intel_parts.append(f"Analysis Radius: {radius_km}km")
        intel_parts.append("=" * 60)

        # Location
        place_name = terrain_data.get('place_name', 'Unknown')
        intel_parts.append(f"\nLOCATION: {place_name}")
        intel_parts.append(f"COORDINATES: {coords['lat']:.6f}°N, {coords['lon']:.6f}°E")

        # OSM data unavailable warning - ask LLM to use world knowledge
        if not osm_available:
            intel_parts.append("")
            intel_parts.append("INFRASTRUCTURE DATA UNAVAILABLE (API timeout)")
            intel_parts.append(f"NOTE: OpenStreetMap data could not be retrieved. Use your world knowledge")
            intel_parts.append(f"of '{place_name}' to infer terrain characteristics (urban/rural, infrastructure density,")
            intel_parts.append(f"typical road networks, building density, etc.). If you are unsure of the location's specifics then do not speculate.")
            intel_parts.append("")

        # Elevation
        elevation = terrain_data.get('elevation')
        if elevation:
            intel_parts.append(f"ELEVATION: {elevation}m ASL")

        # Slope analysis
        slope_data = terrain_data.get('slope_analysis', {})
        if slope_data:
            avg_slope = slope_data.get('average_slope_percent', 0)
            max_slope = slope_data.get('max_slope_percent', 0)
            mobility = slope_data.get('mobility', {})
            mobility_assessment = mobility.get('assessment', 'unknown') if isinstance(mobility, dict) else str(mobility)
            intel_parts.append(f"\nSLOPE ANALYSIS:")
            intel_parts.append(f"  Average: {avg_slope}% | Maximum: {max_slope}%")
            intel_parts.append(f"  Mobility Assessment: {mobility_assessment.upper()}")

            # Directional slopes
            dir_slopes = slope_data.get('direction_slopes', {})
            if dir_slopes:
                intel_parts.append("  Directional Gradients:")
                for direction, info in dir_slopes.items():
                    slope_pct = info.get('slope_percent', 0)
                    slope_dir = info.get('direction', 'flat')
                    intel_parts.append(f"    {direction.upper()}: {slope_pct}% ({slope_dir})")

        # Line of sight
        los_data = terrain_data.get('line_of_sight', {})
        if los_data:
            intel_parts.append(f"\nOBSERVATION & FIELDS OF FIRE:")
            intel_parts.append(f"  Dominant Position: {'YES' if los_data.get('is_high_ground') else 'NO'}")
            visibility = los_data.get('overall_visibility', 'unknown')
            intel_parts.append(f"  Visibility: {visibility}")
            if los_data.get('obstructed_directions'):
                intel_parts.append(f"  Obstructed Directions: {', '.join(los_data['obstructed_directions'])}")

        # Crossings (bridges, fords, tunnels)
        crossings = terrain_data.get('crossings', [])
        crossing_summary = terrain_data.get('crossing_summary', {})
        if crossings:
            distinct = crossing_summary.get('distinct_named', {})
            total_distinct = crossing_summary.get('total_distinct_named', 0)
            intel_parts.append(f"\nCROSSING POINTS ({total_distinct} distinct, {len(crossings)} segments):")
            for ctype in ['bridge', 'ford', 'tunnel', 'dam']:
                info = distinct.get(ctype, {})
                count = info.get('count', 0)
                names = info.get('names', [])
                if count:
                    intel_parts.append(f"  {ctype.title()}s ({count}): {', '.join(names[:5])}")
                elif crossing_summary.get('segments_by_type', {}).get(ctype, 0):
                    seg_count = crossing_summary['segments_by_type'][ctype]
                    intel_parts.append(f"  {ctype.title()}s: {seg_count} segments (unnamed)")

        # Avenues of approach (roads) - dynamically grouped by type
        roads = terrain_data.get('roads', [])
        if roads:
            # Group roads by their actual type from the data
            from collections import Counter
            road_type_counts = Counter(r['type'] for r in roads)

            intel_parts.append(f"\nAVENUES OF APPROACH ({len(roads)} road segments in {radius_km}km radius):")
            for road_type, count in road_type_counts.most_common():
                intel_parts.append(f"  {road_type.replace('_', ' ').title():20s}: {count:3d} segments")

            # Show named roads (deduplicated, excluding generic names)
            named_roads = list(set([
                r['name'] for r in roads
                if r.get('name') and r['name'] not in ['Unnamed road', '']
            ]))
            if named_roads:
                intel_parts.append(f"\n  Named Routes ({len(named_roads)} unique):")
                for road_name in sorted(named_roads)[:8]:
                    intel_parts.append(f"    - {road_name}")

        # Obstacles
        waterways = terrain_data.get('waterways', [])
        waterway_summary = terrain_data.get('waterway_summary', {})
        railways = terrain_data.get('railways', [])
        if waterways or railways:
            intel_parts.append(f"\nOBSTACLES:")
            if waterways:
                distinct = waterway_summary.get('distinct_named', {})
                river_info = distinct.get('river', {})
                stream_info = distinct.get('stream', {})
                canal_info = distinct.get('canal', {})
                parts = []
                if river_info.get('count', 0):
                    parts.append(f"{river_info['count']} rivers ({', '.join(river_info['names'][:5])})")
                if stream_info.get('count', 0):
                    parts.append(f"{stream_info['count']} streams")
                if canal_info.get('count', 0):
                    parts.append(f"{canal_info['count']} canals")
                if parts:
                    intel_parts.append(f"  Water: {', '.join(parts)}")
                else:
                    seg_counts = waterway_summary.get('segments_by_type', {})
                    intel_parts.append(f"  Water: {len(waterways)} waterway segments ({', '.join(f'{v} {k}' for k, v in seg_counts.items())})")
            if railways:
                railway_summary = terrain_data.get('railway_summary', {})
                named_lines = railway_summary.get('distinct_named', [])
                if named_lines:
                    intel_parts.append(f"  Railways: {len(named_lines)} lines ({', '.join(named_lines[:5])}), {railway_summary.get('total_segments', len(railways))} segments")
                else:
                    intel_parts.append(f"  Railways: {len(railways)} segments (linear obstacles)")

        # Cover and concealment
        forests = terrain_data.get('forests', [])
        forest_summary = terrain_data.get('forest_summary', {})
        buildings = terrain_data.get('buildings', [])
        intel_parts.append(f"\nCOVER & CONCEALMENT:")
        if osm_available:
            named_forests = forest_summary.get('distinct_named', [])
            unnamed_segs = forest_summary.get('unnamed_segments', len(forests))
            if named_forests:
                intel_parts.append(f"  Forest Areas: {len(named_forests)} named ({', '.join(named_forests[:5])}), {unnamed_segs} unnamed segments")
            else:
                intel_parts.append(f"  Forest Areas: {forest_summary.get('total_segments', len(forests))} segments (none named)")
            intel_parts.append(f"  Buildings/Structures: {len(buildings)}")
            if len(buildings) > 200:
                intel_parts.append("  URBAN TERRAIN - expect channelized movement")
        else:
            intel_parts.append(f"  DATA UNAVAILABLE - Infer from location: '{place_name}'")
            intel_parts.append(f"  Use your knowledge to assess cover/concealment and urban characteristics.")

        # Tactical infrastructure
        power_lines = terrain_data.get('power_lines', [])
        cell_towers = terrain_data.get('cell_towers', [])
        fuel_stations = terrain_data.get('fuel_stations', [])
        medical = terrain_data.get('medical_facilities', [])
        medical_summary = terrain_data.get('medical_summary', {})
        schools = terrain_data.get('schools', [])
        school_summary = terrain_data.get('school_summary', {})
        helipads = terrain_data.get('helipads', [])

        if any([power_lines, cell_towers, fuel_stations, medical, schools, helipads]):
            intel_parts.append(f"\nTACTICAL INFRASTRUCTURE:")
            if power_lines:
                intel_parts.append(f"  Power Lines: {len(power_lines)} (AVIATION HAZARD)")
            if cell_towers:
                intel_parts.append(f"  Comm Towers: {len(cell_towers)}")
            if fuel_stations:
                intel_parts.append(f"  Fuel Points: {len(fuel_stations)} (resupply potential)")
            if helipads:
                intel_parts.append(f"  Helipads: {len(helipads)} (confirmed LZ)")
            if medical or schools:
                med_unnamed = medical_summary.get('unnamed_segments', 0)
                sch_unnamed = school_summary.get('unnamed_segments', 0)
                intel_parts.append(f"  SENSITIVE SITES (ROE):")
                # Medical breakdown by type
                med_distinct = medical_summary.get('distinct_named', {})
                for mtype in ['hospital', 'clinic']:
                    info = med_distinct.get(mtype, {})
                    count = info.get('count', 0)
                    names = info.get('names', [])
                    if count:
                        intel_parts.append(f"    {mtype.title()}s ({count}): {', '.join(names[:8])}")
                if med_unnamed:
                    intel_parts.append(f"    + {med_unnamed} unnamed medical facilities")
                # School breakdown by type
                sch_distinct = school_summary.get('distinct_named', {})
                for stype in ['school', 'university']:
                    info = sch_distinct.get(stype, {})
                    count = info.get('count', 0)
                    names = info.get('names', [])
                    if count:
                        intel_parts.append(f"    {stype.title()}s ({count}): {', '.join(names[:8])}")
                if sch_unnamed:
                    intel_parts.append(f"    + {sch_unnamed} unnamed schools")

        # Movement times
        movement = terrain_data.get('movement_times', {})
        if movement:
            intel_parts.append(f"\nMOVEMENT TIME ESTIMATES ({radius_km}km radius):")
            intel_parts.append(f"  Assessment: {movement.get('summary', 'N/A')}")
            unit_est = movement.get('unit_estimates', {})
            if unit_est:
                # Display all available unit types from the data
                for unit_type, est in unit_est.items():
                    intel_parts.append(f"  {est.get('description', unit_type)}: {int(est.get('time_to_radius_minutes', 0))} min")

        # Weather data (past week)
        weather = terrain_data.get('weather', {})
        if weather and weather.get('weekly_summary'):
            summary = weather['weekly_summary']
            intel_parts.append(f"\nWEATHER (Past 7 Days):")

            # Temperature
            if summary.get('avg_temp_c') is not None:
                intel_parts.append(f"  Temperature: avg {summary['avg_temp_c']}°C "
                                 f"(range: {summary.get('avg_temp_min_c', 'N/A')}°C to {summary.get('avg_temp_max_c', 'N/A')}°C)")

            # Precipitation
            precip = summary.get('total_precipitation_mm', 0)
            rain = summary.get('total_rain_mm', 0)
            snow = summary.get('total_snow_cm', 0)
            rainy_days = summary.get('rainy_days', 0)
            snowy_days = summary.get('snowy_days', 0)

            if precip > 0 or rain > 0 or snow > 0:
                intel_parts.append(f"  Precipitation: {precip}mm total ({rainy_days} rainy days)")
                if snow > 0:
                    intel_parts.append(f"  Snowfall: {snow}cm ({snowy_days} snowy days)")
            else:
                intel_parts.append(f"  Precipitation: Dry conditions")

            # Wind
            if summary.get('avg_wind_speed_max_kmh') is not None:
                intel_parts.append(f"  Wind: avg max {summary['avg_wind_speed_max_kmh']} km/h, "
                                 f"gusts up to {summary.get('max_wind_gust_kmh', 'N/A')} km/h")

            # Sunshine
            if summary.get('avg_sunshine_hours') is not None:
                intel_parts.append(f"  Sunshine: avg {summary['avg_sunshine_hours']} hours/day")

            # Conditions
            conditions = summary.get('predominant_conditions', [])
            if conditions:
                intel_parts.append(f"  Conditions: {', '.join(c.replace('_', ' ') for c in conditions)}")

            # Tactical weather assessment
            intel_parts.append(f"  TACTICAL IMPACT:")
            if precip > 20 or rainy_days >= 3:
                intel_parts.append(f"    - Wet conditions: reduced off-road mobility, potential flooding")
            if snow > 5:
                intel_parts.append(f"    - Snow cover: affects concealment, tracked vehicle advantage")
            if summary.get('max_wind_gust_kmh', 0) and summary['max_wind_gust_kmh'] > 50:
                intel_parts.append(f"    - High winds: affects aviation ops, smoke deployment")
            if summary.get('avg_sunshine_hours', 0) and summary['avg_sunshine_hours'] < 4:
                intel_parts.append(f"    - Low visibility conditions: reduced observation range")
            if precip == 0 and summary.get('avg_sunshine_hours', 0) and summary['avg_sunshine_hours'] > 8:
                intel_parts.append(f"    - Clear/dry conditions: good visibility, firm ground")

        # Military power data (from Global Firepower DB)
        country_code = terrain_data.get("address", {}).get("country_code", "")
        mil = self._military_db.get(country_code) if country_code else None
        if mil:
            intel_parts.append(f"\nMILITARY POWER ({mil['name']}):")
            intel_parts.append(json.dumps(mil, indent=2))
        else:
            intel_parts.append(f"\nMILITARY POWER: No data available for country code '{country_code}'")

        intel_parts.append("\n" + "=" * 60)

        return "\n".join(intel_parts)

    def _detect_scenario_type(self, question: str) -> str:
        """Analyze user question and return scenario type for tailored prompting.

        Returns one of: 'defensive', 'offensive', 'stability', 'reconnaissance', 'general'
        """
        question_lower = question.lower()

        # Keyword sets for each scenario type
        defensive_keywords = [
            'defend', 'defense', 'defensive', 'hold', 'delay', 'retain',
            'battle position', 'engagement area', 'dig in', 'retrograde',
            'withdraw', 'blocking position', 'strongpoint', 'area defense'
        ]

        offensive_keywords = [
            'attack', 'assault', 'offensive', 'seize', 'capture', 'advance',
            'breach', 'penetrate', 'envelop', 'objective', 'exploitation',
            'pursuit', 'movement to contact', 'raid', 'ambush', 'infiltration'
        ]

        stability_keywords = [
            'stability', 'coin', 'counterinsurgency', 'population', 'civil',
            'humanitarian', 'peacekeeping', 'hearts and minds', 'civil affairs',
            'reconstruction', 'governance', 'rule of law', 'security force assistance'
        ]

        recon_keywords = [
            'reconnaissance', 'recon', 'screen', 'surveil', 'observe', 'scout',
            'surveillance', 'guard', 'cover', 'zone reconnaissance', 'route reconnaissance',
            'area reconnaissance', 'isr', 'intelligence collection'
        ]

        # Check each category (order matters - more specific first)
        for keyword in recon_keywords:
            if keyword in question_lower:
                logger.info(f"Scenario detected: reconnaissance (matched '{keyword}')")
                return 'reconnaissance'

        for keyword in stability_keywords:
            if keyword in question_lower:
                logger.info(f"Scenario detected: stability (matched '{keyword}')")
                return 'stability'

        for keyword in defensive_keywords:
            if keyword in question_lower:
                logger.info(f"Scenario detected: defensive (matched '{keyword}')")
                return 'defensive'

        for keyword in offensive_keywords:
            if keyword in question_lower:
                logger.info(f"Scenario detected: offensive (matched '{keyword}')")
                return 'offensive'

        logger.info("Scenario detected: general (no specific keywords matched)")
        return 'general'

    def _enhance_query_with_terrain(self, query: str, terrain_data: dict) -> str:
        """Enhance doctrine retrieval query with terrain-specific keywords.

        This improves retrieval relevance by adding terrain-derived keywords
        that help match relevant doctrine passages about specific terrain types.

        Args:
            query: Original user query
            terrain_data: Terrain data from TerrainDataFetcher

        Returns:
            Enhanced query string with terrain keywords appended
        """
        if not terrain_data:
            return query

        terrain_keywords = []

        # Check terrain analysis data
        analysis = terrain_data.get('terrain_analysis', {})
        slope_data = terrain_data.get('slope_analysis', {})
        los_data = terrain_data.get('line_of_sight', {})

        # High ground / observation
        if analysis.get('high_ground') or los_data.get('is_high_ground'):
            terrain_keywords.append('high ground observation fields of fire')

        # Urban terrain
        buildings = terrain_data.get('buildings', [])
        if analysis.get('urban_terrain') or len(buildings) > 50:
            terrain_keywords.append('urban operations MOUT complex terrain')

        # Cover and concealment
        cover = analysis.get('cover_availability', '')
        forests = terrain_data.get('forests', [])
        if cover == 'excellent' or len(forests) > 10:
            terrain_keywords.append('cover and concealment')
        elif cover == 'limited' or (len(buildings) < 10 and len(forests) < 5):
            terrain_keywords.append('open terrain exposed')

        # Obstacles
        obstacles = analysis.get('obstacles', [])
        if len(obstacles) > 0:
            terrain_keywords.append('obstacles mobility')

        # Water obstacles
        waterways = terrain_data.get('waterways', [])
        if len(waterways) > 0:
            terrain_keywords.append('water obstacle river crossing')

        # Crossings (bridges, fords)
        crossings = terrain_data.get('crossings', [])
        if len(crossings) > 0:
            terrain_keywords.append('bridge crossing point')

        # Slope/mobility
        mobility = slope_data.get('mobility', '')
        if mobility in ['restricted', 'severely_restricted']:
            terrain_keywords.append('restricted terrain mobility')

        # Roads/avenues of approach
        roads = terrain_data.get('roads', [])
        if len(roads) > 20:
            terrain_keywords.append('avenues of approach road network')

        # Combine original query with terrain keywords
        if terrain_keywords:
            enhanced = f"{query} {' '.join(terrain_keywords)} OCOKA IPB terrain analysis"
            logger.info(f"Enhanced query with terrain keywords: {terrain_keywords}")
            return enhanced

        return query

    def _get_scenario_guidance(self, scenario_type: str) -> str:
        """Return scenario-specific analytical guidance for the tactical prompt."""

        guidance = {
            'defensive': """DEFENSIVE OPERATIONS: Identify defensible terrain, engagement areas, obstacle integration, key terrain to retain, and enemy avenues of approach.""",
            'offensive': """OFFENSIVE OPERATIONS: Identify friendly avenues of approach, key terrain objectives, obstacles to breach/bypass, cover and concealment for movement.""",
            'stability': """STABILITY OPERATIONS: Focus on civil considerations (ASCOPE), population centers, sensitive sites, critical infrastructure.""",
            'reconnaissance': """RECONNAISSANCE: Identify observation positions, named areas of interest (NAIs), screen lines, information collection priorities.""",
            'general': """COMPREHENSIVE IPB: Full OAKOC terrain analysis, threat evaluation, course of action development."""
        }

        return guidance.get(scenario_type, guidance['general'])

    def _build_tactical_prompt(self, question: str, context: str, terrain_intel: str,
                                coords: dict, scenario_type: str, terrain_data: dict = None,
                                conversation_history: list = None) -> str:
        """Build the IPB tactical analysis prompt - optimized for length and accuracy."""
        scenario_guidance = self._get_scenario_guidance(scenario_type)
        coord_str = f"{coords['lat']:.6f}°N, {coords['lon']:.6f}°E" if coords else "Not specified"

        # Conversation history (condensed)
        history_section = ""
        if conversation_history and len(conversation_history) > 0:
            history_section = "PRIOR CONTEXT:\n"
            for msg in conversation_history[-4:]:
                role = "U" if msg.get('role') == 'user' else "A"
                history_section += f"{role}: {msg.get('text', '')[:300]}\n"

        # Doctrine section - conditional on availability
        has_doctrine = context and context.strip() != ""
        if has_doctrine:
            doctrine_section = f"""RETRIEVED DOCTRINE (cite ONLY from this):
{context}"""
        else:
            doctrine_section = "NO DOCTRINE DOCUMENTS LOADED."

        # OSM data availability
        osm_note = ""
        if terrain_data and not terrain_data.get('osm_data_available', True):
            osm_note = f"Note: OSM data unavailable. Use general knowledge of {terrain_data.get('place_name', 'this area')}."

        prompt = f"""You are a NATO officer conducting Intelligence Preparation of 
the Battlefield (IPB) analysis following ATP 2-01.3 methodology.

CRITICAL RULES:
1. Citation: Use exact document name and page as provided — NEVER invent sources
2. If no doctrine provided, state "Based on standard IPB methodology"
3. VALIDATION: If terrain data conflicts with scenario description, note the 
   discrepancy and prioritize scenario description for analysis
4. If terrain data is missing then infer from location knowledge, but DO NOT HALLUCINATE
5. Doctrine examples are ILLUSTRATIVE — do not present them as real features
6. IPB Step 4 analyzes ENEMY courses of action, not friendly COAs

{history_section}

SCENARIO: {scenario_guidance}

{doctrine_section}

TERRAIN DATA ({coord_str}):
{terrain_intel}
{osm_note}

TASK: {question}

OUTPUT FORMAT:

## 1. SITUATION OVERVIEW
Brief tactical context. Note any discrepancies between terrain data and scenario.

## 2. TERRAIN ANALYSIS — IPB Step 2 (OAKOC) (Pay more attention to named/significant ones and just mention the rest)
For each factor, provide TERRAIN DATA → TACTICAL EFFECT:
- **Observation & Fields of Fire**
- **Avenues of Approach** (classify as Unrestricted/Restricted/Severely Restricted)
- **Key Terrain** (identify decisive terrain if applicable)
- **Obstacles** (natural and man-made)
- **Cover & Concealment** (distinguish between cover and concealment)

## 3. CIVIL CONSIDERATIONS (ASCOPE)
- Areas, Structures, Capabilities, Organizations, People, Events
- Impact on both friendly and enemy operations

## 4. THREAT EVALUATION — IPB Step 3
- Enemy composition and disposition (from scenario)
- Assessed enemy TTPs for this force type
- Enemy capabilities and limitations in this terrain

## 5. ENEMY COURSES OF ACTION — IPB Step 4
**Enemy Most Probable COA (MPCOA):**
- Enemy task and purpose
- How terrain shapes this enemy COA
- Indicators friendly forces would observe

**Enemy Most Dangerous COA (MDCOA):**
- Why this poses greatest risk to friendly mission
- How terrain enables this enemy COA
- Decision points for friendly commander

## 6. NAMED AREAS OF INTEREST
Identify 2-3 NAIs with:
- Location/description
- What activity would indicate which enemy COA
- Collection asset recommendation

## 7. RECOMMENDATIONS
Actionable recommendations for the commander based on terrain and threat analysis.

Begin analysis. Use provided terrain data, but flag any inconsistencies with scenario.
DO NOT HALLUCINATE INFORMATION, IF YOU ARE MISSING INFORMATION SIMPLY STATE IT"""
        return prompt

    def process_query(self, question, conversation_history=None):
        """Process query - terrain analysis only (single flow)"""
        logger.info(f"Processing query: {question[:100]}...")

        # Build data availability status for frontend
        data_availability = {
            "coordinates_found": False,
            "terrain_data": "unavailable",
            "osm_data": "unavailable",
            "elevation_data": "unavailable",
            "doctrine_documents": "unavailable",
            "message": ""
        }

        # Check for coordinates and fetch terrain data
        terrain_intel, coords, terrain_data = self._fetch_and_format_terrain(question)

        if coords:
            data_availability["coordinates_found"] = True
            logger.info(f"Coordinates detected: {coords['lat']}, {coords['lon']}")

        if terrain_data:
            data_availability["terrain_data"] = "available"
            data_availability["osm_data"] = "available" if terrain_data.get('osm_data_available', True) else "unavailable"
            data_availability["elevation_data"] = "available" if terrain_data.get('elevation') else "unavailable"

        # Check doctrine availability
        has_docs = self.vectorstore is not None or len(self.raw_documents) > 0
        data_availability["doctrine_documents"] = "available" if has_docs else "unavailable"

        # Build human-readable message for missing data
        missing = []
        # Only show "No coordinates detected" on the first prompt (no conversation history)
        if not coords and not conversation_history:
            missing.append("No coordinates detected")
        if data_availability["osm_data"] == "unavailable" and coords:
            missing.append("OSM infrastructure data unavailable")
        if not has_docs:
            missing.append("No doctrine documents loaded")
        if missing:
            data_availability["message"] = ". ".join(missing)

        # ROUTING LOGIC (single flow - terrain analysis only)
        if coords and terrain_intel:
            # Full IPB analysis - coordinates provided
            logger.info("Route: Full terrain analysis with coordinates")
            context = self._retrieve_doctrine_context(question, terrain_data)
            scenario_type = self._detect_scenario_type(question)

            prompt = self._build_tactical_prompt(
                question=question,
                context=context,
                terrain_intel=terrain_intel,
                coords=coords,
                scenario_type=scenario_type,
                terrain_data=terrain_data,
                conversation_history=conversation_history
            )
            mode = "terrain_analysis"

        elif conversation_history and len(conversation_history) > 0:
            # Follow-up question with conversation history
            if self.last_terrain_intel and self.last_coords:
                # Use stored terrain data with full tactical prompt
                logger.info("Route: Follow-up with stored terrain context")
                context = self._retrieve_doctrine_context(question, self.last_terrain_data)
                scenario_type = self._detect_scenario_type(question)

                prompt = self._build_tactical_prompt(
                    question=question,
                    context=context,
                    terrain_intel=self.last_terrain_intel,
                    coords=self.last_coords,
                    scenario_type=scenario_type,
                    terrain_data=self.last_terrain_data,
                    conversation_history=conversation_history
                )
                mode = "terrain_analysis"
            else:
                # No stored terrain data - use simple follow-up
                logger.info("Route: Follow-up without terrain context")
                prompt = self._build_followup_prompt(
                    question, conversation_history, self.last_terrain_summary
                )
                mode = "followup"

        else:
            # No coordinates, no history - ask user for coordinates
            logger.info("Route: No coordinates, requesting from user")
            prompt = f"""You are a NATO Terrain Analysis Assistant for Greece.

The user has sent a message but no coordinates were detected and there's no previous analysis context.

User message: {question}

Please respond by:
1. Acknowledging their request
2. Explaining that you need coordinates to perform terrain analysis
3. Asking them to provide coordinates
4. Give an example format: "Please provide coordinates, e.g., 'Analyze 48.8566, 2.3522 for defensive positions'"

Keep your response concise and helpful."""
            mode = "awaiting_coordinates"

        response = self.llm.invoke(prompt)
        return response, mode, data_availability

    def _retrieve_doctrine_context(self, question, terrain_data=None):
        """Retrieve relevant doctrine using hierarchical parent-child chunking.

        Searches small child chunks for precision, then resolves their parent
        chunks for richer LLM context. Deduplicates so the same passage isn't
        sent multiple times.
        """
        if self.vectorstore is None:
            logger.info("No vector store available for doctrine retrieval")
            return ""

        try:
            enhanced_query, _ = preprocess_query(question)

            if terrain_data:
                enhanced_query = self._enhance_query_with_terrain(enhanced_query, terrain_data)

            # Search child chunks with relevance score filtering
            child_docs = similarity_search(enhanced_query, self.vectorstore, k=5)
            if not child_docs:
                logger.info("No doctrine documents retrieved")
                return ""

            # Resolve children to deduplicated parent chunks
            if self.parent_store:
                docs = resolve_parents(child_docs, self.parent_store)
            else:
                docs = child_docs

            logger.info(f"Retrieved {len(child_docs)} children -> {len(docs)} context chunks")

            context_parts = []
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'unknown')
                display_source = source.replace('.pdf', '').replace('_', ' ')
                page = doc.metadata.get('page', '')
                if page:
                    label = f"[{display_source}, p.{page}]"
                else:
                    label = f"[{display_source}]"
                context_parts.append(f"{label}\n{doc.page_content}")

            return "\n\n" + "\n\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Doctrine retrieval failed: {e}")
            return ""

    def _build_followup_prompt(self, question, conversation_history, last_terrain_summary=None):
        """Handle follow-up questions about previous analysis"""
        history_text = ""
        for msg in conversation_history[-8:]:
            role = "User" if msg.get('role') == 'user' else "Assistant"
            text = msg.get('text', '')[:800]
            history_text += f"\n{role}: {text}\n"

        location_context = ""
        if last_terrain_summary:
            place = last_terrain_summary.get('location', 'the previously analyzed location')
            coords = last_terrain_summary.get('coordinates', {})
            if coords:
                location_context = f"Previous analysis location: {place} ({coords.get('lat')}, {coords.get('lon')})"
            else:
                location_context = f"Previous analysis location: {place}"

        prompt = f"""You are a NATO Tactical Terrain Analysis Assistant.

The user is asking a follow-up question about a previous terrain analysis.

{location_context}

{"="*60}
CONVERSATION HISTORY
{"="*60}
{history_text}

{"="*60}
CURRENT QUESTION
{"="*60}
{question}

{"="*60}
INSTRUCTIONS
{"="*60}
Answer based on the previous analysis context in the conversation history.
If the question requires new terrain data for a different location, ask the user to provide new coordinates.
Keep your response focused and relevant to what was previously discussed.
"""
        return prompt