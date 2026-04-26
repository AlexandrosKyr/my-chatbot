#!/usr/bin/env python3
"""
Integrated Map Analysis Pipeline

Combines:
- OCR for Greek text and location extraction
- ResNet50 terrain classification
- Google Maps API for elevation/terrain data
- Traditional CV analysis
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from dotenv import load_dotenv

# Use standard logging instead of loguru for Flask compatibility
logger = logging.getLogger(__name__)


class IntegratedMapAnalyzer:
    """Complete map analysis pipeline"""

    def __init__(self, google_api_key: Optional[str] = None):
        """
        Initialize integrated analyzer

        Args:
            google_api_key: Google Maps API key (optional)
        """
        # Lazy imports to avoid dependency issues
        from topographic_analyzer import TopographicMapAnalyzer

        self.google_api_key = google_api_key

        # Initialize components
        logger.info("Initializing analyzers...")

        # 1. Topographic analyzer (OCR, CV, Google API)
        self.topo_analyzer = TopographicMapAnalyzer(google_api_key)

        # 2. ResNet50 terrain classifier (lazy load)
        self.terrain_classifier = None
        model_path = './models/map_classifier_resnet50.pth'
        if Path(model_path).exists():
            try:
                from map_classifier_inference import MapClassifier
                import torch
                device = 'mps' if torch.backends.mps.is_available() else 'cpu'
                self.terrain_classifier = MapClassifier(model_path, device=device)
                logger.info("ResNet50 terrain classifier loaded")
            except Exception as e:
                logger.warning(f"ResNet50 model failed to load: {e}")
                self.terrain_classifier = None
        else:
            logger.warning("ResNet50 model not found - terrain classification disabled")

        logger.info("Integrated analyzer initialized")

    def extract_location_from_text(self, text_data: Dict) -> Optional[Dict]:
        """
        Extract location information from OCR text

        Args:
            text_data: OCR results with place names and coordinates

        Returns:
            Dict with extracted location info or None
        """
        location_info = {
            'place_names': [],
            'coordinates': [],
            'detected_lat': None,
            'detected_lon': None
        }

        # Get place names
        location_info['place_names'] = text_data.get('place_names', [])[:10]  # Top 10

        # Look for coordinate patterns in text
        raw_text = text_data.get('raw_text', '')

        # Pattern 1: Decimal degrees (e.g., 40.7128, 23.3653)
        decimal_coords = re.findall(r'(\d{2}\.\d{4,6})[,\s]+(\d{2}\.\d{4,6})', raw_text)
        if decimal_coords:
            lat, lon = decimal_coords[0]
            location_info['detected_lat'] = float(lat)
            location_info['detected_lon'] = float(lon)

        # Pattern 2: DMS format (e.g., 40°42'46"N, 23°21'55"E)
        # Common in Greek military maps
        dms_pattern = r'(\d{2})[°]\s?(\d{2})[\']\s?(\d{2})["]?\s?([NS])'
        lat_match = re.search(dms_pattern, raw_text)
        if lat_match:
            deg, min, sec, dir = lat_match.groups()
            lat = int(deg) + int(min)/60 + int(sec)/3600
            if dir == 'S':
                lat = -lat
            location_info['detected_lat'] = lat

        dms_pattern_lon = r'(\d{2})[°]\s?(\d{2})[\']\s?(\d{2})["]?\s?([EW])'
        lon_match = re.search(dms_pattern_lon, raw_text)
        if lon_match:
            deg, min, sec, dir = lon_match.groups()
            lon = int(deg) + int(min)/60 + int(sec)/3600
            if dir == 'W':
                lon = -lon
            location_info['detected_lon'] = lon

        return location_info

    def prompt_user_for_location(self, detected_location: Dict) -> Tuple[float, float]:
        """
        Prompt user to confirm or provide location

        Args:
            detected_location: Auto-detected location info

        Returns:
            Tuple of (latitude, longitude)
        """
        print("\n" + "="*70)
        print("LOCATION CONFIRMATION")
        print("="*70)

        # Show detected place names
        if detected_location['place_names']:
            print("\nDetected place names from map:")
            for i, name in enumerate(detected_location['place_names'][:5], 1):
                print(f"  {i}. {name}")

        # Show detected coordinates
        if detected_location['detected_lat'] and detected_location['detected_lon']:
            print(f"\nDetected coordinates:")
            print(f"  Latitude:  {detected_location['detected_lat']:.6f}")
            print(f"  Longitude: {detected_location['detected_lon']:.6f}")
            print("\nUse detected coordinates? (y/n): ", end='')
            choice = input().strip().lower()

            if choice == 'y':
                return detected_location['detected_lat'], detected_location['detected_lon']

        # Manual input
        print("\nPlease provide map center coordinates:")
        print("(For Greek maps, typically: Lat 37-41°N, Lon 20-28°E)")

        while True:
            try:
                lat_str = input("Latitude (e.g., 40.6401): ").strip()
                lon_str = input("Longitude (e.g., 22.9444): ").strip()

                lat = float(lat_str)
                lon = float(lon_str)

                # Validate Greece region
                if 35 <= lat <= 42 and 19 <= lon <= 30:
                    return lat, lon
                else:
                    print("⚠ Coordinates outside Greece region. Please try again.")
            except ValueError:
                print("⚠ Invalid format. Please enter decimal degrees (e.g., 40.6401)")

    def analyze_map_complete(self, image_path: str,
                            lat: Optional[float] = None,
                            lon: Optional[float] = None,
                            auto_location: bool = True) -> Dict:
        """
        Complete integrated map analysis

        Args:
            image_path: Path to map image
            lat: Optional latitude (if known)
            lon: Optional longitude (if known)
            auto_location: Try to auto-detect location from map

        Returns:
            Complete analysis results
        """
        logger.info("="*70)
        logger.info("INTEGRATED MAP ANALYSIS PIPELINE")
        logger.info("="*70)

        results = {
            'image_path': image_path,
            'location': {},
            'topographic': {},
            'terrain_classification': {},
            'google_maps': {},
            'summary': {}
        }

        # PHASE 1: OCR and Location Detection
        logger.info("\n[1/4] Text Extraction & Location Detection")
        text_data = self.topo_analyzer.extract_greek_text(image_path)
        results['topographic']['text'] = text_data

        # Extract and confirm location
        if auto_location and (lat is None or lon is None):
            detected_location = self.extract_location_from_text(text_data)
            lat, lon = self.prompt_user_for_location(detected_location)

        results['location'] = {
            'latitude': lat,
            'longitude': lon,
            'place_names': text_data.get('place_names', [])[:5]
        }

        logger.info(f"Location confirmed: {lat:.6f}, {lon:.6f}")

        # PHASE 2: Traditional CV Analysis
        logger.info("\n[2/4] Traditional Computer Vision Analysis")

        # Grid coordinates
        grid_data = self.topo_analyzer.extract_grid_coordinates(image_path)
        results['topographic']['grid'] = grid_data

        # Terrain colors
        terrain_colors = self.topo_analyzer.analyze_terrain_colors(image_path)
        results['topographic']['terrain_colors'] = terrain_colors

        # Road/river extraction
        roads = self.topo_analyzer.extract_road_network(image_path)
        rivers = self.topo_analyzer.extract_river_network(image_path)
        contours = self.topo_analyzer.extract_contour_lines(image_path)

        results['topographic']['roads'] = {'num_segments': roads['num_segments']}
        results['topographic']['rivers'] = {'num_segments': rivers['num_segments']}
        results['topographic']['contours'] = {'num_contours': contours['num_contours']}

        # PHASE 3: ResNet50 Terrain Classification
        logger.info("\n[3/4] ResNet50 Terrain Classification")
        if self.terrain_classifier:
            terrain_results = self.terrain_classifier.predict_map_regions(
                image_path, patch_size=224, stride=200
            )
            results['terrain_classification'] = terrain_results
            logger.info(f"Terrain distribution: {terrain_results['terrain_distribution']}")
        else:
            logger.warning("Terrain classifier not available")
            results['terrain_classification'] = None

        # PHASE 4: Google Maps API
        logger.info("\n[4/4] Google Maps API Integration")
        if self.google_api_key and lat and lon:
            google_data = self.topo_analyzer.get_google_terrain_info(lat, lon, radius=5000)
            results['google_maps'] = google_data

            if 'elevation' in google_data:
                logger.info(f"Elevation: {google_data['elevation']:.1f}m")
        else:
            logger.warning("Google Maps API not available or no coordinates")
            results['google_maps'] = {}

        # Generate Summary
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*70)

        results['summary'] = self._generate_summary(results)

        return results

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate human-readable summary"""
        summary = {}

        # Location
        if results['location'].get('latitude'):
            summary['location'] = f"{results['location']['latitude']:.6f}, {results['location']['longitude']:.6f}"
            if results['location']['place_names']:
                summary['nearest_places'] = ', '.join(results['location']['place_names'][:3])

        # Elevation
        if results['google_maps'].get('elevation'):
            summary['elevation'] = f"{results['google_maps']['elevation']:.0f}m"

        # Terrain (ResNet50)
        if results['terrain_classification']:
            terrain_dist = results['terrain_classification']['terrain_distribution']
            dominant_terrain = max(terrain_dist.items(), key=lambda x: x[1]) if terrain_dist else None
            if dominant_terrain:
                summary['dominant_terrain'] = f"{dominant_terrain[0]} ({dominant_terrain[1]*100:.0f}%)"

            tactical_dist = results['terrain_classification']['tactical_distribution']
            if tactical_dist:
                summary['tactical_features'] = ', '.join([
                    f"{k} ({v*100:.0f}%)" for k, v in sorted(tactical_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                ])

        # Topographic features
        topo = results['topographic']
        summary['infrastructure'] = f"{topo['roads']['num_segments']} road segments, {topo['rivers']['num_segments']} river segments"
        summary['elevation_detail'] = f"{topo['contours']['num_contours']} contour lines detected"

        return summary

    def print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*70)
        print("COMPLETE MAP ANALYSIS RESULTS")
        print("="*70)

        summary = results['summary']

        print("\n📍 LOCATION:")
        print(f"   Coordinates: {summary.get('location', 'Unknown')}")
        if 'nearest_places' in summary:
            print(f"   Near: {summary['nearest_places']}")
        if 'elevation' in summary:
            print(f"   Elevation: {summary['elevation']}")

        print("\n🗺️  TERRAIN ANALYSIS:")
        if 'dominant_terrain' in summary:
            print(f"   Dominant: {summary['dominant_terrain']}")
        if 'tactical_features' in summary:
            print(f"   Tactical: {summary['tactical_features']}")

        print("\n🛣️  INFRASTRUCTURE:")
        print(f"   {summary.get('infrastructure', 'N/A')}")

        print("\n⛰️  TOPOGRAPHY:")
        print(f"   {summary.get('elevation_detail', 'N/A')}")

        # Color-based terrain
        if 'terrain_colors' in results['topographic']:
            print("\n🎨 COLOR ANALYSIS:")
            for terrain, data in results['topographic']['terrain_colors'].items():
                if data['coverage_percent'] > 1:
                    print(f"   {terrain}: {data['coverage_percent']:.1f}%")

        print("\n" + "="*70)


def main():
    """Test the integrated pipeline"""
    load_dotenv()
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')

    if not api_key or api_key == 'your_api_key_here':
        logger.warning("Google Maps API key not configured in .env")
        api_key = None

    # Initialize analyzer
    analyzer = IntegratedMapAnalyzer(google_api_key=api_key)

    # Test map
    test_map = './uploads/1000033857.jpg'

    if not Path(test_map).exists():
        logger.error(f"Test map not found: {test_map}")
        return

    # Run complete analysis
    results = analyzer.analyze_map_complete(
        test_map,
        auto_location=True  # Will prompt user for location
    )

    # Print results
    analyzer.print_results(results)

    # Optionally save to JSON
    import json
    output_file = './map_analysis_results.json'
    with open(output_file, 'w') as f:
        # Remove non-serializable objects
        clean_results = {
            'location': results['location'],
            'summary': results['summary'],
            'terrain_classification': results['terrain_classification'],
            'google_maps': results['google_maps']
        }
        json.dump(clean_results, f, indent=2)
    logger.info(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
