#!/usr/bin/env python3
"""
Topographic Map Analyzer for Greek Military Maps

Extracts coordinates, terrain features, road/river networks,
and integrates with Google Maps API for elevation data.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import requests
from typing import Dict, List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)


class TopographicMapAnalyzer:
    """Analyze topographic maps with coordinate extraction and terrain analysis"""

    def __init__(self, google_maps_api_key: Optional[str] = None):
        """
        Initialize topographic map analyzer

        Args:
            google_maps_api_key: Optional Google Maps API key for elevation/terrain data
        """
        self.google_api_key = google_maps_api_key

        # Greek language support for OCR
        self.tesseract_config = '--oem 3 --psm 6 -l ell+eng'

        # Color ranges for terrain detection (HSV)
        self.terrain_colors = {
            'vegetation': {
                'lower': np.array([35, 40, 40]),   # Light green
                'upper': np.array([85, 255, 255])   # Dark green
            },
            'water': {
                'lower': np.array([90, 50, 50]),    # Light blue
                'upper': np.array([130, 255, 255])  # Dark blue
            },
            'urban': {
                'lower': np.array([0, 0, 0]),       # Dark/black areas
                'upper': np.array([180, 50, 100])
            },
            'elevation': {
                'lower': np.array([10, 30, 100]),   # Brown/tan contours
                'upper': np.array([25, 150, 200])
            }
        }

    def extract_greek_text(self, image_path: str) -> Dict[str, List[str]]:
        """
        Extract Greek and English text from map

        Args:
            image_path: Path to map image

        Returns:
            Dict with place_names and coordinates
        """
        logger.info("Extracting text (Greek + English) from map...")

        try:
            img = Image.open(image_path)

            # Extract all text
            text = pytesseract.image_to_string(img, config=self.tesseract_config)

            # Parse text for place names and coordinates
            lines = text.split('\n')
            place_names = []
            coordinates = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for coordinate patterns (various formats)
                # Format: XX°XX'XX" or decimal degrees
                coord_pattern = r'\d{1,3}[°]\s?\d{1,2}[\']\s?\d{1,2}["]?'
                if re.search(coord_pattern, line):
                    coordinates.append(line)
                # Look for grid references (e.g., FL, numbers)
                elif re.search(r'^\d{1,3}$', line) or 'FL' in line:
                    coordinates.append(line)
                # Assume everything else is place names
                elif len(line) > 2:
                    place_names.append(line)

            logger.info(f"Found {len(place_names)} place names, {len(coordinates)} coordinate references")

            return {
                'place_names': place_names,
                'coordinates': coordinates,
                'raw_text': text
            }

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {'place_names': [], 'coordinates': [], 'raw_text': ''}

    def extract_grid_coordinates(self, image_path: str) -> Dict[str, any]:
        """
        Extract grid coordinate system from map edges

        Args:
            image_path: Path to map image

        Returns:
            Dict with grid info and bounding coordinates
        """
        logger.info("Extracting grid coordinates...")

        try:
            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            # Extract text from edges (where grid coords usually are)
            edge_margin = 100

            edges = {
                'top': img[0:edge_margin, :],
                'bottom': img[h-edge_margin:h, :],
                'left': img[:, 0:edge_margin],
                'right': img[:, w-edge_margin:w]
            }

            grid_info = {}

            for edge_name, edge_img in edges.items():
                # Convert to PIL for tesseract
                edge_pil = Image.fromarray(cv2.cvtColor(edge_img, cv2.COLOR_BGR2RGB))
                text = pytesseract.image_to_string(edge_pil, config=self.tesseract_config)

                # Extract numbers (grid coordinates)
                numbers = re.findall(r'\d+', text)
                if numbers:
                    grid_info[edge_name] = numbers

            logger.info(f"Grid info extracted: {grid_info}")

            # Try to determine coordinate system
            # Greek military maps typically use MGRS or Greek Grid
            return {
                'grid_numbers': grid_info,
                'coordinate_system': self._detect_coordinate_system(grid_info)
            }

        except Exception as e:
            logger.error(f"Grid extraction failed: {e}")
            return {'grid_numbers': {}, 'coordinate_system': 'unknown'}

    def _detect_coordinate_system(self, grid_info: Dict) -> str:
        """Detect which coordinate system is used"""
        # Simple heuristic based on number ranges
        all_numbers = []
        for numbers in grid_info.values():
            all_numbers.extend([int(n) for n in numbers if n.isdigit()])

        if not all_numbers:
            return 'unknown'

        max_num = max(all_numbers)

        if max_num > 500000:
            return 'UTM/MGRS'
        elif max_num > 10000:
            return 'Greek Grid (EGSA87)'
        else:
            return 'Local Grid'

    def analyze_terrain_colors(self, image_path: str) -> Dict[str, Dict]:
        """
        Analyze terrain types based on color segmentation

        Args:
            image_path: Path to map image

        Returns:
            Dict with terrain coverage percentages and masks
        """
        logger.info("Analyzing terrain colors...")

        try:
            img = cv2.imread(image_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            terrain_analysis = {}
            total_pixels = img.shape[0] * img.shape[1]

            for terrain_type, color_range in self.terrain_colors.items():
                # Create mask for this terrain type
                mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])

                # Calculate coverage
                terrain_pixels = cv2.countNonZero(mask)
                coverage_pct = (terrain_pixels / total_pixels) * 100

                terrain_analysis[terrain_type] = {
                    'coverage_percent': round(coverage_pct, 2),
                    'pixel_count': terrain_pixels
                }

                logger.info(f"{terrain_type}: {coverage_pct:.1f}% coverage")

            return terrain_analysis

        except Exception as e:
            logger.error(f"Terrain color analysis failed: {e}")
            return {}

    def extract_road_network(self, image_path: str) -> Dict[str, any]:
        """
        Extract road network using line detection

        Args:
            image_path: Path to map image

        Returns:
            Dict with road segments and characteristics
        """
        logger.info("Extracting road network...")

        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect red/brown roads (typical for main roads)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Red roads (highways/main roads)
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 100, 100])
            red_upper2 = np.array([180, 255, 255])

            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            road_mask = cv2.bitwise_or(mask_red1, mask_red2)

            # Brown roads (secondary roads)
            brown_lower = np.array([10, 50, 50])
            brown_upper = np.array([20, 200, 200])
            brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

            # Combine masks
            combined_mask = cv2.bitwise_or(road_mask, brown_mask)

            # Find contours (road segments)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter by size (remove noise)
            road_segments = [c for c in contours if cv2.contourArea(c) > 50]

            logger.info(f"Found {len(road_segments)} road segments")

            return {
                'num_segments': len(road_segments),
                'road_mask': combined_mask,
                'segments': road_segments[:100]  # Limit to avoid huge data
            }

        except Exception as e:
            logger.error(f"Road extraction failed: {e}")
            return {'num_segments': 0, 'segments': []}

    def extract_river_network(self, image_path: str) -> Dict[str, any]:
        """
        Extract river/water network using blue line detection

        Args:
            image_path: Path to map image

        Returns:
            Dict with river segments
        """
        logger.info("Extracting river network...")

        try:
            img = cv2.imread(image_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Blue rivers/streams
            blue_lower = np.array([95, 80, 80])
            blue_upper = np.array([115, 255, 255])

            river_mask = cv2.inRange(hsv, blue_lower, blue_upper)

            # Morphological operations to connect broken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            river_mask = cv2.morphologyEx(river_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(river_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter by size and shape (rivers are elongated)
            river_segments = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 30:
                    x, y, w, h = cv2.boundingRect(c)
                    aspect_ratio = max(w, h) / (min(w, h) + 1)
                    if aspect_ratio > 2:  # Elongated shape
                        river_segments.append(c)

            logger.info(f"Found {len(river_segments)} river segments")

            return {
                'num_segments': len(river_segments),
                'river_mask': river_mask,
                'segments': river_segments[:100]
            }

        except Exception as e:
            logger.error(f"River extraction failed: {e}")
            return {'num_segments': 0, 'segments': []}

    def extract_contour_lines(self, image_path: str) -> Dict[str, any]:
        """
        Extract elevation contour lines

        Args:
            image_path: Path to map image

        Returns:
            Dict with contour info
        """
        logger.info("Extracting contour lines...")

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Apply edge detection to find contour lines
            edges = cv2.Canny(img, 50, 150)

            # Morphological closing to connect broken contours
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Filter for contour-like shapes (closed or nearly closed curves)
            elevation_contours = [c for c in contours if len(c) > 20]

            logger.info(f"Found {len(elevation_contours)} potential contour lines")

            return {
                'num_contours': len(elevation_contours),
                'contour_mask': edges
            }

        except Exception as e:
            logger.error(f"Contour extraction failed: {e}")
            return {'num_contours': 0}

    def get_google_elevation(self, lat: float, lon: float) -> Optional[float]:
        """
        Get elevation from Google Maps Elevation API

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Elevation in meters or None
        """
        if not self.google_api_key:
            logger.warning("No Google API key provided")
            return None

        try:
            url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={self.google_api_key}"
            response = requests.get(url, timeout=5)
            data = response.json()

            if data['status'] == 'OK':
                elevation = data['results'][0]['elevation']
                logger.info(f"Elevation at ({lat}, {lon}): {elevation}m")
                return elevation
            else:
                logger.error(f"Google API error: {data['status']}")
                return None

        except Exception as e:
            logger.error(f"Google API request failed: {e}")
            return None

    def get_google_terrain_info(self, lat: float, lon: float, radius: int = 1000) -> Dict:
        """
        Get terrain characteristics from Google Maps APIs

        Args:
            lat: Latitude
            lon: Longitude
            radius: Radius in meters for nearby places search

        Returns:
            Dict with terrain characteristics
        """
        if not self.google_api_key:
            logger.warning("No Google API key provided")
            return {}

        terrain_info = {}

        # Get elevation
        elevation = self.get_google_elevation(lat, lon)
        if elevation:
            terrain_info['elevation'] = elevation

        # Get nearby places (for terrain context)
        try:
            places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius={radius}&key={self.google_api_key}"
            response = requests.get(places_url, timeout=5)
            data = response.json()

            if data['status'] == 'OK':
                place_types = []
                for place in data.get('results', [])[:10]:
                    place_types.extend(place.get('types', []))

                # Categorize terrain based on place types
                terrain_info['nearby_features'] = {
                    'natural_features': [t for t in place_types if t in ['natural_feature', 'park', 'forest']],
                    'water_bodies': [t for t in place_types if t in ['river', 'lake', 'reservoir']],
                    'urban_areas': [t for t in place_types if t in ['locality', 'sublocality', 'neighborhood']]
                }
        except Exception as e:
            logger.error(f"Places API request failed: {e}")

        return terrain_info

    def analyze_map_comprehensive(self, image_path: str, center_lat: Optional[float] = None,
                                  center_lon: Optional[float] = None) -> Dict:
        """
        Comprehensive map analysis combining all methods

        Args:
            image_path: Path to map image
            center_lat: Optional center latitude for Google API queries
            center_lon: Optional center longitude for Google API queries

        Returns:
            Complete analysis dict
        """
        logger.info("="*70)
        logger.info("COMPREHENSIVE TOPOGRAPHIC MAP ANALYSIS")
        logger.info("="*70)

        analysis = {}

        # 1. Text extraction
        logger.info("\n[1/7] Greek Text & Coordinate Extraction")
        analysis['text'] = self.extract_greek_text(image_path)

        # 2. Grid coordinates
        logger.info("\n[2/7] Grid Coordinate System")
        analysis['grid'] = self.extract_grid_coordinates(image_path)

        # 3. Terrain colors
        logger.info("\n[3/7] Terrain Color Analysis")
        analysis['terrain'] = self.analyze_terrain_colors(image_path)

        # 4. Road network
        logger.info("\n[4/7] Road Network Extraction")
        analysis['roads'] = self.extract_road_network(image_path)

        # 5. River network
        logger.info("\n[5/7] River Network Extraction")
        analysis['rivers'] = self.extract_river_network(image_path)

        # 6. Contour lines
        logger.info("\n[6/7] Elevation Contour Detection")
        analysis['contours'] = self.extract_contour_lines(image_path)

        # 7. Google API integration
        if center_lat and center_lon and self.google_api_key:
            logger.info("\n[7/7] Google Maps API Integration")
            analysis['google_data'] = self.get_google_terrain_info(center_lat, center_lon)
        else:
            logger.info("\n[7/7] Skipping Google API (no coordinates or API key)")
            analysis['google_data'] = {}

        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70)

        return analysis


def main():
    """Test the analyzer on sample maps"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')

    analyzer = TopographicMapAnalyzer(google_maps_api_key=api_key)

    # Test on one of the uploaded maps
    test_map = './uploads/1000033857.jpg'

    if os.path.exists(test_map):
        results = analyzer.analyze_map_comprehensive(test_map)

        # Print summary
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"Place names found: {len(results['text']['place_names'])}")
        print(f"Coordinate references: {len(results['text']['coordinates'])}")
        print(f"Grid system: {results['grid']['coordinate_system']}")
        print(f"\nTerrain coverage:")
        for terrain_type, info in results['terrain'].items():
            print(f"  {terrain_type}: {info['coverage_percent']}%")
        print(f"\nRoad segments: {results['roads']['num_segments']}")
        print(f"River segments: {results['rivers']['num_segments']}")
        print(f"Contour lines: {results['contours']['num_contours']}")
    else:
        print(f"Test map not found: {test_map}")


if __name__ == "__main__":
    main()
