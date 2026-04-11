#!/usr/bin/env python3
"""
End-to-End Test Suite for Coordinate Functionality
Tests coordinate parsing, terrain fetching, and full tactical analysis pipeline
"""

import sys
import os
import json
import logging
from typing import Dict, List
import requests

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordinate_parser import CoordinateParser, extract_coordinates
from terrain_data_fetcher import TerrainDataFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoordinateFunctionalityTester:
    """Comprehensive test suite for coordinate functionality"""

    def __init__(self):
        self.parser = CoordinateParser()
        self.terrain_fetcher = TerrainDataFetcher()
        self.backend_url = "http://localhost:5001"  # Updated to match config.py

        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []

    def print_header(self, text: str):
        """Print a formatted test section header"""
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80)

    def print_test(self, name: str, passed: bool, details: str = ""):
        """Print test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"

        print(f"{color}{status}{reset} - {name}")
        if details:
            print(f"       {details}")

        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1

        self.test_results.append({
            'name': name,
            'passed': passed,
            'details': details
        })

    def test_coordinate_parser(self):
        """Test coordinate parsing with various formats"""
        self.print_header("TEST SUITE 1: Coordinate Parser")

        test_cases = [
            {
                'name': 'Decimal coordinates (basic)',
                'input': '40.7128, -74.0060',
                'expected': {'lat': 40.7128, 'lon': -74.0060}
            },
            {
                'name': 'Decimal coordinates (negative)',
                'input': '-33.8688, 151.2093',
                'expected': {'lat': -33.8688, 'lon': 151.2093}
            },
            {
                'name': 'Labeled coordinates (lat/lon)',
                'input': 'lat: 51.5074, lon: -0.1278',
                'expected': {'lat': 51.5074, 'lon': -0.1278}
            },
            {
                'name': 'Labeled coordinates (latitude/longitude)',
                'input': 'latitude: 35.6762, longitude: 139.6503',
                'expected': {'lat': 35.6762, 'lon': 139.6503}
            },
            {
                'name': 'DMS format (degrees, minutes, seconds)',
                'input': '40°42\'51"N, 74°00\'21"W',
                'expected_approx': {'lat': 40.7141, 'lon': -74.0058}
            },
            {
                'name': 'Coordinates in text',
                'input': 'Please analyze coordinates 48.8566, 2.3522 for defensive positions',
                'expected': {'lat': 48.8566, 'lon': 2.3522}
            },
            {
                'name': 'Invalid coordinates (out of range latitude)',
                'input': '95.0000, 100.0000',
                'expected': None
            },
            {
                'name': 'No coordinates',
                'input': 'This is just text without any coordinates',
                'expected': None
            }
        ]

        for test in test_cases:
            result = self.parser.parse(test['input'])

            if 'expected_approx' in test:
                # For DMS, check approximate match (within 0.01 degrees)
                if result and 'lat' in result and 'lon' in result:
                    lat_close = abs(result['lat'] - test['expected_approx']['lat']) < 0.01
                    lon_close = abs(result['lon'] - test['expected_approx']['lon']) < 0.01
                    passed = lat_close and lon_close
                    details = f"Got: {result}, Expected ~{test['expected_approx']}"
                else:
                    passed = False
                    details = f"Got: {result}, Expected: {test['expected_approx']}"
            else:
                passed = result == test['expected']
                details = f"Got: {result}, Expected: {test['expected']}"

            self.print_test(test['name'], passed, details)

    def test_coordinate_formatting(self):
        """Test coordinate formatting"""
        self.print_header("TEST SUITE 2: Coordinate Formatting")

        test_cases = [
            {
                'name': 'Format positive coordinates',
                'lat': 40.7128,
                'lon': 74.0060,
                'expected_contains': ['40.7128', 'N', '74.0060', 'E']
            },
            {
                'name': 'Format negative coordinates',
                'lat': -33.8688,
                'lon': -151.2093,
                'expected_contains': ['33.8688', 'S', '151.2093', 'W']
            }
        ]

        for test in test_cases:
            result = self.parser.format_coordinates(test['lat'], test['lon'])
            passed = all(expected in result for expected in test['expected_contains'])
            self.print_test(test['name'], passed, f"Formatted: {result}")

    def test_terrain_data_fetcher(self):
        """Test terrain data fetching from APIs"""
        self.print_header("TEST SUITE 3: Terrain Data Fetcher")

        # Test with known location (Paris, France)
        lat, lon = 48.8566, 2.3522

        try:
            print(f"\nFetching terrain data for Paris ({lat}, {lon})...")
            terrain_data = self.terrain_fetcher.fetch_terrain_data(lat, lon, radius_km=2)

            # Test 1: Basic structure
            required_keys = ['location', 'place_name', 'roads', 'waterways', 'buildings',
                           'forests', 'terrain_analysis']
            has_keys = all(key in terrain_data for key in required_keys)
            self.print_test(
                'Terrain data structure',
                has_keys,
                f"Has all required keys: {has_keys}"
            )

            # Test 2: Reverse geocoding
            place_name = terrain_data.get('place_name', '')
            has_place = len(place_name) > 0 and 'Paris' in place_name
            self.print_test(
                'Reverse geocoding (place name)',
                has_place,
                f"Place: {place_name}"
            )

            # Test 3: Elevation data (Open-Meteo API)
            has_elevation = terrain_data.get('elevation') is not None
            elevation = terrain_data.get('elevation', 'N/A')
            self.print_test(
                'Elevation data from Open-Meteo API',
                has_elevation,
                f"Elevation: {elevation}m"
            )

            # Test 4: OSM features
            roads = terrain_data.get('roads', [])
            buildings = terrain_data.get('buildings', [])
            has_osm_data = len(roads) > 0 or len(buildings) > 0
            self.print_test(
                'OpenStreetMap features',
                has_osm_data,
                f"Roads: {len(roads)}, Buildings: {len(buildings)}"
            )

            # Test 5: Terrain analysis
            analysis = terrain_data.get('terrain_analysis', {})
            has_analysis = 'cover_availability' in analysis and 'urban_terrain' in analysis
            self.print_test(
                'Terrain tactical analysis',
                has_analysis,
                f"Urban: {analysis.get('urban_terrain')}, Cover: {analysis.get('cover_availability')}"
            )

        except Exception as e:
            self.print_test('Terrain data fetcher', False, f"Exception: {str(e)}")

    def test_api_endpoint(self):
        """Test the /analyze_coordinates API endpoint"""
        self.print_header("TEST SUITE 4: API Endpoint Testing")

        # Check if backend is running
        try:
            health_response = requests.get(f"{self.backend_url}/health", timeout=5)
            backend_running = health_response.status_code == 200
        except:
            backend_running = False

        if not backend_running:
            print("       ⚠ Backend not running on localhost:5000")
            print("       ⚠ Skipping API endpoint tests")
            print("       ℹ Start backend with: cd backend && python app.py")
            return

        print(f"       ✓ Backend is running")

        # Test case 1: Valid coordinates in prompt
        test_prompts = [
            {
                'name': 'Simple decimal coordinates',
                'message': 'Analyze defensive positions at 48.8566, 2.3522',
                'scenario': 'Urban defensive analysis',
                'should_succeed': True
            },
            {
                'name': 'Labeled coordinates with radius',
                'message': 'Tactical analysis for lat: 40.7128, lon: -74.0060 within 3km radius',
                'scenario': 'Coastal defense',
                'should_succeed': True
            },
            {
                'name': 'DMS coordinates',
                'message': 'OCOKA analysis at 51°30\'26"N, 0°7\'39"W',
                'scenario': 'Key terrain evaluation',
                'should_succeed': True
            },
            {
                'name': 'No coordinates (should fail)',
                'message': 'What is a good defensive position?',
                'scenario': 'General query',
                'should_succeed': False
            }
        ]

        for test in test_prompts:
            try:
                print(f"\n   Testing: {test['name']}")
                response = requests.post(
                    f"{self.backend_url}/analyze_coordinates",
                    json={
                        'message': test['message'],
                        'scenario': test['scenario']
                    },
                    timeout=60  # Longer timeout for LLM processing
                )

                success = response.status_code == 200

                if test['should_succeed']:
                    if success:
                        data = response.json()
                        has_required = all(k in data for k in ['success', 'strategy', 'coordinates', 'terrain_data'])
                        passed = has_required

                        if passed:
                            coords = data.get('coordinates', {})
                            details = f"✓ Coords: {coords.get('lat')}, {coords.get('lon')}"
                        else:
                            details = f"Missing required fields in response"
                    else:
                        passed = False
                        details = f"Expected success but got status {response.status_code}"
                else:
                    # Should fail
                    passed = not success
                    if passed:
                        error_data = response.json()
                        details = f"✓ Correctly rejected: {error_data.get('error', '')[:50]}"
                    else:
                        details = f"Should have failed but succeeded"

                self.print_test(test['name'], passed, details)

            except Exception as e:
                self.print_test(test['name'], False, f"Exception: {str(e)}")

    def test_error_handling(self):
        """Test error handling for edge cases"""
        self.print_header("TEST SUITE 5: Error Handling")

        # Test invalid coordinate ranges
        test_cases = [
            {
                'name': 'Invalid latitude (> 90)',
                'lat': 95.0,
                'lon': 100.0,
                'should_be_valid': False
            },
            {
                'name': 'Invalid latitude (< -90)',
                'lat': -95.0,
                'lon': 50.0,
                'should_be_valid': False
            },
            {
                'name': 'Invalid longitude (> 180)',
                'lat': 45.0,
                'lon': 185.0,
                'should_be_valid': False
            },
            {
                'name': 'Valid edge case (North Pole)',
                'lat': 90.0,
                'lon': 0.0,
                'should_be_valid': True
            },
            {
                'name': 'Valid edge case (International Date Line)',
                'lat': 0.0,
                'lon': 180.0,
                'should_be_valid': True
            }
        ]

        for test in test_cases:
            is_valid = self.parser._validate_coordinates(test['lat'], test['lon'])
            passed = is_valid == test['should_be_valid']
            self.print_test(
                test['name'],
                passed,
                f"Lat: {test['lat']}, Lon: {test['lon']}, Valid: {is_valid}"
            )

    def print_summary(self):
        """Print test summary"""
        self.print_header("TEST SUMMARY")

        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0

        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {self.tests_passed} (\033[92m{pass_rate:.1f}%\033[0m)")
        print(f"Failed: {self.tests_failed}")

        if self.tests_failed > 0:
            print("\n\033[91mFailed Tests:\033[0m")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  ✗ {result['name']}")
                    if result['details']:
                        print(f"    {result['details']}")
        else:
            print("\n\033[92m✓ All tests passed!\033[0m")

        print("\n" + "="*80)

        # Save results to JSON
        results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'total': total_tests,
                'passed': self.tests_passed,
                'failed': self.tests_failed,
                'pass_rate': pass_rate,
                'results': self.test_results
            }, f, indent=2)

        print("Test results saved to: backend/test_results.json")

    def test_location_accuracy(self):
        """Test that specific coordinates return correct locations and terrain data"""
        self.print_header("TEST SUITE 6: Location Accuracy & Data Verification")

        # Test well-known locations with expected data
        test_locations = [
            {
                'name': 'Eiffel Tower, Paris',
                'lat': 48.8584,
                'lon': 2.2945,
                'expected_place_contains': ['Paris', 'France'],
                'expected_terrain': {
                    'urban_terrain': True,
                    'has_buildings': True
                }
            },
            {
                'name': 'Statue of Liberty, New York',
                'lat': 40.6892,
                'lon': -74.0445,
                'expected_place_contains': ['New York', 'United States'],
                'expected_terrain': {
                    'urban_terrain': False,  # Island location
                    'has_waterways': True
                }
            },
            {
                'name': 'Big Ben, London',
                'lat': 51.5007,
                'lon': -0.1246,
                'expected_place_contains': ['London', 'United Kingdom'],
                'expected_terrain': {
                    'urban_terrain': True,
                    'has_buildings': True,
                    'has_waterways': True  # Thames river nearby
                }
            },
            {
                'name': 'Sydney Opera House, Australia',
                'lat': -33.8568,
                'lon': 151.2153,
                'expected_place_contains': ['Sydney', 'Australia'],
                'expected_terrain': {
                    'urban_terrain': True,
                    'has_waterways': True  # Harbor location
                }
            },
            {
                'name': 'Mount Fuji, Japan',
                'lat': 35.3606,
                'lon': 138.7274,
                'expected_place_contains': ['Japan'],
                'expected_terrain': {
                    'urban_terrain': False,  # Mountain location
                    'has_buildings': False
                }
            }
        ]

        for location in test_locations:
            try:
                print(f"\n   Testing: {location['name']}")
                terrain_data = self.terrain_fetcher.fetch_terrain_data(
                    location['lat'],
                    location['lon'],
                    radius_km=2
                )

                # Test 1: Place name accuracy
                place_name = terrain_data.get('place_name', '')
                place_contains_expected = all(
                    expected.lower() in place_name.lower()
                    for expected in location['expected_place_contains']
                )

                self.print_test(
                    f"{location['name']} - Location name",
                    place_contains_expected,
                    f"Place: {place_name}"
                )

                # Test 2: Terrain characteristics
                analysis = terrain_data.get('terrain_analysis', {})
                buildings = terrain_data.get('buildings', [])
                waterways = terrain_data.get('waterways', [])

                if 'urban_terrain' in location['expected_terrain']:
                    expected_urban = location['expected_terrain']['urban_terrain']
                    actual_urban = analysis.get('urban_terrain', False)
                    self.print_test(
                        f"{location['name']} - Urban terrain detection",
                        actual_urban == expected_urban,
                        f"Expected urban: {expected_urban}, Got: {actual_urban}"
                    )

                if 'has_buildings' in location['expected_terrain']:
                    expected_buildings = location['expected_terrain']['has_buildings']
                    has_buildings = len(buildings) > 0
                    self.print_test(
                        f"{location['name']} - Building detection",
                        has_buildings == expected_buildings,
                        f"Buildings found: {len(buildings)}"
                    )

                if 'has_waterways' in location['expected_terrain']:
                    expected_waterways = location['expected_terrain']['has_waterways']
                    has_waterways = len(waterways) > 0
                    self.print_test(
                        f"{location['name']} - Waterway detection",
                        has_waterways == expected_waterways,
                        f"Waterways found: {len(waterways)}"
                    )

            except Exception as e:
                self.print_test(
                    f"{location['name']} - Data fetch",
                    False,
                    f"Exception: {str(e)}"
                )

    def run_all_tests(self):
        """Run all test suites"""
        print("\n" + "="*80)
        print("  COORDINATE FUNCTIONALITY END-TO-END TEST SUITE")
        print("="*80)

        self.test_coordinate_parser()
        self.test_coordinate_formatting()
        self.test_terrain_data_fetcher()
        self.test_error_handling()
        self.test_location_accuracy()
        self.test_api_endpoint()

        self.print_summary()


if __name__ == '__main__':
    tester = CoordinateFunctionalityTester()
    tester.run_all_tests()
