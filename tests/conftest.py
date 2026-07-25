"""
Adds the backend directory to sys.path so tests can import backend modules
regardless of which directory pytest is invoked from.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "tools"))
