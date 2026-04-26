#!/usr/bin/env python3
"""
Topographic Map Dataset Generator

Splits maps into patches and generates training labels based on
color/texture analysis for terrain classification.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from loguru import logger
from typing import Dict, List, Tuple
import random


class MapDatasetGenerator:
    """Generate training dataset from topographic maps"""

    def __init__(self, patch_size: int = 224, stride: int = 150):
        """
        Initialize dataset generator

        Args:
            patch_size: Size of square patches (224 for ResNet)
            stride: Stride for sliding window (overlap if < patch_size)
        """
        self.patch_size = patch_size
        self.stride = stride

        # Terrain classification thresholds (HSV)
        self.terrain_rules = {
            'forest': {
                'hsv_range': [(35, 40, 40), (85, 255, 255)],  # Green
                'min_coverage': 0.30  # 30% green pixels
            },
            'water': {
                'hsv_range': [(90, 50, 50), (130, 255, 255)],  # Blue
                'min_coverage': 0.20
            },
            'urban': {
                'hsv_range': [(0, 0, 0), (180, 50, 100)],  # Dark/black
                'min_coverage': 0.15
            },
            'open_terrain': {
                'hsv_range': [(15, 0, 150), (35, 80, 255)],  # Tan/beige
                'min_coverage': 0.40
            },
            'mountainous': {
                # Detected by high density of contour lines
                'contour_density': 50  # Number of contour lines per patch
            }
        }

        # Tactical features
        self.tactical_rules = {
            'high_ground': {
                # Many contour lines + elevation markers
                'contour_density_min': 40
            },
            'chokepoint': {
                # Narrow passage with terrain on both sides
                'width_ratio_max': 0.3  # Narrow in one direction
            },
            'cover': {
                # Forest or urban areas
                'requires': ['forest', 'urban']
            },
            'killzone': {
                # Open terrain with clear lines of sight
                'requires': ['open_terrain'],
                'vegetation_max': 0.10  # Less than 10% vegetation
            }
        }

    def extract_patches(self, image_path: str, output_dir: str) -> List[Dict]:
        """
        Extract patches from map image

        Args:
            image_path: Path to map image
            output_dir: Directory to save patches

        Returns:
            List of patch metadata
        """
        logger.info(f"Extracting patches from {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return []

        h, w = img.shape[:2]
        output_path = Path(output_dir) / 'patches'
        output_path.mkdir(parents=True, exist_ok=True)

        patches = []
        patch_id = 0

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # Extract patch
                patch = img[y:y+self.patch_size, x:x+self.patch_size]

                # Auto-label based on content
                labels = self._auto_label_patch(patch)

                # Skip mostly empty/white patches (map borders)
                if labels.get('skip', False):
                    continue

                # Save patch
                patch_filename = f"patch_{patch_id:05d}.jpg"
                patch_path = output_path / patch_filename
                cv2.imwrite(str(patch_path), patch)

                patches.append({
                    'id': patch_id,
                    'filename': patch_filename,
                    'position': (x, y),
                    'labels': labels,
                    'source_map': Path(image_path).name
                })

                patch_id += 1

        logger.info(f"Extracted {len(patches)} patches")
        return patches

    def _auto_label_patch(self, patch: np.ndarray) -> Dict:
        """
        Automatically label patch based on visual features

        Args:
            patch: Image patch (BGR)

        Returns:
            Dict of labels
        """
        labels = {}

        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        total_pixels = patch.shape[0] * patch.shape[1]

        # Skip if mostly white (map margins)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        white_ratio = np.sum(gray > 240) / total_pixels
        if white_ratio > 0.7:
            labels['skip'] = True
            return labels

        # Terrain type detection
        terrain_scores = {}

        # Forest
        forest_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        forest_ratio = cv2.countNonZero(forest_mask) / total_pixels
        terrain_scores['forest'] = forest_ratio

        # Water
        water_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
        water_ratio = cv2.countNonZero(water_mask) / total_pixels
        terrain_scores['water'] = water_ratio

        # Urban
        urban_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 100]))
        urban_ratio = cv2.countNonZero(urban_mask) / total_pixels
        terrain_scores['urban'] = urban_ratio

        # Open terrain (tan/beige)
        open_mask = cv2.inRange(hsv, np.array([15, 0, 150]), np.array([35, 80, 255]))
        open_ratio = cv2.countNonZero(open_mask) / total_pixels
        terrain_scores['open_terrain'] = open_ratio

        # Contour density (for mountainous terrain)
        edges = cv2.Canny(gray, 50, 150)
        contour_density = np.sum(edges > 0) / total_pixels
        terrain_scores['mountainous'] = contour_density

        # Assign primary terrain type
        if forest_ratio > 0.30:
            labels['terrain_forest'] = 1
        if water_ratio > 0.20:
            labels['terrain_water'] = 1
        if urban_ratio > 0.15:
            labels['terrain_urban'] = 1
        if open_ratio > 0.40:
            labels['terrain_open'] = 1
        if contour_density > 0.05:
            labels['terrain_mountainous'] = 1

        # If no clear terrain type, mark as mixed
        if not any(k.startswith('terrain_') for k in labels):
            labels['terrain_mixed'] = 1

        # Tactical features
        # High ground: many contour lines
        if contour_density > 0.06:
            labels['tactical_high_ground'] = 1

        # Cover: forest or urban
        if forest_ratio > 0.30 or urban_ratio > 0.15:
            labels['tactical_cover'] = 1

        # Killzone: open with little vegetation
        if open_ratio > 0.50 and forest_ratio < 0.10:
            labels['tactical_killzone'] = 1

        # Concealment: dense vegetation
        if forest_ratio > 0.50:
            labels['tactical_concealment'] = 1

        return labels

    def generate_dataset(self, map_dir: str, output_dir: str) -> Dict:
        """
        Generate complete dataset from all maps in directory

        Args:
            map_dir: Directory containing map images
            output_dir: Output directory for dataset

        Returns:
            Dataset metadata
        """
        logger.info("Generating dataset from maps...")

        map_dir = Path(map_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_patches = []

        # Process each map
        for map_path in map_dir.glob('*.jpg'):
            patches = self.extract_patches(str(map_path), str(output_dir))
            all_patches.extend(patches)

        # Split into train/val/test
        random.shuffle(all_patches)

        train_size = int(0.7 * len(all_patches))
        val_size = int(0.15 * len(all_patches))

        train_patches = all_patches[:train_size]
        val_patches = all_patches[train_size:train_size + val_size]
        test_patches = all_patches[train_size + val_size:]

        dataset = {
            'train': train_patches,
            'val': val_patches,
            'test': test_patches,
            'metadata': {
                'patch_size': self.patch_size,
                'stride': self.stride,
                'total_patches': len(all_patches),
                'train_size': len(train_patches),
                'val_size': len(val_patches),
                'test_size': len(test_patches)
            }
        }

        # Save dataset metadata
        metadata_path = output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Dataset created: {len(train_patches)} train, {len(val_patches)} val, {len(test_patches)} test")

        return dataset

    def augment_patches(self, dataset_dir: str):
        """
        Apply data augmentation to increase dataset size

        Args:
            dataset_dir: Directory containing patches
        """
        logger.info("Applying data augmentation...")

        patches_dir = Path(dataset_dir) / 'patches'
        augmented_dir = Path(dataset_dir) / 'patches_augmented'
        augmented_dir.mkdir(exist_ok=True)

        augmentations = [
            ('rot90', lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
            ('rot180', lambda img: cv2.rotate(img, cv2.ROTATE_180)),
            ('rot270', lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ('flip_h', lambda img: cv2.flip(img, 1)),
            ('flip_v', lambda img: cv2.flip(img, 0)),
        ]

        count = 0
        for patch_path in patches_dir.glob('*.jpg'):
            img = cv2.imread(str(patch_path))

            # Copy original
            cv2.imwrite(str(augmented_dir / patch_path.name), img)

            # Apply augmentations
            for aug_name, aug_func in augmentations:
                aug_img = aug_func(img)
                aug_filename = f"{patch_path.stem}_{aug_name}.jpg"
                cv2.imwrite(str(augmented_dir / aug_filename), aug_img)
                count += 1

        logger.info(f"Created {count} augmented patches")


def main():
    """Test dataset generation"""
    import sys

    generator = MapDatasetGenerator(patch_size=224, stride=150)

    # Generate dataset from uploaded maps
    maps_dir = './uploads'
    output_dir = './map_dataset'

    if not Path(maps_dir).exists():
        logger.error(f"Maps directory not found: {maps_dir}")
        sys.exit(1)

    # Generate dataset
    dataset = generator.generate_dataset(maps_dir, output_dir)

    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"Total patches: {dataset['metadata']['total_patches']}")
    print(f"Training set: {dataset['metadata']['train_size']}")
    print(f"Validation set: {dataset['metadata']['val_size']}")
    print(f"Test set: {dataset['metadata']['test_size']}")
    print(f"\nOutput directory: {output_dir}")
    print("="*70)

    # Optionally apply augmentation
    print("\nApplying data augmentation...")
    generator.augment_patches(output_dir)


if __name__ == "__main__":
    main()
