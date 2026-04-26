#!/usr/bin/env python3
"""
Map Classifier Inference

Use trained ResNet50 model to classify topographic map patches.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from train_map_classifier import MapResNet50
from loguru import logger
from typing import Dict, List


class MapClassifier:
    """Inference wrapper for trained map classifier"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize classifier

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)

        # Load model
        self.model = MapResNet50(num_classes=10, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Label names
        self.label_names = [
            'terrain_forest',
            'terrain_water',
            'terrain_urban',
            'terrain_open',
            'terrain_mountainous',
            'terrain_mixed',
            'tactical_high_ground',
            'tactical_cover',
            'tactical_killzone',
            'tactical_concealment'
        ]

        logger.info(f"Classifier loaded from {model_path}")

    def predict(self, image_path: str, threshold: float = 0.5) -> Dict:
        """
        Predict labels for an image

        Args:
            image_path: Path to image
            threshold: Confidence threshold for positive prediction

        Returns:
            Dict with predictions and confidences
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            confidences = outputs.squeeze().cpu().numpy()

        # Parse predictions
        predictions = {}
        terrain_types = []
        tactical_features = []

        for i, label_name in enumerate(self.label_names):
            confidence = float(confidences[i])
            predictions[label_name] = {
                'confidence': confidence,
                'prediction': confidence > threshold
            }

            if confidence > threshold:
                if label_name.startswith('terrain_'):
                    terrain_types.append(label_name.replace('terrain_', ''))
                elif label_name.startswith('tactical_'):
                    tactical_features.append(label_name.replace('tactical_', ''))

        return {
            'terrain_types': terrain_types,
            'tactical_features': tactical_features,
            'raw_predictions': predictions
        }

    def predict_map_regions(self, image_path: str, patch_size: int = 224,
                           stride: int = 150) -> Dict:
        """
        Predict across entire map using sliding window

        Args:
            image_path: Path to map image
            patch_size: Size of patches
            stride: Stride for sliding window

        Returns:
            Aggregated predictions for entire map
        """
        import cv2

        logger.info(f"Analyzing map: {image_path}")

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        all_predictions = []

        # Slide window across image
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Extract patch
                patch = img[y:y+patch_size, x:x+patch_size]

                # Save temp patch
                temp_path = '/tmp/temp_patch.jpg'
                cv2.imwrite(temp_path, patch)

                # Predict
                pred = self.predict(temp_path)
                pred['position'] = (x, y)
                all_predictions.append(pred)

        # Aggregate predictions
        terrain_counts = {}
        tactical_counts = {}

        for pred in all_predictions:
            for terrain in pred['terrain_types']:
                terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
            for tactical in pred['tactical_features']:
                tactical_counts[tactical] = tactical_counts.get(tactical, 0) + 1

        total_patches = len(all_predictions)

        return {
            'terrain_distribution': {
                k: v / total_patches for k, v in terrain_counts.items()
            },
            'tactical_distribution': {
                k: v / total_patches for k, v in tactical_counts.items()
            },
            'patch_predictions': all_predictions,
            'total_patches': total_patches
        }


def main():
    """Test inference"""
    import sys

    model_path = './models/map_classifier_resnet50.pth'

    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Train the model first using: python train_map_classifier.py")
        sys.exit(1)

    # Initialize classifier
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    classifier = MapClassifier(model_path, device=device)

    # Test on multiple maps
    import sys
    test_map = sys.argv[1] if len(sys.argv) > 1 else './uploads/1000033857.jpg'

    if not Path(test_map).exists():
        logger.error(f"Test map not found: {test_map}")
        sys.exit(1)

    logger.info("Running inference on full map...")
    results = classifier.predict_map_regions(test_map, patch_size=224, stride=200)

    print("\n" + "="*70)
    print("MAP ANALYSIS RESULTS")
    print("="*70)
    print(f"Total patches analyzed: {results['total_patches']}")
    print("\nTerrain Distribution:")
    for terrain, percentage in sorted(results['terrain_distribution'].items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"  {terrain:20s}: {percentage*100:5.1f}%")

    print("\nTactical Features:")
    for feature, percentage in sorted(results['tactical_distribution'].items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"  {feature:20s}: {percentage*100:5.1f}%")


if __name__ == "__main__":
    main()
