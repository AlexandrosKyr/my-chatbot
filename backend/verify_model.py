#!/usr/bin/env python3
"""
Verify NATO Symbol Classifier Model Parameters

Quick check that trained model and class mapping are properly saved.
"""

import os
import sys
import json
import torch

def main():
    print("="*60)
    print("MODEL PARAMETERS VERIFICATION")
    print("="*60)

    model_path = './models/symbol_classifier.pth'
    mapping_path = './symbol_dataset/class_mapping.json'

    # Check model file
    print("\n[1/3] Checking model file...")
    if not os.path.exists(model_path):
        print(f"✗ Model NOT FOUND: {model_path}")
        print("Run: python symbol_classifier.py")
        return 1

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Model found: {model_path} ({size_mb:.2f} MB)")

    # Verify model contents
    print("\n[2/3] Verifying model parameters...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' not in checkpoint:
            print("✗ Invalid checkpoint: missing model_state_dict")
            return 1

        state_dict = checkpoint['model_state_dict']
        num_classes = checkpoint.get('num_classes', 'unknown')

        print(f"✓ Model parameters loaded")
        print(f"  Layers: {len(state_dict)}")
        print(f"  Classes: {num_classes}")

        if 'fc.weight' in state_dict:
            fc_shape = state_dict['fc.weight'].shape
            expected_classes = num_classes if isinstance(num_classes, int) else 35
            print(f"  Final layer: {fc_shape} {'✓' if fc_shape[0] == expected_classes else f'✗ Expected {expected_classes} classes!'}")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return 1

    # Check class mapping
    print("\n[3/3] Checking class mapping...")
    if not os.path.exists(mapping_path):
        print(f"✗ Mapping NOT FOUND: {mapping_path}")
        return 1

    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

        print(f"✓ Class mapping loaded: {len(mapping)} classes")

        # Verify mapping matches model
        if num_classes != 'unknown' and len(mapping) != num_classes:
            print(f"⚠️  Warning: Mapping has {len(mapping)} classes but model expects {num_classes}")

    except Exception as e:
        print(f"✗ Failed to load mapping: {e}")
        return 1

    # GPU check
    print(f"\n{'='*60}")
    if torch.cuda.is_available():
        print(f"GPU (CUDA): {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("GPU (MPS): Apple Silicon available")
    else:
        print("GPU: Not available (using CPU)")

    # Summary
    print(f"{'='*60}")
    print("✓ MODEL PARAMETERS VERIFIED")
    print("✓ Ready to use!")
    print(f"{'='*60}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
