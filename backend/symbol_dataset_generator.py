"""
NATO Symbol Dataset Generator

Generates synthetic training data for ResNet18 symbol classifier using military-symbol library.
Target: ~3,000 images across 50 symbol classes with augmentation.
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import io

logger = logging.getLogger(__name__)

try:
    from military_symbol import get_symbol_svg_string_from_name
    import cairosvg
    LIBS_AVAILABLE = True
except ImportError as e:
    LIBS_AVAILABLE = False
    logger.error(f"Required libraries not available: {e}")
    logger.error("Please install: pip install military-symbol cairosvg")


class NATOSymbolDatasetGenerator:
    """Generate synthetic NATO symbol dataset for training"""

    def __init__(self, output_dir: str = "./symbol_dataset"):
        self.output_dir = Path(output_dir)
        self.image_size = 128  # ResNet input size

        # Define 50 common NATO symbols (unit_type + echelon + affiliation)
        self.symbol_definitions = self._define_symbol_classes()

        # Augmentation parameters
        self.augmentations = {
            'rotation_range': (-15, 15),  # degrees
            'scale_range': (0.8, 1.2),
            'brightness_range': (0.7, 1.3),
            'backgrounds': ['white', 'tan', 'light_green', 'light_gray'],
            'noise_levels': [0, 0.02, 0.05]
        }

    def _define_symbol_classes(self) -> List[Dict]:
        """Define 50 NATO symbol classes to generate"""

        symbols = []

        # Core unit types
        unit_types = [
            'infantry', 'armor', 'artillery', 'reconnaissance',
            'engineer', 'headquarters', 'medical', 'logistics'
        ]

        # Echelons (unit sizes)
        echelons = ['section', 'platoon', 'company', 'battalion']

        # Affiliations
        affiliations = ['friendly', 'enemy', 'neutral']

        # Generate combinations (selecting 50 most common)
        # Priority: Infantry, Armor, Artillery with multiple echelons for each affiliation

        # High priority combinations (30 symbols)
        for unit in ['infantry', 'armor', 'artillery']:
            for echelon in ['platoon', 'company', 'battalion']:
                for affiliation in ['friendly', 'enemy']:
                    symbols.append({
                        'unit_type': unit,
                        'echelon': echelon,
                        'affiliation': affiliation,
                        'name': f"{affiliation}_{unit}_{echelon}"
                    })

        # Medium priority (15 symbols)
        for unit in ['reconnaissance', 'engineer', 'headquarters']:
            for echelon in ['platoon', 'company']:
                for affiliation in ['friendly', 'enemy']:
                    symbols.append({
                        'unit_type': unit,
                        'echelon': echelon,
                        'affiliation': affiliation,
                        'name': f"{affiliation}_{unit}_{echelon}"
                    })

        # Lower priority (5 symbols)
        for unit in ['medical', 'logistics']:
            for affiliation in ['friendly', 'enemy']:
                symbols.append({
                    'unit_type': unit,
                    'echelon': 'company',
                    'affiliation': affiliation,
                    'name': f"{affiliation}_{unit}_company"
                })

        # Add one neutral example
        symbols.append({
            'unit_type': 'infantry',
            'echelon': 'company',
            'affiliation': 'neutral',
            'name': 'neutral_infantry_company'
        })

        return symbols[:50]  # Ensure exactly 50 classes

    def generate_dataset(self, images_per_class: int = 60):
        """Generate complete dataset with augmentations"""

        if not LIBS_AVAILABLE:
            logger.error("Cannot generate dataset - required libraries not installed")
            return False

        logger.info(f"Generating dataset: {len(self.symbol_definitions)} classes × {images_per_class} images")
        logger.info(f"Total target: ~{len(self.symbol_definitions) * images_per_class} images")

        # Create directory structure
        splits = ['train', 'val', 'test']
        split_ratios = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test

        for split in splits:
            split_dir = self.output_dir / split / 'images'
            split_dir.mkdir(parents=True, exist_ok=True)

        # Generate images for each symbol class
        all_labels = {split: {} for split in splits}
        total_generated = 0

        for symbol_def in self.symbol_definitions:
            logger.info(f"Generating {symbol_def['name']}...")

            # Generate base symbol
            try:
                base_image = self._generate_base_symbol(symbol_def)

                # Generate augmented variations
                generated_images = []
                for i in range(images_per_class):
                    aug_image = self._augment_image(base_image, i)
                    generated_images.append(aug_image)

                # Split into train/val/test
                np.random.shuffle(generated_images)
                split_indices = [
                    0,
                    int(len(generated_images) * split_ratios[0]),
                    int(len(generated_images) * (split_ratios[0] + split_ratios[1]))
                ]

                for split_idx, split in enumerate(splits):
                    start = split_indices[split_idx]
                    end = split_indices[split_idx + 1] if split_idx < len(splits) - 1 else len(generated_images)
                    split_images = generated_images[start:end]

                    # Save images
                    for img_idx, img in enumerate(split_images):
                        filename = f"{symbol_def['name']}_{img_idx:03d}.png"
                        filepath = self.output_dir / split / 'images' / filename
                        img.save(filepath)

                        # Record label
                        all_labels[split][filename] = {
                            'class_name': symbol_def['name'],
                            'unit_type': symbol_def['unit_type'],
                            'echelon': symbol_def['echelon'],
                            'affiliation': symbol_def['affiliation']
                        }

                total_generated += len(generated_images)

            except Exception as e:
                logger.error(f"Failed to generate {symbol_def['name']}: {e}")
                continue

        # Save label files
        for split in splits:
            label_file = self.output_dir / split / 'labels.json'
            with open(label_file, 'w') as f:
                json.dump(all_labels[split], f, indent=2)

        # Save class mapping
        class_mapping = {i: symbol_def['name'] for i, symbol_def in enumerate(self.symbol_definitions)}
        mapping_file = self.output_dir / 'class_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)

        logger.info(f"✓ Dataset generation complete: {total_generated} images generated")
        logger.info(f"  Train: {len(all_labels['train'])} images")
        logger.info(f"  Val: {len(all_labels['val'])} images")
        logger.info(f"  Test: {len(all_labels['test'])} images")
        logger.info(f"  Saved to: {self.output_dir}")

        return True

    def _generate_base_symbol(self, symbol_def: Dict) -> Image.Image:
        """Generate base symbol image from definition"""

        # Create natural language description for military-symbol library
        description = f"{symbol_def['affiliation']} {symbol_def['unit_type']} {symbol_def['echelon']}"

        # Generate symbol SVG string using military-symbol library
        svg_data = get_symbol_svg_string_from_name(description)

        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(
            bytestring=svg_data.encode('utf-8'),
            output_width=self.image_size,
            output_height=self.image_size
        )

        # Load as PIL Image
        image = Image.open(io.BytesIO(png_data)).convert('RGB')

        return image

    def _augment_image(self, base_image: Image.Image, aug_index: int) -> Image.Image:
        """Apply augmentations to base symbol image"""

        # Start with copy of base
        img = base_image.copy()

        # Apply transformations based on aug_index for variety
        np.random.seed(aug_index)  # Reproducible augmentations

        # 1. Rotation
        if aug_index % 10 >= 5:  # 50% of images rotated
            angle = np.random.uniform(*self.augmentations['rotation_range'])
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

        # 2. Scale
        if aug_index % 8 >= 4:  # ~50% scaled
            scale = np.random.uniform(*self.augmentations['scale_range'])
            new_size = int(self.image_size * scale)
            img = img.resize((new_size, new_size), Image.LANCZOS)

            # Pad/crop back to original size
            if scale < 1:
                # Pad
                new_img = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
                offset = (self.image_size - new_size) // 2
                new_img.paste(img, (offset, offset))
                img = new_img
            else:
                # Crop
                offset = (new_size - self.image_size) // 2
                img = img.crop((offset, offset, offset + self.image_size, offset + self.image_size))

        # 3. Brightness
        brightness = np.random.uniform(*self.augmentations['brightness_range'])
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

        # 4. Background color variation
        bg_index = aug_index % len(self.augmentations['backgrounds'])
        img = self._apply_background(img, self.augmentations['backgrounds'][bg_index])

        # 5. Noise
        noise_index = aug_index % len(self.augmentations['noise_levels'])
        noise_level = self.augmentations['noise_levels'][noise_index]
        if noise_level > 0:
            img = self._add_noise(img, noise_level)

        return img

    def _apply_background(self, img: Image.Image, bg_type: str) -> Image.Image:
        """Apply different background colors to simulate map backgrounds"""

        bg_colors = {
            'white': (255, 255, 255),
            'tan': (245, 235, 220),
            'light_green': (230, 245, 230),
            'light_gray': (240, 240, 240)
        }

        bg_color = bg_colors.get(bg_type, (255, 255, 255))

        # Replace white background with chosen color
        img_array = np.array(img)
        mask = np.all(img_array > 250, axis=2)  # Nearly white pixels
        img_array[mask] = bg_color

        return Image.fromarray(img_array)

    def _add_noise(self, img: Image.Image, noise_level: float) -> Image.Image:
        """Add random noise to simulate real-world map quality"""

        img_array = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)


def main():
    """Generate the dataset"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    generator = NATOSymbolDatasetGenerator(output_dir="./symbol_dataset")

    logger.info("=" * 70)
    logger.info("NATO SYMBOL DATASET GENERATOR")
    logger.info("=" * 70)
    logger.info("Target: 50 symbol classes × 60 images = ~3,000 total images")
    logger.info("Split: 70% train, 15% val, 15% test")
    logger.info("Image size: 128×128 RGB")
    logger.info("=" * 70)

    success = generator.generate_dataset(images_per_class=60)

    if success:
        logger.info("✓ Dataset generation successful!")
        logger.info("Next step: Run symbol_classifier.py to train the model")
    else:
        logger.error("✗ Dataset generation failed")
        logger.error("Make sure you have: pip install military-symbol cairosvg")


if __name__ == "__main__":
    main()
