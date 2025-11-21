"""
NATO Symbol Classifier - ResNet18 Transfer Learning

Fast, accurate symbol recognition using pretrained ResNet18.
Training time: 30min-2hrs on GPU for ~3,000 images.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NATOSymbolDataset(Dataset):
    """Dataset loader for NATO symbol images"""

    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.image_dir = self.data_dir / 'images'
        self.transform = transform

        # Load labels
        label_file = self.data_dir / 'labels.json'
        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.image_files = list(self.labels.keys())

        # Load class mapping
        mapping_file = Path(data_dir) / 'class_mapping.json'
        with open(mapping_file, 'r') as f:
            self.class_mapping = json.load(f)

        # Create class name to index mapping
        self.class_to_idx = {name: int(idx) for idx, name in self.class_mapping.items()}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert('RGB')

        label_info = self.labels[img_name]
        class_name = label_info['class_name']
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, class_idx


class ResNetSymbolClassifier:
    """ResNet18-based NATO symbol classifier"""

    def __init__(self, num_classes: int = 50, device: str = 'auto'):
        self.num_classes = num_classes

        # Auto-detect device (prioritize MPS for Mac Silicon, then CUDA, then CPU)
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)

        # Training transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.2),  # Slight augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Val/test transforms (no augmentation)
        self.eval_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_model(self) -> nn.Module:
        """Build ResNet18 with custom final layer"""

        # Load pretrained ResNet18
        model = models.resnet18(pretrained=True)

        # Freeze early layers (transfer learning)
        for param in model.parameters():
            param.requires_grad = False

        # Replace final layer for our symbol classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)

        # Unfreeze final layer
        for param in model.fc.parameters():
            param.requires_grad = True

        logger.info(f"ResNet18 model built: {self.num_classes} classes")
        return model

    def train(
        self,
        dataset_dir: str,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        save_path: str = './models/symbol_classifier.pth'
    ):
        """Train the symbol classifier"""

        logger.info("=" * 70)
        logger.info("TRAINING NATO SYMBOL CLASSIFIER")
        logger.info("=" * 70)

        # Load datasets
        train_dataset = NATOSymbolDataset(dataset_dir, 'train', transform=self.train_transform)
        val_dataset = NATOSymbolDataset(dataset_dir, 'val', transform=self.eval_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Epochs: {epochs}")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 10

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            train_acc = 100 * train_correct / train_total

            # Validate
            val_acc, val_loss = self._validate(val_loader, criterion)

            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Learning rate scheduling
            scheduler.step(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model(save_path)
                logger.info(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        logger.info("=" * 70)
        logger.info(f"✓ Training complete! Best Val Acc: {best_val_acc:.2f}%")
        logger.info(f"Model saved to: {save_path}")

    def _validate(self, val_loader, criterion) -> Tuple[float, float]:
        """Validate model on validation set"""

        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        return val_acc, avg_val_loss

    def evaluate(self, dataset_dir: str):
        """Evaluate on test set"""

        test_dataset = NATOSymbolDataset(dataset_dir, 'test', transform=self.eval_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        self.model.eval()
        test_correct = 0
        test_total = 0

        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                # Per-class accuracy
                for label, pred in zip(labels, predicted):
                    label_item = label.item()
                    class_total[label_item] = class_total.get(label_item, 0) + 1
                    if label == pred:
                        class_correct[label_item] = class_correct.get(label_item, 0) + 1

        test_acc = 100 * test_correct / test_total
        logger.info(f"Test Accuracy: {test_acc:.2f}%")

        # Show per-class accuracy
        logger.info("\nPer-class accuracy:")
        class_mapping = test_dataset.class_mapping
        for class_idx, class_name in class_mapping.items():
            idx = int(class_idx)
            if idx in class_total:
                acc = 100 * class_correct.get(idx, 0) / class_total[idx]
                logger.info(f"  {class_name}: {acc:.1f}%")

        return test_acc

    def predict(self, image: Image.Image) -> Dict:
        """Predict symbol class from PIL image"""

        self.model.eval()

        # Transform image
        image_tensor = self.eval_transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            'class_idx': predicted.item(),
            'confidence': confidence.item()
        }

    def save_model(self, path: str):
        """Save model checkpoint"""

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)

    def load_model(self, path: str):
        """Load model checkpoint"""

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Model loaded from {path}")


def main():
    """Train the symbol classifier"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Configuration
    dataset_dir = './symbol_dataset'
    model_save_path = './models/symbol_classifier.pth'

    # Auto-detect number of classes from dataset
    mapping_file = Path(dataset_dir) / 'class_mapping.json'
    with open(mapping_file, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    logger.info(f"Detected {num_classes} classes in dataset")

    # Initialize classifier
    classifier = ResNetSymbolClassifier(num_classes=num_classes)

    # Train
    classifier.train(
        dataset_dir=dataset_dir,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        save_path=model_save_path
    )

    # Load best model and evaluate
    classifier.load_model(model_save_path)
    classifier.evaluate(dataset_dir)

    logger.info("=" * 70)
    logger.info("✓ Training and evaluation complete!")
    logger.info(f"Model ready for deployment: {model_save_path}")


if __name__ == "__main__":
    main()
