#!/usr/bin/env python3
"""
ResNet50 Transfer Learning for Topographic Map Classification

Trains a multi-label classifier for:
- Terrain types (forest, water, urban, open, mountainous)
- Tactical features (high ground, cover, killzone, concealment)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class MapPatchDataset(Dataset):
    """Dataset for topographic map patches"""

    def __init__(self, patches_data: List[Dict], patches_dir: str, transform=None):
        """
        Initialize dataset

        Args:
            patches_data: List of patch metadata with labels
            patches_dir: Directory containing patch images
            transform: Torchvision transforms
        """
        self.patches = patches_data
        self.patches_dir = Path(patches_dir)
        self.transform = transform

        # Define label mapping
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

        logger.info(f"Dataset initialized with {len(self.patches)} patches")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]

        # Load image
        img_path = self.patches_dir / patch_info['filename']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Create multi-label vector
        labels = torch.zeros(len(self.label_names), dtype=torch.float32)
        for i, label_name in enumerate(self.label_names):
            if label_name in patch_info['labels']:
                labels[i] = 1.0

        return image, labels


class MapResNet50(nn.Module):
    """ResNet50 with custom head for multi-label classification"""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        Initialize model

        Args:
            num_classes: Number of output classes (10 for our task)
            pretrained: Use ImageNet pre-trained weights
        """
        super(MapResNet50, self).__init__()

        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Freeze early layers (keep low-level features)
        for param in list(self.resnet.parameters())[:-30]:  # Freeze all but last 30 params
            param.requires_grad = False

        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Multi-label classification
        )

    def forward(self, x):
        return self.resnet(x)


class MapClassifierTrainer:
    """Trainer for map classification model"""

    def __init__(self, model, device, learning_rate=1e-4):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            device: Device to train on (cpu/cuda/mps)
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()  # Binary cross-entropy for multi-label
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy (threshold 0.5 for multi-label)
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.numel()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        return avg_loss, accuracy

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.numel()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs, save_path):
        """
        Train model for multiple epochs

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_path: Path to save best model
        """
        logger.info(f"Training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, save_path)
                logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")

    def plot_training_history(self, save_path):
        """Plot training curves"""
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Training plot saved: {save_path}")


def test_model(model, test_loader, device, label_names):
    """
    Test model and show per-class metrics

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        label_names: List of label names
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            predictions = (outputs > 0.5).float()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)

    # Calculate per-class metrics
    print("\n" + "="*70)
    print("PER-CLASS METRICS")
    print("="*70)

    for i, label_name in enumerate(label_names):
        preds = all_predictions[:, i]
        labels = all_labels[:, i]

        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{label_name:30s} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # Overall accuracy
    accuracy = np.mean(all_predictions == all_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")


def main():
    """Main training script"""
    logger.info("Starting Map Classifier Training")

    # Configuration
    DATASET_DIR = Path('./map_dataset')
    PATCHES_DIR = DATASET_DIR / 'patches_augmented'  # Use augmented if available
    if not PATCHES_DIR.exists():
        PATCHES_DIR = DATASET_DIR / 'patches'

    MODEL_SAVE_PATH = Path('./models/map_classifier_resnet50.pth')
    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)

    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4

    # Check if dataset exists
    metadata_path = DATASET_DIR / 'dataset_metadata.json'
    if not metadata_path.exists():
        logger.error(f"Dataset not found. Run map_dataset_generator.py first!")
        return

    # Load dataset metadata
    with open(metadata_path, 'r') as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset: {dataset['metadata']['total_patches']} patches")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = MapPatchDataset(dataset['train'], PATCHES_DIR, train_transform)
    val_dataset = MapPatchDataset(dataset['val'], PATCHES_DIR, val_transform)
    test_dataset = MapPatchDataset(dataset['test'], PATCHES_DIR, val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # Create model
    model = MapResNet50(num_classes=10, pretrained=True)
    logger.info(f"Model created: ResNet50 with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Train
    trainer = MapClassifierTrainer(model, device, learning_rate=LEARNING_RATE)
    trainer.train(train_loader, val_loader, NUM_EPOCHS, MODEL_SAVE_PATH)

    # Plot training history
    trainer.plot_training_history(DATASET_DIR / 'training_history.png')

    # Test
    logger.info("\nEvaluating on test set...")
    test_model(model, test_loader, device, train_dataset.label_names)

    logger.info(f"\n✓ Training complete! Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
