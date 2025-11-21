# NATO Symbol Classifier - Quick Reference

## Overview

ResNet18 symbol classifier with saved model parameters for easy deployment across computers.

## Full Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Developer guide for training the model
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - User guide for deploying pre-trained model

## Quick Commands

## Model Files Structure

```
backend/
├── symbol_classifier.py          # Classifier code
├── symbol_dataset_generator.py   # Dataset generation
├── models/
│   └── symbol_classifier.pth     # Trained model weights (save this!)
└── symbol_dataset/
    ├── class_mapping.json        # Class index to name mapping (save this!)
    ├── train/                    # Training data (don't commit - too large)
    ├── val/                      # Validation data (don't commit)
    └── test/                     # Test data (don't commit)
```

## Step 1: Train the Model (First Computer)

### Generate Dataset

```bash
cd backend
python symbol_dataset_generator.py
```

This creates ~3,000 images in `symbol_dataset/` directory.

### Train Classifier

```bash
python symbol_classifier.py
```

**Training Process**:
- Monitors validation accuracy each epoch
- Automatically saves best model to `models/symbol_classifier.pth`
- Shows progress with training/validation metrics
- Stops early if no improvement for 10 epochs

**Expected Output**:
```
Epoch 1/50
  Train Loss: 2.3421, Train Acc: 35.20%
  Val Loss: 1.9234, Val Acc: 45.30%
  ✓ New best model saved! Val Acc: 45.30%

Epoch 2/50
  Train Loss: 1.5432, Train Acc: 58.70%
  Val Loss: 1.2345, Val Acc: 67.80%
  ✓ New best model saved! Val Acc: 67.80%

...

Epoch 28/50
  Train Loss: 0.1234, Train Acc: 96.50%
  Val Loss: 0.2345, Val Acc: 93.20%
  ✓ New best model saved! Val Acc: 93.20%

✓ Training complete! Best Val Acc: 93.20%
Model saved to: ./models/symbol_classifier.pth
```

### Test Final Model

```bash
python symbol_classifier.py
```

The script will load the saved model and evaluate it on the test set, showing per-class accuracy.

## Step 2: Prepare for GitHub

### What to Commit

**Required Files** (small, must commit):
```bash
backend/models/symbol_classifier.pth      # ~45 MB (trained weights)
backend/symbol_dataset/class_mapping.json # ~2 KB (class names)
backend/symbol_classifier.py              # Python code
backend/symbol_dataset_generator.py       # Dataset generator
backend/symbol_helper.py                  # Helper functions
```

**Do NOT Commit** (too large):
```bash
backend/symbol_dataset/train/    # ~1.5 GB of images
backend/symbol_dataset/val/      # ~300 MB
backend/symbol_dataset/test/     # ~300 MB
```

### Update .gitignore

Add to `.gitignore`:
```bash
# Ignore training data (too large for GitHub)
backend/symbol_dataset/train/
backend/symbol_dataset/val/
backend/symbol_dataset/test/

# Keep the model and mapping
!backend/models/symbol_classifier.pth
!backend/symbol_dataset/class_mapping.json
```

### Git Commands

```bash
cd /Users/alexandroskyriakopoulos/my-chatbot

# Add model files
git add backend/models/symbol_classifier.pth
git add backend/symbol_dataset/class_mapping.json
git add backend/symbol_classifier.py
git add backend/symbol_dataset_generator.py
git add backend/symbol_helper.py

# Commit
git commit -m "Add trained ResNet18 NATO symbol classifier (93% accuracy)"

# Push to GitHub
git push origin master
```

## Step 3: Deploy on Another Computer

### Clone Repository

```bash
git clone https://github.com/your-username/my-chatbot.git
cd my-chatbot
```

### Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Verify Model Files

```bash
# Check model exists
ls -lh models/symbol_classifier.pth

# Check mapping exists
ls -lh symbol_dataset/class_mapping.json
```

**Expected**:
```
-rw-r--r--  1 user  staff   45M  symbol_classifier.pth
-rw-r--r--  1 user  staff   2K   class_mapping.json
```

### Test the Model

Create a test script `test_classifier.py`:

```python
from symbol_classifier import ResNetSymbolClassifier
from PIL import Image
import json

# Load classifier
classifier = ResNetSymbolClassifier(num_classes=50, device='auto')
classifier.load_model('./models/symbol_classifier.pth')

# Load class mapping
with open('./symbol_dataset/class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

print("✓ Model loaded successfully!")
print(f"Device: {classifier.device}")
print(f"Classes: {len(class_mapping)}")

# Test prediction on an image
# image = Image.open('path/to/test/symbol.png')
# result = classifier.predict(image)
# symbol_name = class_mapping[str(result['class_idx'])]
# print(f"Detected: {symbol_name} ({result['confidence']:.1%} confidence)")
```

Run:
```bash
python test_classifier.py
```

### Run the Application

```bash
python app.py
```

The tactical analyzer will automatically load the model on startup:
```
INFO - NATO symbol classifier loaded successfully
```

## Model Architecture Details

### What's Saved in the .pth File

The checkpoint contains:
```python
{
    'model_state_dict': OrderedDict([
        ('conv1.weight', tensor(...)),
        ('bn1.weight', tensor(...)),
        # ... all ResNet18 layers
        ('fc.weight', tensor(...)),      # Final layer: 512 → 50 classes
        ('fc.bias', tensor(...))
    ]),
    'num_classes': 50
}
```

### File Size

- **Full model**: ~45 MB
- **Breakdown**:
  - ResNet18 backbone: ~44 MB (pretrained weights frozen, but saved)
  - Custom final layer: ~1 MB (512 × 50 weights + 50 biases)

### Loading Process

1. Creates fresh ResNet18 architecture
2. Loads pretrained ImageNet weights (for frozen layers)
3. Replaces final layer with 50-class layer
4. Loads your trained weights from `.pth` file
5. Sets model to evaluation mode

## Re-training or Fine-tuning

### If Accuracy Is Too Low

Option 1: Generate more data
```bash
# Edit symbol_dataset_generator.py line 316
generator.generate_dataset(images_per_class=100)  # was 60
```

Option 2: Train longer
```bash
# Edit symbol_classifier.py line 334
classifier.train(epochs=100, learning_rate=0.001)  # was 50
```

Option 3: Unfreeze more layers
```python
# In symbol_classifier.py, modify _build_model():
# Unfreeze last residual block for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True
```

### Saving a New Model

Training automatically overwrites `models/symbol_classifier.pth` with the best checkpoint. Commit the new version:

```bash
git add backend/models/symbol_classifier.pth
git commit -m "Update symbol classifier to 95% accuracy"
git push
```

## Best Practices

### Version Control for Models

Consider using Git LFS for large model files:

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or: apt install git-lfs  # Linux

# Initialize
git lfs install

# Track model files
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track model files with Git LFS"
```

This prevents bloating your repository with large binary files.

### Model Versioning

Tag releases with accuracy:
```bash
git tag -a v1.0-acc93 -m "ResNet18 classifier with 93% test accuracy"
git push origin v1.0-acc93
```

### Alternative: Model Hosting

For very large models, consider:
1. **Hugging Face Hub** (free hosting for models)
2. **Google Drive** (download via script)
3. **GitHub Releases** (attach .pth file to release)

Example auto-download script:
```python
import os
import requests

MODEL_URL = "https://github.com/user/repo/releases/download/v1.0/symbol_classifier.pth"
MODEL_PATH = "./models/symbol_classifier.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    os.makedirs("./models", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("✓ Model downloaded")
```

## Troubleshooting

### "Model not found" on New Computer

**Error**:
```
WARNING - Symbol classifier not found - symbol detection disabled
  Model: ./models/symbol_classifier.pth
```

**Solution**:
```bash
# Check if file exists
ls backend/models/symbol_classifier.pth

# If missing, pull from GitHub
cd backend
git pull origin master

# Verify file size
ls -lh models/symbol_classifier.pth  # Should be ~45 MB
```

### Model Loaded but Poor Accuracy

**Possible Issues**:
1. Wrong class_mapping.json (mismatch between training and inference)
2. Model trained on different image preprocessing
3. YOLO cropping symbols poorly

**Solution**:
```bash
# Verify mapping matches model
python -c "import torch; print(torch.load('models/symbol_classifier.pth')['num_classes'])"
# Should output: 50

# Count classes in mapping
python -c "import json; print(len(json.load(open('symbol_dataset/class_mapping.json'))))"
# Should output: 50
```

### GPU Not Available on New Computer

The model automatically falls back to CPU:
```python
Device: cpu  # Instead of cuda:0
```

Training/inference will be slower but still work.

## Summary

### To Train and Commit:
1. `python symbol_dataset_generator.py` (generates data)
2. `python symbol_classifier.py` (trains and saves model)
3. `git add models/symbol_classifier.pth symbol_dataset/class_mapping.json`
4. `git commit -m "Add trained model"`
5. `git push`

### To Deploy on New Computer:
1. `git clone` your repository
2. `pip install -r requirements.txt`
3. `python app.py` (model auto-loads)

The model weights are portable and work on any computer with PyTorch installed!
