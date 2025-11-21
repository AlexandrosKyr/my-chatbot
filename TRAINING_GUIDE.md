# Training Guide (For Developer)

## Quick Training Workflow

### 1. Generate Training Dataset

```bash
cd backend
python symbol_dataset_generator.py
```

**Output**: ~3,000 images in `symbol_dataset/` (~2 GB total)
- 50 NATO symbol classes
- 70% train, 15% val, 15% test split

### 2. Train ResNet18 Model

```bash
python symbol_classifier.py
```

**What happens**:
- Loads pretrained ResNet18 from PyTorch
- Trains only final layer (512 → 50 classes)
- Auto-saves best model based on validation accuracy
- Early stopping after 10 epochs without improvement

**Expected time**: 30 minutes - 2 hours (GPU), 4-8 hours (CPU)

**Training output**:
```
Epoch 1/50
  Train Loss: 2.3421, Train Acc: 35.20%
  Val Loss: 1.9234, Val Acc: 45.30%
  ✓ New best model saved! Val Acc: 45.30%

...

Epoch 28/50
  Train Loss: 0.1234, Train Acc: 96.50%
  Val Loss: 0.2345, Val Acc: 93.20%
  ✓ New best model saved! Val Acc: 93.20%

✓ Training complete! Best Val Acc: 93.20%
Model saved to: ./models/symbol_classifier.pth

Test Accuracy: 91.50%
```

### 3. Verify Model Parameters

```bash
python verify_model.py
```

**Checks**:
- ✓ Model file exists (~45 MB)
- ✓ Contains 50 classes
- ✓ Class mapping JSON exists
- ✓ Parameters can be loaded

### 4. Commit to GitHub

```bash
# Model parameters and mapping are small enough for GitHub
git add backend/models/symbol_classifier.pth
git add backend/symbol_dataset/class_mapping.json

git commit -m "Add trained ResNet18 NATO symbol classifier (93% acc)"
git push
```

**Note**: Training data (`symbol_dataset/train/`, `val/`, `test/`) is already ignored by `.gitignore`

---

## Adjusting Training

### If Accuracy Too Low (<85%)

**Option 1: More data**
```python
# Edit symbol_dataset_generator.py line 316
generator.generate_dataset(images_per_class=100)  # was 60
```

**Option 2: Train longer**
```python
# Edit symbol_classifier.py line 334
classifier.train(epochs=100)  # was 50
```

**Option 3: Fine-tune more layers**
```python
# Edit symbol_classifier.py _build_model() method
# After line 108, add:
for param in model.layer4.parameters():
    param.requires_grad = True
```

### Monitor Training

Watch validation accuracy:
- Should reach 60-70% by epoch 5
- Should reach 80-85% by epoch 15
- Should reach 90-95% by epoch 25-30
- If stuck below 80%, regenerate dataset with more augmentation

---

## What Gets Saved

### Model Checkpoint (`models/symbol_classifier.pth`)

Contains:
```python
{
    'model_state_dict': {
        # All ResNet18 weights (pretrained + trained)
        'conv1.weight': tensor(...),
        'layer1.0.conv1.weight': tensor(...),
        ...
        'fc.weight': tensor([50, 512]),  # Your trained layer
        'fc.bias': tensor([50])
    },
    'num_classes': 50
}
```

Size: ~45 MB

### Class Mapping (`symbol_dataset/class_mapping.json`)

```json
{
  "0": "friendly_infantry_platoon",
  "1": "friendly_infantry_company",
  ...
  "49": "neutral_infantry_company"
}
```

Size: ~2 KB

---

## GPU Training

Check GPU availability:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

If CUDA available, training uses GPU automatically. Monitor:
```bash
# In another terminal
nvidia-smi  # Watch GPU memory usage
```

---

## Testing Trained Model

Quick test script:
```python
from symbol_classifier import ResNetSymbolClassifier
from PIL import Image
import json

# Load model
classifier = ResNetSymbolClassifier(num_classes=50, device='auto')
classifier.load_model('./models/symbol_classifier.pth')

# Load mapping
with open('./symbol_dataset/class_mapping.json', 'r') as f:
    mapping = json.load(f)

# Test on a symbol image
img = Image.open('./symbol_dataset/test/images/friendly_infantry_platoon_000.png')
result = classifier.predict(img)

symbol = mapping[str(result['class_idx'])]
print(f"Detected: {symbol} ({result['confidence']:.1%})")
```

---

## Files Summary

**Commit to GitHub** (required for deployment):
- ✅ `backend/models/symbol_classifier.pth` (~45 MB)
- ✅ `backend/symbol_dataset/class_mapping.json` (~2 KB)

**Keep local** (regenerate on each machine):
- ❌ `backend/symbol_dataset/train/` (~1.5 GB)
- ❌ `backend/symbol_dataset/val/` (~300 MB)
- ❌ `backend/symbol_dataset/test/` (~300 MB)

Total to commit: **~45 MB** (GitHub allows up to 100 MB per file)
