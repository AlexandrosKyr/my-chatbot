# Deployment Guide (For Users)

## Prerequisites

- Python 3.8+
- Git

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-username/my-chatbot.git
cd my-chatbot
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note**: This installs PyTorch, which is large (~2 GB). Be patient.

### 3. Verify Model Files

The trained model parameters are included in the repository:

```bash
ls -lh models/symbol_classifier.pth        # Should show ~45 MB
ls -lh symbol_dataset/class_mapping.json  # Should show ~2 KB
```

If files are missing, ensure Git LFS is installed (if used) or re-clone.

### 4. Run Application

```bash
python app.py
```

Expected startup output:
```
INFO - NATO symbol classifier loaded successfully
INFO - Server running on http://localhost:5000
```

## How It Works

### NATO Symbol Detection Pipeline

1. **Upload Map**: User uploads tactical map with NATO symbols
2. **YOLO Detection**: Detects objects on map
3. **Symbol Classification**: ResNet18 classifies each object as NATO symbol
4. **Display Results**: Lists detected symbols with confidence scores

Example output:
```
═══════════════════════════════════════════════════════════════════
NATO MILITARY SYMBOLS DETECTED ON MAP (ResNet18 Classification)
═══════════════════════════════════════════════════════════════════

Identified 3 NATO symbols:

1. Friendly Infantry Platoon
   Location: Northeast sector
   Confidence: 94.2%

2. Enemy Armor Company
   Location: Southwest sector
   Confidence: 89.7%

3. Friendly Artillery Battery
   Location: Center sector
   Confidence: 82.1% ⚠️ (Low Confidence)

Note: Symbols marked with ⚠️ have <85% confidence.
If any symbols appear incorrect or missing, please notify me.
═══════════════════════════════════════════════════════════════════
```

## Supported Symbols

The model recognizes **50 NATO APP-6 symbol classes**:

### Infantry
- Friendly/Enemy Infantry: Section, Platoon, Company, Battalion

### Armor
- Friendly/Enemy Armor: Platoon, Company, Battalion

### Artillery
- Friendly/Enemy Artillery: Battery, Battalion
- Friendly/Enemy Mortar: Section

### Support Units
- Reconnaissance: Team, Platoon
- Engineer: Platoon
- Medical: Company
- Logistics: Company

### Headquarters
- Company HQ, Battalion HQ, Brigade HQ

### Affiliations
- Friendly (Blue rectangle frame)
- Enemy (Red diamond frame)
- Neutral (Green square frame)

## No Training Required

The model weights are **already trained** and included in the repository. You do NOT need to:
- Generate training data
- Run training scripts
- Have a GPU

The application works immediately after cloning and installing dependencies.

## Troubleshooting

### Model Not Found

**Error**:
```
WARNING - Symbol classifier not found - symbol detection disabled
```

**Solution**:
```bash
# Verify files exist
ls backend/models/symbol_classifier.pth
ls backend/symbol_dataset/class_mapping.json

# If missing, try:
git lfs pull  # If repository uses Git LFS
```

### Import Errors

**Error**:
```
ImportError: No module named 'torch'
```

**Solution**:
```bash
cd backend
pip install -r requirements.txt
```

### Low Detection Accuracy

If symbols are frequently misclassified:
1. Ensure map has clear, standard NATO symbols
2. Symbols should be reasonably sized (not tiny)
3. Check that YOLO is detecting the symbols (in logs)

Contact the developer if accuracy is consistently poor.

## System Requirements

### Minimum
- CPU: Any modern processor
- RAM: 4 GB
- Disk: 5 GB free space

### Recommended
- CPU: Multi-core processor
- RAM: 8 GB
- GPU: NVIDIA GPU with CUDA (optional, improves speed)

### Performance
- **With GPU**: Near real-time symbol detection
- **With CPU**: 2-5 seconds per map (acceptable for tactical planning)

## Updates

To update to the latest version:

```bash
git pull origin master
pip install -r requirements.txt --upgrade
python app.py
```

Model parameters will update automatically if developer has retrained.

## Support

If you encounter issues:
1. Check logs in terminal for error messages
2. Verify model files exist and are correct size
3. Ensure all dependencies are installed
4. Report issues to repository maintainer

## Technical Details

### Model Architecture
- **Base**: ResNet18 (pretrained on ImageNet)
- **Custom**: Final layer trained for 50 NATO symbol classes
- **Input**: 128×128 RGB images
- **Output**: Symbol class + confidence score

### Model Size
- Total: ~45 MB
- Architecture: Standard PyTorch checkpoint format
- Portable: Works on any OS with PyTorch

### Inference Speed
- CPU: ~50-100ms per symbol
- GPU: ~5-10ms per symbol

For a typical map with 5-10 symbols, total detection time is <1 second.
