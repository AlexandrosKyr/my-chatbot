# NATO Military Symbology Integration

## Overview

The chatbot now includes NATO APP-6(E) military symbology support to better recognize and interpret military symbols on tactical maps.

## Installation

To enable the military symbol library:

```bash
cd backend
pip install -r requirements.txt
```

This will install `military-symbol==1.0.8` along with other dependencies.

## What Was Added

### 1. **Symbol Helper Module** ([symbol_helper.py](backend/symbol_helper.py))
- Provides NATO symbology reference guide for LLaVA
- Maps unit types to doctrine keywords
- Extracts unit categories from visual analysis
- Generates unit-specific doctrine queries

### 2. **Enhanced LLaVA Prompts**
The visual analysis now includes:
- Complete NATO APP-6(E) symbol reference guide
- Frame shape/color meanings (friendly/enemy/neutral)
- Echelon indicators (platoon, company, battalion, etc.)
- Unit type icons (infantry, armor, artillery, etc.)
- Mobility indicators and tactical graphics

### 3. **Smart Doctrine Retrieval**
The system now:
- Analyzes LLaVA's output to detect unit types
- Generates targeted doctrine queries based on detected units
- Retrieves unit-specific tactical doctrine (e.g., "armor tactics", "fire support coordination")

## How It Works

```
Map Upload → LLaVA Analysis with Symbol Guide
           ↓
    Detects: "infantry platoon at NE, armor company at center"
           ↓
    Extracts unit categories: ["infantry", "armor"]
           ↓
    Doctrine Query: "infantry tactics dismounted operations
                     armor tactics armored operations tank deployment"
           ↓
    Retrieves targeted doctrine chapters
           ↓
    Strategy incorporates unit-specific guidance
```

## Benefits

✅ **Better Symbol Recognition**: LLaVA gets comprehensive reference guide
✅ **Precise Doctrine Matching**: Queries doctrine based on actual units detected
✅ **Consistent Terminology**: Enforces NATO nomenclature (platoon, company, battalion)
✅ **Smarter Analysis**: Understands WHY units are positioned in specific locations

## Example

**Before**: Generic query → "How does NATO doctrine address offensive operations?"

**After**: Unit-aware query → "How does NATO doctrine address offensive operations? What are the doctrinal principles for infantry tactics dismounted operations close combat? What are the doctrinal principles for armor tactics armored operations tank deployment?"

Result: More relevant, unit-specific doctrine retrieved from KB!

## Future Enhancements (Optional)

If needed, we can add:
- Symbol validation (cross-check LLaVA's interpretation with library)
- SIDC code generation for detected units
- Symbol-based metadata tagging for better search
- Visual symbol generation for recommendations

## Testing

To test the integration:

1. Upload a map with NATO symbols
2. Check logs for "Detected unit types: infantry, armor..."
3. Verify doctrine query includes unit-specific keywords
4. Confirm strategy references appropriate unit tactics

## Troubleshooting

**If library fails to import**: The system will gracefully fall back to text-only mode. The symbol reference guide will still be provided to LLaVA via prompts.

**Check logs for**:
```
INFO - Military symbol library loaded successfully
INFO - Detected unit types: infantry, armor
INFO - Doctrine query: How does NATO doctrine address...
```
