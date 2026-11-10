# Enhanced Synthetic Data Generation System

## Overview

This system generates synthetic training data for YOLO object detection, set up for Coca-Cola can defect detection: PBR materials, randomized lighting and camera, simulated defects, and automatic label and quality checks. It produces training data only; no detector trained on it is evaluated in this repository.

## System Components

### Core Scripts

1. **enhanced_generator_v282.py** - Main generation engine (Blender 2.82 compatible)
   - Advanced PBR material system
   - Three-point lighting setup
   - Intelligent camera positioning
   - Realistic defect simulation
   - Quality validation

2. **dataset_analyzer.py** - Comprehensive dataset analysis tool
   - Image quality metrics
   - Label validation
   - Distribution analysis
   - Quality violation detection
   - Automated reporting

3. **enhanced_test_runner.py** - Test suite for validation
   - System requirements check
   - Small-scale generation test
   - Quality validation
   - Performance benchmarking

4. **enhanced_runner.py** - Large batch dataset generation
   - Batch generation (default target 5000 images)
   - Progress tracking
   - Quality monitoring
   - YOLO dataset organization

## Features Implemented

### Advanced Material System
- Full PBR workflow with all texture maps
- Base Color, Normal, Metallic, Roughness textures
- Procedural material variations
- Enhanced surface properties

### Realistic Defect Simulation
- **Pristine**: Perfect can condition
- **Scratches**: Surface roughness variations
- **Dents**: Material darkening and texture changes
- **Punctures**: Severe damage with material property changes
- Configurable intensity levels

### Lighting
- Three-point lighting setup (Key, Fill, Rim)
- Dynamic light intensity variations
- Color temperature variations
- HDRI environment support

### Intelligent Camera System
- Spherical coordinate positioning
- Automatic can framing
- Variable focal lengths (35-85mm)
- Natural camera movements

### Quality Validation
- Automatic bounding box validation
- Image quality metrics (sharpness, contrast, noise)
- Size and occlusion checks
- Comprehensive quality reporting

### Comprehensive Analysis
- Dataset distribution analysis
- Quality metrics visualization
- Label validation
- Performance statistics

## Quick Start

### 1. Test the System
```bash
python3 enhanced_test_runner.py
```

This will:
- Generate 10 test images
- Validate quality
- Create analysis report
- Show performance metrics

### 2. Generate a Full Dataset
```bash
python3 enhanced_runner.py
```

This will:
- Generate up to 5000 images (the default target)
- Create train/val/test splits
- Generate YOLO-ready dataset
- Provide comprehensive analysis

### 3. Analyze Results
```bash
python3 dataset_analyzer.py enhanced_dataset/
```

## Configuration

The system uses JSON configuration files for customization:

```json
{
  "render": {
    "resolution": [1024, 1024],
    "samples": 64,
    "engine": "BLENDER_EEVEE"
  },
  "defects": {
    "pristine_probability": 0.3,
    "scratch_probability": 0.25,
    "dent_probability": 0.25,
    "puncture_probability": 0.2
  },
  "quality": {
    "min_can_area": 0.05,
    "max_can_area": 0.8
  }
}
```

## Output Structure

```
enhanced_dataset/
├── images/              # High-resolution synthetic images
├── labels/              # YOLO format labels
├── metadata/            # Generation parameters and stats
├── analysis/            # Quality analysis reports
└── yolo_dataset/        # YOLO-ready train/val/test splits
    ├── train/
    ├── val/
    ├── test/
    └── data.yaml
```

## Quality Metrics

The system tracks multiple quality indicators:

- **Image Quality**: Sharpness, contrast, noise levels
- **Bounding Boxes**: Size, position, accuracy
- **Class Distribution**: Balanced defect representation

## Performance

- **Generation speed**: about 8 seconds per 1024x1024 image on my machine (not logged in this repository)

## Training Integration

The generated dataset can be passed to Ultralytics YOLO for training:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo11s.pt')  # the bundled run used YOLO11s; yolov8s.pt also works

# Train with enhanced dataset
model.train(
    data='enhanced_dataset/yolo_dataset/data.yaml',
    epochs=200,
    imgsz=1024,
    batch=8
)
```

## System Requirements

- **Blender**: 2.82+ (tested with 2.82.7)
- **Python**: 3.8+ with packages:
  - opencv-python
  - matplotlib
  - seaborn
  - pandas
  - numpy
- **Hardware**: GPU recommended for rendering

## Usage Examples

### Generate a Dataset
```bash
# Target count is set by target_total in enhanced_runner.py (default 5000)
python3 enhanced_runner.py
```

### Analyze Existing Dataset
```bash
# Analyze any YOLO dataset
python3 dataset_analyzer.py /path/to/dataset/
```


## Troubleshooting

### Common Issues

1. **Blender Version Compatibility**
   - Use enhanced_generator_v282.py for Blender 2.82
   - Use enhanced_generator.py for Blender 3.0+

2. **Missing Packages**
   ```bash
   pip install opencv-python matplotlib seaborn pandas
   ```

3. **Performance Issues**
   - Reduce render samples in config
   - Use smaller resolution for testing
   - Enable GPU acceleration in Blender

### Quality Issues

1. **Low Image Quality**
   - Increase render samples
   - Check lighting configuration
   - Verify texture files

2. **Poor Bounding Boxes**
   - Adjust camera positioning range
   - Check quality thresholds
   - Verify model scaling
