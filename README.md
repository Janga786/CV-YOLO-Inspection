# CV-YOLO-Inspection

Synthetic-data pipeline and YOLO training workflow for automated visual inspection of beverage cans.

This project builds an end-to-end computer vision workflow for detecting cans and surface defects (scratches, dents, punctures) without hand-labeling a large real-world dataset. Photorealistic training images are generated in Blender from 3D can models, composited over varied backgrounds, and rendered with automatically generated YOLO-format bounding-box labels. The resulting datasets are used to train Ultralytics YOLO detectors, complemented by a convolutional autoencoder for unsupervised anomaly detection. The pipeline was developed as the perception component for a robotic inspection workflow, where a camera-equipped arm examines products and flags defective units.

![Training batch of synthetic can images](docs/images/train_batch0.jpg)

*A training batch from the bundled model run: Blender-rendered cans composited onto varied real backgrounds, with mosaic augmentation applied by the Ultralytics trainer.*

## Pipeline

1. **3D assets** — Can meshes in FBX format (`YOLO_2/source/Coke Can.fbx`, `YOLO_2/Synthetic_coke_pepsi.fbx`) with PBR texture support (base color, normal, metallic, roughness).
2. **Background sourcing** — Scripts that pull Creative Commons background imagery from Flickr and Openverse (`flickr_image_finder*.py`, `openverse_image_finder.py`, `cc_image_finder*.py`, `download_backgrounds.py`), plus optional HDRI environments.
3. **Synthetic rendering** — Headless Blender scripts (`enhanced_generator.py`, `enhanced_generator_v282.py`, `generate_detector_data.py`) that randomize camera position, focal length, three-point lighting, and materials; simulate scratch, dent, and puncture defects at configurable probabilities; and write YOLO labels for every render. `known_good_config.json` holds a validated render configuration.
4. **Quality validation and analysis** — `dataset_analyzer.py`, `check_images.py`, and `diagnose_model.py` validate bounding boxes, image quality, and class distribution before training.
5. **Detector training** — Datasets are exported in Ultralytics format (see `YOLO_2/defect_dataset/data.yaml`, classes: `can`, `scratch`, `dent`, `puncture`) and trained with Ultralytics YOLO. The bundled model was trained on Google Colab (`train_autoencoder_colab.py` covers the autoencoder equivalent).
6. **Anomaly detection** — A TensorFlow/Keras convolutional autoencoder (`train_autoencoder.py`) is trained on pristine renders so that defective cans can also be flagged by reconstruction error, independent of the detector's class list. Trained weights live in `YOLO_2/autoencoder_model/`.

## Repository layout

```
CV-YOLO-Inspection/
├── YOLO_2/                          # Main pipeline
│   ├── enhanced_generator.py        # Blender generation engine (Cycles)
│   ├── enhanced_generator_v282.py   # Blender 2.82-compatible engine
│   ├── generate_detector_data.py    # Render + auto-label driver
│   ├── enhanced_runner.py           # Production-scale generation runner
│   ├── run_continuous.py            # Long-running / resumable generation
│   ├── dataset_analyzer.py          # Dataset quality analysis
│   ├── *_image_finder*.py           # CC background sourcing (Flickr, Openverse)
│   ├── train_autoencoder.py         # Convolutional autoencoder (anomaly detection)
│   ├── autoencoder_model/           # Trained autoencoder weights (TF variables)
│   ├── defect_dataset/              # YOLO labels + data.yaml (4 classes)
│   ├── source/                      # 3D can assets (FBX)
│   ├── known_good_config.json       # Validated render configuration
│   └── README_ENHANCED_SYSTEM.md    # Detailed pipeline documentation
├── models/
│   └── trained_model.zip            # Trained YOLO11s weights + training artifacts
├── docs/images/                     # Sample training visualizations
├── LICENSE
└── README.md
```

Large generated datasets, HDRI files, and standalone model checkpoints are intentionally excluded from version control (see `.gitignore`).

## Trained model

`models/trained_model.zip` contains:

- `my_model.pt` — Ultralytics YOLO11s detector weights from a single-class can-detection run
- `train/args.yaml` — the full training configuration (YOLO11s, 20 epochs, 640 px images, batch 16, mosaic augmentation)
- `train/train_batch*.jpg`, `train/labels.jpg` — training-batch mosaics and label-distribution plots

Label statistics from that run (roughly 900 annotated instances, uniformly distributed positions):

![Label distribution for the bundled training run](docs/images/labels.jpg)

## Setup

Prerequisites:

- Python 3.8+
- Blender 2.82+ available on `PATH` (rendering runs headless via `blender --background`)
- A CUDA-capable GPU is recommended for rendering and training

```bash
git clone https://github.com/Janga786/CV-YOLO-Inspection.git
cd CV-YOLO-Inspection/YOLO_2
pip install ultralytics tensorflow opencv-python numpy matplotlib pandas pillow tqdm
```

Note: several scripts contain absolute paths from the original development machine (e.g., `/home/janga/YOLO_2/...` in `defect_dataset/data.yaml` and `train_autoencoder.py`). Update these to your local paths before running.

## Usage

Smoke-test the generation pipeline (renders a handful of images to verify model placement, textures, and labels):

```bash
cd YOLO_2
python3 generate_dataset.py
```

Run production-scale synthetic generation with quality monitoring:

```bash
python3 enhanced_runner.py
```

Analyze a generated dataset before training:

```bash
python3 dataset_analyzer.py <dataset_dir>
```

Train a detector on the generated data:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(data="defect_dataset/data.yaml", epochs=20, imgsz=640, batch=16)
```

Run inference with the bundled weights:

```python
from ultralytics import YOLO

model = YOLO("my_model.pt")  # extracted from models/trained_model.zip
results = model.predict("path/to/image.jpg")
```

Train the autoencoder for anomaly detection (after updating `DATASET_PATH`):

```bash
python3 train_autoencoder.py
```

See `YOLO_2/README_ENHANCED_SYSTEM.md` for the full pipeline documentation, configuration options, and output structure.

## Results

The repository ships the artifacts of a completed training run rather than claimed benchmark numbers: the trained YOLO11s weights, the exact training configuration, and the training-batch and label-distribution plots shown above. Rendering throughput during dataset generation was approximately 8 seconds per 1024x1024 image, and the pipeline has been exercised at the scale of several thousand images per dataset.

## License

MIT — see [LICENSE](LICENSE).
