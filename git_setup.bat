@echo off
REM Git setup script for Inspection Computer Vision

echo 🔍 Initializing Git repository for Inspection Computer Vision...

git init
if errorlevel 1 (
    echo ❌ Git init failed. Make sure Git is installed.
    pause
    exit /b 1
)

git add .
if errorlevel 1 (
    echo ❌ Git add failed.
    pause
    exit /b 1
)

git commit -m "Initial commit: Inspection Computer Vision System

- Complete YOLO-based defect detection pipeline
- Synthetic data generation using Blender
- Autoencoder integration for anomaly detection
- Multi-modal training with real + synthetic data
- Industrial inspection applications
- Advanced computer vision research tools

Features:
- YOLO v1 and v2 implementations
- Automated 3D scene generation
- HDRI-based realistic lighting
- Comprehensive defect simulation
- Real-time detection capabilities
- Batch processing tools
- Jupyter notebook tutorials

Ready for research and industrial applications."

if errorlevel 1 (
    echo ❌ Git commit failed.
    pause
    exit /b 1
)

echo ✅ Git repository initialized successfully!
echo.
echo 📤 To push to GitHub:
echo 1. Create a new repository on GitHub: https://github.com/Janga786/Inspection-Computer-Vision
echo 2. Run: git remote add origin https://github.com/Janga786/Inspection-Computer-Vision.git
echo 3. Run: git push -u origin main
echo.
echo ⚠️  Note: Large files (datasets, models) are gitignored.
echo    Consider using Git LFS for model files if needed.
echo.
pause