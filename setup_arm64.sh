#!/bin/bash
# Full LayerPano3D setup for macOS ARM64 (Apple Silicon)
# This script installs the required dependencies for the MPS workflow.

set -e  # Exit on error

echo "=========================================="
echo "LayerPano3D Setup for macOS ARM64"
echo "=========================================="

# 1. Remove old environment if it exists
echo -e "\n[1/6] Removing old conda environment..."
conda remove -n layerpano3d --all -y 2>/dev/null || echo "No previous environment found"

# 2. Create a new environment with Python 3.10
echo -e "\n[2/6] Creating conda environment with Python 3.10..."
conda create -n layerpano3d python=3.10 -y

# 3. Activate environment
echo -e "\n[3/6] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate layerpano3d

# 4. Install updated setuptools
echo -e "\n[4/6] Installing updated setuptools..."
pip install "setuptools>=65" wheel

# 5. Install PyTorch CPU wheels (required to build local modules)
echo -e "\n[5/6] Installing PyTorch CPU wheels..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install dependencies from requirements.txt
echo -e "\n[6/6] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 7. Install local build submodules (diff-gaussian-rasterization, simple-knn)
echo -e "\n[EXTRA] Installing local submodules..."
pip install -e submodules/diff-gaussian-rasterization && echo "✓ diff-gaussian-rasterization installed" || echo "⚠ Warning: diff-gaussian-rasterization failed"
pip install -e submodules/simple-knn && echo "✓ simple-knn installed" || echo "⚠ Warning: simple-knn failed"

# Download checkpoints
echo -e "\n=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download checkpoints (see README/SETUP_ARM64.md):"
echo "   - Panorama LoRA"
echo "   - Lama"
echo "   - SAM model"
echo "   - Depth-Anything-V2"
echo "   - Infusion"
echo ""
echo "2. To validate the setup:"
echo "   conda activate layerpano3d"
echo "   python -c \"import torch; print(f'PyTorch: {torch.__version__}')\""
echo ""
echo "3. Login to Hugging Face and run:"
echo "   huggingface-cli login"
echo "   bash run.sh"
echo ""
echo "If needed, disable 360monodepth and keep Depth-Anything-V2 only in gen_panodepth.py"
