# LayerPano3D Setup for macOS ARM64

Complete setup guide for running this MPS-focused fork on Apple Silicon (M1/M2/M3+).

## Prerequisites

- macOS 11+
- Apple Silicon CPU
- Conda or Miniforge installed
- At least 20 GB free disk space

## Quick Setup

```bash
cd /Users/nicola/Documents/GitHub/Tesi/LayerPano3D
chmod +x setup_arm64.sh
./setup_arm64.sh
```

The setup script will:

1. Create a fresh `layerpano3d` conda environment (Python 3.10)
2. Install PyTorch CPU wheels (MPS is provided by PyTorch on macOS)
3. Install project dependencies from `requirements.txt`
4. Install local submodules (`diff-gaussian-rasterization`, `simple-knn`)

## Checkpoint Download

After setup:

```bash
conda activate layerpano3d
mkdir -p checkpoints/Infusion
```

Download required files:

- `checkpoints/pano_lora_720x1440_v1.safetensors`
- `checkpoints/ControlNetLama.pth`
- `checkpoints/sam_vit_h_4b8939.pth`
- `checkpoints/depth_anything_v2_vitl.pth`
- Infusion files under `checkpoints/Infusion/`

## Verify Installation

```bash
conda activate layerpano3d
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

## Run

```bash
huggingface-cli login
bash run.sh
```

## ARM64 Notes

- This fork defaults to an MPS-first pipeline.
- CUDA-only optional paths are not required for normal usage.
- `DepthAnythingv2` is the default depth path in this workflow.

## Troubleshooting

### Missing torch

```bash
conda activate layerpano3d
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Submodule build issues

```bash
pip install -e submodules/diff-gaussian-rasterization --no-cache-dir
pip install -e submodules/simple-knn --no-cache-dir
```
