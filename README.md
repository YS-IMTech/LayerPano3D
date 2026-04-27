# LayerPano3D MPS Fork

This repository is an Apple Silicon and MPS-focused fork of [3DTopia/LayerPano3D](https://github.com/3DTopia/LayerPano3D).
It keeps the original LayerPano3D goals while streamlining the default workflow for macOS + PyTorch MPS.

## What This Fork Changes

- MPS-first end-to-end launcher via `inference_mps.py`
- MPS-only training entrypoint in `run_layerpano.py` (`splat-apple` backend)
- Cleaner local workflow via `run.sh` and `.gitignore`
- ARM64 setup docs and scripts aligned with this fork

## Original Project Links

- Project page: https://ys-imtech.github.io/projects/LayerPano3D/
- Paper: https://arxiv.org/abs/2408.13252
- Dataset: https://huggingface.co/datasets/ysmikey/Layerpano3D_PanoData
- Panorama LoRA: https://huggingface.co/ysmikey/LayerPano3D-FLUX-Panorama-LoRA

## Quick Start (Apple Silicon)

1. Setup environment:

```bash
chmod +x setup_arm64.sh
./setup_arm64.sh
```

2. Download required checkpoints (see `SETUP_ARM64.md`).

3. Login to Hugging Face:

```bash
huggingface-cli login
```

4. Run the full pipeline:

```bash
bash run.sh
```

## MPS Pipeline (Direct)

```bash
python inference_mps.py \
  --prompt "A realistic mountain valley at sunrise" \
  --save_dir outputs \
  --scene_type outdoor \
  --num_layers 3 \
  --quality high
```

### Quality Presets

- `standard`: fastest, lower resolution
- `high`: balanced quality and speed
- `ultra`: best quality, highest memory/time cost

## Rendering

```bash
python -m rendering.render_video_360 --save_dir outputs/scene --elevation 0
python -m rendering.render_video_zigzag --save_dir outputs/scene
```

## Notes

- This fork prioritizes MPS behavior and memory stability on Apple Silicon.
- CUDA-specific optional paths are intentionally not part of the default workflow.

## Citation

If LayerPano3D is useful for your work, please cite the original paper:

```bibtex
@article{yang2024layerpano3d,
  title={LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation},
  author={Yang, Shuai and Tan, Jing and Zhang, Mengchen and Wu, Tong and Li, Yixuan and Wetzstein, Gordon and Liu, Ziwei and Lin, Dahua},
  journal={arXiv preprint arXiv:2408.13252},
  year={2024}
}
```
