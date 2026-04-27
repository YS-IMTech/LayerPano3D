import argparse
import os
import subprocess
import sys


def run_step(name, cmd):
    print(f"\n===== {name} =====")
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def quality_profile(quality):
    profiles = {
        "standard": {
            "gen_h": 720,
            "gen_w": 1440,
            "out_h": 1024,
            "out_w": 2048,
            "ref_steps": 50,
            "ref_guidance": 7.0,
            "maxsize": 2048,
            "layer_steps": 50,
            "layer_guidance": 30.0,
            "train_h": 1024,
            "train_w": 2048,
            "outlier_thresh": 4,
            "train_quality": "standard",
        },
        "high": {
            "gen_h": 1024,
            "gen_w": 2048,
            "out_h": 1536,
            "out_w": 3072,
            "ref_steps": 60,
            "ref_guidance": 7.5,
            "maxsize": 2048,
            "layer_steps": 60,
            "layer_guidance": 32.0,
            "train_h": 1536,
            "train_w": 3072,
            "outlier_thresh": 3,
            "train_quality": "high",
        },
        "ultra": {
            "gen_h": 1280,
            "gen_w": 2560,
            "out_h": 2048,
            "out_w": 4096,
            "ref_steps": 70,
            "ref_guidance": 8.0,
            "maxsize": 4096,
            "layer_steps": 70,
            "layer_guidance": 35.0,
            "train_h": 2048,
            "train_w": 4096,
            "outlier_thresh": 2,
            "train_quality": "ultra",
        },
    }
    return profiles[quality]


def main():
    parser = argparse.ArgumentParser(description="End-to-end LayerPano3D inference on MPS")
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--save_dir", default="outputs", type=str)
    parser.add_argument("--scene_type", default="outdoor", type=str, choices=["indoor", "outdoor"])
    parser.add_argument("--seed", default=16806, type=int)
    parser.add_argument("--lora_path", default="checkpoints/pano_lora_720x1440_v1.safetensors", type=str)
    parser.add_argument("--depth_model", default="DepthAnythingv2", type=str)
    parser.add_argument("--quality", default="high", choices=["standard", "high", "ultra"], type=str)
    parser.add_argument("--run_sr", action="store_true")
    parser.add_argument("--mps_rasterizer", default="cpp", choices=["python", "cpp"], type=str)
    args = parser.parse_args()

    py = sys.executable
    cfg = quality_profile(args.quality)

    os.makedirs(args.save_dir, exist_ok=True)

    run_step(
        "Step 1 - Reference Panorama",
        [
            py,
            "gen_refpano.py",
            "--save_dir",
            args.save_dir,
            "--prompt",
            args.prompt,
            "--lora_path",
            args.lora_path,
            "--seed",
            str(args.seed),
            "--gen_height",
            str(cfg["gen_h"]),
            "--gen_width",
            str(cfg["gen_w"]),
            "--out_height",
            str(cfg["out_h"]),
            "--out_width",
            str(cfg["out_w"]),
            "--num_inference_steps",
            str(cfg["ref_steps"]),
            "--guidance_scale",
            str(cfg["ref_guidance"]),
        ],
    )

    run_step(
        "Step 2 - Panorama Depth",
        [
            py,
            "gen_panodepth.py",
            "--save_dir",
            args.save_dir,
            "--depth_model",
            args.depth_model,
            "--input_path",
            f"{args.save_dir}/rgb.png",
        ],
    )

    run_step(
        "Step 3 - Auto Layering",
        [
            py,
            "gen_autolayering.py",
            "--input_dir",
            args.save_dir,
            "--scene_type",
            args.scene_type,
            "--max_layers",
            str(args.num_layers),
        ],
    )

    run_step(
        "Step 4 - Layer Data Construction",
        [
            py,
            "gen_layerdata.py",
            "--lora_path",
            args.lora_path,
            "--base_dir",
            f"{args.save_dir}/layering",
            "--seed",
            str(args.seed),
            "--maxsize",
            str(cfg["maxsize"]),
            "--num_inference_steps",
            str(cfg["layer_steps"]),
            "--guidance_scale",
            str(cfg["layer_guidance"]),
        ],
    )

    if args.run_sr:
        run_step(
            "Optional - PASD SR",
            [
                py,
                "pasd/run_layers_pasd.py",
                "--inputs_dir",
                f"{args.save_dir}/layering",
                "--upscale",
                "2",
            ],
        )

    train_cmd = [
        py,
        "gen_traindata.py",
        "--save_dir",
        args.save_dir,
        "--depth_model",
        args.depth_model,
        "--layerpano_dir",
        f"{args.save_dir}/layering",
        "--layerexp_h",
        str(cfg["train_h"]),
        "--layerexp_w",
        str(cfg["train_w"]),
    ]
    if args.run_sr:
        train_cmd.append("--sr")

    run_step("Step 5 - Training Data", train_cmd)

    run_step(
        "Step 6 - 3DGS Training (MPS backend)",
        [
            py,
            "run_layerpano.py",
            "--input_dir",
            args.save_dir,
            "--save_dir",
            args.save_dir,
            "--outlier_thresh",
            str(cfg["outlier_thresh"]),
            "--mps_rasterizer",
            args.mps_rasterizer,
            "--quality",
            cfg["train_quality"],
        ],
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
