import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from tqdm import tqdm


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _estimate_log_scales(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.full((points.shape[0], 3), -3.0, dtype=np.float32)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        nn = np.maximum(distances[:, 1] * 0.5, 1e-3)
        log_scale = np.log(nn).astype(np.float32)
        return np.repeat(log_scale[:, None], 3, axis=1)
    except Exception:
        return np.full((points.shape[0], 3), -3.0, dtype=np.float32)


def _build_optimizer(gaussians):
    return torch.optim.Adam(
        [
            {"params": [gaussians.means], "lr": 0.00016},
            {"params": [gaussians.scales], "lr": 0.005},
            {"params": [gaussians.quaternions], "lr": 0.001},
            {"params": [gaussians.opacities], "lr": 0.05},
            {"params": [gaussians.sh_coeffs], "lr": 0.0025},
        ],
        lr=0.001,
        eps=1e-15,
    )


def _adaptive_topology_update(
    gaussians,
    prune_threshold=0.02,
    clone_fraction=0.08,
    min_points=128,
    max_points=None,
):
    means = gaussians.means.detach().cpu().numpy().astype(np.float32)
    scales = gaussians.scales.detach().cpu().numpy().astype(np.float32)
    quats = gaussians.quaternions.detach().cpu().numpy().astype(np.float32)
    opacities = gaussians.opacities.detach().cpu().numpy().astype(np.float32)
    sh_coeffs = gaussians.sh_coeffs.detach().cpu().numpy().astype(np.float32)

    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacities[:, 0]))
    score = opacity_sigmoid * np.exp(scales).mean(axis=1)

    keep_mask = opacity_sigmoid >= prune_threshold
    if keep_mask.sum() < min_points and len(score) > 0:
        keep_mask = np.zeros(len(score), dtype=bool)
        keep_mask[np.argsort(score)[-min(min_points, len(score)) :]] = True

    means = means[keep_mask]
    scales = scales[keep_mask]
    quats = quats[keep_mask]
    opacities = opacities[keep_mask]
    sh_coeffs = sh_coeffs[keep_mask]
    score = score[keep_mask]

    if max_points is not None and len(score) > int(max_points):
        keep_top = int(max(max_points, min_points))
        top_idx = np.argsort(score)[-keep_top:]
        means = means[top_idx]
        scales = scales[top_idx]
        quats = quats[top_idx]
        opacities = opacities[top_idx]
        sh_coeffs = sh_coeffs[top_idx]
        score = score[top_idx]

    clone_count = int(max(1, round(len(score) * clone_fraction))) if len(score) and clone_fraction > 0 else 0
    if max_points is not None and clone_count > 0:
        clone_count = max(0, min(clone_count, int(max_points) - len(score)))
    if clone_count > 0:
        clone_indices = np.argsort(score)[-clone_count:]
        noise_scale = np.maximum(np.exp(scales[clone_indices]) * 0.25, 1e-3)
        clone_means = means[clone_indices] + np.random.normal(loc=0.0, scale=noise_scale).astype(np.float32)
        clone_scales = scales[clone_indices] - np.float32(np.log(1.35))
        clone_quats = quats[clone_indices]
        clone_opacities = np.full_like(opacities[clone_indices], _logit(0.08), dtype=np.float32)
        clone_sh = sh_coeffs[clone_indices]

        means = np.concatenate([means, clone_means], axis=0)
        scales = np.concatenate([scales, clone_scales], axis=0)
        quats = np.concatenate([quats, clone_quats], axis=0)
        opacities = np.concatenate([opacities, clone_opacities], axis=0)
        sh_coeffs = np.concatenate([sh_coeffs, clone_sh], axis=0)

    gaussians.means = torch.nn.Parameter(torch.tensor(means, dtype=torch.float32, device=gaussians.means.device))
    gaussians.scales = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=gaussians.scales.device))
    gaussians.quaternions = torch.nn.Parameter(torch.tensor(quats, dtype=torch.float32, device=gaussians.quaternions.device))
    gaussians.opacities = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=gaussians.opacities.device))
    gaussians.sh_coeffs = torch.nn.Parameter(torch.tensor(sh_coeffs, dtype=torch.float32, device=gaussians.sh_coeffs.device))
    return gaussians


def _ensure_torch_gs_importable():
    try:
        import torch_gs  # noqa: F401
        return
    except Exception:
        pass

    candidates = []
    env_path = os.environ.get("SPLAT_APPLE_PATH")
    if env_path:
        candidates.append(env_path)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(repo_root, "submodules", "splat-apple"))
    candidates.append(os.path.join(repo_root, "external", "splat-apple"))

    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
            try:
                import torch_gs  # noqa: F401
                return
            except Exception:
                continue

    raise ModuleNotFoundError(
        "Modulo torch_gs non trovato. Installa splat-apple e compila estensioni: "
        "`python setup.py build_ext --inplace` nel repo splat-apple; "
        "poi esporta SPLAT_APPLE_PATH o clona in submodules/splat-apple."
    )


def _resolve_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_target_tensor(image, device: str) -> torch.Tensor:
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).to(device)


def _pose_to_w2c(pose: np.ndarray) -> np.ndarray:
    # Keep pose convention aligned with LayerPano's MiniCam2.
    w2c = np.linalg.inv(pose)
    w2c[1:3, :3] *= -1
    w2c[:3, 3] *= -1
    return w2c.astype(np.float32)


def _build_training_batch(traindata: Dict, device: str):
    from torch_gs.training.trainer import Camera

    fov_deg = float(traindata["fov"])
    width = int(traindata["W"])
    height = int(traindata["H"])
    fovx = math.radians(fov_deg)
    fovy = height * fovx / width
    fx = width / (2.0 * math.tan(fovx / 2.0))
    fy = height / (2.0 * math.tan(fovy / 2.0))
    cx = width / 2.0
    cy = height / 2.0

    cameras: List[Camera] = []
    targets: List[torch.Tensor] = []

    for frame in traindata["frames"]:
        w2c = _pose_to_w2c(np.array(frame["transform_matrix"], dtype=np.float32))
        cameras.append(
            Camera(
                W=width,
                H=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                W2C=torch.tensor(w2c, dtype=torch.float32, device=device),
            )
        )
        targets.append(_to_target_tensor(frame["image"], device))

    xyz = np.asarray(traindata["pcd_points"], dtype=np.float32)
    rgb = np.asarray(traindata["pcd_colors"], dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    return xyz, rgb, cameras, targets


def _save_layerpano_compatible_ply(path: str, gaussians, sh_degree: int = 3):
    xyz = gaussians.means.detach().cpu().numpy().astype(np.float32)
    normals = np.zeros_like(xyz, dtype=np.float32)

    f_dc = gaussians.sh_coeffs[:, 0, :].detach().cpu().numpy().astype(np.float32)
    expected_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    f_rest = np.zeros((xyz.shape[0], expected_rest), dtype=np.float32)

    opacities = gaussians.opacities.detach().cpu().numpy().astype(np.float32)
    scales = gaussians.scales.detach().cpu().numpy().astype(np.float32)
    rots = gaussians.quaternions.detach().cpu().numpy().astype(np.float32)

    finite_mask = np.isfinite(xyz).all(axis=1)
    finite_mask &= np.isfinite(f_dc).all(axis=1)
    finite_mask &= np.isfinite(opacities).all(axis=1)
    finite_mask &= np.isfinite(scales).all(axis=1)
    finite_mask &= np.isfinite(rots).all(axis=1)

    if not np.all(finite_mask):
        print(f"Filtering out {int((~finite_mask).sum())} non-finite gaussians before export.")
        xyz = xyz[finite_mask]
        normals = normals[finite_mask]
        f_dc = f_dc[finite_mask]
        f_rest = f_rest[finite_mask]
        opacities = opacities[finite_mask]
        scales = scales[finite_mask]
        rots = rots[finite_mask]

    attrs: List[Tuple[str, str]] = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]
    attrs += [(f"f_dc_{i}", "f4") for i in range(3)]
    attrs += [(f"f_rest_{i}", "f4") for i in range(expected_rest)]
    attrs += [("opacity", "f4")]
    attrs += [(f"scale_{i}", "f4") for i in range(3)]
    attrs += [(f"rot_{i}", "f4") for i in range(4)]

    elements = np.empty(xyz.shape[0], dtype=attrs)
    packed = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scales, rots], axis=1)
    elements[:] = list(map(tuple, packed))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def train_with_splat_apple(
    traindata: Dict,
    out_ply_path: str,
    num_iterations: int,
    rasterizer: str = "cpp",
    device: str = "mps",
    adaptive: bool = True,
    densify_interval: int = 120,
    prune_threshold: float = 0.02,
    clone_fraction: float = 0.08,
    max_points: int = None,
) -> str:
    """Train one layer using splat-apple (torch_gs) backend and export LayerPano-compatible PLY."""
    _ensure_torch_gs_importable()
    from torch_gs.core.gaussians import init_gaussians_from_pcd
    from torch_gs.training.trainer import train_step

    if device == "auto":
        device = _resolve_device()
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; cannot use the splat-apple backend on this host")

    xyz, rgb, cameras, targets = _build_training_batch(traindata, device)

    scale_init = _estimate_log_scales(xyz)

    gaussians = init_gaussians_from_pcd(
        torch.tensor(xyz, dtype=torch.float32),
        torch.tensor(rgb, dtype=torch.float32),
        device=device,
    )

    gaussians.scales = torch.nn.Parameter(torch.tensor(scale_init, dtype=torch.float32, device=device))
    gaussians.opacities = torch.nn.Parameter(torch.full((gaussians.opacities.shape[0], 1), _logit(0.1), dtype=torch.float32, device=device))

    gaussians.means.requires_grad = True
    gaussians.scales.requires_grad = True
    gaussians.quaternions.requires_grad = True
    gaussians.opacities.requires_grad = True
    gaussians.sh_coeffs.requires_grad = True

    optimizer = _build_optimizer(gaussians)

    # On Apple Silicon, cap adaptive growth to avoid MPS OOM on long ultra runs.
    if device == "mps" and max_points is None:
        init_points = int(gaussians.means.shape[0])
        max_points = int(min(max(init_points * 1.35, 20000), 70000))
        densify_interval = max(densify_interval, 200)
        clone_fraction = min(clone_fraction, 0.03)

    progress = tqdm(range(num_iterations), desc="Splat-Apple training")
    t0 = time.time()
    for i in progress:
        idx = random.randint(0, len(cameras) - 1)
        try:
            loss, psnr, _ = train_step(
                gaussians,
                optimizer,
                targets[idx],
                cameras[idx],
                lambda_ssim=0.2,
                device=device,
                rasterizer_type=rasterizer,
            )
        except RuntimeError as e:
            if device == "mps" and "MPS backend out of memory" in str(e):
                print("[splat-apple] MPS OOM detected, pruning gaussians and continuing.", flush=True)
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

                current_n = int(gaussians.means.shape[0])
                prune_to = int(max(4096, current_n * 0.8))
                if max_points is not None:
                    prune_to = min(prune_to, int(max_points))

                gaussians = _adaptive_topology_update(
                    gaussians,
                    prune_threshold=max(prune_threshold, 0.05),
                    clone_fraction=0.0,
                    min_points=512,
                    max_points=prune_to,
                )
                gaussians.means.requires_grad = True
                gaussians.scales.requires_grad = True
                gaussians.quaternions.requires_grad = True
                gaussians.opacities.requires_grad = True
                gaussians.sh_coeffs.requires_grad = True
                optimizer = _build_optimizer(gaussians)
                continue
            raise
        if i % 20 == 0:
            progress.set_postfix({"loss": f"{loss:.4f}", "psnr": f"{psnr:.2f}"})
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"[splat-apple] iter={i}/{num_iterations} loss={loss:.5f} psnr={psnr:.2f} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if adaptive and i > 0 and i % densify_interval == 0:
            gaussians = _adaptive_topology_update(
                gaussians,
                prune_threshold=prune_threshold,
                clone_fraction=clone_fraction,
                max_points=max_points,
            )
            gaussians.means.requires_grad = True
            gaussians.scales.requires_grad = True
            gaussians.quaternions.requires_grad = True
            gaussians.opacities.requires_grad = True
            gaussians.sh_coeffs.requires_grad = True
            optimizer = _build_optimizer(gaussians)

    _save_layerpano_compatible_ply(out_ply_path, gaussians, sh_degree=3)
    return out_ply_path
