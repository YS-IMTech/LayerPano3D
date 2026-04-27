"""
Fallback for simple_knn._C module when CUDA compilation is unavailable.
This module provides a pure-Python implementation of distCUDA2 for CPU/MPS devices.
"""

import torch
import numpy as np
from scipy.spatial import cKDTree


def distCUDA2(points):
    """
    Compute squared distances to 3-nearest neighbors (CPU fallback for distCUDA2).
    
    This is a drop-in replacement for simple_knn._C.distCUDA2 that works on CPU and MPS.
    Uses KDTree for efficient nearest neighbor search.
    
    Args:
        points: (N, 3) tensor of point coordinates on any device
    Returns:
        dist2: (N,) tensor of squared distances to 3-NN
    """
    device = points.device
    points_np = points.detach().cpu().numpy()
    
    # Build KDTree for efficient neighbor search
    tree = cKDTree(points_np)
    
    # Query for 4 nearest neighbors (including self) and take the 4-th one
    # k=4 because index 0 is itself, indices 1,2,3 are the actual 3-NN
    distances, _ = tree.query(points_np, k=4)
    
    # Get distance to 3rd nearest neighbor (index 3) and square it
    dist2 = torch.from_numpy(distances[:, 3] ** 2).float()
    
    # Move result back to original device
    return torch.clamp_min(dist2.to(device), 0.0000001)
