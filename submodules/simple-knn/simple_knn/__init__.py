#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import platform

# Try to import the C++ extension, with fallback on macOS/CPU.
try:
    from . import _C
    HAS_CUDA_EXTENSION = True
except (ImportError, ModuleNotFoundError):
    HAS_CUDA_EXTENSION = False
    if platform.system() == "Darwin":
        print("[warning] simple_knn on macOS without CUDA extension")
        print("          inference will run in CPU-only mode")
    else:
        print("[warning] CUDA extension is not available")
