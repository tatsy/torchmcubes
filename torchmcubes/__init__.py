import os
import warnings

__version__ = "0.1.0"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Tuple

import torch
import torchmcubes_module as mc

# True when the extension module was compiled with CUDA kernels.
HAS_CUDA = hasattr(mc, "mcubes_cuda")


def _warn_cpu_fallback() -> None:
    warnings.warn(
        "torchmcubes was built without CUDA support, so CUDA tensors are processed on the CPU "
        "and the results are copied back to the original device. Rebuild torchmcubes in an "
        "environment where the CUDA toolkit is available to run on the GPU.",
        RuntimeWarning,
        stacklevel=3,
    )


def marching_cubes(vol: torch.Tensor, thresh: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    vol: 3D torch tensor
    thresh: threshold
    """

    if vol.is_cuda:
        if HAS_CUDA:
            return mc.mcubes_cuda(vol, thresh)
        _warn_cpu_fallback()
        verts, faces = mc.mcubes_cpu(vol.cpu(), thresh)
        return verts.to(vol.device), faces.to(vol.device)
    return mc.mcubes_cpu(vol, thresh)


def grid_interp(vol: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Interpolate volume data at given points

    Inputs:
        vol: 4D torch tensor (C, Nz, Ny, Nx)
        points: point locations (Np, 3)
    Outputs:
        output: interpolated data (Np, C)
    """

    if vol.is_cuda:
        if HAS_CUDA:
            return mc.grid_interp_cuda(vol, points)
        _warn_cpu_fallback()
        return mc.grid_interp_cpu(vol.cpu(), points.cpu()).to(vol.device)
    return mc.grid_interp_cpu(vol, points)
