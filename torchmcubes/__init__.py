import torch
import torchmcubes_module as mc


def marching_cubes(vol, thresh):
    """
    vol: 3D torch tensor
    thresh: threshold
    """

    if vol.is_cuda:
        return mc.mcubes_cuda(vol, thresh)
    else:
        return mc.mcubes_cpu(vol, thresh)


def grid_interp(vol, points):
    """
    Interpolate volume data at given points

    Inputs:
        vol: 4D torch tensor (C, Nz, Ny, Nx)
        points: point locations (Np, 3)
    Outputs:
        output: interpolated data (Np, C)    
    """

    if vol.is_cuda:
        return mc.grid_interp_cuda(vol, points)
    else:
        return mc.grid_interp_cpu(vol, points)
