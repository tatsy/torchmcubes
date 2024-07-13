import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from torchmcubes import grid_interp, marching_cubes


def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5 * np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)


def translate(x, y, z):
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def visualize(V, F, C):
    """
    This function is inspired by the following URL:
    https://blog.scientific-python.org/matplotlib/custom-3d-engine/
    """

    V = (V - (V.max(0) + V.min(0)) / 2) / max(V.max(0) - V.min(0))
    MVP = perspective(40, 1, 1, 100) @ \
          translate(0, 0, -2.5) @ \
          xrotate(0.0) @ \
          yrotate(0.0)

    V = np.c_[V, np.ones(len(V))] @ MVP.T
    V /= V[:, 3].reshape(-1, 1)
    V = V[F]
    C = C[F].mean(axis=1)

    T = V[:, :, :2]
    Z = -V[:, :, 2].mean(axis=1)
    zmin, zmax = Z.min(), Z.max()
    Z = (Z - zmin) / (zmax - zmin)
    I = np.argsort(Z)
    T, C = T[I, :], C[I, :]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes(
        (0, 0, 1, 1),
        xlim=(-1, 1),
        ylim=(-1, 1),
        aspect=1,
        frameon=False,
    )

    collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor='black')
    ax.add_collection(collection)

    plt.show()


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Grid data
    N = 128
    Nx, Ny, Nz = N - 8, N, N + 8
    x, y, z = np.mgrid[:Nx, :Ny, :Nz]
    x = (x / N).astype('float32')
    y = (y / N).astype('float32')
    z = (z / N).astype('float32')

    # Implicit function (metaball)
    f0 = (x - 0.35)**2 + (y - 0.35)**2 + (z - 0.35)**2
    f1 = (x - 0.65)**2 + (y - 0.65)**2 + (z - 0.65)**2
    u = 1.0 / f0 + 1.0 / f1
    rgb = np.stack((x, y, z), axis=-1)
    rgb = np.transpose(rgb, axes=(3, 2, 1, 0)).copy()

    # Test (CPU)
    u = torch.from_numpy(u)
    rgb = torch.from_numpy(rgb)
    verts, faces = marching_cubes(u, 15.0)
    colrs = grid_interp(rgb, verts)

    verts = verts.numpy()
    faces = faces.numpy()
    colrs = colrs.numpy()
    visualize(verts, faces, colrs)

    # Test (GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
        u = u.to(device)
        rgb = rgb.to(device)
        verts, faces = marching_cubes(u, 15.0)
        colrs = grid_interp(rgb, verts)

        verts = verts.cpu().numpy()
        faces = faces.cpu().numpy()
        colrs = colrs.cpu().numpy()
        visualize(verts, faces, colrs)

    else:
        print('CUDA is not available in this environment. Skip testing.')


if __name__ == '__main__':
    main()
