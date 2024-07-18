import time
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
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def visualize(V, F, C):
    """
    This function is inspired by the following URL:
    https://blog.scientific-python.org/matplotlib/custom-3d-engine/
    """

    V = (V - (V.max(0) + V.min(0)) / 2) / max(V.max(0) - V.min(0))
    MVP = (perspective(40, 1, 1, 100) @ translate(0, 0, -2.5) @ xrotate(0.0) @ yrotate(0.0))

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

    collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="#00000033")
    ax.add_collection(collection)

    plt.show()


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Grid data
    N = 128
    xs = np.linspace(-1.0, 1.0, N, endpoint=True, dtype="float32")
    ys = np.linspace(-1.0, 1.0, N, endpoint=True, dtype="float32")
    zs = np.linspace(-1.0, 1.0, N, endpoint=True, dtype="float32")
    zs, ys, xs = np.meshgrid(zs, ys, xs)

    # Implicit function (metaball)
    f0 = (xs - 0.35)**2 + (ys - 0.35)**2 + (zs - 0.35)**2
    f1 = (xs + 0.35)**2 + (ys + 0.35)**2 + (zs + 0.35)**2
    u = 4.0 / (f0 + 1.0e-6) + 4.0 / (f1 + 1.0e-6)

    rgb = np.stack((xs, ys, zs), axis=-1) * 0.5 + 0.5
    rgb = np.transpose(rgb, axes=(3, 2, 1, 0))
    rgb = np.ascontiguousarray(rgb)

    # Test (CPU)
    u = torch.from_numpy(u)
    rgb = torch.from_numpy(rgb)

    t_start = time.time()
    verts, faces = marching_cubes(u, 15.0)
    colors = grid_interp(rgb, verts)
    t_end = time.time()
    print(f"verts: {verts.size(0)}, faces: {faces.size(0)}, time: {t_end - t_start:.2f}s")

    verts = verts.numpy()
    faces = faces.numpy()
    colors = colors.numpy()
    verts = (verts / (N - 1)) * 2.0 - 1.0  # Get back to the original space
    visualize(verts, faces, colors)

    # Test (GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
        u = u.to(device)
        rgb = rgb.to(device)

        t_start = time.time()
        verts, faces = marching_cubes(u, 15.0)
        colors = grid_interp(rgb, verts)
        t_end = time.time()
        print(f"verts: {verts.size(0)}, faces: {faces.size(0)}, time: {t_end - t_start:.2f}s")

        verts = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        colors = colors.detach().cpu().numpy()
        verts = (verts / (N - 1)) * 2.0 - 1.0  # Get back to the original space
        visualize(verts, faces, colors)

    else:
        print("CUDA is not available in this environment. Skip testing.")


if __name__ == "__main__":
    main()
