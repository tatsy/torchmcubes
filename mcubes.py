import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))
import mcubes_module as mc

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


if __name__ == '__main__':
    # Modules needed for testing
    import numpy as np
    import open3d as o3d

    # Grid data
    N = 128
    x, y, z = np.mgrid[:N, :N, :N]
    x = (x / N).astype('float32')
    y = (y / N).astype('float32')
    z = (z / N).astype('float32')

    # Implicit function (metaball)
    f0 = (x - 0.35) ** 2 + (y - 0.35) ** 2 + (z - 0.35) ** 2
    f1 = (x - 0.65) ** 2 + (y - 0.65) ** 2 + (z - 0.65) ** 2
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

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colrs)
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    o3d.visualization.draw_geometries([mesh, wire], window_name='Marching cubes (CPU)')

    # Test (GPU)
    if torch.cuda.is_available():
        u = u.cuda()
        rgb = rgb.cuda()
        verts, faces = marching_cubes(u, 15.0)
        colrs = grid_interp(rgb, verts)

        verts = verts.cpu().numpy()
        faces = faces.cpu().numpy()
        colrs = colrs.cpu().numpy()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colrs)
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        o3d.visualization.draw_geometries([mesh, wire], window_name='Marching cubes (CUDA)')

    else:
        print('CUDA is not available in this environment. Skip testing.')
