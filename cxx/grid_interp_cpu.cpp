#include <torch/extension.h>

#define NOMIMAX
#include <cstdlib>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "macros.h"

torch::Tensor grid_interp_cpu(torch::Tensor vol, torch::Tensor points) {
    CHECK_CPU(vol);
    CHECK_CONTIGUOUS(vol);
    CHECK_N_DIM(vol, 4);

    CHECK_CPU(points);
    CHECK_CONTIGUOUS(points);
    CHECK_N_DIM(points, 2);

    const int Nx = vol.size(3);
    const int Ny = vol.size(2);
    const int Nz = vol.size(1);
    const int C = vol.size(0);
    const int Np = points.size(0);

    torch::Tensor output = torch::zeros({Np, C},
        torch::TensorOptions().dtype(torch::kFloat32).device(vol.device()));

    auto vol_ascr = vol.accessor<float, 4>();
    auto pts_ascr = points.accessor<float, 2>();
    auto out_ascr = output.accessor<float, 2>();

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < Np; i++) {
        const float x = pts_ascr[i][0];
        const float y = pts_ascr[i][1];
        const float z = pts_ascr[i][2];

        const int ix = (int)x;
        const int iy = (int)y;
        const int iz = (int)z;
        const float fx = x - ix;
        const float fy = y - iy;
        const float fz = z - iz;

        for (int c = 0; c < C; c++) {
            const int x0 = std::max(0, std::min(ix, Nx - 1));
            const int x1 = std::max(0, std::min(ix + 1, Nx - 1));
            const int y0 = std::max(0, std::min(iy, Ny - 1));
            const int y1 = std::max(0, std::min(iy + 1, Ny - 1));
            const int z0 = std::max(0, std::min(iz, Nx - 1));
            const int z1 = std::max(0, std::min(iz + 1, Nz - 1));

            const float v00 = (1.0 - fx) * vol_ascr[c][z0][y0][x0] + fx * vol_ascr[c][z0][y0][x1];
            const float v01 = (1.0 - fx) * vol_ascr[c][z0][y1][x0] + fx * vol_ascr[c][z0][y1][x1];
            const float v10 = (1.0 - fx) * vol_ascr[c][z1][y0][x0] + fx * vol_ascr[c][z1][y0][x1];
            const float v11 = (1.0 - fx) * vol_ascr[c][z1][y1][x0] + fx * vol_ascr[c][z1][y1][x1];
            
            const float v0 = (1.0 - fy) * v00 + fy * v01;
            const float v1 = (1.0 - fy) * v10 + fy * v11;

            out_ascr[i][c] = (1.0 - fz) * v0 + fz * v1;
        }         
    }

    return output;
}
