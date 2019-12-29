#include <torch/extension.h>

#include <iostream>
#include <vector>

// Forward declarations

std::vector<torch::Tensor> mcubes_cpu(
    torch::Tensor func,
    float threshold
);

#if defined(__CUDACC__)
std::vector<torch::Tensor> mcubes_cuda(
    torch::Tensor func,
    float threshold
);
#endif

torch::Tensor grid_interp_cpu(
    torch::Tensor vol,
    torch::Tensor points
);

#if defined(__CUDACC__)
torch::Tensor grid_interp_cuda(
    torch::Tensor vol,
    torch::Tensor points
);
#endif

// Pybind11 exports
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mcubes_cpu", &mcubes_cpu, "Marching cubes (CPU)");
    m.def("grid_interp_cpu", &grid_interp_cpu, "Grid interpolation (CPU)");
    #if defined(__CUDACC__)
    m.def("mcubes_cuda", &mcubes_cuda, "Marching cubes (CUDA)");
    m.def("grid_interp_cuda", &grid_interp_cuda, "Grid interpolation (CUDA)");
    #endif
}
