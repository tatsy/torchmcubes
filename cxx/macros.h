#ifdef _MSC_VER
#pragma once
#endif

#ifndef _MACROS_H_
#define _MACROS_H_

#define CHECK_CUDA(x) TORCH_CHECK(vol.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.device() == torch::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_N_DIM(x, n) TORCH_CHECK(x.sizes().size() == n, #x " must be N-dimension")

#endif  // _MACROS_H_
