#ifdef _MSC_VER
#pragma once
#endif

#ifndef _MACROS_H_
#define _MACROS_H_

#include <torch/extension.h>

#include <cstdio>

#define CHECK_CUDA(x)                                                       \
    do {                                                                    \
        TORCH_CHECK(vol.type().is_cuda(), #x " must be a CUDA tensor");     \
    } while(0)

#define CHECK_CPU(x)                                                        \
    do {                                                                    \
        TORCH_CHECK(x.device() == torch::kCPU, #x " must be a CPU tensor"); \
    } while(0)

#define CHECK_CONTIGUOUS(x)                                             \
    do {                                                                \
        TORCH_CHECK(x.is_contiguous(), #x " must be contiguous");       \
    } while(0)

#define CHECK_IS_INT(x)                                                 \
    do {                                                                \
        TORCH_CHECK(x.scalar_type() == at::ScalarType::Int,             \
                    #x " must be an int tensor");                       \
    } while(0)

#define CHECK_IS_FLOAT(x)                                               \
    do {                                                                \
        TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,           \
                    #x " must be a float tensor");                      \
    } while(0)

#define CHECK_N_DIM(x, n)                                               \
    do {                                                                \
        char msg[256];                                                  \
        sprintf(msg, "%s must be %d-dimension", #x, n);                 \
        TORCH_CHECK(x.sizes().size() == n, msg);                        \
    } while(0)

#endif  // _MACROS_H_
