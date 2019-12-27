#ifdef _MSC_VER
#pragma once
#endif

#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <cuda_runtime.h>

#define CUDA_CHECK_ERRORS()                                                 \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (cudaSuccess != err) {                                           \
            fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
                    cudaGetErrorString(err), __PRETTY_FUNCTION__,           \
                    __LINE__, __FILE__);                                    \
            std::exit(1);                                                   \
        }                                                                   \
    } while(0)

#endif  // _CUDA_UTILS_H_
