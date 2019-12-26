#include "pscan.h"

#include <iostream>

#include <cuda_runtime.h>

static const int THREADS_PER_BLOCK = 128;
static const int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

__host__ int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

__global__ void prescan_small_kernel(int *input, int *output, int n, int pow2) {
    extern __shared__ int buffer[];
    const int threadID = threadIdx.x;

    if (threadID < n) {
        buffer[2 * threadID] = input[2 * threadID];
        buffer[2 * threadID + 1] = input[2 * threadID + 1];
    } else {
        buffer[2 * threadID] = 0.0;
        buffer[2 * threadID + 1] = 0.0;
    }

    int offset = 1;
    for (int d = pow2 >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadID < d) {
            const int ai = offset * (2 * threadID + 1) - 1;
            const int bi = offset * (2 * threadID + 2) - 1;
            buffer[bi] += buffer[ai];
        }
        offset *= 2;
    }

    if (threadID == 0) { buffer[pow2 - 1] = 0; }

    for (int d = 1; d < pow2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadID < d) {
            const int ai = offset * (2 * threadID + 1) - 1;
            const int bi = offset * (2 * threadID + 2) - 1;
            const int t = buffer[ai];
            buffer[ai] = buffer[bi];
            buffer[bi] += t;           
        }
    }
    __syncthreads();

    if (threadID < n) {
        output[2 * threadID] = buffer[2 * threadID];
        output[2 * threadID + 1] = buffer[2 * threadID + 1];
    }
}

__global__ void prescan_large_kernel(int *input, int *output, int n, int *sums) {
    const int blockID = blockIdx.x;
    const int threadID = threadIdx.x;
    const int blockOffset = blockID * n;

    extern __shared__ int buffer[];
    buffer[2 * threadID] = input[blockOffset + (2 * threadID)];
    buffer[2 * threadID + 1] = input[blockOffset + (2 * threadID + 1)];

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadID < d) {
            const int ai = offset * (2 * threadID + 1) - 1;
            const int bi = offset * (2 * threadID + 2) - 1;
            buffer[bi] += buffer[ai];
        }
        offset *= 2;
    }
    __syncthreads();

    if (threadID == 0) {
        sums[blockID] = buffer[n - 1];
        buffer[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadID < d) {
            const int ai = offset * (2 * threadID + 1) - 1;
            const int bi = offset * (2 * threadID + 2) - 1;
            const int t = buffer[ai];
            buffer[ai] = buffer[bi];
            buffer[bi] += t;           
        }
    }
    __syncthreads();

    output[blockOffset + (2 * threadID)] = buffer[2 * threadID];
    output[blockOffset + (2 * threadID + 1)] = buffer[2 * threadID + 1];
}

__global__ void add(int *output, int length, int *n) {
    const int blockID = blockIdx.x;
    const int threadID = threadIdx.x;
    const int blockOffset = blockID * length;
    output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
    const int blockID = blockIdx.x;
    const int threadID = threadIdx.x;
    const int blockOffset = blockID * length;
    output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

void prescan_small(int *d_in, int *d_out, int n) {
    const int pow2 = nextPowerOfTwo(n);
    prescan_small_kernel<<<1, (n + 1) / 2, 2 * pow2 * sizeof(int)>>>(d_in, d_out, n, pow2);
}

void prescan_large(int *d_in, int *d_out, int n) {
    const int blocks = n / ELEMENTS_PER_BLOCK;
    const int sharedSize = ELEMENTS_PER_BLOCK * sizeof(int);

    int *d_sums, *d_incr;
    cudaMalloc((void**)&d_sums, blocks * sizeof(int));
    cudaMalloc((void**)&d_incr, blocks * sizeof(int));

    prescan_large_kernel<<<blocks, THREADS_PER_BLOCK, 2 * sharedSize>>>(d_in, d_out, ELEMENTS_PER_BLOCK, d_sums);

    const int sumThreadsNeeded = (blocks + 1) / 2;
    if (sumThreadsNeeded > THREADS_PER_BLOCK) {
        prescan_large(d_sums, d_incr, blocks);
    } else {
        prescan_small(d_sums, d_incr, blocks);
    }

    add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

    cudaFree(d_sums);
    cudaFree(d_incr);
}

void prescan(int *d_in, int *d_out, int size) {
    const size_t residue = size % ELEMENTS_PER_BLOCK;
    if (size < ELEMENTS_PER_BLOCK) {
        prescan_small(d_in, d_out, size);
    } else if (residue == 0) {
        prescan_large(d_in, d_out, size);
    } else {
        const size_t tail = size - residue;
        prescan_large(d_in, d_out, tail);
        prescan_small(&d_in[tail], &d_out[tail], residue);
        add<<<1, residue>>>(&d_out[tail], residue, &d_in[tail - 1], &d_out[tail - 1]);
    }
}
