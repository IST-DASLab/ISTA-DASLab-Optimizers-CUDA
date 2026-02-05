#include "../utils.h"
#include <float.h>
#include <limits>
// #include "parallel_reduce.h"

__device__ inline void generic_parallel_reduce(float *mem, int N, const long THREADS, const long Tid) {
	/*
	    Compute parallel reduce on a shared memory array of `n` elements using `T` threads, even when n > T.
	*/

	// perform addition to the first T elements (when n > T)
	for(int i = Tid + THREADS; i < N; i += THREADS) {
	    mem[Tid] += mem[i];
	}
	__syncthreads();
	for(int stride = (THREADS >> 1); stride > 0; stride >>= 1) {
	    if(Tid < stride && Tid + stride < N) {
	        mem[Tid] += mem[Tid + stride];
	    }
	    __syncthreads();
	}
}

__global__ void
compute_row_dense_kernel(float *global_V, float *global_g, float *global_q, float *global_out, int row_start, int row_end, int m, float damp, int N, int B, int grad_const)
{
    const long Bid = blockIdx.x; // block id
	const long THREADS = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

    extern __shared__ float mem[];
    float *V = mem; // size B, stores one row of V, e.g. V[i, Bid, :]
    float *g = mem + B; // size B, stores one row of g, e.g. g[Bid, :]
    float *prods = mem + 2 * B; // size B, stores products V*g before summing up.
    float *Vout = mem + 3 * B; // size B, accumulates dot * V

    // predefined constants to avoid computing the same quantities multiple times
    long N_B = N * B;
    long Bid_B = Bid * B;

    int i, j, j_global;
    float dot, q, delta;
    long V_start;

    // g = global_g[Bid, :]
    copy_global_to_shmem(global_g, g, Bid_B, Bid_B + B, THREADS, Tid);
    __syncthreads();

//     copy_global_to_shmem(global_out, Vout, Bid_B, Bid_B + B, THREADS, Tid); // Vout = out[Bid, :]
    if(row_end < m) { // we call the kernel to compute rows of V
//         V_start = 0 * N_B + Bid_B;
//         copy_global_to_shmem<T>(global_V, Vout, V_start, V_start + B, THREADS, Tid);
        for(i = Tid; i < B; i += THREADS) {
            if(should_skip(g[i], grad_const)) {
                Vout[i] = static_cast<float>(0);
            } else {
                Vout[i] = static_cast<float>(damp) * static_cast<float>(g[i]);
            }
        }
    } else if(row_end == m) { // we call the kernel to compute the final update that prunes the model
        for(i = Tid; i < B; i += THREADS) {
            Vout[i] = static_cast<float>(0);
        }
    }
    __syncthreads();

    for(j = row_start; j < row_end; ++j) {
        // V = global_V[j, Bid, :]
        V_start = j * N_B + Bid_B;
        copy_global_to_shmem(global_V, V, V_start, V_start + B, THREADS, Tid);
        __syncthreads();

        // (1) compute dot products
        for(i = Tid; i < B; i += THREADS) {
            if(should_skip(g[i], grad_const)) {
                prods[i] = static_cast<float>(0);
            } else {
                prods[i] = V[i] * g[i];
            }
        }

        __syncthreads();

        generic_parallel_reduce(prods, B, THREADS, Tid);
        dot = prods[0];

        // read q from global memory: q = global_q[j, Bid]
        if(Tid == 0) {
            prods[0] = global_q[j * N + Bid];
        }
        __syncthreads();
        q = prods[0];
        delta = dot / q;

        for(i = Tid; i < B; i += THREADS) {
            Vout[i] -= delta * V[i];
        }
    } // end for j < row

    // out[Bid, :] = Vout
    for(j_global = Bid_B + Tid, j = Tid;
        j_global < Bid_B + B;
        j_global += THREADS, j += THREADS)
    {
        global_out[j_global] = Vout[j];
    }

    // TODO: compute q here, based on Vout: q[row, Bid] = m + dot_product(Vout, g)
}

void
compute_row_dense_cuda (TT V, TT g, TT q, TT out, int row_start, int row_end, int m, float damp, int N, int B, int nblocks, int nthreads, int grad_const)
{
    dim3 blocks(nblocks, 1, 1);
    dim3 threads(nthreads, 1, 1);
    long sh_mem_size_bytes = 4 * B * sizeof(float);

    if(sh_mem_size_bytes > 48 * 1024) {
        cudaFuncSetAttribute(compute_row_dense_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_mem_size_bytes);
    }

    float* fpV = (float*) V.data_ptr();
    float* fpg = (float*) g.data_ptr();
    float* fpq = (float*) q.data_ptr();
    float* fpout = (float*) out.data_ptr();

    compute_row_dense_kernel<<<blocks, threads, sh_mem_size_bytes>>>(fpV, fpg, fpq, fpout, row_start, row_end, m, damp, N, B, grad_const);

	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
// 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
