#include "../utils.h"
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

__device__ inline void unpack(float packed, int &i, float &f) { // ct stands for constant
    /*
        *i = static_cast<int>(packed);
        *f = packed - (*i);
        if((*i) < 0) {
            (*i) = -(*i);
        }
    */
    i = static_cast<int>(packed);
    f = packed - i;
    if(i < 0) {
        i = -i;
    }
}

__global__ void
compute_row_sparse_kernel(float *global_V, int16 *global_gi, float *global_gv, float *global_q, float *global_out, int row_start, int row_end, int m, float damp, int density, int N, int B, int do_init)
{
    const long Bid = blockIdx.x; // block id
	const long THREADS = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

    extern __shared__ unsigned char shmem[];

//     sh_mem_size_bytes += B * sizeof(float); // for a row of V
//     sh_mem_size_bytes += B * sizeof(float); // for a row of Vout
//     sh_mem_size_bytes += density * sizeof(float); // for prods
//     sh_mem_size_bytes += density * sizeof(float); // for gv
//     sh_mem_size_bytes += density * sizeof(int16); // for gi
    float *V = (float*) shmem;
    float *Vout = V + B;
    float *prods = V + 2 * B;
    float *gv = V + 2 * B + density;
    int16 *gi = (int16*)(shmem + (2 * B + 2 * density) * sizeof(float));

    // predefined constants to avoid computing the same quantities multiple times
    long N_B = N * B;
    long Bid_B = Bid * B;
    long Bid_density = Bid * density;

    long V_start;

    int i, j, j_global;
    float dot, q, delta;

    copy_global_to_shmem(global_gv, gv, Bid_density, Bid_density + density, THREADS, Tid);
    __syncthreads();

    copy_global_to_shmem(global_gi, gi, Bid_density, Bid_density + density, THREADS, Tid);
    __syncthreads();

//     for(i = Tid; i < density; i += THREADS) {
//         printf("[Bid=%ld][Tid=%ld][i=%d] gi=%d, gv=%.8f\n", Bid, Tid, i, gi[i], gv[i]);
//     }

    // initialize Vout with zeros in the first place
    for(i = Tid; i < B; i += THREADS) {
        Vout[i] = static_cast<float>(0);
    }
    __syncthreads();

    if(do_init) {
        // initialize with damp * grad
        for(i = Tid; i < density; i += THREADS) {
            Vout[gi[i]] = damp * gv[i];
        }
    }
    __syncthreads();

    for(j = row_start; j < row_end; ++j) {
        // V = global_V[j, Bid, :]
        V_start = j * N_B + Bid_B;
        copy_global_to_shmem(global_V, V, V_start, V_start + B, THREADS, Tid);
        __syncthreads();

        // (1) compute dot products
        for(i = Tid; i < density; i += THREADS) {
            prods[i] = V[gi[i]] * gv[i];
        }
        __syncthreads();

        generic_parallel_reduce(prods, density, THREADS, Tid);
        dot = prods[0];

        // read q from global memory: q = global_q[j, Bid]
        if(Tid == 0) {
            prods[0] = static_cast<float>(global_q[j * N + Bid]);
        }
        __syncthreads();
        q = prods[0];
        delta = dot / q;

        for(i = Tid; i < B; i += THREADS) {
            Vout[i] -= delta * V[i];
        }
    } // end for j < row

    for(j_global = Bid_B + Tid, j = Tid;
        j_global < Bid_B + B;
        j_global += THREADS, j += THREADS)
    {
        global_out[j_global] = Vout[j];
    }
    // TODO: compute q here, based on Vout: q[row, Bid] = m + dot_product(Vout, g)
}

void compute_row_sparse_cuda (TT V, TT gi, TT gv, TT q, TT out, int row_start, int row_end, int m, float damp, int density, int N, int B, int nblocks, int nthreads, int do_init)
{
    dim3 blocks(nblocks, 1, 1);
    dim3 threads(nthreads, 1, 1);

    long sh_mem_size_bytes = 0;
    sh_mem_size_bytes += B * sizeof(float); // for a row of V
    sh_mem_size_bytes += B * sizeof(float); // for a row of Vout
    sh_mem_size_bytes += density * sizeof(float); // for prods
    sh_mem_size_bytes += density * sizeof(float); // for gv
    sh_mem_size_bytes += density * sizeof(int16); // for gi

    if(sh_mem_size_bytes > 48 * 1024) {
        //// if we want to allocate more than 48KB, then we have to call this method
        cudaFuncSetAttribute(compute_row_sparse_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_mem_size_bytes);
    }

    float* pV = (float*) V.data_ptr();
    int16* pgi = (int16*) gi.data_ptr();
    float* pgv = (float*) gv.data_ptr();
    float* pq = (float*) q.data_ptr();
    float* pout = (float*) out.data_ptr();

    compute_row_sparse_kernel<<<blocks, threads, sh_mem_size_bytes>>>(pV, pgi, pgv, pq, pout, row_start, row_end, m, damp, density, N, B, do_init);

	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
 	// GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
