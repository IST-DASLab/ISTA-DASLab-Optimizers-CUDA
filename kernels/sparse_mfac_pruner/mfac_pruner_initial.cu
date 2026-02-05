#include "utils.h"
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

__device__ inline void kahan_parallel_reduce(float* mem, int N, int THREADS, int Tid) {
   // initially, we sum everything in interval [THREADS, N-1] to [0, THREADS-1]
   // the already existing values in mem[Tid] serve as initial values, so sum=mem[Tid]
   double sum = static_cast<double>(mem[Tid]);
   double c = static_cast<double>(0);
   double y, t;

   // the following for-loop implements mem[Tid] += mem[i] using Kahan summation
   // for the values at indices i > THREADS
   for(int i = Tid + THREADS; i < N; i += THREADS) {
       y = static_cast<double>(mem[i]) - c;
       t = sum + y;
       c = (t - sum) - y;
       sum = t;
   }
   mem[Tid] = static_cast<float>(sum);
   __syncthreads();

   // the following for-loop implements mem[Tid] += mem[Tid + stride] using
   // Kahan summation and parallel reduce in logarithmic time
   for(int stride = (THREADS >> 1); stride > 0; stride >>= 1) {
       if(Tid < stride && Tid + stride < N) {
           y = static_cast<double>(mem[Tid + stride]) - c; // mem[Tid+stride] is the value to be summed up
           t = sum + y; // mem[Tid] stores the sum
           c = (t - sum) - y;
           sum = t; // update sum
           mem[Tid] = static_cast<float>(sum);
       }
       __syncthreads();
   }
}

__global__ void compute_row_initial_kernel(float *global_V, float *global_g, float *global_q, float *global_out, int row_start, int row_end, int m, float damp, int N, int B, int nbits, int use_kahan, int grad_const, int do_init, int do_debug)
{
    const long Bid = blockIdx.x; // block id
	const long THREADS = blockDim.x; // number of threads
	const long Tid = threadIdx.x; // thread id

    if(do_debug) {
        printf("[Bid=%ld][Tid=%ld][rows=%d-%d] THREADS=%ld, damp=%f\n", Bid, Tid, row_start, row_end, THREADS, damp);
    }

    extern __shared__ unsigned char shmem[];
    float *mem = reinterpret_cast<float*>(shmem);
    float *V = mem; // size B, stores one row of V, e.g. V[i, Bid, :]
    float *g = mem + B; // size B, stores one row of g, e.g. g[Bid, :]
    float *prods = mem + 2 * B; // size B, stores products V*g before summing up.
    float *Vout = mem + 3 * B; // size B, accumulates dot * V
    double *comps = 0;
    if(use_kahan) { //
        comps = reinterpret_cast<double*>(shmem); // Kahan compensations for each component in Vout
        if(nbits == 32) {
            comps += 2 * B;
        } else if(nbits == 64) {
            comps += 4 * B;
        }
    }

    // predefined constants to avoid computing the same quantities multiple times
    long N_B = N * B;
    long Bid_B = Bid * B;

    long g_start = Bid_B;
    long V_start;

    int i, j, j_global;
    float dot, q, delta;

    // g = global_g[Bid, :]
    copy_global_to_shmem(global_g, g, g_start, g_start + B, THREADS, Tid);
    __syncthreads();

    if(do_debug) {
        for(i = Tid; i < B; i += THREADS) {
            printf("[Bid=%ld][Tid=%ld][rows=%d-%d][step-1] g[%d]=%lf\n", Bid, Tid, row_start, row_end, i, g[i]);
        }
    }

    if(row_end < m) { // we call the kernel to compute rows of V
//         V_start = 0 * N_B + Bid_B;
//         copy_global_to_shmem(global_V, Vout, V_start, V_start + B, THREADS, Tid);
        for(i = Tid; i < B; i += THREADS) {
            if(do_init) {
                if(should_skip(g[i], grad_const)) {
                    Vout[i] = static_cast<float>(0);
                } else {
                    Vout[i] = static_cast<float>(damp) * static_cast<float>(g[i]);
                }
            } else {
                Vout[i] = static_cast<float>(0);
            }
        }

    } else if(row_end == m) { // we call the kernel to compute the final update that prunes the model
        for(i = Tid; i < B; i += THREADS) {
            Vout[i] = static_cast<float>(0);
        }
    }
    __syncthreads();

    double y, t; // for Kahan

    // initialize compensations to zero
    if(use_kahan) {
        for(i = Tid; i < B; i += THREADS) {
            comps[i] = static_cast<float>(0);
        }
    }

    for(j = row_start; j < row_end; ++j) {
        // V = global_V[j, Bid, :]
        V_start = j * N_B + Bid_B;
        copy_global_to_shmem(global_V, V, V_start, V_start + B, THREADS, Tid);
        __syncthreads();

        if(do_debug) {
            for(i = Tid; i < B; i += THREADS) {
                printf("[Bid=%ld][Tid=%ld][rows=%d-%d][step-2] v[%d, %ld, %d]=%lf\n", Bid, Tid, row_start, row_end, j, Bid, i, V[i]);
            }
        }

        // (1) compute dot products
        for(i = Tid; i < B; i += THREADS) {
            if(should_skip(g[i], grad_const)) {
                prods[i] = static_cast<float>(0);
            } else {
                prods[i] = V[i] * g[i];
            }
        }

        if(do_debug) {
            for(i = Tid; i < B; i += THREADS) {
                printf("[Bid=%ld][Tid=%ld][rows=%d-%d][step-3] prods[%d]=%lf (pre-reduce)\n", Bid, Tid, row_start, row_end, i, prods[i]);
            }
        }
        __syncthreads();
        if(use_kahan) {
            kahan_parallel_reduce(prods, B, THREADS, Tid);
        } else {
            generic_parallel_reduce(prods, B, THREADS, Tid);
        }
        dot = prods[0];

        if(do_debug) {
            for(i = Tid; i < B; i += THREADS) {
                printf("[Bid=%ld][Tid=%ld][rows=%d-%d][step-4] prods[%d]=%lf (post-reduce)\n", Bid, Tid, row_start, row_end, i, prods[i]);
            }
            printf("[Bid=%ld][Tid=%ld][rows=%d-%d][step-5] dot=%lf\n", Bid, Tid, row_start, row_end, dot);
        }

        // read q from global memory: q = global_q[j, Bid]
        if(Tid == 0) {
            prods[0] = static_cast<float>(global_q[j * N + Bid]);
        }
        __syncthreads();
        q = prods[0];
        delta = dot / q;

        if(do_debug) {
            printf("[Bid=%ld][Tid=%ld][rows=%d-%d][step-6] q=%lf, delta=%lf\n", Bid, Tid, row_start, row_end, q, delta);
        }

        if(use_kahan) {
            for(i = Tid; i < B; i += THREADS) {
                y = static_cast<double>(-delta) * static_cast<double>(V[i]) - comps[i];
                t = static_cast<double>(Vout[i]) + y;
                comps[i] = (t - static_cast<double>(Vout[i])) - y;
                Vout[i] = t;
            }
        } else {
            for(i = Tid; i < B; i += THREADS) {
                Vout[i] -= delta * V[i];
                if(do_debug) {
                    printf("[Bid=%ld][Tid=%ld][rows=%d-%d][step-7] delta*V[%d, %ld, %d]=%lf\n", Bid, Tid, row_start, row_end, j, Bid, i, delta * V[i]);
                }
            }
        }
    } // end for j < row

    for(j_global = Bid_B + Tid, j = Tid;
        j_global < Bid_B + B;
        j_global += THREADS, j += THREADS)
    {
        global_out[j_global] = static_cast<float>(Vout[j]);
    }
    if(do_debug) {
        for(i = Tid; i < B; i += THREADS) {
            printf("[Bid=%ld][Tid=%ld][rows=%d-%d][step-8] vout[%d]=%lf[OUT]\n", Bid, Tid, row_start, row_end, i, Vout[i]);
        }
    }

    // TODO: compute q here, based on Vout: q[row, Bid] = m + dot_product(Vout, g)
}

void
compute_row_initial_cuda (TT V, TT g, TT q, TT out, int row_start, int row_end, int m, float damp, int N, int B, int nblocks, int nthreads, int nbits, int use_kahan, int grad_const, int do_init, int do_debug) {
    assert(nbits == 32);
    dim3 blocks(nblocks, 1, 1);
    dim3 threads(nthreads, 1, 1);
    long sh_mem_size_bytes = 4 * B * ((nbits == 32) ? sizeof(float) : sizeof(double));
    if(use_kahan){
        sh_mem_size_bytes += B * sizeof(double); // add shared memory space for the Kahan compensations
    }

//     printf("row=%d, N=%d, B=%d, blocks=%d, threads=%d, sh_mem_size_bytes=%ld\n", row, N, B, nblocks, nthreads, sh_mem_size_bytes);

    if(sh_mem_size_bytes > 48 * 1024) {
        //// if we want to allocate more than 48KB, then we have to call this method
        cudaFuncSetAttribute(compute_row_initial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sh_mem_size_bytes);
    }

    float* fpV = (float*) V.data_ptr();
    float* fpg = (float*) g.data_ptr();
    float* fpq = (float*) q.data_ptr();
    float* fpout = (float*) out.data_ptr();

    compute_row_initial_kernel<<<blocks, threads, sh_mem_size_bytes>>>(fpV, fpg, fpq, fpout, row_start, row_end, m, damp, N, B, nbits, use_kahan, grad_const, do_init, do_debug);


	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
// 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
