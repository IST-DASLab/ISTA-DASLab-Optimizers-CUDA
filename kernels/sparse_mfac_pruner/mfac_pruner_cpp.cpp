#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "utils.h"
//#include "parallel_reduce.h"

__global__ void compute_row_initial_kernel (float *global_V, float *global_g, float *global_q, float *global_out, int row_start, int row_end, int m, float damp, int N, int B, int nbits, int use_kahan, int grad_const, int do_init, int do_debug);
void compute_row_initial_cuda (TT V, TT g, TT q, TT out, int row_start, int row_end, int m, float damp, int N, int B, int nblocks, int nthreads, int nbits, int use_kahan, int grad_const, int do_init, int do_debug);
void compute_row_initial      (TT V, TT g, TT q, TT out, int row_start, int row_end, int m, float damp, int N, int B, int nblocks, int nthreads, int nbits, int use_kahan, int grad_const, int do_init, int do_debug)
{
    assert((nbits == 32) || (nbits == 64));
    assert((use_kahan == 0) || (use_kahan == 1));
    assert((grad_const == 0) || (grad_const == 512));
    assert((0 <= row_start) && (row_start < row_end) && (row_end <= m));
	CHECK_INPUT(V);
	CHECK_INPUT(g);
	CHECK_INPUT(q);
	CHECK_INPUT(out);

	const at::cuda::OptionalCUDAGuard device_guard(device_of(V));
    compute_row_initial_cuda(V, g, q, out, row_start, row_end, m, damp, N, B, nblocks, nthreads, nbits, use_kahan, grad_const, do_init, do_debug);
}

__global__ void compute_row_dense_kernel(float *global_V, float *global_g, float *global_q, float *global_out, int row_start, int row_end, int m, float damp, int N, int B, int grad_const);
void compute_row_dense_cuda (TT V, TT g, TT q, TT out, int row_start, int row_end, int m, float damp, int N, int B, int nblocks, int nthreads, int grad_const);
void compute_row_dense      (TT V, TT g, TT q, TT out, int row_start, int row_end, int m, float damp, int N, int B, int nblocks, int nthreads, int grad_const)
{
    //assert((grad_const == 0) || (grad_const == 512));
    assert((0 <= row_start) && (row_start < row_end) && (row_end <= m));
	CHECK_INPUT(V);
	CHECK_INPUT(g);
	CHECK_INPUT(q);
	CHECK_INPUT(out);

	const at::cuda::OptionalCUDAGuard device_guard(device_of(V));
    compute_row_dense_cuda(V, g, q, out, row_start, row_end, m, damp, N, B, nblocks, nthreads, grad_const);
}

__global__ void compute_row_sparse_kernel(float *global_V, int16 *global_gi, float *global_gv, float *global_q, float *global_out, int row_start, int row_end, int m, float damp, int density, int N, int B, int do_init);
void compute_row_sparse_cuda (TT V, TT gi, TT gv, TT q, TT out, int row_start, int row_end, int m, float damp, int density, int N, int B, int nblocks, int nthreads, int do_init);
void compute_row_sparse      (TT V, TT gi, TT gv, TT q, TT out, int row_start, int row_end, int m, float damp, int density, int N, int B, int nblocks, int nthreads, int do_init)
{
    assert((0 <= row_start) && (row_start < row_end) && (row_end <= m));
	CHECK_INPUT(V);
	CHECK_INPUT(gi);
	CHECK_INPUT(gv);
	CHECK_INPUT(q);
	CHECK_INPUT(out);

	const at::cuda::OptionalCUDAGuard device_guard(device_of(V));
    compute_row_sparse_cuda(V, gi, gv, q, out, row_start, row_end, m, damp, density, N, B, nblocks, nthreads, do_init);
}
/*
void pipeline_copy_compute(TT Vcpu, TT Vgpu0, TT Vgpu1,
                           TT Qcpu, TT Qgpu0, TT Qgpu1,
                           TT Vtmp, TT grad, TT gi, TT gv,
                           int start_copy_cpu, int end_copy_cpu,
                           int start_copy_gpu, int end_copy_gpu,
                           int start_comp_gpu, int end_comp_gpu,
                           int half_copy, int half_compute,
                           int m, int N, int B, float damp,
                           int grad_const, int kernel_call_count, int topk_type,
                           int nblocks, int nthreads)
{
    float *pVcpu = (float*) Vcpu.data_ptr();
    float *pVgpu0 = (float*) Vgpu0.data_ptr();
    float *pVgpu1 = (float*) Vgpu1.data_ptr();
    float *pQcpu = (float*) Qcpu.data_ptr();
    float *pQgpu0 = (float*) Qgpu0.data_ptr();
    float *pQgpu1 = (float*) Qgpu1.data_ptr();
    float *pVtmp = (float*) Vtmp.data_ptr();
    float *pgrad = (float*) grad.data_ptr();
    int16 *pgi = (int16*) gi.data_ptr();
    float *pgv = (float*) gv.data_ptr();

    dim3 blocks(nblocks, 1, 1);
    dim3 threads(nthreads, 1, 1);

    cudaStream_t stream_copy_V, stream_copy_Q, stream_compute;
    cudaStreamCreate(&stream_copy_V);
    cudaStreamCreate(&stream_copy_Q);
    cudaStreamCreate(&stream_compute);

    /// START SECTION COPY
    int NB = N * B;
    int rows_copy = end_copy_cpu - start_copy_cpu;
    int sizeVcpu = rows_copy * N * B;
    int sizeQcpu = rows_copy * N;
    int offsetVcpu = start_copy_cpu * NB;
    int offsetQcpu = start_copy_cpu * N;

    float *copyVgpu = (half_copy == 0) ? pVgpu0 : pVgpu1;
    float *copyQgpu = (half_copy == 0) ? pQgpu0 : pQgpu1;
    /// END SECTION COPY

    /// START SECTION COMPUTE
    int density = gi.sizes()[1];
    long shmem_initial = 4 * B * sizeof(float);
    long shmem_sparse = (2 * B + 2 * density) * sizeof(float) + density * sizeof(int16);

    float *compVgpu = (half_compute == 0) ? pVgpu0 : pVgpu1;
    float *compQgpu = (half_compute == 0) ? pQgpu0 : pQgpu1;

    // kernel_call_count=-1 means kernel_call_count=None
    int do_init = (kernel_call_count == -1) ? 0 : static_cast<int>(kernel_call_count == 1);
    /// END SECTION COMPUTE

    /// COPY V
    cudaMemcpyAsync(
        copyVgpu,               // device pointer
        pVcpu + offsetVcpu,     // host pointer
        sizeVcpu,               // size
        cudaMemcpyHostToDevice, // direction
        stream_copy_V           // stream
    );

    /// COPY Q
    cudaMemcpyAsync(
        copyQgpu,               // device pointer
        pQcpu + offsetQcpu,     // host pointer
        sizeQcpu,               // size
        cudaMemcpyHostToDevice, // direction
        stream_copy_Q           // stream
    );

    /// COMPUTE
    if(topk_type == 0) { // global topk
        if(shmem_initial > 48 * 1024) {
            cudaFuncSetAttribute(compute_row_initial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_initial);
        }
        compute_row_initial_kernel<<<blocks, threads, shmem_initial, stream_compute>>>(compVgpu, pgrad, compQgpu, pVtmp, start_comp_gpu, end_comp_gpu, m, damp, N, B, 32, 0, grad_const, do_init, 0);
    } else { // row topk
        if(shmem_sparse > 48 * 1024){
            cudaFuncSetAttribute(shmem_sparse, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sparse);
        }
        compute_row_sparse_kernel<<<blocks, threads, shmem_sparse, stream_compute>>>(compVgpu, pgi, pgv, compQgpu, pVtmp, start_comp_gpu, end_comp_gpu, m, damp, density, N, B, do_init);
    }

	GPU_ERROR_CHECK(cudaGetLastError());
	GPU_ERROR_CHECK(cudaPeekAtLastError());
 	GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("compute_row_initial", &compute_row_initial, "Computes one row of matrix V used for pruning");
	m.def("compute_row_dense", &compute_row_dense, "Computes one row of matrix V used for pruning using dense gradients");
	m.def("compute_row_sparse", &compute_row_sparse, "Computes one row of matrix V used for pruning using sparse gradients");
	//m.def("pipeline_copy_compute", &pipeline_copy_compute, "CPU-GPU transfer and GPU computation using streams");
}


