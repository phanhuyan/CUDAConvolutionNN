// convolution_im2col_gemm.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
using namespace nvcuda::wmma;

// Define constants
#define TILE_WIDTH 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Forward declarations of kernels
__global__ void im2col_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int B, const int C,
    const int H, const int W,
    const int K, const int S,
    const int outH, const int outW);

__global__ void wmma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K);

/**
 * @brief CUDA kernel for im2col operation.
 * 
 * @param input     Pointer to the input tensor of shape (B, C, H, W).
 * @param output    Pointer to the output matrix of shape (B * outH * outW, C * K * K).
 * @param B         Batch size.
 * @param C         Number of input channels.
 * @param H         Input height.
 * @param W         Input width.
 * @param K         Kernel size (assuming square kernels).
 * @param S         Stride.
 * @param outH      Output height.
 * @param outW      Output width.
 */
__global__ void im2col_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int B, const int C,
    const int H, const int W,
    const int K, const int S,
    const int outH, const int outW)
{
    // Calculate the global thread indices
    int b = blockIdx.z; // Batch index
    int oc = blockIdx.y * blockDim.y + threadIdx.y; // Output column index (C * K * K)
    int oh = blockIdx.x * blockDim.x + threadIdx.x; // Output height index

    // Each thread processes multiple output widths (ow) if necessary
    for (int ow = 0; ow < outW; ow += blockDim.x) {
        if (oh < outH && (ow + threadIdx.x) < outW && oc < C * K * K) {
            // Calculate the position in the output matrix
            int out_row = b * outH * outW + oh * outW + ow + threadIdx.x;
            int out_col = oc;

            // Determine the corresponding input channel, kernel row, and kernel column
            int c = oc / (K * K);
            int k = (oc % (K * K)) / K;
            int q = (oc % (K * K)) % K;

            // Calculate the input spatial location
            int in_h = oh * S + k;
            int in_w = (ow + threadIdx.x) * S + q;

            // Boundary check (assuming no padding; set to 0 if out of bounds)
            float val = 0.0f;
            if (in_h < H && in_w < W) {
                val = input[((b * C + c) * H + in_h) * W + in_w];
            }

            // Write to the output matrix
            output[out_row * (C * K * K) + out_col] = val;
        }
    }
}

/**
 * @brief CUDA kernel for Matrix Multiplication using Tensor Cores (WMMA).
 * 
 * C = A * B
 * A: (M x K) matrix in row-major
 * B: (K x N) matrix in column-major
 * C: (M x N) matrix in row-major
 * 
 * @param A         Pointer to matrix A in half-precision.
 * @param B         Pointer to matrix B in half-precision.
 * @param C         Pointer to matrix C in float.
 * @param M         Number of rows in A and C.
 * @param N         Number of columns in B and C.
 * @param K         Number of columns in A and rows in B.
 */
__global__ void wmma_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Define WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Calculate global row and column indices
    int row = blockIdx.y * WMMA_M + threadIdx.y;
    int col = blockIdx.x * WMMA_N + threadIdx.x;

    // Initialize the output fragment to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over tiles of K dimension
    for (int tile = 0; tile < (K + WMMA_K - 1) / WMMA_K; tile++) {
        // Load the A and B fragments
        int a_row = row;
        int a_col = tile * WMMA_K;
        int b_row = tile * WMMA_K;
        int b_col = col;

        if (a_row < M && (a_col + WMMA_K) <= K && (b_row + WMMA_K) <= K && b_col < N) {
            wmma::load_matrix_sync(a_frag, A + a_row * K + a_col, K);
            wmma::load_matrix_sync(b_frag, B + b_row + b_col * K, K);
            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store the accumulated result
    if (row < M && col < N) {
        // Convert accumulator to float and write to global memory
        float value = 0.0f;
        for (int i = 0; i < c_frag.num_elements; i++) {
            value += c_frag.x[i];
        }
        C[row * N + col] = value;
    }
}

/**
 * @brief Host function to perform im2col on GPU.
 */
void im2col_gpu(
    const float* d_input,
    float* d_output,
    const int B, const int C,
    const int H, const int W,
    const int K, const int S)
{
    // Calculate output dimensions
    int outH = (H - K) / S + 1;
    int outW = (W - K) / S + 1;

    // Define block and grid dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // (oh, oc)
    dim3 gridDim((outH + TILE_WIDTH - 1) / TILE_WIDTH,
                 (C * K * K + TILE_WIDTH - 1) / TILE_WIDTH,
                 B);

    // Launch the kernel
    im2col_kernel<<<gridDim, blockDim>>>(
        d_input,
        d_output,
        B, C,
        H, W,
        K, S,
        outH, outW);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error launching im2col_kernel: %s\n", cudaGetErrorString(err));
    }
}

/**
 * @brief Host function to perform convolution using im2col and GEMM with Tensor Cores.
 */
void convolution_im2col_gemm(
    const float* h_input,
    const float* h_filter,
    float* h_output,
    const int B, const int M, const int C,
    const int H, const int W,
    const int K, const int S)
{
    // Calculate output dimensions
    int outH = (H - K) / S + 1;
    int outW = (W - K) / S + 1;

    // Allocate device memory
    float *d_input, *d_im2col;
    cudaMalloc(&d_input, B * C * H * W * sizeof(float));

    // im2col output dimensions: (B * outH * outW, C * K * K)
    int im2col_rows = B * outH * outW;
    int im2col_cols = C * K * K;
    cudaMalloc(&d_im2col, im2col_rows * im2col_cols * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);

    // Perform im2col
    im2col_gpu(d_input, d_im2col, B, C, H, W, K, S);

    // Prepare filter matrix for GEMM
    // Filters shape: (M, C, K, K) -> (M, C*K*K)
    float* h_filter_reshaped = (float*)malloc(M * C * K * K * sizeof(float));
    for (int m = 0; m < M; ++m) {
        for (int c = 0; c < C; ++c) {
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    h_filter_reshaped[m * (C * K * K) + c * (K * K) + p * K + q] = h_filter[((m * C + c) * K + p) * K + q];
                }
            }
        }
    }

    // Allocate device memory for filters
    half* d_filter_half;
    cudaMalloc(&d_filter_half, M * C * K * K * sizeof(half));

    // Convert filters to half-precision
    half* h_filter_half = (half*)malloc(M * C * K * K * sizeof(half));
    for (int i = 0; i < M * C * K * K; ++i) {
        h_filter_half[i] = __float2half(h_filter_reshaped[i]);
    }
    cudaMemcpy(d_filter_half, h_filter_half, M * C * K * K * sizeof(half), cudaMemcpyHostToDevice);

    // Allocate device memory for im2col in half-precision
    half* d_im2col_half;
    cudaMalloc(&d_im2col_half, im2col_rows * im2col_cols * sizeof(half));

    // Convert im2col to half-precision
    // This can be done via a separate kernel or on the host if data is small
    // For simplicity, we'll perform it on the host here

    float* h_im2col = (float*)malloc(im2col_rows * im2col_cols * sizeof(float));
    cudaMemcpy(h_im2col, d_im2col, im2col_rows * im2col_cols * sizeof(float), cudaMemcpyDeviceToHost);

    half* h_im2col_half = (half*)malloc(im2col_rows * im2col_cols * sizeof(half));
    for (int i = 0; i < im2col_rows * im2col_cols; ++i) {
        h_im2col_half[i] = __float2half(h_im2col[i]);
    }
    cudaMemcpy(d_im2col_half, h_im2col_half, im2col_rows * im2col_cols * sizeof(half), cudaMemcpyHostToDevice);

    // Allocate device memory for GEMM output
    // GEMM: (M x (C*K*K)) * ((C*K*K) x (B*outH*outW)) = (M x (B*outH*outW))
    // We'll store it as (M x (B*outH*outW)) in row-major
    float* d_gemm_output;
    cudaMalloc(&d_gemm_output, M * im2col_rows * sizeof(float));

    // Define block and grid dimensions for GEMM
    dim3 blockDimGEMM(WMMA_N, WMMA_M, 1); // (threadIdx.x, threadIdx.y)
    dim3 gridDimGEMM((im2col_rows + WMMA_N - 1) / WMMA_N,
                    (M + WMMA_M - 1) / WMMA_M,
                    1);

    // Launch GEMM kernel
    wmma_gemm_kernel<<<gridDimGEMM, blockDimGEMM>>>(
        d_filter_half, // A: (M x K) in row-major
        d_im2col_half, // B: (K x N) in column-major
        d_gemm_output, // C: (M x N) in row-major
        M, im2col_rows, im2col_cols // M, N, K
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error launching wmma_gemm_kernel: %s\n", cudaGetErrorString(err));
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy GEMM result back to host
    float* h_gemm_output = (float*)malloc(M * im2col_rows * sizeof(float));
    cudaMemcpy(h_gemm_output, d_gemm_output, M * im2col_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Reshape GEMM output to (B, M, outH, outW)
    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int oh = 0; oh < outH; ++oh) {
                for (int ow = 0; ow < outW; ++ow) {
                    int im2col_idx = b * outH * outW + oh * outW + ow;
                    float value = h_gemm_output[m * im2col_rows + im2col_idx];
                    h_output[((b * M + m) * outH + oh) * outW + ow] = value;
                }
            }
        }
    }

    // Clean up device memory
    cudaFree(d_input);
    cudaFree(d_im2col);
    cudaFree(d_filter_half);
    cudaFree(d_im2col_half);
    cudaFree(d_gemm_output);

    // Free host memory
    free(h_filter_reshaped);
    free(h_filter_half);
    free(h_im2col);
    free(h_im2col_half);
    free(h_gemm_output);
}

int main()
{
    // Seed for random number generation
    srand(time(NULL));

    // Define convolution parameters
    const int B = 1;  // Batch size
    const int C = 3;  // Input channels
    const int H = 5;  // Input height
    const int W = 5;  // Input width
    const int M = 2;  // Number of output feature maps
    const int K = 3;  // Kernel size
    const int S = 1;  // Stride

    // Calculate output dimensions
    int outH = (H - K) / S + 1;
    int outW = (W - K) / S + 1;

    // Allocate and initialize host input
    size_t input_size = B * C * H * W * sizeof(float);
    float* h_input = (float*)malloc(input_size);
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    h_input[((b * C + c) * H + h) * W + w] = static_cast<float>(rand() % 10);
                }
            }
        }
    }

    // Allocate and initialize host filters
    size_t filter_size = M * C * K * K * sizeof(float);
    float* h_filter = (float*)malloc(filter_size);
    for (int m = 0; m < M; ++m) {
        for (int c = 0; c < C; ++c) {
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    h_filter[((m * C + c) * K + p) * K + q] = static_cast<float>(rand() % 3);
                }
            }
        }
    }

    // Allocate host memory for output
    size_t output_size = B * M * outH * outW * sizeof(float);
    float* h_output = (float*)malloc(output_size);
    // Initialize output to zero
    for (int i = 0; i < B * M * outH * outW; ++i) {
        h_output[i] = 0.0f;
    }

    // Perform convolution using im2col and GEMM
    convolution_im2col_gemm(h_input, h_filter, h_output, B, M, C, H, W, K, S);

    // (Optional) Print the input, filters, and output for verification
    printf("Input Tensor (B=1, C=3, H=5, W=5):\n");
    for (int c = 0; c < C; ++c) {
        printf("Channel %d:\n", c);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                printf("%.1f ", h_input[((0 * C + c) * H + h) * W + w]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("Filters (M=2, C=3, K=3, K=3):\n");
    for (int m = 0; m < M; ++m) {
        printf("Filter %d:\n", m);
        for (int c = 0; c < C; ++c) {
            printf("  Channel %d:\n", c);
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    printf("%.1f ", h_filter[((m * C + c) * K + p) * K + q]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    printf("Convolution Output (B=1, M=2, outH=%d, outW=%d):\n", outH, outW);
    for (int m = 0; m < M; ++m) {
        printf("Output Feature Map %d:\n", m);
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                printf("%.1f ", h_output[((0 * M + m) * outH + oh) * outW + ow]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Clean up host memory
    free(h_input);
    free(h_filter);
    free(h_output);

    // Reset the device and exit
    cudaDeviceReset();
    return 0;
}
