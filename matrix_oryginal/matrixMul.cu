/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#define DEFAULT_BLOCK_SIZE 32
#define INIT_TYPE 0
#define NUM_ITER 10
#define MAT_DIM 3200

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
    float *B, int wA,
    int wB) {
  // Indeks bloku
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Indeks wątku w obrębie bloku
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Indeks pierwszej submacierzy macierzy A przetwarzanej przez blok
  int aBegin = wA * BLOCK_SIZE * by;

  // Indeks ostatniej submacierzy macierzy A przetwarzanej przez blok
  int aEnd   = aBegin + wA - 1;

  // Wielkość kroku do iteracji przez submacierze A
  int aStep  = BLOCK_SIZE;

  // Indeks pierwszej submacierzy macierzy B przetwarzanej przez blok
  int bBegin = BLOCK_SIZE * bx;

  // Wielkość kroku do iteracji przez submacierze B
  int bStep  = BLOCK_SIZE * wB;

  // Zmienna przechowująca obliczany element submacierzy bloku
  float Csub = 0;

  // Pętla przechodząca przez wszystkie submacierze A i B
  // wymagane do obliczenia submacierzy bloku
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {
    // Deklaracja tablicy w pamięci współdzielonej dla submacierzy A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Deklaracja tablicy w pamięci współdzielonej dla submacierzy B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Załadowanie submacierzy A i B z pamięci globalnej do pamięci współdzielonej;
    // każdy wątek ładuje jeden element z każdej macierzy
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronizacja wątków, aby upewnić się, że wszystkie dane zostały załadowane
    __syncthreads();

    // Mnożenie submacierzy; każdy wątek oblicza jeden element
    // submacierzy wynikowej
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronizacja wątków, aby upewnić się, że poprzednie
    // obliczenia zostały zakończone przed załadowaniem nowych submacierzy
    __syncthreads();
  }

  // Zapis submacierzy wyniku do pamięci globalnej;
  // każdy wątek zapisuje jeden element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

void RandomInit(float* data, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void PatternInit(float* data, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            data[i * cols + j] = static_cast<float>(i * cols + j) * 0.1f; // Example: unique value based on (row, col)
        }
    }
}

// Check the result of the matrix multiplication with the CPU
// Test relative error by the formula
//     |<x, y>_cpu - <x,y>_gpu|  /  <|x|, |y|>  <  eps
bool CheckConstantInitMatrixMulResult(const float* h_C, const dim3& dims_C, const dim3& dims_A, float val_B, double eps = 1.e-4) {
    bool correct = true;
    for (uint32_t i = 0; i < static_cast<uint32_t>(dims_C.x * dims_C.y); i++) {
        double abs_err = fabs(h_C[i] - (dims_A.x * val_B));
        double dot_length = dims_A.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                i, h_C[i], dims_A.x * val_B, eps);
            correct = false;
        }
    }
    return correct;
}

// Check the result of the matrix multiplication with the CPU
// Test relative error by the formula
//     |<x, y>_cpu - <x,y>_gpu|  /  <|x|, |y|>  <  eps
bool CheckRandomInitMatrixMulResult(const float* h_C, const float* h_A, const float* h_B,
    const dim3& dims_A, const dim3& dims_B, const dim3& dims_C, double eps = 1.e-4) {
    bool correct = true;
    for (uint32_t row = 0; row < dims_C.y; ++row) {
        for (uint32_t col = 0; col < dims_C.x; ++col) {
            double ref_value = 0.0;
            for (uint32_t k = 0; k < dims_A.x; ++k) {
                ref_value += h_A[row * dims_A.x + k] * h_B[k * dims_B.x + col];
            }
            double abs_err = fabs(h_C[row * dims_C.x + col] - ref_value);
            double abs_val = fabs(ref_value);
            double rel_err = abs_err / (abs_val + 1e-8);

            if (rel_err > eps) {
                printf("Error! Matrix[%05d][%05d]=%.8f, ref=%.8f error term is > %E\n",
                    row, col, h_C[row * dims_C.x + col], ref_value, eps);
                correct = false;
            }
        }
    }
    return correct;
}

// Check the result of the matrix multiplication with the CPU
// Test relative error by the formula
//     |<x, y>_cpu - <x,y>_gpu|  /  <|x|, |y|>  <  eps
bool CheckPatternInitMatrixMulResult(const float* h_C, const dim3& dims_C, const float* h_A, const float* h_B,
    const dim3& dims_A, const dim3& dims_B, double eps = 1.e-4) {
    bool correct = true;

    // Iterate through all elements in the resulting matrix C
    for (uint32_t i = 0; i < dims_C.y; i++) {
        for (uint32_t j = 0; j < dims_C.x; j++) {
            // Compute reference value for C[i][j]
            double ref = 0.0;
            for (uint32_t k = 0; k < dims_A.x; k++) { // Shared dimension
                ref += static_cast<double>(h_A[i * dims_A.x + k] * h_B[k * dims_B.x + j]);
            }

            // Compare result with reference
            double abs_err = fabs(h_C[i * dims_C.x + j] - ref);
            double dot_length = dims_A.x;
            double abs_val = fabs(h_C[i * dims_C.x + j]);
            double rel_err = abs_err / (abs_val + eps) / dot_length; // Adding eps to avoid division by zero

            if (rel_err > eps) {
                printf("Error! Matrix[%05d,%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, j, h_C[i * dims_C.x + j], ref, eps);
                correct = false;
            }
        }
    }

    return correct;
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char** argv, int block_size, int init_type, const dim3& dimsA, const dim3& dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
    cudaStream_t stream;

    // Initialization of host memory (on CPU)
    float val_A = 1.0f;
    float val_B = 0.01f;
    if (init_type == 0) {
        ConstantInit(h_A, size_A, val_A);
        ConstantInit(h_B, size_B, val_B);

    }
    else if (init_type == 1) {
        srand(static_cast<unsigned int>(time(0)));
        RandomInit(h_A, size_A);
        RandomInit(h_B, size_B);
    }
    else if (init_type == 2) {
        PatternInit(h_A, dimsA.y, dimsA.x);
        PatternInit(h_B, dimsB.y, dimsB.x);
    }
    else {
        fprintf(stderr, "Invalid init_type parameter!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate device memory (on GPU)
    float* d_A, * d_B, * d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float* h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulCUDA<16> <<<grid, threads, 0, stream>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else {
        MatrixMulCUDA<32> <<<grid, threads, 0, stream>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    int nIter = NUM_ITER;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16> <<<grid, threads, 0, stream>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else {
            MatrixMulCUDA<32> <<<grid, threads, 0, stream>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
        static_cast<double>(dimsA.y) *
        static_cast<double>(dimsB.x);
    double gigaFlops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
        " WorkgroupSize= %u threads/block\n",
        gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(
        cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Checking computed result for correctness: ");

    double eps = 1.e-4;  // tolerance threshold
    bool correct = true;
    if (init_type == 0) {
        correct = CheckConstantInitMatrixMulResult(h_C, dimsC, dimsA, val_B, eps);
    }
    else if (init_type == 1) {
        correct = CheckRandomInitMatrixMulResult(h_C, h_A, h_B, dimsA, dimsB, dimsC, eps);
    }
    else if (init_type == 2) {
        correct = CheckPatternInitMatrixMulResult(h_C, dimsC, h_A, h_B, dimsA, dimsB, eps);
    }
    else {
        fprintf(stderr, "Invalid init_type parameter!\n");
        exit(EXIT_FAILURE);
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}

/**
 * Program main
 */
int main(int argc, char** argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char**)argv, "help") || checkCmdLineFlag(argc, (const char**)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -wB=HeightB\n");
        printf("      -init=init_type (0 - constant, 1 - random (0..1), 2 - pattern\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");
        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char**)argv);
    int init_type = INIT_TYPE;
    int block_size = DEFAULT_BLOCK_SIZE;
    int matrix_size = MAT_DIM;
    dim3 dimsA(matrix_size, matrix_size, 1);
    dim3 dimsB(matrix_size, matrix_size, 1);

    // width and height of Matrix A
    if (checkCmdLineFlag(argc, (const char**)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char**)argv, "wA");
        dimsA.y = getCmdLineArgumentInt(argc, (const char**)argv, "wA");
    }


    // width and height of Matrix B
    if (checkCmdLineFlag(argc, (const char**)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char**)argv, "wB");
        dimsB.y = getCmdLineArgumentInt(argc, (const char**)argv, "wB");
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "init")) {
        init_type = getCmdLineArgumentInt(argc, (const char**)argv, "init");
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "bS")) {
        block_size = getCmdLineArgumentInt(argc, (const char**)argv, "bS");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    checkCudaErrors(cudaProfilerStart());
    int matrix_result = MatrixMultiply(argc, argv, block_size, init_type, dimsA, dimsB);
    checkCudaErrors(cudaProfilerStop());

    exit(matrix_result);
}
