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
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define INIT_BLOCK_SIZE 32
#define MAT_DIM 3200

template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA_Modified(float* C, float* A, float* B, int width) {

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    __shared__ float shared_A[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[4][BLOCK_SIZE][BLOCK_SIZE];

    float Csub[8] = {0};

    int row_start_A[2] = {by * BLOCK_SIZE * width, by * BLOCK_SIZE * width + (width * width / 2)};
    int col_start_B[4] = {
        bx * BLOCK_SIZE,
        bx * BLOCK_SIZE + (width / 4),
        bx * BLOCK_SIZE + (width / 2),
        bx * BLOCK_SIZE + (3 * width / 4)
    };

    for (int k = 0; k < width; k += BLOCK_SIZE) {
        shared_A[0][ty][tx] = A[row_start_A[0] + k + ty * width + tx];
        shared_A[1][ty][tx] = A[row_start_A[1] + k + ty * width + tx];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            shared_B[i][ty][tx] = B[k * width + col_start_B[i] + ty * width + tx];
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            float val_A[2] = {shared_A[0][ty][i], shared_A[1][ty][i]};
            float val_B[4] = {
                shared_B[0][i][tx],
                shared_B[1][i][tx],
                shared_B[2][i][tx],
                shared_B[3][i][tx]
            };

            Csub[0] += val_A[0] * val_B[0];
            Csub[1] += val_A[0] * val_B[1];
            Csub[2] += val_A[1] * val_B[0];
            Csub[3] += val_A[1] * val_B[1];
            Csub[4] += val_A[0] * val_B[2];
            Csub[5] += val_A[0] * val_B[3];
            Csub[6] += val_A[1] * val_B[2];
            Csub[7] += val_A[1] * val_B[3];
        }

        __syncthreads();
    }

    int c = by * BLOCK_SIZE * width + bx * BLOCK_SIZE;
    C[c + ty * width + tx] = Csub[0];                                  
    C[c + (width / 4) + ty * width + tx] = Csub[1];                   
    C[c + (width / 2) * width + ty * width + tx] = Csub[2];            
    C[c + (width / 2) * width + (width / 4) + ty * width + tx] = Csub[3]; 
    C[c + (width / 2) + ty * width + tx] = Csub[4];                   
    C[c + (3 * width / 4) + ty * width + tx] = Csub[5];               
    C[c + (width / 2) * width + (width / 2) + ty * width + tx] = Csub[6]; 
    C[c + (width / 2) * width + (3 * width / 4) + ty * width + tx] = Csub[7];
}



void StaticInit(float* data, int size, float value)
{
	for (int i = 0; i < size; ++i) {
		data[i] = value;
	}
}

void LinearInit(float* data, int size, bool reverse)
{
	if (reverse) {
		for (int i = 0; i < size; ++i) {
			data[i] = static_cast<float>(size - i);
		}
	}
	else {
		for (int i = 0; i < size; ++i) {
			data[i] = static_cast<float>(i);
		}
	}
}

void RandomInit(float* data, int size) {
	for (int i = 0; i < size; ++i) {
		data[i] = static_cast<float>(rand()) / RAND_MAX; // Random numbers in [0, 1]
	}
}

int MatrixMultiply(int argc,
	char** argv,
	int block_size,
	const dim3& dimsA,
	const dim3& dimsB,
	const bool omit)
{
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

	// Initialize host memory
	const float valB = 0.01f;
	if (omit)
	{
		StaticInit(h_A, size_A, 1.0f);
		StaticInit(h_B, size_B, valB);
	}
	else 
	{
		LinearInit(h_A, size_A, false);
		LinearInit(h_B, size_B, true);
		//RandomInit(h_A, size_A);
		//RandomInit(h_B, size_B);
	}

	// Allocate device memory
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
	checkCudaErrors(
		cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(
		cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid_Modified(ceil((float)dimsB.x / (threads.x * 4)), ceil((float)dimsA.y / (threads.y * 2)));
	printf("Threads: %d\nGrid_Modified: %d\n", threads.x, grid_Modified.x);

	// Performs warmup operation using matrixMul CUDA kernel
	if (block_size == 16) {
		MatrixMulCUDA_Modified<16>
			<< <grid_Modified, threads, 0, stream >> > (d_C, d_A, d_B, dimsA.x);
	}
	else {
		MatrixMulCUDA_Modified<32>
			<< <grid_Modified, threads, 0, stream >> > (d_C, d_A, d_B, dimsA.x);
	}

	checkCudaErrors(cudaStreamSynchronize(stream));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, stream));

	// Execute the kernel
	int nIter = 10;

	for (int j = 0; j < nIter; j++) {
		if (block_size == 16) {
			MatrixMulCUDA_Modified<16>
				<< <grid_Modified, threads, 0, stream >> > (d_C, d_A, d_B, dimsA.x);
		}
		else {
			MatrixMulCUDA_Modified<32>
				<< <grid_Modified, threads, 0, stream >> > (d_C, d_A, d_B, dimsA.x);
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
	double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf(
		"Modified\nPerformance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
		" WorkgroupSize= %u threads/block\n",
		gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

	// Copy result from device to host
	checkCudaErrors(
		cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));

	printf("Checking computed result for correctness: \n");

	bool correct = true;

	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
	double eps = 1.e-4; // 1.e-6;  // machine zero
	int counter = 0;
	for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
		double abs_err;
		if (omit) {
			abs_err = fabs(h_C[i] - (dimsA.x * valB));
		}
		double dot_length = dimsA.x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / abs_val / dot_length;
		if (rel_err > eps) {
			//if (correct)
			//{
				if (omit) {
					printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x * valB, eps);
				}
				else {
					printf("Error! Matrix[%05d]=%.8f error term is > %E\n", i, h_C[i], eps);
				}
				correct = false;
				counter++;
			//}
			//else counter++;
		}
	}
	if (!correct)
	{
		printf("Error count: %d, Total count: %d, %f%%\n", counter, dimsA.x* dimsA.y, ((double)counter / (dimsA.x * dimsA.y)) * 100);
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
int main(int argc, char** argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	if (checkCmdLineFlag(argc, (const char**)argv, "help") || checkCmdLineFlag(argc, (const char**)argv, "?")) {
		printf("Usage -device=n (n >= 0 for deviceID)\n");
		printf("      -size=n (Width and Height of Matrix A and B)\n");
		printf("      -block=n ( n = 16 || 32)\n");
		printf("      -omit (Omit original)\n");
		exit(EXIT_SUCCESS);
	}

	// This will pick the best possible CUDA capable device, otherwise
	// override the device ID based on input provided at the command line
	int dev = findCudaDevice(argc, (const char**)argv);

	int block_size = INIT_BLOCK_SIZE;

	dim3 dimsA(MAT_DIM, MAT_DIM, 1);
	dim3 dimsB(MAT_DIM, MAT_DIM, 1);

	if (checkCmdLineFlag(argc, (const char**)argv, "size")) {
		dimsA.x = getCmdLineArgumentInt(argc, (const char**)argv, "size");
		dimsA.y = dimsA.x;
		dimsB.x = dimsA.x;
		dimsB.y = dimsA.x;
	}

	if (checkCmdLineFlag(argc, (const char**)argv, "block")) {
		block_size = getCmdLineArgumentInt(argc, (const char**)argv, "block");
	}

	bool omit = checkCmdLineFlag(argc, (const char**)argv, "omit");

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
		dimsB.y);

	checkCudaErrors(cudaProfilerStart());
	int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, omit);
	checkCudaErrors(cudaProfilerStop());

	exit(matrix_result);
}