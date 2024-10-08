/*
* Class: EE 5885 GPGPU
* Instructor: Dr. Suresh 
* Assignment: 3
* Problem: 4
* Author: Josh Blaney
* Date: 10/25/2022
*/

#include "Prob4.h"
#include "GPUErrors.h"

__global__ void MatrixVectorMult(float* g_Matrix, float* g_V, float* g_P, const int Size)
{
	// Calculate the global index of each thread 
	int id = threadIdx.x + (blockDim.x * blockIdx.x);
	float sum = 0.0f;

	// Ensure that the thread will access a valid memory location
	if (id < Size) {
		// Iterate over one row of the matrix and multiply it with the postmultiplied vector
		for (int j = 0; j < Size; j++) {
			sum += g_Matrix[(id*Size) + j] * g_V[j];
		}
		// Store the result of the dot product in the resultant vector
		g_P[id] = sum;
	}
}

__host__ void gpuMultHelper(float* h_Matrix, float* h_V, float* h_P, const int Size)
{
	float* d_Matrix, * d_V, * d_P;
	const int MatrixSizeInBytes = Size * Size * sizeof(float);
	const int VectorSize = Size * sizeof(float);

	chrono::time_point<std::chrono::high_resolution_clock> start, end;
	std::chrono::duration<double> elapsed_time;

	start = chrono::high_resolution_clock::now();
	//Allocate device memory on the global memory
	// Allocate the necessary memory to store the matrix on the device
	if (!HandleCUDAError(cudaMalloc((void**) &d_Matrix, MatrixSizeInBytes))) {
		cout << "[ERROR] Memory Allocation ( d_Matrix ) Failed" << endl;
	}

	// Allocate the necessary memory to store the postmultiplied vector on the device
	if (!HandleCUDAError(cudaMalloc((void**) &d_V, VectorSize))) {
		cout << "[ERROR] Memory Allocation ( d_V ) Failed" << endl;
	}

	// Allocate the necessary memory to store the resultant vector on the device
	if (!HandleCUDAError(cudaMalloc((void**) &d_P, VectorSize))) {
		cout << "[ERROR] Memory Allocation ( d_V ) Failed" << endl;
	}

	//Transfer data from CPU Memory to GPU Memory
	// Transfer the matrix from the host to the device memory
	if (!HandleCUDAError(cudaMemcpy(d_Matrix, h_Matrix, MatrixSizeInBytes, cudaMemcpyHostToDevice))) {
		cout << "[ERROR] Memory Copy ( D -> H ) <d_Matrix> Failed" << endl;
	}

	// Trnasfer the post multiplied vector to the device memory
	if (!HandleCUDAError(cudaMemcpy(d_V, h_V, VectorSize, cudaMemcpyHostToDevice))) {
		cout << "[ERROR] Memory Copy ( D -> H ) <d_V> Failed" << endl;
	}
	end = chrono::high_resolution_clock::now();
	elapsed_time = end - start;

	cout << "[INFO] Memory allocation and copy complete in " << duration_cast<microseconds>(elapsed_time).count() << " usecs" << endl;

	//Kernel Execution Configuration Parameters 
	// Force the number of threads per block to be an integer
	int threads_per_block = 128;
	// Force the number of blocks per grid to be an integer
	int blocks_per_grid = (Size + threads_per_block - 1) / threads_per_block;

	//Launch Kernel and collect execution time
	start = chrono::high_resolution_clock::now();
	MatrixVectorMult <<< blocks_per_grid, threads_per_block >>> (d_Matrix, d_V, d_P, Size);
	HandleCUDAError(cudaDeviceSynchronize());
	end = chrono::high_resolution_clock::now();
	elapsed_time = end - start;
	
	cout << "[INFO] Kernel execution completed in " << duration_cast<microseconds>(elapsed_time).count() << " usecs" << endl;

	//Transfer data from GPU Memory to CPU memory
	start = chrono::high_resolution_clock::now();
	// Only copy the resultant vector from the device to the host
	if (!HandleCUDAError(cudaMemcpy(h_P, d_P, VectorSize, cudaMemcpyDeviceToHost))) {
		cout << "[ERROR] Memory Copy ( H -> D ) Failed" << endl;
	}

	//Release device memory
	// On the device release the memory reserved for the matrix
	if (!HandleCUDAError(cudaFree(d_Matrix))) {
		cout << "[ERROR] Memory Free <d_Matrix> Failed" << endl;
	}

	// On the device release the memory reserved for the post multiplied vector
	if (!HandleCUDAError(cudaFree(d_V))) {
		cout << "[ERROR] Memory Free <d_V> Failed" << endl;
	}

	// On the device release the memory reserved for the resultant vector
	if (!HandleCUDAError(cudaFree(d_P))) {
		cout << "[ERROR] Memory Free <d_P> Failed" << endl;
	}
	end = chrono::high_resolution_clock::now();
	elapsed_time = end - start;
	
	HandleCUDAError(cudaDeviceReset());

	cout << "[INFO] Memory deallocation complete in " << duration_cast<microseconds>(elapsed_time).count() << " usecs" << endl;
}