/*
* Author: Josh Blaney
* Date: 11/04/2024
* 
* Implements the functionality for adding, multiplying (dot product), 
* printing, and subtracting vectors on the GPU. The functions are 
* organized in alphabetical order and all functions include timing
* functionality.
*/

#include "common.cuh"
#include "Linear Algebra.h"


__global__ void add_vectors_gpu_helper(float* a, float* b, float* c, int length) {
	/*
	* Performs vector addition on two vectors (a minus b) of known length
	* (length) using one thread per data element and storing the result in a
	* different vector (c).
	*/

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);	// Global thread id

	// Threads beyond the vector length will return without processing
	if (idx >= length) { return; }

	// Each thread will perform one addition according to it's global id
	c[idx] = a[idx] + b[idx];
}

__host__ void add_vectors_gpu(float* a, float* b, float* c, int length, bool time) {

	// Device vectors
	float* da, * db, * dc;
	const int vector_size = length * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = 128;
	int blocks_per_grid = (length + threads_per_block - 1) / threads_per_block;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, vector_size))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, vector_size))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, vector_size))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	add_vectors_gpu_helper << < blocks_per_grid, threads_per_block >> > (da, db, dc, length);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, vector_size, cudaMemcpyDeviceToHost))) {
		std::cout << "[ERROR] An error occured transfering dc to the host\n";
	}

	if (!HandleCUDAError(cudaFree(da))) {
		std::cout << "[ERROR] An error occured freeing da from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(db))) {
		std::cout << "[ERROR] An error occured freeing db from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(dc))) {
		std::cout << "[ERROR] An error occured freeing dc from device memory\n";
	}

	HandleCUDAError(cudaDeviceReset());

	stop = std::chrono::high_resolution_clock::now();

	if (time) {
		compute_time = compute_time / 1E6;
		wall_time = stop - start;
		std::cout << "Time elapsed adding vectors (GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}


__global__ void dot_product_gpu_helper_1block1(float* a, float* b, float* c, int length) {
	/*
	* Coalesced memory access
	* Performs a vector dot product between two vectors (a and b) of known
	* length (length) and stores the result in a different vector (c) at the 
	* first index. This kernel assumes one block will be used for computations
	* and performs a __syncthreads() operation before combining the partial
	* computations from each thread.
	*/

	int bin_length = length / blockDim.x;	// The number of data elements each thread will process
	int offset = threadIdx.x * bin_length;	// The starting index for processing
	int position;							// The current index for processing
	float sum = 0.0f;						// Store the partial sum in register memory

	// Threads beyond the vector length will return without processing
	if (offset > length) { return; }

	/*
	 * Similar to a reduction, compute the dot product
	 * of sub vectors and store the result in the starting 
	 * index for processing
	*/
	for (int i = 0; i < bin_length; i++) {
		position = offset + i;
		if (position < length) {
			sum += a[position] * b[position];
		}		
	}

	c[offset] = sum;

	// Sync all threads in the block, this only works with single block algorithms
	__syncthreads();

	// Use the zero index thread to combine the partial computations.
	if (offset == 0) {
		for (int i = 1; i < blockDim.x; i++) {
			sum += c[i * bin_length];
		}

		c[offset] = sum;
	}
}

__host__ void dot_product_gpu_1block1(float* a, float* b, float* c, int length, bool time) {

	// Device vectors
	float* da, * db, * dc;
	const int vector_size = length * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = (length <= 1024) ? length : 512;
	int blocks_per_grid = 1;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, vector_size))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, vector_size))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, vector_size))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	dot_product_gpu_helper_1block1 <<< blocks_per_grid, threads_per_block >> > (da, db, dc, length);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, vector_size, cudaMemcpyDeviceToHost))) {
		std::cout << "[ERROR] An error occured transfering dc to the host\n";
	}

	if (!HandleCUDAError(cudaFree(da))) {
		std::cout << "[ERROR] An error occured freeing da from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(db))) {
		std::cout << "[ERROR] An error occured freeing db from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(dc))) {
		std::cout << "[ERROR] An error occured freeing dc from device memory\n";
	}

	HandleCUDAError(cudaDeviceReset());

	stop = std::chrono::high_resolution_clock::now();

	if (time) {
		compute_time = compute_time / 1E6;
		wall_time = stop - start;
		std::cout << "Time elapsed performing dot product (1Block Coalesced GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}


__global__ void dot_product_gpu_helper_1block2(float* a, float* b, float* c, int length) {
	/*
	* Interleaved memory access
	* Performs a vector dot product between two vectors (a and b) of known
	* length (length) and stores the result in a different vector (c) at the
	* first index. This kernel assumes one block will be used for computations
	* and performs a __syncthreads() operation before combining the partial
	* computations from each thread.
	*/

	int bin_length = length / blockDim.x;	// The number of data elements each thread will process
	int idx = threadIdx.x;					// The starting index for processing
	int position;							// The current index for processing
	float sum = 0.0f;						// Store the partial sum in register memory

	// Threads beyond the vector length will return without processing
	if (idx > length) { return; }

	/*
	 * Similar to a reduction, compute the dot product
	 * of sub vectors and store the result in the starting
	 * index for processing
	*/
	for (int i = 0; i < bin_length; i++) {
		position = idx + i * blockDim.x;
		if (position < length) {
			sum += a[position] * b[position];
		}
	}

	c[idx] = sum;

	// Sync all threads in the block, this only works with single block algorithms
	__syncthreads();

	// Use the zero index thread to combine the partial computations.
	if (idx == 0) {
		for (int i = 1; i < blockDim.x; i++) {
			sum += c[i];
		}

		c[idx] = sum;
	}
}

__host__ void dot_product_gpu_1block2(float* a, float* b, float* c, int length, bool time) {

	// Device vectors
	float* da, * db, * dc;
	const int vector_size = length * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = (length <= 1024) ? length : 512;
	int blocks_per_grid = 1;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, vector_size))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, vector_size))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, vector_size))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	dot_product_gpu_helper_1block2 << < blocks_per_grid, threads_per_block >> > (da, db, dc, length);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, vector_size, cudaMemcpyDeviceToHost))) {
		std::cout << "[ERROR] An error occured transfering dc to the host\n";
	}

	if (!HandleCUDAError(cudaFree(da))) {
		std::cout << "[ERROR] An error occured freeing da from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(db))) {
		std::cout << "[ERROR] An error occured freeing db from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(dc))) {
		std::cout << "[ERROR] An error occured freeing dc from device memory\n";
	}

	HandleCUDAError(cudaDeviceReset());

	stop = std::chrono::high_resolution_clock::now();

	if (time) {
		compute_time = compute_time / 1E6;
		wall_time = stop - start;
		std::cout << "Time elapsed performing dot product (1Block Interleaved GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}


__global__ void dot_product_gpu_helper_multiblock1(float* a, float* b, float* c, int length) {
	/*
	* Coalesced memory access
	* Performs a partial vector dot product between two vectors (a and b) of 
	* known length (length) and stores the partial result in a different 
	* vector (c) at the starting index for processing computed by each thread.
	* The results from this kernel must be further processed on the CPU to 
	* compute the final dot product by combining the partial results computed
	* by each thread.
	*/

	int bin_length = length / (blockDim.x * gridDim.x);	// The number of elements each thread will process
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);	// The global id of the current thread
	int offset = idx * bin_length;						// The starting index for processing
	int position;										// The current index for processing
	float sum = 0.0f;									// Store the partial sum in register memory

	// Threads beyond the vector length will return without processing
	if (offset >= length) { return; }

	 /*
	 * Perform a dot product on the sub-vector assigned to this thread,
	 * storing the result in the starting index for processing.
	 */
	for (int i = 0; i < bin_length; i++) {
		position = offset + i;
		if (position < length) {
			sum += a[position] * b[position];
		}
	}

	c[offset] = sum;
}

__host__ void dot_product_gpu_multiblock1(float* a, float* b, float* c, int length, bool time) {

	// Device vectors
	float* da, * db, * dc;
	const int bin_count = (length < 2048) ? length : 2048;
	const int bin_length = length / bin_count;
	const int vector_size = length * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = (length < 1024) ? length : 128;
	int blocks_per_grid = std::max(((length + threads_per_block - 1 ) / threads_per_block) / bin_count, 1);

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, vector_size))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, vector_size))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, vector_size))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	dot_product_gpu_helper_multiblock1 << < blocks_per_grid, threads_per_block >> > (da, db, dc, length);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, vector_size, cudaMemcpyDeviceToHost))) {
		std::cout << "[ERROR] An error occured transfering dc to the host\n";
	}

	if (!HandleCUDAError(cudaFree(da))) {
		std::cout << "[ERROR] An error occured freeing da from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(db))) {
		std::cout << "[ERROR] An error occured freeing db from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(dc))) {
		std::cout << "[ERROR] An error occured freeing dc from device memory\n";
	}

	HandleCUDAError(cudaDeviceReset());

	// Combining the partial computations into the final result
	for (int i = 1; i < bin_count; i++) {
		c[0] = c[0] + c[i * bin_length];
	}

	stop = std::chrono::high_resolution_clock::now();

	if (time) {
		compute_time = compute_time / 1E6;
		wall_time = stop - start;
		std::cout << "Time elapsed performing dot product (Coalesced Multiblock GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}

__global__ void dot_product_gpu_helper_multiblock2(float* a, float* b, float* c, int length) {
	/*
	* Interleaved memory access
	* Performs a partial vector dot product between two vectors (a and b) of
	* known length (length) and stores the partial result in a different
	* vector (c) at the starting index for processing computed by each thread.
	* The results from this kernel must be further processed on the CPU to
	* compute the final dot product by combining the partial results computed
	* by each thread.
	*/

	int bin_count = length / (blockDim.x * gridDim.x);	// The number of elements each thread will process
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);	// The global id of the current thread
	int offset = blockDim.x * gridDim.x;				// The number of elements to skip between computations
	int position;										// The current index for processing
	float sum = 0.0f;									// Store the partial sum in register memory

	// Threads beyond the vector length will return without processing
	if (idx >= length) { return; }

	/*
	* Perform a dot product on the sub-vector assigned to this thread,
	* storing the result in the starting index for processing.
	*/
	for (int i = 0; i < bin_count; i++) {
		position = idx + i * offset;
		if (position < length) {
			sum += a[position] * b[position];
		}
	}

	c[idx] = sum;
}

__host__ void dot_product_gpu_multiblock2(float* a, float* b, float* c, int length, bool time) {

	// Device vectors
	float* da, * db, * dc;
	const int bin_count = (length < 2048) ? length : 2048;
	const int bin_length = length / bin_count;
	const int vector_size = length * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = (length < 1024) ? length : 128;
	int blocks_per_grid = std::max(((length + threads_per_block - 1) / threads_per_block) / bin_count, 1);
	const int thread_count = threads_per_block * blocks_per_grid;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, vector_size))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, vector_size))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, vector_size))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	dot_product_gpu_helper_multiblock2 << < blocks_per_grid, threads_per_block >> > (da, db, dc, length);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, vector_size, cudaMemcpyDeviceToHost))) {
		std::cout << "[ERROR] An error occured transfering dc to the host\n";
	}

	if (!HandleCUDAError(cudaFree(da))) {
		std::cout << "[ERROR] An error occured freeing da from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(db))) {
		std::cout << "[ERROR] An error occured freeing db from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(dc))) {
		std::cout << "[ERROR] An error occured freeing dc from device memory\n";
	}

	HandleCUDAError(cudaDeviceReset());

	// Combining the partial computations into the final result
	for (int i = 1; i < thread_count; i++) {
		c[0] = c[0] + c[i];
	}

	stop = std::chrono::high_resolution_clock::now();

	if (time) {
		compute_time = compute_time / 1E6;
		wall_time = stop - start;
		std::cout << "Time elapsed performing dot product (Interleaved Multiblock GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}


__global__ void sub_vectors_gpu_helper(float* a, float* b, float* c, int length) {
	/*
	* Performs vector subtraction on two vectors (a minus b) of known length
	* (length) using one thread per data element and storing the result in a
	* different vector (c).
	*/

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);	// Global thread id

	// Threads beyond the vector length will return without processing
	if (idx >= length) { return; }

	// Each thread will perform one subtraction according to it's global id
	c[idx] = a[idx] - b[idx];
}

__host__ void sub_vectors_gpu(float* a, float* b, float* c, int length, bool time) {

	// Device vectors
	float* da, * db, * dc;
	const int vector_size = length * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = 128;
	int blocks_per_grid = (length + threads_per_block - 1) / threads_per_block;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, vector_size))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, vector_size))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, vector_size))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, vector_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	sub_vectors_gpu_helper << < blocks_per_grid, threads_per_block >> > (da, db, dc, length);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, vector_size, cudaMemcpyDeviceToHost))) {
		std::cout << "[ERROR] An error occured transfering dc to the host\n";
	}

	if (!HandleCUDAError(cudaFree(da))) {
		std::cout << "[ERROR] An error occured freeing da from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(db))) {
		std::cout << "[ERROR] An error occured freeing db from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(dc))) {
		std::cout << "[ERROR] An error occured freeing dc from device memory\n";
	}

	HandleCUDAError(cudaDeviceReset());

	stop = std::chrono::high_resolution_clock::now();

	if (time) {
		compute_time = compute_time / 1E6;
		wall_time = stop - start;
		std::cout << "Time elapsed subtracting vectors (GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}