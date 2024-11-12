/*
* Author: Josh Blaney
* Date: 11/04/2024
* 
* Description:
* Implements the functionality for adding, multiplying, subtracting, 
* and printing matrices on the GPU. The functionas are organized in 
* alphabetical order (except transpose) and all functions include 
* timing functionality.
*/


#include "common.cuh"
#include "Linear Algebra.h"


__global__ void add_matrices_gpu_helper(float* a, float* b, float* c, int rows, int cols) {
	/*
	* Performs matrix addition between two matrices (a and b) of known sizes
	* (rows and cols) and stores the result in a different matrix (c), using
	* one thread per data element.
	*/

	int idx = threadIdx.x + (blockIdx.x * blockDim.x); // Global id of the current thread

	// Threads beyond the matrix size will return without processing
	if (idx >= rows * cols) { return; }

	// Each thread will perform one addition according to it's global id
	c[idx] = a[idx] + b[idx];
}

__host__ void add_matrices_gpu(float* a, float* b, float* c, int rows, int cols, bool time) {

	// Device matrices
	float* da, * db, * dc;
	const int matrix_size = rows * cols * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = 128;
	int blocks_per_grid = (rows * cols + threads_per_block - 1) / threads_per_block;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, matrix_size))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, matrix_size))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, matrix_size))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, matrix_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, matrix_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	add_matrices_gpu_helper <<< blocks_per_grid, threads_per_block >>> (da, db, dc, rows, cols);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, matrix_size, cudaMemcpyDeviceToHost))) {
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
		std::cout << "Time elapsed adding matrices (GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}


__global__ void mult_matrices_gpu_niave_helper1(float* a, float* b, float* c, int rowsa, int colsa, int colsb) {
	/*
	* Performs A * B
	* Performs matrix multiplication between two matrices (a and b) of
	* known shapes (rowsa, colsa, and colsb) and stores the result in 
	* a different matrix (c).
	*/

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);	// Global id of the current thread
	int row, col;												// The current row and col for processing
	float sum = 0.0f;											// Store the partial sum in register memory

	// Threads beyond the matrix size will return without processing
	if (idx >= rowsa * colsb) { return; }

	// Identify the row and column to iterate along
	row = int(idx / colsb) * colsa;
	col = idx % colsb;

	// Iterate along the entries and compute the multiplication for one index
	for (int i = 0; i < colsa; i++) {
		sum += a[row + i] * b[col + i * colsb];
	}

	c[idx] = sum;
}

__host__ void mult_matrices_gpu_niave1(float* a, float* b, float* c, int rowsa, int colsa, int colsb, bool time) {

	// Device matrices
	float* da, * db, * dc;
	const int matrix_size_a = rowsa * colsa * sizeof(float);
	const int matrix_size_b = colsa * colsb * sizeof(float);
	const int matrix_size_c = rowsa * colsb * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = 256;
	int blocks_per_grid = (rowsa * colsb + threads_per_block - 1) / threads_per_block;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, matrix_size_a))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, matrix_size_b))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, matrix_size_c))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, matrix_size_a, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, matrix_size_b, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	mult_matrices_gpu_niave_helper1 <<< blocks_per_grid, threads_per_block >>> (da, db, dc, rowsa, colsa, colsb);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, matrix_size_c, cudaMemcpyDeviceToHost))) {
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
		std::cout << "Time elapsed multiplying matrices (Coalesced GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}


__global__ void mult_matrices_gpu_niave_helper2(float* a, float* b, float* c, int rowsa, int colsa, int colsb) {
	/*
	* Performs Transpose(A) * B
	* Performs matrix multiplication between two matrices (a and b) of
	* known shapes (rowsa, colsa, and colsb) and stores the result in 
	* a different matrix (c).
	*/

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);	// Global id of the current thread
	int cola, colb;												// The current row and col for processing
	float sum = 0.0f;											// Store the partial sum in register memory

	// Threads beyond the matrix size will return without processing
	if (idx >= rowsa * colsb) { return; }

	// Identify the row and column to iterate along
	cola = int(idx / colsb);
	colb = idx % colsb;

	// Iterate along the entries and compute the multiplication for one index
	for (int i = 0; i < colsa; i++) {
		sum += (a[cola + i * rowsa] * b[colb + i * colsb]);
	}

	c[idx] = sum;
}

__host__ void mult_matrices_gpu_niave2(float* a, float* b, float* c, int rowsa, int colsa, int colsb, bool time) {

	// Device matrices
	float* da, * db, * dc, * transpose_a;
	const int matrix_size_a = rowsa * colsa * sizeof(float);
	const int matrix_size_b = colsa * colsb * sizeof(float);
	const int matrix_size_c = rowsa * colsb * sizeof(float);
	
	transpose_a = new float[rowsa * colsa]();

	// Transpose matrix a to allow coallesced memory access
	transpose_matrices_gpu(a, transpose_a, rowsa, colsa, time);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = (rowsa * colsb < 1024) ? rowsa * colsb : 256;
	int blocks_per_grid = max((rowsa * colsb + threads_per_block - 1) / threads_per_block, 1);

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, matrix_size_a))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, matrix_size_b))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, matrix_size_c))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, transpose_a, matrix_size_a, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, matrix_size_b, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	mult_matrices_gpu_niave_helper2 << < blocks_per_grid, threads_per_block >> > (da, db, dc, rowsa, colsa, colsb);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, matrix_size_c, cudaMemcpyDeviceToHost))) {
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

	free(transpose_a);

	stop = std::chrono::high_resolution_clock::now();

	if (time) {
		compute_time = compute_time / 1E6;
		wall_time = stop - start;
		std::cout << "Time elapsed multiplying matrices (Interleaved GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}


__global__ void mult_matrices_gpu_tiled_helper(float* a, float* b, float* c, int rowsa, int colsa, int colsb) {
	/*
	* Performs A * B
	* Performs matrix multiplication between two matrices (a and b) of
	* known shapes (rowsa, colsa, and colsb) and stores the result in
	* a different matrix (c).
	*/

	__shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int gidx = tidx + (blockIdx.x * blockDim.x);		// Index of the row in c
	int gidy = tidy + (blockIdx.y * blockDim.y);		// Index of the column in c
	int row = gidy * colsa + tidx;
	int col = tidy * colsb + gidx;
	int tiles = colsa / TILE_WIDTH;						// The number of tiles to process
	float sum = 0.0f;									// Store the partial sum in register memory

	// Iterate along the entries and compute the multiplication for one index
	for (int tile = 0; tile < tiles; tile++) { 
		A_tile[tidy][tidx] = a[tile * TILE_WIDTH + gidy * colsa + tidx];
		B_tile[tidy][tidx] = b[(tile * TILE_WIDTH + tidy) * colsb + gidx];

		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; i++) {
			sum += A_tile[tidy][i] * B_tile[i][tidx];
		}
		
		__syncthreads();
	}

	c[gidy * colsb + gidx] = sum;
}

__host__ void mult_matrices_gpu_tiled(float* a, float* b, float* c, int rowsa, int colsa, int colsb, bool time) {

	// Device matrices
	float* da, * db, * dc;
	const int matrix_size_a = rowsa * colsa * sizeof(float);
	const int matrix_size_b = colsa * colsb * sizeof(float);
	const int matrix_size_c = rowsa * colsb * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int blocks_x = (colsb + TILE_WIDTH - 1) / TILE_WIDTH;
	int blocks_y = (rowsa + TILE_WIDTH - 1) / TILE_WIDTH;

	dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
	dim3 blocks_per_grid(blocks_x, blocks_y);

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, matrix_size_a))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, matrix_size_b))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, matrix_size_c))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, matrix_size_a, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, matrix_size_b, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	mult_matrices_gpu_tiled_helper << < blocks_per_grid, threads_per_block >> > (da, db, dc, rowsa, colsa, colsb);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, matrix_size_c, cudaMemcpyDeviceToHost))) {
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
		std::cout << "Time elapsed multiplying matrices (Tiled GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}


__global__ void transpose_matrices_gpu_helper(float* src, float* dst, int rows, int cols) {
	/*
	* Performs a matrix transpose on a source matrix (src) and stores
	* the result in a destination matrix (dst) both of which have known
	* size (rows and cols).
	*/

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);	// Global id of the current thread
	int row = idx * rows;									// The index of the current processing row

	// Threads beyond the matrix size will return without processing
	if (idx >= rows * cols) { return; }

	// Iterate along the entries and compute the multiplication for one index
	for (int i = 0; i < rows; i++) {
		dst[row + i] = src[idx + i * cols];
	}
}

__host__ void transpose_matrices_gpu(float* src, float* dst, int rows, int cols, bool time) {

	// Device matrices
	float* dsrc, * ddst;
	const int matrix_size = rows * cols * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = 128;
	int blocks_per_grid = (cols + threads_per_block - 1) / threads_per_block;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&dsrc, matrix_size))) {
		std::cout << "[ERROR] An error occured allocating dsrc on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&ddst, matrix_size))) {
		std::cout << "[ERROR] An error occured allocating ddst on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(dsrc, src, matrix_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	transpose_matrices_gpu_helper << < blocks_per_grid, threads_per_block >> > (dsrc, ddst, rows, cols);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(dst, ddst, matrix_size, cudaMemcpyDeviceToHost))) {
		std::cout << "[ERROR] An error occured transfering dc to the host\n";
	}

	if (!HandleCUDAError(cudaFree(dsrc))) {
		std::cout << "[ERROR] An error occured freeing dsrc from device memory\n";
	}

	if (!HandleCUDAError(cudaFree(ddst))) {
		std::cout << "[ERROR] An error occured freeing ddst from device memory\n";
	}

	HandleCUDAError(cudaDeviceReset());

	stop = std::chrono::high_resolution_clock::now();

	if (time) {
		compute_time = compute_time / 1E6;
		wall_time = stop - start;
		std::cout << "Time elapsed transposing matrices (GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}


__global__ void sub_matrices_gpu_helper(float* a, float* b, float* c, int rows, int cols) {
	/*
	* Performs matrix subtraction between two matrices (a minus b) of known sizes
	* (rows and cols) and stores the result in a different matrix (c), using
	* one thread per data element.
	*/

	int idx = threadIdx.x + (blockIdx.x * blockDim.x); // Global id of the current thread

	// Threads beyond the vector length will return without processing
	if (idx >= rows * cols) { return; }

	// Each thread will perform one addition according to it's global id
	c[idx] = a[idx] - b[idx];
}

__host__ void sub_matrices_gpu(float* a, float* b, float* c, int rows, int cols, bool time) {

	// Device matrices
	float* da, * db, * dc;
	const int matrix_size = rows * cols * sizeof(float);

	float compute_time = 0.0f;
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	int threads_per_block = 128;
	int blocks_per_grid = (rows * cols + threads_per_block - 1) / threads_per_block;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::high_resolution_clock::duration wall_time;

	start = std::chrono::high_resolution_clock::now();

	// Memory allocation on and data transfers to the device
	if (!HandleCUDAError(cudaMalloc(&da, matrix_size))) {
		std::cout << "[ERROR] An error occured allocating da on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&db, matrix_size))) {
		std::cout << "[ERROR] An error occured allocating db on the device\n";
	}

	if (!HandleCUDAError(cudaMalloc(&dc, matrix_size))) {
		std::cout << "[ERROR] An error occured allocating dc on the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(da, a, matrix_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering da to the device\n";
	}

	if (!HandleCUDAError(cudaMemcpy(db, b, matrix_size, cudaMemcpyHostToDevice))) {
		std::cout << "[ERROR] An error occured transfering db to the device\n";
	}

	// Launching the kernel and timing events
	cudaEventRecord(begin);
	sub_matrices_gpu_helper <<< blocks_per_grid, threads_per_block >>> (da, db, dc, rows, cols);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&compute_time, begin, end);

	// Memory transfers back to the host and freeing device memory
	if (!HandleCUDAError(cudaMemcpy(c, dc, matrix_size, cudaMemcpyDeviceToHost))) {
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
		std::cout << "Time elapsed subtracting matrices (GPU):\nComp Time:\t" << compute_time << "ms\nWall time:\t" << wall_time.count() / 1E6 << "ms" << std::endl;
	}
}