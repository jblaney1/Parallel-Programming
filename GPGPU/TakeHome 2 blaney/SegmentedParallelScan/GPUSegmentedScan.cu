/* 
 * Author: Josh Blaney
 * Date: 12/02/2022
 * Class: EE 5885 GPGPU
 * Instructor: Dr. Suresh
 * Assignment: Exam 2 Takehome
 *
 */

#include "ParallelSegmentedScan.h"

#define BLOCK_SIZE B

//Kernel to perform parallel scan of only a single block wiht maximum of 1024 elements using the Efficient Kogge Stone Algorithm
__global__ void SingleBlockScan(float* In, float* Out, const int SIZE)
{
	__shared__ float In_Shared[BLOCK_SIZE];
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int tid = threadIdx.x;

	if (idx < SIZE)
	{
		In_Shared[tid] = In[idx];
	}
	else {
		In_Shared[tid] = 0.0f;
	}

	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		__syncthreads();
		int index = ((tid + 1) * 2 * stride) - 1;
		if (index < blockDim.x)
		{
			In_Shared[index] += In_Shared[index - stride];
		}
	}

	for (int stride = (BLOCK_SIZE / 4); stride > 0; stride /= 2)
	{
		__syncthreads();
		int index = ((tid + 1) * 2 * stride) - 1;
		if ((index + stride) < blockDim.x)
		{
			In_Shared[(index + stride)] += In_Shared[index];
		}
	}
	__syncthreads();

	if (idx < SIZE)
	{
		Out[idx] = In_Shared[tid];
	}
}

/*Kernel for performing parallel scan as follows:
1. Store the intermediate scanned outputs across multiple blocks in global memory
pointed by the pointer AuxOut
2. Store the last element output of each block in the global memory pointed by the blockSum
*/
__global__ void MultipleBlockScanIntermediateOutput(float* In, float* AuxOut, float* blockSum, const int SIZE)
{
	//Kernel to perform parallel scan of only a single block wiht maximum of 1024 elements using the Efficient Kogge Stone Algorithm
	__shared__ float In_Shared[BLOCK_SIZE];
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int tid = threadIdx.x;

	if (idx < SIZE)
	{
		In_Shared[tid] = In[idx];
	}
	else {
		In_Shared[tid] = 0.0f;
	}

	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		__syncthreads();
		int index = ((tid + 1) * 2 * stride) - 1;
		if (index < blockDim.x)
		{
			In_Shared[index] += In_Shared[index - stride];
		}
	}

	for (int stride = (BLOCK_SIZE / 4); stride > 0; stride /= 2)
	{
		__syncthreads();
		int index = ((tid + 1) * 2 * stride) - 1;
		if ((index + stride) < blockDim.x)
		{
			In_Shared[(index + stride)] += In_Shared[index];
		}
	}
	__syncthreads();

	/* Alteration to the Kogge Stone algorithm which stores the result of the reduction
	 * in the global variable AuxOut but then also stores the final value from each Scan 
	 * Block in a separate global variable blockSum according to the block id. 
	 */
	if (idx < SIZE)
	{
		AuxOut[idx] = In_Shared[tid];				// Store the Kogge Stone reduction in global memory

		if (tid + 1 == BLOCK_SIZE) {				// Check the local thread id against the last block index 
			blockSum[blockIdx.x] = In_Shared[tid];	// Store last scanned value in global blockSum by blockId
		}
	}
}

/*Kernel to add the block sum of the last element to the Intermediate Outputs to produce the final answer*/
__global__ void AddBlockSumValues(float* AuxIn, float* blockSum, const int SIZE)
{
	/* The execution configuration allocates an adaquate number of threads to allow one thread to process each
	 * vector element; therefore, each thread can perform one addition betweeen the reduction vector and the 
	 * scanned block sums. All of the threads within a block will use the same element from the scanned block 
	 * sums but will add the element to a unique position in the reduction vector based on the global thread
	 * id. 
	 * 
	 * A more coarse approach can be used by limiting the number of threads to the number of blocks and 
	 * utilizing only one block per grid but this configuration slows down kernel execution considerably.
	 */

	// Compute the global thread id adjusted by the BLOCK_SIZE
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x + BLOCK_SIZE;
	
	// Limit the array access to threads with an id less than than the size of the reduction vector
	if (idx < SIZE) {
		AuxIn[idx] += blockSum[blockIdx.x];	// Perform one addition per thread
	}
}

__host__ void Helper_Scan(float* Input, float* Output, float* RefOutputData, int SIZE)
{
	float* d_in{};//Global Memory pointer for the input vector
	float* d_out{}; //Global Memory pointer for the intermediate and final output 
	float* d_blockSum{}; //Global Memory pointer for storing the last element output of each block

	//Execution Configuration parameters
	int threadsPerBlock = BLOCK_SIZE;
	int blocksPerGrid = ceil(VECTOR_SIZE / threadsPerBlock);
	
	cout << "[INFO] Vector Size = " << VECTOR_SIZE << endl;
	cout << "[INFO] Threads Per Block = " << threadsPerBlock << endl;
	cout << "[INFO] Block Per Grid = " << blocksPerGrid << endl;


	// ======================= Memory Allocation Section =======================

	/* Allocate memory on the GPU using cudaMalloc for global variables.
	 * The cudaMalloc calls are within HandleCUDAError calls so that if an error occurs
	 * while attemptimg to allocate memory a human readable message can be written
	 * to the console and the program can exit gracefully.
	 */	

	/* d_in is used to store the input vector passed to the helper function by Input.
	 * USED BY: MultipleBlockScanIntermediateOutput
	 */ 
	HandleCUDAError(cudaMalloc((void**)&d_in, VECTOR_SIZE_IN_BYTES));
	
	/* d_out is used to store the output vector 
	 * USED BY: MultipleBlockScanIntermediateOutput and AddBlockSumValues
	 */
	HandleCUDAError(cudaMalloc((void**)&d_out, VECTOR_SIZE_IN_BYTES));
	
	/* d_blockSum is used to store the last value in the intermediate scan blocks
	 * so that the last computation from each block can be propagated in the
	 * reduction.
	 * USED BY: MultipleBlockScanIntermediateOutput and SingleBlockScan
	 */
	HandleCUDAError(cudaMalloc((void**)&d_blockSum, blocksPerGrid * sizeof(float)));


	// ===================== Memory Instantiation Section =====================

	/* Instantiate the memory pointed to by d_in by transfering the contents of Input
	 * from the host to the GPU. The cudaMemcpy call is within a HandleCUDAError call
	 * so that if an error occurs while attemptimg to transfer the data a human 
	 * readable message can be written to the console and the program can exit gracefully.
	 */

	 /* d_in is being used to store the Input vector which is being operated on. This line 
	  * of code performs a transfer from the Host memory to the GPU memory according to the 
	  * number of bytes specified. VECTOR_SIZE_IN_BYTES is defined in the header file 
	  * ParallelSegmentedScan.h and is created by multiplying the number of elements defined
	  * by SIZE by the predefined size of a single precision float.
	  * USED BY: MultipleBlockScanIntermediateOutput
	  */
	HandleCUDAError(cudaMemcpy(d_in, Input, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice));
	

	// ========================= Kernel Launch Section =========================

	/* This section contains all of the kernel launch statements used by this program
	 * to perform the block scan Kogge Stone reduction algorithm. 
	 * 
	 * First the MultipleBlockScanIntermediateOutput kernel is used to compute a scan
	 * reduction on blocks of size BLOCK_SIZE (defined in ParallelSegmenetedScan.h) 
	 * and store the intermediate reduction along with the last element in each block 
	 * in two outputs (d_out and d_blockSum respectively). The number of blocks 
	 * dedicated to this task is configured by blocksPerGrid and the number of threads
	 * dedicated to this task is configured by threadsPerBlock such that there are at 
	 * least as many threads as there are array elements.
	 * 
	 * Second the SingleBlockScan kernel takes the scanned block which form the 
	 * intermediate reduction and performs a reduction on the scanned block creating
	 * a vector of scanned block sums which can be distributed into the output vector
	 * to complete the algorithm. The number of blocks used for this task is 1 and 
	 * the number of threads per block is set to blocksPerGrid. This aligns the number
	 * of threads per block with the number of scan block elements.
	 * 
	 * Third the AddBlockSumValues kernel takes the scanned block sums and distributes 
	 * them among the reduction array such that the resutlant arrray is the complete
	 * output for the block scan Kogge Stone reduction algorithm. The number of blocks 
	 * dedicated to this task is configured by blocksPerGrid and the number of threads
	 * dedicated to this task is configured by threadsPerBlock such that there are at 
	 * least as many threads as there are array elements.
	 */

	MultipleBlockScanIntermediateOutput <<< blocksPerGrid, threadsPerBlock >>> (d_in, d_out, d_blockSum, SIZE);
	SingleBlockScan <<< 1, blocksPerGrid >>> (d_blockSum, d_blockSum, SIZE);
	AddBlockSumValues <<< blocksPerGrid, threadsPerBlock >>> (d_out, d_blockSum, SIZE);

	// ===================== Memory Retrieval Section =====================

	/* Retrieve the final output array (d_out) from GPU memeory and store it in the Host's 
	 * memory (Output) by transfering VECTOR_SIZE_IN_BYTES bytes. The cudaMemcpy call is 
	 * within a HandleCUDAError call so that if an error occurs while attemptimg to transfer 
	 * the data a human readable message can be written to the console and the program can 
	 * exit gracefully.
	 */
	HandleCUDAError(cudaMemcpy(Output, d_out, VECTOR_SIZE_IN_BYTES, cudaMemcpyDeviceToHost));


	cout << "[INFO] Efficient Kunne Stone Parallel Segmented Scan GPU Results" << endl;
	Verify(RefOutputData, Output, SIZE);
	PrintVectors(Output, SIZE);

	HandleCUDAError(cudaFree(d_in));
	HandleCUDAError(cudaFree(d_out));
	HandleCUDAError(cudaFree(d_blockSum));
	HandleCUDAError(cudaDeviceReset());
}