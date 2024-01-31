#include "ParallelHistogram.h"


//Using private copies on the global memory
__global__ void gpu_PCOnGlobal_Histogram(unsigned char* in, unsigned int* priv_out, unsigned int* out, unsigned int h, unsigned int w, bool commit)
{
	unsigned int global_idx = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int local_idx = threadIdx.x;
	//Computing the private histrograms on the Global Memory
	if (global_idx < (w * h))
	{
		// Use the local ID and the number of threads in a block to decide how many pixels each thread should process.
		// If fewer threads per block are requested than the number of bins, some threads will be idle while the others
		// process more than one pixel. Store the result in a histogram which is private to the block and is located on 
		// the global memory.
		for (int i = local_idx; i < BINS; i += blockDim.x) {
			atomicAdd(&(priv_out[blockIdx.x * BINS + in[global_idx]]), 1);
		}
	}
	__syncthreads();

	if (commit)
	{
		// If the shared histograms are supposed to be committed to the global histogram by the device,
		// iterate over the shared histograms in the global memory and use an atomic add to bring the 
		// shared histograms into one global histogram without collisions.
		for (int i = local_idx; i < BINS; i += blockDim.x) {
			atomicAdd(&(out[i]), priv_out[blockIdx.x * BINS + i]);
		}
	}
	//If commit = False, the CPU will commit the private histograms on the GPU
}

__host__ void gpu_PCOnGlobalHistogramHelper(unsigned char* h_in,
	unsigned int* histogram,
	unsigned int graySIZE,
	unsigned int h,
	unsigned int w,
	unsigned int* cpu_hist,
	bool CommitPrivateCopies)
{
	//Cleaning previous histogram data
	for (unsigned int i = 0; i < BINS; i++)
	{
		histogram[i] = 0;
	}

	unsigned char* d_in;
	unsigned int* d_out;
	//Allocating device memory for GrayScale Image and Histogram
	if (!HandleCUDAError(cudaMalloc((void**)&d_in, graySIZE)))
	{
		cout << "[ERROR] Unable to allocate memory on GPU for the GrayScale image" << endl;
		return;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_out, BINS * sizeof(unsigned int))))
	{
		cout << "[ERROR] Unable to allocate memory on GPU for the Histogram" << endl;
		return;
	}
	//Copying the GrayScale image to the device
	if (!HandleCUDAError(cudaMemcpy(d_in, h_in, graySIZE, cudaMemcpyHostToDevice)))
	{
		cout << "[ERROR] Unable to copy Gray Scale image from Host to GPU" << endl;
		return;
	}
	//Initialize the device memory for the histogram with zero
	if (!HandleCUDAError(cudaMemset(d_out, 0, BINS * sizeof(unsigned int))))
	{
		cout << "[ERROR] Unable to initialize the histogram device memory on the  GPU" << endl;
		return;
	}

	//Setup Execution Configuration Parameters
	unsigned int threadsPerBlock = 256;
	unsigned int blocksPerGrid = ((w * h) / threadsPerBlock) + 1;

	/*Allocating memory on the Host to store
	histogram private copy/block computed on the GPU*/
	unsigned int copies = BINS * blocksPerGrid;
	unsigned int* private_histograms;
	private_histograms = new unsigned int[copies];

	//Create private copies of the histogram on the global memory
	//The number of private copies each of size BINS, should be equal to the number of blocks
	unsigned int* d_priv_out; //pointer to the collection of private copies	

	// Allocate the private output memory on the device
	if (!HandleCUDAError(cudaMalloc((void**)&d_priv_out, (copies * sizeof(unsigned int))))) {
		cout << "[ERROR] Unable to allocate the private copies on the gpu" << endl;
		return;
	}

	//Initialize the private histrogram copies
	if (!HandleCUDAError(cudaMemset(d_priv_out, 0, (copies * sizeof(unsigned int)))))
	{
		cout << "[ERROR] Unable to initialize the private histogram copies device memory on the  GPU" << endl;
		return;
	}

	//Launching the Private Histogram Copies on Global Memory 
	gpu_PCOnGlobal_Histogram << <blocksPerGrid, threadsPerBlock >> > (d_in,
		d_priv_out,
		d_out,
		h,
		w,
		CommitPrivateCopies);
	cudaDeviceSynchronize();
	if (CommitPrivateCopies)
	{
		// If the private copies have been committed to the global histogram by the device, retrieve the global histogram only
		if (!HandleCUDAError(cudaMemcpy(histogram, d_out, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost))) {
			cout << "[ERROR] Unable to copy histogram from device to host" << endl;
			return;
		}
	}
	else {
		
		// Else if the private copies have not been committed to the global histogram by the device, retrieve the private histogram copies
		if (!HandleCUDAError(cudaMemcpy(private_histograms, d_priv_out, (copies * sizeof(unsigned int)), cudaMemcpyDeviceToHost))) {
			cout << "[ERROR] Unable to copy private histograms from device to host" << endl;
			return;
		}

		// Use the CPU to commit the  private histogram copies to the global histogram using either Naive or Optimized code
		//HistogramCommitNaive(private_histograms, histogram, BINS, blocksPerGrid);
		HistogramCommitOptimized(private_histograms, histogram, BINS, blocksPerGrid);
	}
	Verify(cpu_hist, histogram, BINS);
	WriteHistograms("GlobalCopy.csv", cpu_hist, histogram, BINS);

	if (!HandleCUDAError(cudaFree(d_in)))
	{
		cout << "[ERROR] Unable to free RGB image memory" << endl;
		return;
	}
	if (!HandleCUDAError(cudaFree(d_out)))
	{
		cout << "[ERROR] Unable to free Histogram memory" << endl;
		return;
	}
	if (!HandleCUDAError(cudaFree(d_priv_out)))
	{
		cout << "[ERROR] Unable to free histogram private copies memory" << endl;
		return;
	}
	HandleCUDAError(cudaDeviceReset());
	delete[] private_histograms;
}
