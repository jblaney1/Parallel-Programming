#include "ParallelHistogram.h"

/*Using private copies on the Shared memory*/
__global__ void gpu_PCOnShared_Histogram(unsigned char* in, unsigned int* priv_out, unsigned int* out, unsigned int h, unsigned int w, bool commit)
{
	unsigned int global_idx = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int local_idx = threadIdx.x;
	__shared__ int local_histogram[BINS];

	//Initializing the private histogram copies on the Shared Memory
	for (int i = local_idx; i < BINS; i += blockDim.x)
	{
		local_histogram[i] = 0;
	}
	__syncthreads();

	//Computing the private histrograms on the Shared Memory
	if (global_idx < (w * h))
	{
		int temp = in[global_idx];
		atomicAdd(&(local_histogram[temp]), 1);
	}
	__syncthreads();

	if (commit)
	{
		/*Code to commit the private histogram copies on the shared memory to the
		public or global histogram on the global memory*/
		for (int i = local_idx; i < BINS; i += blockDim.x)
		{
			atomicAdd(&(out[i]), local_histogram[i]);
		}
	}
	else {

		// If the device is not supposed to commit the private histograms to the global histogram,
		// take the shared memory private histograms which are shared among a block and transfer 
		// the data to the global memory private histogram.
		for (int i = local_idx; i < BINS; i += blockDim.x) {
			priv_out[blockIdx.x*BINS + i] += local_histogram[i];
		}
		__syncthreads();
	}
}

__host__ void gpu_PCOnSharedHistogramHelper(unsigned char* h_in,
	unsigned int* histogram,
	unsigned int graySIZE,
	unsigned int h,
	unsigned int w,
	unsigned int* cpu_hist,
	bool CommitPrivateCopies)
{
	//Initializing histogram data
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
	//Allocating device memory for the Global Histogram
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
	//Initialize the device memory for the Global histogram with zeros
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
	unsigned int* d_priv_out; //pointer to the private copies of the histogram

	// Allocate the private output memory on the device
	if (!HandleCUDAError(cudaMalloc((void**)&d_priv_out, copies * sizeof(unsigned int)))) {
		cout << "[ERROR] Unable to allocate the private copies on the gpu" << endl;
		return;
	}

	//Initialize the private histrogram copies
	if (!HandleCUDAError(cudaMemset(d_priv_out, 0, (copies * sizeof(unsigned int)))))
	{
		cout << "[ERROR] Unable to initialize the private histogram copies device memory on the  GPU" << endl;
		return;
	}

	gpu_PCOnShared_Histogram << <blocksPerGrid, threadsPerBlock >> > (d_in,
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
	WriteHistograms("SharedCopy.csv", cpu_hist, histogram, BINS);

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
		cout << "[ERROR] Unable to free Histogram private copies memory" << endl;
		return;
	}
	HandleCUDAError(cudaDeviceReset());
	delete[] private_histograms;

}