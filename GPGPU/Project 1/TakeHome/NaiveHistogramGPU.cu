#include "ParallelHistogram.h"
//Naive Version
__global__ void gpu_NaiveHistogram(unsigned char* in, unsigned int* out, unsigned int h, unsigned int w)
{
	unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx < (w * h))
	{
		int temp = in[idx];
		atomicAdd(&(out[temp]),1);
	}
}

//Host Helper function for Naive Kernel
__host__ void gpu_NaiveHistogramHelper(unsigned char* h_in,
	unsigned int* histogram,
	unsigned int graySIZE,
	unsigned int h,
	unsigned int w,
	unsigned int* cpu_hist)
{
	//Initializing histogram data
	for (unsigned int i = 0; i < BINS; i++)
	{
		histogram[i] = 0;
	}
	
	unsigned char* d_in;
	unsigned int *d_out;

	//Allocating device memory for GrayScale Image and Histogram
	if (!HandleCUDAError(cudaMalloc((void**)&d_in, graySIZE)))
	{
		cout << "Error Allocating memory on GPU for the GrayScale image" << endl;
		return;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_out, BINS*sizeof(unsigned int))))
	{
		cout << "Error Allocating memory on GPU for the Histogram" << endl;
		return;
	}
	//Copying the GrayScale image to the device
	if (!HandleCUDAError(cudaMemcpy(d_in,h_in,graySIZE,cudaMemcpyHostToDevice)))
	{
		cout << "Error copying Gray Scale image from Host to GPU" << endl;
		return;
	}
	//Initialize the device memory for the histogram with zero
	if (!HandleCUDAError(cudaMemset(d_out,0,BINS*sizeof(unsigned int))))
	{
		cout << "Error initializing the histogram device memory on the  GPU" << endl;
		return;
	}
	//Setup Execution Configuration Parameters
	unsigned int threadsPerBlock = 256;
	unsigned int blocksPerGrid = ((w * h) / threadsPerBlock) + 1;
	cout << "Image Grid Size = " << (w * h) << " pixels" << endl;
	cout << "Number of threads per block = " << threadsPerBlock << endl;
	cout << "Number of blocks per Grid = " << blocksPerGrid << endl;
	cout << "Total Number of Threads in the Grid = " << threadsPerBlock * blocksPerGrid << endl;

	cout << "Executing Naive Histogram Computation" << endl;
	gpu_NaiveHistogram << <blocksPerGrid, threadsPerBlock >> > (d_in,
		d_out,
		h,
		w);
	cudaDeviceSynchronize();
	if (!HandleCUDAError(cudaMemcpy(histogram, d_out, (BINS*sizeof(unsigned int)), cudaMemcpyDeviceToHost)))
	{
		cout << "Error copying Histogram from GPU to Host" << endl;
		return;
	}
	Verify(cpu_hist, histogram, BINS);
	WriteHistograms("NaiveHistogram.csv", cpu_hist,histogram, BINS);

	if (!HandleCUDAError(cudaFree(d_in)))
	{
		cout << "Error freeing RGB image memory" << endl;
		return;
	}
	if (!HandleCUDAError(cudaFree(d_out)))
	{
		cout << "Error freeing Histogram memory" << endl;
		return;
	}
	HandleCUDAError(cudaDeviceReset());
}