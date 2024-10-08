#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//Image Processing Routines
#include "CImg.h"
using namespace cimg_library;

//GPU Error Handling
#include "GPUErrors.h"

//CPU Functions
void cpu__RGBtoGrayScale(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg);
void Verify(unsigned int* cpu_histogram, unsigned int* gpu_histogram, unsigned int bins);
void WriteHistograms(string FileName, unsigned int* cpu_histogram, unsigned int* gpu_histogram, unsigned int bins);

//CPU Helper Functions to compute histogram
void cpu_Histogram(unsigned char* in, unsigned int bins, unsigned int* hist, unsigned int h, unsigned int w);

//CPU Functions to commit private histogram copies to public or global histogram
void HistogramCommitNaive(unsigned int* hist_private_copies,
	unsigned int* hist_global,
	unsigned int bins,
	unsigned int copies);
void HistogramCommitOptimized(unsigned int* hist_private_copies,
	unsigned int* hist_global,
	unsigned int bins,
	unsigned int copies);

//Define Histrogram Size
#define BINS 256

//GPU Helper Function for Naive Histogram
__host__ void gpu_NaiveHistogramHelper(unsigned char* h_in, 
	unsigned int *histogram,
	unsigned int graySIZE,
	unsigned int h, 
	unsigned int w,
	unsigned int *cpu_hist);

//GPU Kernel: Naive Version without Privatization
__global__ void gpu_NaiveHistogram(unsigned char *in, unsigned int *out, unsigned int h, unsigned int w);

//GPU Helper Function for private copy on Global Memory
__host__ void gpu_PCOnGlobalHistogramHelper(unsigned char* h_in, 
	unsigned int* histogram,
	unsigned int graySIZE,
	unsigned int h, 
	unsigned int w,
	unsigned int* cpu_hist, 
	bool CommitPrivateCopies);

/*GPU Kernel: Privatization with private histogram copies on the global memory,
and commit private histrogram copies to the global histogram on the GPU*/
__global__ void gpu_PCOnGlobal_Histogram(unsigned char *in, unsigned int *priv_out,unsigned int *out, unsigned int h, unsigned int w,bool commit);


//GPU Helper Function for private copy on Shared Memory
__host__ void gpu_PCOnSharedHistogramHelper(unsigned char* h_in,
	unsigned int* histogram,
	unsigned int graySIZE,
	unsigned int h,
	unsigned int w,
	unsigned int* cpu_hist,
	bool CommitPrivateCopies);

/*GPU Kernel: Privatization with private histogram copies on the shared memory,
and commit private histrogram copies to the global histogram on the GPU*/
__global__ void gpu_PCOnShared_Histogram(unsigned char *in, unsigned int* priv_out,unsigned int *out, unsigned int h, unsigned int w, bool commit);







