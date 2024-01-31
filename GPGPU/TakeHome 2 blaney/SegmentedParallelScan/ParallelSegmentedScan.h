#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"

#define B 1024 // Represents the size of each block, and can be set to a maximum size of 1024
#define Z 1024 // Integer Multiplier
#define VECTOR_SIZE B*Z // Number of Elements
#define VECTOR_SIZE_IN_BYTES (VECTOR_SIZE*sizeof(float))

//CPU Functions
void OnInitializeInputData(float* vectorTemp, int SIZE);
void CopyInputData(float* vectorTemp, float* ref,int SIZE);
void ZeroData(float* in, int SIZE);
void OnSequentialScan(float* in, float * out,int SIZE);
void PrintVectors(float* vector,int SIZE);
void Verify(float* ref, float* in, int SIZE);

//GPU Helper
__host__ void Helper_Scan(float* Input, float* Output, float* RefOutputData, int SIZE);

//GPU Kernels
//Kernel to perform parallel scan of only a single block with maximum of 1024 elements using the Efficient Kogge Stone Algorithm
__global__ void SingleBlockScan(float* In, float* Out, const int SIZE);
/*Kernel to perform parallel scan on individual blocks and produce  intermediate output and output of the
last element in each block*/
__global__ void MultipleBlockScanIntermediateOutput(float* In, float* AuxOut, float *blockSum, const int SIZE);
/*Kernel to add the block sum of the last element to the Intermediate Outputs to produce the final answer*/
__global__ void AddBlockSumValues(float* AuxIn, float* blockSum, const int SIZE);





