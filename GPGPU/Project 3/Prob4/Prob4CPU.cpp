/*
* Class: EE 5885 GPGPU
* Instructor: Dr. Suresh 
* Assignment: 3
* Problem: 4
* Author: Josh Blaney
* Date: 10/25/2022
*/

#include "Prob4.h"

void cpuMatrixVectorMult(float* matrix, float* v, float* p, const int Size)
{
	chrono::time_point<high_resolution_clock> start, end;
	double computeTime{};
	start = high_resolution_clock::now();
	//Write code to perform Matrix Vector Multiplication
	
	float sum = 0.0f;

	// i is used to track the number of rows processed
	// j is used to track the current column being processed
	for (int i = 0; i < Size; i++) {
		sum = 0.0f;
		for (int j = 0; j < Size; j++) {
			sum += matrix[(i * Size) + j] * v[j];
		}
		p[i] = sum;
	}

	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "CPU Execution time: " << computeTime << " usecs" << endl;
}