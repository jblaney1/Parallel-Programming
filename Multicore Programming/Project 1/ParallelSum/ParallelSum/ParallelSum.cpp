#include "Sum.h"

//Parallelization with Manual Partitioning
double ArraySumParallel_ManualPartition(double* AverageTime, double* array, int Size, int ThreadCount)
{
	chrono::time_point<std::chrono::system_clock> start, end;
	*AverageTime = 0.0f;
	double sum;
	int ArrayDivision; //Variable to store the number of elements per thread of the array
	//Compute the number of elements per thread
	ArrayDivision = Size / ThreadCount;

	//Define an additional array to implement manual partitioning
	double* Results = new double[ThreadCount];

	for (int iter = 0; iter < NUM_IT; iter++)
	{
		//Start Timer
		start = std::chrono::system_clock::now();
		//Implement OMP based Parallelization with manual partitioning using only Parallel region and desired number of threads
#pragma omp parallel num_threads(ThreadCount)
		{
			_int64 thread_id = omp_get_thread_num();							// Store this threads id as a 64bit int
			int Thread_Start = thread_id * ArrayDivision;						// Calculate the starting point based on the thread
			int Thread_End = Thread_Start + ArrayDivision;						// Calculate the ending point based on the starting point
			Results[thread_id] = 0;												// Set the initial value of Results[] to zero each loop

			// Iterate through <array> and store the summation in the proper
			// position in Results[]
			for (int i = Thread_Start; i < Thread_End; i++) {
				Results[thread_id] += array[i];
			}
			
		}

		//Stop timer
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		*AverageTime += elasped_seconds.count();
	}
	*AverageTime /= NUM_IT;
	*AverageTime *= 1000.0f; //Converting seconds to milliseconds
	
	// Put the final interations calculation in sum
	sum = 0;
	for (int i = 0; i < ThreadCount; i++) {
		sum += Results[i];
	}

	//Release dynamically allocated Memory
	delete[] Results;

	return sum; //Return the sum of the last iteration
}


//Parallelization with Reduction
double ArraySumParallel_Reduction(double* AverageTime, double* array, int Size, int ThreadCount)
{
	chrono::time_point<std::chrono::system_clock> start, end;
	*AverageTime = 0.0f;
	double sum;
	int ArrayDivision; //Variable to store the number of elements per thread of the array
	//Compute the number of elements per thread
	ArrayDivision = Size / ThreadCount;

	for (int iter = 0; iter < NUM_IT; iter++)
	{
		sum = 0.0f;
		//Start Timer
		start = std::chrono::system_clock::now();
		//Implement OMP based Parallelization with Reduction and Parallel region and desired number of threads
#pragma omp parallel num_threads(ThreadCount), reduction(+:sum)
		{
			_int64 thread_id = omp_get_thread_num();							// Store this threads id as a 64bit int
			int Thread_Start = thread_id * ArrayDivision;						// Calculate the starting point based on the thread
			int Thread_End = Thread_Start + ArrayDivision;						// Calculate the ending point based on the starting point

			// Because of the <Reduction> clause I use double sum to store the results
			// rather than a specific point in an array.
			for (int i = Thread_Start; i < Thread_End; i++) {
				sum += array[i];
			}
		}

		//Stop timer
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		*AverageTime += elasped_seconds.count();
	}
	*AverageTime /= NUM_IT;
	*AverageTime *= 1000.0f; //Converting seconds to milliseconds
	return sum; //Return the sum of the last iteration
}