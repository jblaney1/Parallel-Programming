#include "Sum.h"

//Function to compute the array sum sequentially
double ArraySumSequential(double* AverageTime, double* array, int Size) 
{
	chrono::time_point<std::chrono::system_clock> start, end;
	*AverageTime = 0.0f;
	double sum;
	for (int iter = 0;iter < NUM_IT; iter++)
	{
		start = std::chrono::system_clock::now();
		sum = array[0];
		for (int j = 1; j < Size; j++)
		{
			sum += array[j];
		}
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		*AverageTime += elasped_seconds.count();
	}
	*AverageTime /= NUM_IT;
	*AverageTime *= 1000.0f; //Converting seconds to milliseconds
	return sum; //Return the sum of the last iteration
}