//A Program to compute the sum of the elements of an array
#include "Sum.h"

int main()
{
	//Maximum number of hyper-threads
	int LogicalProcessorCount = omp_get_max_threads();
	cout << "Number of Logical Processors: " << LogicalProcessorCount << endl;

	int ThreadsPerLogicalProcessor{ 1 }; //Thread Multiplier to multiply the number of logical processor counts
	cout << "Enter the number of threads per logical processors to be spawned: ";
	cin >> ThreadsPerLogicalProcessor;

	int SpawnedThreads = LogicalProcessorCount * ThreadsPerLogicalProcessor;

	//Array Size divisible equally by the number of threads
	int ArraySize = 96000000;
	/*int ArraySizeRemainder = ArraySize % SpawnedThreads;
	ArraySize += ArraySizeRemainder;*/
	int ArraySizeRemainder = ArraySize / SpawnedThreads;
	ArraySize = ArraySizeRemainder * SpawnedThreads + SpawnedThreads;

	//Display Threads per Logical Processor, Total Number of Threads Spawned, and Array Size
	cout << "Number of Threads/Logial Processor: " << ThreadsPerLogicalProcessor << endl;
	cout << "Number of Threads to be Spawned: " << SpawnedThreads << endl;
	cout << "Array Size: " << ArraySize << endl;

	//Allocate memory dynamically for a single dimensional Array
	double* Array = new double[ArraySize];

	//Initialize the array with random double precision numbers between RANGE_MIN and RANGE_MAX
	InitializeArray(Array, ArraySize);

	double AverageTime{ 0.0f }; //Average Computation Time
	double Sum{ 0.0 }; //Sum of the elements of the Array

	//Compute the sum of the array sequentially
	Sum = ArraySumSequential(&AverageTime, Array, ArraySize);
	//Display Average Computation Time and the results
	cout << "Sequential Execution: Average Computation Time: " << AverageTime << " msecs" << endl;
	cout << "Sequential Execution: Array Sum: " << Sum << endl;

	//Reset
	Sum = 0.0f;
	AverageTime = 0.0f;
	//Compute the sum of the array in parallel with manual partitioning
	Sum = ArraySumParallel_ManualPartition(&AverageTime, Array, ArraySize, SpawnedThreads);
	//Display parallel with manual partitioning Average Computation Time and the results
	cout << "Parallel Execution with Manual Partition: Average Computation Time: " << AverageTime << " msecs" << endl;
	cout << "Parallel Execution with Manual Partition: Array Sum: " << Sum << endl;

	//Reset
	Sum = 0.0f;
	AverageTime = 0.0f;
	//Compute the sum of the array in parallel with reduction
	Sum = ArraySumParallel_Reduction(&AverageTime, Array, ArraySize, SpawnedThreads);
	//Display parallel with manual partitioning Average Computation Time and the results
	cout << "Parallel Execution with Reduction: Average Computation Time: " << AverageTime << " msecs" << endl;
	cout << "Parallel Execution with Reduction: Array Sum: " << Sum << endl;

	//Release dynamically allocated memory
	delete[] Array;
	return 0;
}