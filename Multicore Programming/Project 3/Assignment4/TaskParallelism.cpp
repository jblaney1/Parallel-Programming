#include "LinearRegression.h"

float Sum(const float* VectorX, int n)
{
	float fSumOf = 0.0f;

	//Write code to compute the sum of x values in parallel
#pragma omp parallel for reduction(+:fSumOf)
	for (int j = 0; j < n; j++) {
		fSumOf += VectorX[j];
	}

	return fSumOf;
}

float SumOfXSquare(const float* VectorX, int n)
{
	float fSumOfXSquare = 0.0f;
	//Write code to compute the sum of x^2 values in parallel
#pragma omp parallel for reduction(+:fSumOfXSquare)
	for (int j = 0; j < n; j++) {
		fSumOfXSquare += VectorX[j] * VectorX[j];
	}

	return fSumOfXSquare;
}

float SumOfXYProduct(const float* VectorX, const float* VectorY, int n)
{
	float fSumOfXYProduct = 0.0f;
	//Write code to compute the sum of the product of xy values in parallel
#pragma omp parallel for reduction(+:fSumOfXYProduct)
	for (int j = 0; j < n; j++) {
		fSumOfXYProduct += VectorX[j] * VectorY[j];
	}

	return fSumOfXYProduct;
}

//Function to implement task parallelism to compute the four individual components
bool ParallelTasksLeastSquareFitting(const float* VectorX, const float* VectorY, int n, float* m, float* b)
{
	float fSumOfX = 0.0f;
	float fSumOfY = 0.0f;
	float fSumOfXSquare = 0.0f;
	float fSumOfXYProduct = 0.0f;

	//Write code to create four tasks, with each task computing an individual component
	//First spawn four threads and create four tasks
#pragma omp parallel num_threads(4)
	{
#pragma omp single nowait
		{
#pragma omp task untied
			{
				fSumOfX = Sum(VectorX, n);
			}
		}
#pragma omp single nowait
		{
#pragma omp task untied
			{
				fSumOfY = Sum(VectorY, n);
			}
		}
#pragma omp single nowait
		{
#pragma omp task untied
			{
				fSumOfXSquare = SumOfXSquare(VectorX, n);
			}
		}
#pragma omp single
		{
#pragma omp task untied
			{
				fSumOfXYProduct = SumOfXYProduct(VectorX, VectorY, n);
			}
		}
		//Write code to wait for the four tasks to complete their job
#pragma omp barrier
	}

	//Write code to compute m and b of the simple linear regression
	float denom = n * fSumOfXSquare - (fSumOfX * fSumOfX);

	if (Epsilon >= fabs(denom)) //Avoid divide by zero
	{
		return false;
	}

	*m = ((n * fSumOfXYProduct) - (fSumOfX * fSumOfY)) / denom;
	*b = ((fSumOfXSquare * fSumOfY) - (fSumOfX * fSumOfXYProduct)) / denom;

	return true;
}