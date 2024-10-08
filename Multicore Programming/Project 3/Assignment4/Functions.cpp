#include "LinearRegression.h"

void GenerateRandomDataSamples(float* VectorX, float* VectorY, int n)
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < n; i++)
	{
		VectorX[i] = ((float)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		VectorY[i] = ((float)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
	}
}