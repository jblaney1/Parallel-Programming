#include "Sum.h"

void InitializeArray(double* array, int nSize)
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < nSize; i++)
	{
		array[i] = ((double)rand() / ((double)RAND_MAX + 1.0f) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
	}
}