#include "LinearRegression.h"

bool ComputeLeastSquareFittingSeq(const float* VectorX, const float* VectorY, int n, float* m, float* b)
{
	float fSumOfX = 0.0f;
	float fSumOfY = 0.0f;
	float fSumOfXSquare = 0.0f;
	float fSumOfXYProduct = 0.0f;

	//For loop to compute the individual components of m and b
	for (int i = 0; i < n; i++)
	{
		fSumOfX += VectorX[i];
		fSumOfXSquare += VectorX[i] * VectorX[i];
		fSumOfXYProduct += VectorX[i] * VectorY[i];
		fSumOfY += VectorY[i];
	}

	//Computing m and b of the simple linear regression
	float denom = n * fSumOfXSquare - (fSumOfX * fSumOfX);

	if (Epsilon >= fabs(denom)) //Avoid divide by zero
	{
		return false;
	}
	
	*m = ((n * fSumOfXYProduct) - (fSumOfX * fSumOfY)) / denom;
	*b = ((fSumOfXSquare * fSumOfY) - (fSumOfX * fSumOfXYProduct)) / denom;
	return true;
}