#include "MatrixMult.h"

void getDim(string name, Dim* dim)
{
	cout << "Enter dimensions of Matrix: " << name << endl;
	cout << " Rows: ";
	cin >> dim->Rows;
	cout << " Cols: ";
	cin >> dim->Cols;
}

bool VerifyDim(Dim dimA, Dim dimB, Dim* dimC)
{
	if (dimA.Cols == dimB.Rows)
	{
		//Assigning dimensions to the product matrix C
		dimC->Rows = dimA.Rows;
		dimC->Cols = dimB.Cols;
		return true;
	}
	return false;
}

void InitializeArray(double** array, Dim dim)
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < dim.Rows; i++)
	{
		for (int j = 0; j < dim.Cols; j++)
		{
			array[i][j] = ((double)rand() / ((double)RAND_MAX + 1.0f) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
	}
}

void DisplayArray(string text, double** array, Dim dim)
{
	if (dim.Rows <= 6 && dim.Cols <= 6)
	{
		cout << text << endl;
		for (int i = 0; i < dim.Rows; i++)
		{
			for (int j = 0; j < dim.Cols; j++)
			{
				cout << setprecision(4) << array[i][j] << '\t';
			}
			cout << endl;
		}
	}
}

void CopyArray(double** arrayData, double** array, Dim dim)
{
	for (int i = 0; i < dim.Rows; i++)
	{
		for (int j = 0; j < dim.Cols; j++)
		{
			arrayData[i][j] = array[i][j];
			array[i][j] = 0.0f;
		}
	}
}

void Verify(double** arrayData, double** array, Dim dim)
{
	double tol = 1.0E-06;
	for (int i = 0; i < dim.Rows; i++)
	{
		for (int j = 0; j < dim.Cols; j++)
		{
			if (fabs(arrayData[i][j] - array[i][j]) > tol)
			{
				cout << "Error Value: "<< fabs(arrayData[i][j] - array[i][j]) <<" i = " << i << " j = " << j << " " << arrayData[i][j] << "," << array[i][j] << endl;
				return;
			}
		}
	}
}