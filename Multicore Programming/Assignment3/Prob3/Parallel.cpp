#include "MatrixMult.h"

/* MatrixTranspose()
 * A simple function which transposes the given matrix 
 * "arrayA" and stores it in the second give matrix 
 * "arrayC" using the relationship A^T[j][i] = A[i][j].
*/
void MatrixTranspose(double** arrayA, double** arrayC, Dim A){
	for (int i = 0; i < A.Rows; i++) {
		for (int j = 0; j < A.Cols; j++) {
			arrayC[j][i] = arrayA[i][j];
		}
	}
}

/* MatrixMultParallel()
 * The two outer parallelized for loops do not need
 * any clauses as they do not actually handle any data.
 * The innermost parallelized for loop needs to make
 * "sum" private so that multiple threads do not write
 * it. The easiest way to do this is with a <reduction>
 * clause.
*/
void MatrixMultParallel(double** arrayA, double** arrayB, double** arrayC, Dim A, Dim B)
{
#pragma omp parallel for
	for (int i = 0; i < A.Rows; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < B.Cols; j++)
		{
			// The difference between the class code and the assignment code is this variable
			// "sum" which is being used to store the matrix value during multiplication.
			float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
			for (int k = 0; k < A.Cols; k++)
			{
				sum += arrayA[i][k] * arrayB[k][j];
			}
			arrayC[i][j] = sum;
		}
	}
}

/* MatrixMultParallelTrans()
 * This function is essentially the same as MatrixMultParallel
 * except that it uses a transposed matrix in place of "arrayB".
 * To accomodate this the iteration variables "j" and "k" are swapped
 * in the inner most for loop where "arrayA" and "arrayB" are multiplied.
 * This allows the algorithm to iterate along rows of "arrayA" and 
 * "arrayB" which should require fewer fetches from RAM than iterating
 * along columns.
*/
void MatrixMultParallelTrans(double** arrayA, double** arrayB, double** arrayC, Dim A, Dim B)
{
#pragma omp parallel for
	for (int i = 0; i < A.Rows; i++)
	{
#pragma omp parallel for
		for (int j = 0; j < B.Cols; j++)
		{
			float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
			for (int k = 0; k < A.Cols; k++)
			{
				sum += arrayA[i][k] * arrayB[j][k];
			}
			arrayC[i][j] = sum;
		}
	}
}