#include "MatrixMult.h"

void MatrixMultSeq(double** arrayA, double** arrayB, double** arrayC, Dim A, Dim B)
{
	for (int i = 0; i < A.Rows; i++)
	{
		for (int j = 0; j < B.Cols; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < A.Cols; k++)
			{
				sum += arrayA[i][k] * arrayB[k][j];
			}
			arrayC[i][j] = sum;
		}
	}
}

/* MatrixMultSeqTrans()
 * This function is essentially the same as MatrixMultSeq
 * except that it uses a transposed matrix in place of "arrayB".
 * To accomodate this the iteration variables "j" and "k" are swapped
 * in the inner most for loop where "arrayA" and "arrayB" are multiplied.
 * This allows the algorithm to iterate along rows of "arrayA" and
 * "arrayB" which should require fewer fetches from RAM than iterating
 * along columns.
*/
void MatrixMultSeqTrans(double** arrayA, double** arrayB, double** arrayC, Dim A, Dim B)
{
	for (int i = 0; i < A.Rows; i++)
	{
		for (int j = 0; j < B.Cols; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < A.Cols; k++)
			{
				sum += arrayA[i][k] * arrayB[j][k];
			}
			arrayC[i][j] = sum;
		}
	}
}