#include "MatrixMult.h"

int main()
{
	Dim dimA, dimB,dimC;
	getDim("A", &dimA);
	getDim("B", &dimB);
	if (!VerifyDim(dimA, dimB, &dimC))
	{
		cout << "The inner matrix dimensions of A and B are not equal for matrix multiplication" << endl;
		return 0;
	}

	double** A = new double* [dimA.Rows];
	double** B = new double* [dimB.Rows];
	double** C = new double* [dimC.Rows];
	double** CRef = new double* [dimC.Rows];

	for (int i = 0; i < dimA.Rows; i++)
	{
		A[i] = new double[dimA.Cols];
	}
	for (int i = 0; i < dimB.Rows; i++)
	{
		B[i] = new double[dimB.Cols];
	}
	for (int i = 0; i < dimC.Rows; i++)
	{
		C[i] = new double[dimC.Cols];
		CRef[i] = new double[dimC.Cols];
	}

	//Initialize Matrices with Random data
	InitializeArray(A, dimA);
	InitializeArray(B, dimB);

	//Sequential Operation
	chrono::time_point<std::chrono::system_clock> start, end;
	double AverageTime{ 0.0f }; //Average Computation Time

	for (int i = 0; i < NUM_IT; i++)
	{
		start = std::chrono::system_clock::now();
		MatrixMultSeq(A, B, C,dimA, dimB);
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		AverageTime += elasped_seconds.count();
	}
	AverageTime /= NUM_IT;
	AverageTime *= 1000.0f; //Converting seconds to milliseconds

	//Display Average Computation Time and the results
	cout << "Matrix Multiplication: Sequential Execution: Average Computation Time: " << AverageTime << " msecs" << endl;
	DisplayArray("Sequential: Matrix Multiplication", C, dimC);
	//Copying results to reference array
	CopyArray(CRef, C, dimC);

	/*
	 * Josh Blaney EE5885 Multicore Programming
	 *
	 * Call student written functions and test their validity.
	 * 
	 * Functions:
	 * MatrixTranspose(double** arrayA, double** arrayC, Dim A)
	 * MatrixMultParallel(double** arrayA, double** arrayB, double** arrayC, Dim A, Dim B)
	 * MatrixMultSeqTrans(double** arrayA, double** arrayB, double** arrayC, Dim A, Dim B)
	 * MatrixMultParallelTrans(double** arrayA, double** arrayB, double** arrayC, Dim A, Dim B)
	 * 
	*/

	// A new matrix to hold the transpose of B
	double** D = new double* [dimB.Cols];

	// Initialize the second dimension of D
	for (int i = 0; i < dimB.Cols; i++) {
		D[i] = new double[dimB.Rows];
	}

	// Transpose B and store the result in D
	MatrixTranspose(B, D, dimB);

	//Parallel Execution - Matrix Multiplication
	AverageTime = 0.0f; //Average Computation Time
	for (int i = 0; i < NUM_IT; i++)
	{
		start = std::chrono::system_clock::now();
		MatrixMultParallel(A, B, C, dimA, dimB);
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		AverageTime += elasped_seconds.count();
	}
	AverageTime /= NUM_IT;
	AverageTime *= 1000.0f; //Converting seconds to milliseconds

	//Display Average Computation Time and the results
	cout << "Matrix Multiplication: Parallel Execution: Average Computation Time: " << AverageTime << " msecs" << endl;
	DisplayArray("Parallel: Matrix Multiplication", C, dimC);
	Verify(CRef, C, dimC);

	// Sequential matrix multiplication utilizing a transposed matrix
	AverageTime = 0.0f; //Average Computation Time
	for (int i = 0; i < NUM_IT; i++)
	{
		start = std::chrono::system_clock::now();
		MatrixMultSeqTrans(A, D, C, dimA, dimB);
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		AverageTime += elasped_seconds.count();
	}
	AverageTime /= NUM_IT;
	AverageTime *= 1000.0f; //Converting seconds to milliseconds

	cout << "Matrix Multiplication: Sequential Execution with Transpose: Average Computation Time: " << AverageTime << " msecs" << endl;
	DisplayArray("Sequential: Matrix Multiplication with Transpose", C, dimC);
	Verify(CRef, C, dimC);

	// Parallel matrix multiplication utilizing a transposed matrix
	AverageTime = 0.0f; //Average Computation Time
	for (int i = 0; i < NUM_IT; i++)
	{
		start = std::chrono::system_clock::now();
		MatrixMultParallelTrans(A, D, C, dimA, dimB);
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		AverageTime += elasped_seconds.count();
	}
	AverageTime /= NUM_IT;
	AverageTime *= 1000.0f; //Converting seconds to milliseconds

	//Display Average Computation Time and the results
	cout << "Matrix Multiplication: Parallel Execution with Transpose: Average Computation Time: " << AverageTime << " msecs" << endl;
	DisplayArray("Parallel: Matrix Multiplication with Transpose", C, dimC);
	Verify(CRef, C, dimC);

	delete[] A;
	delete[] B;
	delete[] C;
	delete[] D;
	delete[] CRef;
	return 0;
}