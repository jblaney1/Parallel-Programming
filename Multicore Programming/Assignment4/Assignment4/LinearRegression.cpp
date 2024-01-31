#include "LinearRegression.h"
double Epsilon = 1.0e-12;

int main()
{
	int n; //Number of data samples
	cout << "Enter the number of data samples: ";
	cin >> n;

	float* X, // X is the vector to store the samples of the independent variable
		* Y; // Y is the vector to store the corresponding dependent variable.

	X = new float[n];
	Y = new float[n];

	GenerateRandomDataSamples(X, Y, n);

	float Slope;
	float YIntercept;
	bool no_error;

	//Sequential Implementation
	chrono::time_point<std::chrono::system_clock> start, end;
	double AverageTime{ 0.0f }; //Average Computation Time

	for (int i = 0; i < NUM_IT; i++)
	{
		Slope = 0.0f;
		YIntercept = 0.0f;
		start = std::chrono::system_clock::now();
		no_error = ComputeLeastSquareFittingSeq(X, Y, n, &Slope, &YIntercept);
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		if (no_error)
		{
			AverageTime += elasped_seconds.count();
		}
		else {
			AverageTime += 0.0f;
		}
	}
	AverageTime /= NUM_IT;
	AverageTime *= 1000.0f; //Converting seconds to milliseconds
	//Display Average Computation Time and the results
	cout << "Simple Linear Regression: Sequential Execution" << endl;
	cout << "\tAverage Computation Time: " << AverageTime << " msecs" << endl;
	cout << "\tm: " << Slope << "\tb: " << YIntercept<<endl;

	//Parallel Implementation using only Parallel for directives with appropriate clauses
	AverageTime = 0.0f; //Average Computation Time
	for (int i = 0; i < NUM_IT; i++)
	{
		Slope = 0.0f;
		YIntercept = 0.0f;
		start = std::chrono::system_clock::now();
		no_error = ParallelLeastSquareFitting(X, Y, n, &Slope, &YIntercept);
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		if (no_error)
		{
			AverageTime += elasped_seconds.count();
		}
		else {
			cout << "Error" << endl;
			AverageTime += 0.0f;
		}
	}
	AverageTime /= NUM_IT;
	AverageTime *= 1000.0f; //Converting seconds to milliseconds
	//Display Average Computation Time and the results
	cout << "Simple Linear Regression: Parallel Execution" << endl;
	cout << "\tAverage Computation Time: " << AverageTime << " msecs" << endl;
	cout << "\tm: " << Slope << "\tb: " << YIntercept << endl;

	//Parallel Implementation using Tasks
	AverageTime = 0.0f; //Average Computation Time
	for (int i = 0; i < NUM_IT; i++)
	{
		Slope = 0.0f;
		YIntercept = 0.0f;
		start = std::chrono::system_clock::now();
		no_error = ParallelTasksLeastSquareFitting(X, Y, n, &Slope, &YIntercept);
		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elasped_seconds = end - start;
		if (no_error)
		{
			AverageTime += elasped_seconds.count();
		}
		else {
			AverageTime += 0.0f;
		}
	}
	AverageTime /= NUM_IT;
	AverageTime *= 1000.0f; //Converting seconds to milliseconds
	//Display Average Computation Time and the results
	cout << "Simple Linear Regression: Parallel Task Execution" << endl;
	cout << "\tAverage Computation Time: " << AverageTime << " msecs" << endl;
	cout << "\tm: " << Slope << "\tb: " << YIntercept << endl;
	

	delete[] X;
	delete[] Y;
	return 0;
}