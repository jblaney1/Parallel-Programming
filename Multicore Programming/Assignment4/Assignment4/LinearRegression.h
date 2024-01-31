#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <string>
using namespace std;

#define RANGE_MIN -255.0
#define RANGE_MAX 255.0

//Iteration Count
constexpr int NUM_IT = 20;

extern double Epsilon;

void GenerateRandomDataSamples(float* VectorX, float* VectorY, int n);

//Function to implement Simple Linear Regression Sequentially
bool ComputeLeastSquareFittingSeq(const float* VectorX, const float* VectorY, int n, float* m, float* b);

//Function to implement Simple Linear Regression in Parallel using only Parallel for with clauses
bool ParallelLeastSquareFitting(const float* VectorX, const float* VectorY, int n, float* m, float* b);

//Functions to be used for Task Parallelism
float Sum(const float* VectorX, int n); // Calculating the sum of the vector function
float SumOfXSquare(const float* VectorX, int n); //function that calculates the sum of the square of the vector elements
float SumOfXYProduct(const float* VectorX, const float* VectorY, int n); // function for calculating the sum of the product of the vector

//Function to implement task parallelism to compute the four individual components
bool ParallelTasksLeastSquareFitting(const float* VectorX, const float* VectorY, int n, float* m, float* b);

