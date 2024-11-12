/*
* Author: Josh Blaney
* Date: 11/04/2024
* 
* Description:
* Implements the functionality for adding, transposing, multiplying, 
* subtracting, and printing matrices on the CPU. The functionas are
* organized in alphabetical order (except transpose) and all functions
* include timing functionality.
*/

#include <chrono>
#include <iomanip>
#include <iostream>


void add_matrices(float* a, float* b, float* c, int rows, int cols, bool time) {
	/*
	* Adds the elements of two matrices (a and b) of known sizes (rows and cols)
	* and stores the result in a different matrix (c). Optionally, this function 
	* can provide information about the execution time (time).
	*/

	int row;
	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::high_resolution_clock::duration elapsed;

	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < rows; i++) {
		row = i * cols;
		for (int j = 0; j < cols; j++) {
			c[row + j] = a[row + j] + b[row + j];
		}
	}

	end = std::chrono::high_resolution_clock::now();

	if (time) {
		elapsed = end - start;
		std::cout << "Time elapsed adding matrices (CPU): "  << elapsed.count() / 1E6 << "ms\n\n";
	}
}


void transpose_matrix(float* src, float* dst, int rows, int cols, bool time) {
	/*
	* Transposes a provided matrix (src) of known size (rows and cols)
	* into a different matrix (dst). Optionally, this function 
	* can provide information about the execution time (time).
	*/

	int row;

	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::high_resolution_clock::duration elapsed;

	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < rows; i++) {
		row = i * cols;
		for (int j = 0; j < cols; j++) {
			dst[rows * j + i] = src[row + j];
		}
	}

	end = std::chrono::high_resolution_clock::now();

	if (time) {
		elapsed = end - start;
		std::cout << "Time elapsed transposing matrices (CPU): " << elapsed.count() / 1E6 << "ms\n";
	}
}


void mult_matrices(float* a, float* b, float* c, int rowsa, int colsa, int colsb, bool time) {
	/*
	* Performs matrix multiplication between two matrices (a and b) leveraging
	* spacial locality in cache by transposing the second matrix (b) and then 
	* reading both matrices in row major order. It is assumed that the number
	* of columns in the first matrix (a) is equal to the number of rows in the 
	* second matrix (b). The result is stored in a different matrix (c) and the
	* timing information for this function can also be printed (time).
	*/

	int rowa, rowb, rowc;
	float sum;
	float* transpose_b = new float[colsa * colsb]();
	transpose_matrix(b, transpose_b, colsa, colsb, time);

	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::high_resolution_clock::duration elapsed;

	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < rowsa; i++) {
		rowa = i * colsa;
		rowc = i * colsb;
		for (int j = 0; j < colsb; j++) {
			rowb = j * colsa;
			sum = 0.0f;
			for (int k = 0; k < colsa; k++) {
				sum += (a[rowa + k] * transpose_b[rowb + k]);
			}
			c[rowc + j] = sum;
		}
	}

	end = std::chrono::high_resolution_clock::now();

	if (time) {
		elapsed = end - start;
		std::cout << "Time elapsed multiplying matrices (CPU): " << elapsed.count() / 1E6 << "ms\n";
	}
}

void print_matrix(float* a, int rows, int cols, char name) {
	/*
	* Prints a limited number of elements from a matrix (a)
	* to the terminal. If the size of the matrix is less than
	* 4 square the entire matrix will be printed. Otherwise, the first
	* 4 square elements will be printed. The output can be named
	* (name) for easy differentiation during execution.
	*/

	int row;
	int p_rows = (rows <= 4) ? rows : 4;
	int p_cols = (cols <= 4) ? cols : 4;

	std::cout << "Printing Matrix " << name << "\n";

	for (int i = 0; i < p_rows; i++) {
		row = i * cols;
		std::cout << "\t";
		for (int j = 0; j < p_cols; j++) {
			std::cout << std::setw(8) << a[row + j] << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}


void sub_matrices(float* a, float* b, float* c, int rows, int cols, bool time) {
	/*
	* Subtracts the elements of two matrices (a and b) of known sizes (rows and cols)
	* and stores the result in a different matrix (c). Optionally, this function
	* can provide information about the execution time (time).
	*/

	int row;

	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::high_resolution_clock::duration elapsed;

	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < rows; i++) {
		row = i * cols;
		for (int j = 0; j < cols; j++) {
			c[row + j] = a[row + j] - b[row + j];
		}
	}

	end = std::chrono::high_resolution_clock::now();

	if (time) {
		elapsed = end - start;
		std::cout << "Time elapsed subtracting matrices (CPU): " << elapsed.count() / 1E6 << "ms\n";
	}
}