/*
* Author: Josh Blaney
* Date: 11/04/2024
* 
* Description:
* Implements the functionality for adding, multiplying (dot product), 
* printing, and subtracting vectors on the CPU. The functions are 
* organized in alphabetical order and all functions include timing
* functionality.
*/


#include <chrono>
#include <iomanip>
#include <iostream>


void add_vectors(float* a, float* b, float* c, int length, bool time) {
	/*
	* Adds the elements of two vectors (a and b) of known length 
	* (length) and stores the results in a different vector (c). 
	* Optionally, this function can provide information about the
	* execution time (time).
	*/

	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::high_resolution_clock::duration elapsed;

	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < length; i++) {
		c[i] = a[i] + b[i];
	}

	end = std::chrono::high_resolution_clock::now();

	if (time) {
		elapsed = end - start;
		std::cout << "Time elapsed adding vectors: " << elapsed.count() / 1E6 << "ms\n";
	}
}


float dot_product(float* a, float* b, int length, bool time) {
	/*
	* Computes the dot product between to vectors (a and b) of
	* known length (lenght) and returns the result as a float (c).
	* Optionally, this function can provide information about
	* the execution time (time).
	*/

	float c = 0.0f;
	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::high_resolution_clock::duration elapsed;

	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < length; i++) {
		c = c + (a[i] * b[i]);
	}

	end = std::chrono::high_resolution_clock::now();

	if (time) {
		elapsed = end - start;
		std::cout << "Time elapsed performing dot product: " << elapsed.count() / 1E6 << "ms\n";
	}

	return c;
}


void print_vector(float* a, int length, char name) {
	/*
	* Prints a limited number of elements from an array (a)
	* to the terminal. If the length of the array is less than
	* 4 the entire array will be printed. Otherwise, the first
	* 4 elements will be printed. The output can be named 
	* (name) for easy differentiation during execution.
	*/

	int p_length = (length < 4) ? length : 4;

	std::cout << "Printing Vector " << name << "\n\t";

	for (int i = 0; i < p_length; i++) {
		std::cout << std::setw(8) << a[i] << "\t";
	}

	std::cout << "\n\n";
}


void sub_vectors(float* a, float* b, float* c, int length, bool time) {
	/*
	* Subtracts the elements of two vectors (a and b) of known length 
	* (length) and stores the results in a different vector (c).
	* Optionally, this function can provide information about the
	* execution time (time).
	*/

	std::chrono::high_resolution_clock::time_point start, end;
	std::chrono::high_resolution_clock::duration elapsed;

	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < length; i++) {
		c[i] = a[i] - b[i];
	}

	end = std::chrono::high_resolution_clock::now();

	if (time) {
		elapsed = end - start;
		std::cout << "Time elapsed subtracting vectors: " << elapsed.count() / 1E6 << "ms\n";
	}
}