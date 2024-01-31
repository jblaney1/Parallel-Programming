#include <iostream>
#include <chrono>

using namespace std;

#define NTIMES 100
#define STREAM_ARRAY_SIZE 80000000

static double a[STREAM_ARRAY_SIZE], b[STREAM_ARRAY_SIZE], c[STREAM_ARRAY_SIZE];

int main() {

	chrono::time_point<std::chrono::system_clock> start, end;

	double scalar = 3.0;

	for (int i = 0; i < STREAM_ARRAY_SIZE; i++) { a[i] = 1.0; b[i] = 2.0; }

	for (int k = 0; k < NTIMES; k++) {
		start = std::chrono::system_clock::now();

		for (int i = 0; i < STREAM_ARRAY_SIZE; i++) { c[i] = a[i] + scalar * b[i]; }

		end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed_seconds = end - start;

		cout << "Iteration: " << (k + 1) << "\tElapsed time: " << elapsed_seconds.count() << endl;

		c[1] = c[2];
	}

	return 0;
}