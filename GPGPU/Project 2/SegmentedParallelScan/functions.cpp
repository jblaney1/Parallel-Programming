#include "ParallelSegmentedScan.h"

void OnInitializeInputData(float* vectorTemp, int SIZE)
{
	const float range_from = -1.00f;
	const float range_to = 1.00f;
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::uniform_real_distribution<float>  distr(range_from, range_to);
	for (int i = 0; i < SIZE; i++)
	{
		//vectorTemp[i] = 1.0f; For debugging purpose all elements of input vector can be set to 1.0
		vectorTemp[i] = distr(generator);
	}
}

void CopyInputData(float* vectorTemp, float* ref,int SIZE)
{
	for (int i = 0; i < SIZE; i++)
	{
		vectorTemp[i] = ref[i];
	}
}

void ZeroData(float* in, int SIZE)
{
	for (int i = 0; i < SIZE; i++)
	{
		in[i] = 0.0f;
	}
}

void PrintVectors(float* vector, int SIZE)
{
	for (int i = 0; i < 4; i++)
	{
		cout << vector[i] << '\t';
	}
	cout << ", . . .\t";
	for (int i = SIZE - 4; i < SIZE; i++)
	{
		cout << vector[i] << '\t';
	}
	cout << endl;
}

void Verify(float* ref, float* in, int SIZE)
{
	float fTolerance = 1.0E-2f;
	for (int i = 0; i < SIZE; i++)
	{
		if (fabs(ref[i] - in[i]) > fTolerance)
		{
			cout << "Error" << endl;
			cout << "\vectRef[" << (i + 1) << "] = " << ref[i] << endl;
			cout << "\vectGPU[" << (i + 1) << "] = " << in[i] << endl;
			return;
		}
	}
}