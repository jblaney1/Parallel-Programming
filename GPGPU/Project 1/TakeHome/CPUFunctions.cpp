#include "ParallelHistogram.h"

void cpu__RGBtoGrayScale(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg)
{
	int height = rgbImg.height();
	int width = rgbImg.width();
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			grayImg(x, y) = 0.21f * rgbImg(x, y, 0, 0) + 0.71f * rgbImg(x, y, 0, 1) + 0.07f * rgbImg(x, y, 0, 2);
		}
	}
}

//A function construct the histogram of a grayscale image
void cpu_Histogram(unsigned char* in, unsigned int bins, unsigned int* hist, unsigned int h, unsigned int w)
{
	chrono::time_point<high_resolution_clock> start, end;
	double computeTime{};
	unsigned int N = h * w;
	//Initialize the histogram counts
	for (unsigned int i = 0; i < bins; i++)
	{
		hist[i] = 0;
	}
	start = high_resolution_clock::now();
	for (unsigned int i = 0; i < N; i++)
	{
		hist[in[i]]++;
	}
	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "Sequential Histogram: CPU Execution time: " << computeTime << " usecs" << endl;
}

//A function verify Histogram Computation
void Verify(unsigned int* cpu_histogram, unsigned int* gpu_histogram, unsigned int bins)
{
	for (unsigned int i = 0; i < bins; i++)
	{
		if (cpu_histogram[i] != gpu_histogram[i])
		{
			cout << "Error in bin: " << i << " CPU Bin Value:" << cpu_histogram[i] << " GPU Bin Value:" << gpu_histogram[i] << endl;
			return;
		}
	}
}

//A Function to write the histogram data to a file
void WriteHistograms(string FileName, unsigned int* cpu_histogram, unsigned int* gpu_histogram, unsigned int bins)
{
	fstream outfile;
	outfile.open(FileName, std::ios_base::out);
	for (unsigned int i = 0; i < bins; i++)
	{
		outfile << i << "," << cpu_histogram[i] << "," << gpu_histogram[i] << endl;
	}
	outfile.close();
}

//CPU Function to commit private histogram copies to global histogram - Unoptimized
void HistogramCommitNaive(unsigned int* hist_private_copies, unsigned int* hist_global, unsigned int bins,unsigned int copies)
{
	chrono::time_point<high_resolution_clock> start, end;
	double computeTime{};
	start = high_resolution_clock::now();
	/*At a time, a single bin across all private copies are summed and then
	committed to the bin on the global histogram*/
	for (unsigned int bin = 0; bin < bins; bin++)
	{
		unsigned int temp = 0;
		for (unsigned int binCopy = 0; binCopy < copies; binCopy++)
		{
			temp += hist_private_copies[bin + (binCopy*bins)];
		}
		hist_global[bin] = temp;
	}
	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "CPU UnOptimized: Global Commit Execution time: " << computeTime << " usecs" << endl;
}

//CPU Function to commit private histogram copies to global histogram - Optimized
void HistogramCommitOptimized(unsigned int* hist_private_copies,
	unsigned int* hist_global,
	unsigned int bins,
	unsigned int copies)
{
	chrono::time_point<high_resolution_clock> start, end;
	double computeTime{};
	start = high_resolution_clock::now();
	/*Sequentially private histogram copies are
	committed to the bin on the global histogram*/
	for (unsigned int binCopy = 0; binCopy < copies; binCopy ++)
	{
		unsigned int idx = binCopy * bins;
		for (unsigned int bin = 0; bin < bins; bin++)
		{
			hist_global[bin] += hist_private_copies[bin + idx];
		}
	}
	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "CPU Optimized: Global Commit Execution time: " << computeTime << " usecs" << endl;
}

