#include "ParallelHistogram.h"

int main()
{
	double cpuComputeTime = 0.0f;
	double gpuComputeTime = 0.0f;
	
	cout << "Loading Images" << endl;
	//Load a RGB images
	CImg<unsigned char> imgRGB = CImg<>("D:\\blaney\\GPGPU\\Images\\lena.jpg");
	//CImg<unsigned char> imgRGB = CImg<>("D:\\blaney\\GPGPU\\Images\\cat.png");
	//CImg<unsigned char> imgRGB = CImg<>("D:\\blaney\\GPGPU\\Images\\scope.jpg");
	//CImg<unsigned char> imgRGB = CImg<>("D:\\blaney\\GPGPU\\Images\\mountain-landscape-reflection.jpg");
	
	cout << "Image Height: " << imgRGB.height()<<" pixels"<<endl;
	cout << "Image Width: " << imgRGB.width() << " pixels" << endl;
	
	//Store RGB image size in bytes
	unsigned int rgbSize = imgRGB.width() * imgRGB.height() * 3*sizeof(unsigned char);

	//Initialize a pointer to the RGB image data stored by CImg
	unsigned char* ptrRGB = imgRGB.data();

	//Create an empty image with a single channel - GrayScale
	CImg<unsigned char> imgGrayScale(imgRGB.width(), imgRGB.height());
	//Store GrayScale image size in bytes
	unsigned int graySize = imgRGB.width() * imgRGB.height() * 1 * sizeof(unsigned char);

	//CPU Version
	cpu__RGBtoGrayScale(imgRGB, imgGrayScale);
	//Initialize a pointer to the GrayScale image data stored by CImg
	unsigned char* ptrGray = imgGrayScale.data();
	CImgDisplay dispGray1(imgGrayScale, "GrayScale Image - CPU - Ver 0");
	
	//Histogram Code
	
	//Define an array to store the histogram data of CPU and GPU
	unsigned int* Histogram = new unsigned int[BINS];
	unsigned int* gpuHistogram = new unsigned int[BINS];
	//CPU function to compute the histrogram
	cpu_Histogram(ptrGray, BINS, Histogram, imgRGB.height(), imgRGB.width());

	//Initialize the GPU Histogram Data
	for (unsigned int i = 0; i < BINS; i++)
	{
		gpuHistogram[i] = 0;
	}
	//Naive Histogram GPU Helper Function
	gpu_NaiveHistogramHelper(ptrGray,
		gpuHistogram, 
		graySize,
		imgRGB.height(), 
		imgRGB.width(), 
		Histogram);

	//Boolean Variable to commit private histogram copies to global histogram
	/*If the Boolean variable is set to FALSE, the operation of 
	committing private histogram copies to global histogram is
	completed by the CPU instead of the GPU*/
	bool CommitGPU = FALSE;

	cout << "Executing Global Private Copy Histogram" << endl;
	//Global Private Copy Histrogram Function
	gpu_PCOnGlobalHistogramHelper(ptrGray,
		gpuHistogram,
		graySize,
		imgRGB.height(),
		imgRGB.width(),
		Histogram,
		CommitGPU);

	cout << "Executing Shared Private Copy Histogram " << endl;
	//Shared Private Copy Histrogram Function
	gpu_PCOnSharedHistogramHelper(ptrGray,
		gpuHistogram,
		graySize,
		imgRGB.height(),
		imgRGB.width(),
		Histogram,
		CommitGPU);

	delete[] Histogram;
	delete[] gpuHistogram;
	return 0;
}