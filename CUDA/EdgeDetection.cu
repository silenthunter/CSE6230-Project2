#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>
#include "bmpLoader.cu"

//Program constants
const int cuda_threadX = 32;
const int cuda_threadY = 16;
const int gaussSize = 3;

__device__ float e = 2.71828f;
__device__ float PI = 3.14f;
__device__ float dev = 1.4f;
__device__ float *GaussMat;
__device__ float *image;
__device__ int width;

__device__ float GaussianBlur(int x, int y)
{
	float p1 = 1.0f / (2 * PI * powf(dev, 2));
	float p2 = p1 * powf(e, -(powf(x, 2) + powf(y, 2)) / (2 * powf(dev, 2)));
	return p2;
}

__global__ void MakeBW(float* d_image, float* d_bw, int d_width)
{
	//TODO: Make local shared array to maximize speed
	int dest = (threadIdx.x + threadIdx.y * blockDim.x) + 
		(blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y);
	int idx = dest * 3;//3 color values

	//http://www.bobpowell.net/grayscale.htm
	float val = d_image[idx] *.3 + d_image[idx + 1] * .59 + d_image[idx + 2] * .11;

	d_bw[dest] = val;
	
	//Have thread 0 assign the global variables
	if(dest == 0)
	{
		image = d_bw;
		width = d_width;
	}
}

__global__ void ComputeGuassian(float* d_gaussMat, int width)
{
	GaussMat = d_gaussMat;
	int idx = threadIdx.x + threadIdx.y * width;
	d_gaussMat[idx] = GaussianBlur(threadIdx.x - width / 2, threadIdx.y - width / 2);
}

__global__ void GaussianBlur(int width)
{
}

int main(int argc, char* argv[])
{
	//Have the card compute the matrix for the gaussion blur
	float* d_gaussMat;
	float* h_gaussMat = (float*)malloc(sizeof(float) * gaussSize * gaussSize);
	cudaMalloc((void**)&d_gaussMat, sizeof(float) * gaussSize * gaussSize);

	dim3 blockSize(gaussSize, gaussSize);
	dim3 gridSize(1,1);
	ComputeGuassian<<<gridSize, blockSize>>>(d_gaussMat, gaussSize);
	//cudaThreadSynchronize();
	cudaMemcpy(h_gaussMat, d_gaussMat, sizeof(float) * gaussSize * gaussSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < gaussSize * gaussSize; i++)
	{
		printf("%f ", h_gaussMat[i]);
		if(i % 3 == 2) printf("\n");
	}
	
	//Load an image
	BITMAPINFOHEADER bmpInfo;
	BITMAPFILEHEADER bitmapHeader;
	unsigned char* cImage = LoadBitmapFile("../Images/sample.bmp", &bitmapHeader, &bmpInfo);
	int imageSize = bmpInfo.biSizeImage;

	float *h_image, *d_image, *d_bw, *h_bw;
	h_image = (float*)malloc(sizeof(float) * imageSize);
	for(int i = 0; i < imageSize; i++)
		h_image[i] = (float)cImage[i] / 256.0f;
	
	cudaMalloc((void**)&d_image, sizeof(float) * imageSize);
	cudaMalloc((void**)&d_bw, sizeof(float) * imageSize / 3);
	cudaMemcpy(d_image, h_image, sizeof(float) * imageSize, cudaMemcpyHostToDevice);
	
	//Convert the image into black and white
	blockSize = dim3(cuda_threadX, cuda_threadY);
	gridSize = dim3(bmpInfo.biWidth / cuda_threadX, bmpInfo.biHeight / cuda_threadY);
	MakeBW<<<gridSize, blockSize>>>(d_image, d_bw, bmpInfo.biWidth);
	h_bw = (float*)malloc(sizeof(float) * imageSize);
	cudaThreadSynchronize();
	cudaMemcpy(h_bw, d_bw, sizeof(float) * imageSize / 3, cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < imageSize; i++)
	{
		cImage[i] = (unsigned char)(h_bw[i / 3] * 256);
	}
	
	SaveBitmapFile("sample.bmp", cImage, &bitmapHeader, &bmpInfo);
}
