#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>

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
	width = d_width;
	image = d_bw;
	//TODO: Make local shared array to maximize speed
	int dest = threadIdx.x + threadIdx.y  * width;
	int idx = dest * 3;//3 color values

	//http://www.bobpowell.net/grayscale.htm
	float val = d_image[idx] *.3 + d_image[idx + 1] * .59 + d_image[idx + 2] * .11;

	d_bw[dest] = val;
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
	int gaussSize = 3;
	float* d_gaussMat;
	float* h_gaussMat = (float*)malloc(sizeof(float) * gaussSize * gaussSize);
	cudaMalloc((void**)&d_gaussMat, sizeof(float) * gaussSize * gaussSize);

	dim3 blockSize(gaussSize, gaussSize);
	dim3 gridSize(1,1);
	ComputeGuassian<<<gridSize, blockSize>>>(d_gaussMat, gaussSize);
	cudaThreadSynchronize();
	cudaMemcpy(h_gaussMat, d_gaussMat, sizeof(float) * gaussSize * gaussSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < gaussSize * gaussSize; i++)
	{
		printf("%f ", h_gaussMat[i]);
		if(i % 3 == 2) printf("\n");
	}

	//float* h_image, d_image;
	//int imageSize = 1;
	//cudaMalloc((void**)&d_image, sizeof(float) * imageSize);
}
