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
__device__ int gaussWidth;
__device__ float *image;
__device__ float *imageBuf;
__device__ float *angles;
__device__ int width;
__device__ float Kgx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__device__ float Kgy[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
extern __shared__ float shared[];

__device__ int GetIdx()
{
	int idx = (threadIdx.x + threadIdx.y * blockDim.x) + 
		(blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y);
	return idx;
}

__device__ float GetGaussVal(int x, int y)
{
	x += gaussWidth / 2;
	y += gaussWidth / 2;
	return GaussMat[x + y * gaussWidth];
}

__device__ float GetSobelValX(int x, int y)
{
	x += 1;
	y += 1;
	return Kgx[x + y * 3];
}

__device__ float GetSobelValY(int x, int y)
{
	x += 1;
	y += 1;
	return Kgy[x + y * 3];
}

__device__ float GetPixel(int cur, int x, int y)
{
	if((int)(threadIdx.x) - x < 0 && blockIdx.x == 0) return imageBuf[cur];
	if(/*threadIdx.x + x >= blockDim.x ||*/ blockIdx.x >= gridDim.x - 1) return imageBuf[cur];
	
	if((int)(threadIdx.y) - y < 0 && blockIdx.y == 0) return imageBuf[cur];
	if(/*threadIdx.y + y >= blockDim.y ||*/ blockIdx.y >= gridDim.y - 1) return imageBuf[cur];
	
	int idx = cur + x + y * width;
	//if(idx < 0) return image[0];
	//if(idx >= 1024 * 768) return image[0];
	return imageBuf[idx];
}

__device__ float GaussianBlur(int x, int y)
{
	float p1 = 1.0f / (2 * PI * powf(dev, 2));
	float p2 = p1 * powf(e, -(powf(x, 2) + powf(y, 2)) / (2 * powf(dev, 2)));
	return p2;
}

__global__ void MakeBW(float* d_image, float* d_bw, float* d_buf, float* d_angles, int d_width)
{	
	//TODO: Make local shared array to maximize speed
	int dest = GetIdx();
	int idx = dest * 3;//3 color values

	//http://www.bobpowell.net/grayscale.htm
	float val = d_image[idx] *.3 + d_image[idx + 1] * .59 + d_image[idx + 2] * .11;
	
	//Have thread 0 assign the global variables
	if(dest == 0)
	{
		image = d_bw;
		width = d_width;
		imageBuf = d_buf;
		angles = d_angles;
	}

	d_bw[dest] = val;
	d_buf[dest] = val;
}

__global__ void ComputeGuassian(float* d_gaussMat, int width)
{
	__shared__ float matSum;
	int idx = threadIdx.x + threadIdx.y * width;
	if(idx == 0)
	{
		gaussWidth = width;
		GaussMat = d_gaussMat;
	}
	float val = GaussianBlur(threadIdx.x - width / 2, threadIdx.y - width / 2);
	d_gaussMat[idx] = val;
	syncthreads();
	
	//Have the first thread compute the sum
	//TODO: make this parallel?
	if(idx == 0)
	{
		matSum = .01;//Starts the sum with a small value to avoid sums > 1.0
		for(int i = 0; i < width * width; i++)
				matSum += d_gaussMat[i];
	}
	syncthreads();
	d_gaussMat[idx] = val / matSum;
}

__global__ void GaussianBlur()
{
	int i, j;
	int idx = GetIdx();
	float val = 0;
	
 	for( i = -gaussWidth / 2; i <= gaussWidth / 2; i++)
	{
		for( j = -gaussWidth / 2; j <= gaussWidth / 2; j++)
		{
			val += GetPixel(idx, i, j) * GetGaussVal(i, j);
		}
	}
	
	image[idx] = val;
	
}

__global__ void CopyToBuffer()
{
	int idx = GetIdx();
	imageBuf[idx] = image[idx];
}

__global__ void FindGradient()
{
	int i, j;
	int idx = GetIdx();
	float Gx = 0, Gy = 0;
	
 	for( i = -1; i <= 1; i++)
	{
		for( j = -1; j <= 1; j++)
		{
			Gx += GetPixel(idx, i, j) * GetSobelValX(i, j);
			Gy += GetPixel(idx, i, j) * GetSobelValY(i, j);
		}
	}
	
	syncthreads();
	image[idx] = sqrt(powf(Gx, 2) + powf(Gy, 2));
	angles[idx] = atan(abs(Gy) / abs(Gx));
}

__global__ void Suppression()
{
	const float step =PI / 4;
	int idx = GetIdx();
	int count = 0;//Use an int to store angle. Better for comparison than float
	float angle = 0;
	
	for(float i = 0; i < 2 * PI; i += step, count++)
	{
		if(angles[idx] - i < 0 || angles[idx] - 1 < step / 2)
		{
			angle = count;
			break;
		}
	}
	
	//syncthreads();
	
	if(angle == 0 || angle == 4)// Up and down
	{
		if(GetPixel(idx, 0, 1) > imageBuf[idx] || GetPixel(idx, 0, -1) > imageBuf[idx])
			image[idx] = 0;
		else
			image[idx] = imageBuf[idx];
	}
	else if(angle == 1 || angle == 5) // UR and DL
	{
		if(GetPixel(idx, 1, 1) > imageBuf[idx] || GetPixel(idx, -1, -1) > imageBuf[idx])
			image[idx] = 0;
		else
			image[idx] = imageBuf[idx];
	}
	else if(angle == 2 || angle == 6) // Left and Right
	{
		if(GetPixel(idx, 1, 0) > imageBuf[idx] || GetPixel(idx, -1, 0) > imageBuf[idx])
			image[idx] = 0;
		else
			image[idx] = imageBuf[idx];
	}
	else if(angle == 3 || angle == 7) // UL and DR
	{
		if(GetPixel(idx, -1, 1) > imageBuf[idx] || GetPixel(idx, 1, -1) > imageBuf[idx])
			image[idx] = 0;
		else
			image[idx] = imageBuf[idx];
	}
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
	unsigned char* cImage = LoadBitmapFile("../Images/SOFIANASA.bmp", &bitmapHeader, &bmpInfo);
	int imageSize = bmpInfo.biSizeImage;

	float *h_image, *d_image, *d_buf, *d_angles, *d_bw, *h_bw;
	h_image = (float*)malloc(sizeof(float) * imageSize);
	for(int i = 0; i < imageSize; i++)
		h_image[i] = (float)cImage[i] / 256.0f;
	
	cudaMalloc((void**)&d_image, sizeof(float) * imageSize);
	cudaMalloc((void**)&d_bw, sizeof(float) * imageSize / 3);
	cudaMalloc((void**)&d_buf, sizeof(float) * imageSize / 3);
	cudaMalloc((void**)&d_angles, sizeof(float) * imageSize / 3);
	cudaMemcpy(d_image, h_image, sizeof(float) * imageSize, cudaMemcpyHostToDevice);
	
	//Convert the image into black and white
	blockSize = dim3(cuda_threadX, cuda_threadY);
	gridSize = dim3(bmpInfo.biWidth / cuda_threadX, bmpInfo.biHeight / cuda_threadY);
	MakeBW<<<gridSize, blockSize>>>(d_image, d_bw, d_buf, d_angles, bmpInfo.biWidth);
	cudaThreadSynchronize();
	
	//Blur the image
	GaussianBlur<<<gridSize, blockSize>>>();
	cudaThreadSynchronize();
	CopyToBuffer<<<gridSize, blockSize>>>();
	cudaThreadSynchronize();
	
	//Find the gradients
	FindGradient<<<gridSize, blockSize, sizeof(float) * 512>>>();
	cudaThreadSynchronize();
	CopyToBuffer<<<gridSize, blockSize>>>();
	cudaThreadSynchronize();
	
	//Non-Maximum suppression
	Suppression<<<gridSize, blockSize>>>();
	cudaThreadSynchronize();
	
	h_bw = (float*)malloc(sizeof(float) * imageSize);
	cudaMemcpy(h_bw, d_bw, sizeof(float) * imageSize / 3, cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < imageSize; i++)
	{
		cImage[i] = (unsigned char)(h_bw[i / 3] * 256);
	}
	
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	SaveBitmapFile("SOFIANASA.bmp", cImage, &bitmapHeader, &bmpInfo);
}
