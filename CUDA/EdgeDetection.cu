#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>
#include "bmpLoader.cu"

//Program constants
const int cuda_threadX = 32;
const int cuda_threadY = 16;
const int gaussSize = 3;

__device__ const float lowThreshold = .1f;
__device__ const float  highThreshold = .3f;

__device__ float e = 2.71828f;
__device__ float PI = 3.14f;
__device__ float dev = 1.4f;
__device__ float *GaussMat;
__device__ int gaussWidth;
__device__ float *image;
__device__ float *imageBuf;
__device__ float *angles;
__device__ int width;
__device__ int height;
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
	cur = threadIdx.x + threadIdx.y * blockDim.x;
	if((int)(threadIdx.x) - x < 0) return shared[cur];
	if(threadIdx.x + x >= blockDim.x) return shared[cur];
	
	if((int)(threadIdx.y) - y < 0) return shared[cur];
	if(threadIdx.y + y >= blockDim.y) return shared[cur];
	
	int idx = cur + x + y * blockDim.x;
	//if(idx < 0) return image[0];
	//if(idx >= 1024 * 768) return image[0];
	return shared[idx];
}

__device__ float GaussianBlur(int x, int y)
{
	float p1 = 1.0f / (2 * PI * powf(dev, 2));
	float p2 = p1 * powf(e, -(powf(x, 2) + powf(y, 2)) / (2 * powf(dev, 2)));
	return p2;
}

__global__ void MakeBW(float* d_image, float* d_bw, float* d_buf, float* d_angles, int d_width, int d_height)
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
		height = d_height;
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
	int localIdx = threadIdx.x + threadIdx.y * blockDim.x;
	float val = 0;
	
	shared[localIdx] = image[idx];
	syncthreads();
	
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
	int localIdx = threadIdx.x + threadIdx.y * blockDim.x;
	float Gx = 0, Gy = 0;
	
	shared[localIdx] = image[idx];
	syncthreads();
	
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
	angles[idx] = atanf(abs(Gy) / abs(Gx));
}

__global__ void Suppression()
{
	const float step = PI / 4;
	int idx = GetIdx();
	int localIdx = threadIdx.x + threadIdx.y * blockDim.x;
	int count = 0;//Use an int to store angle. Better for comparison than float
	int angle = 0;
	
	shared[localIdx] = image[idx];
	syncthreads();
	
	for(float i = -PI / 2; i < PI / 2; i += step, count++)
	{
		if(angles[idx] - i < step / 2)
		{
			angle = count;
			angles[idx] = count;
			break;
		}
	}
	
	//syncthreads();
	
	if(angle == 2)// Up and down
	{
		if(GetPixel(idx, 0, 1) > imageBuf[idx] || GetPixel(idx, 0, -1) > imageBuf[idx])
			image[idx] = 0;
		else
			image[idx] = imageBuf[idx];
	}
	else if(angle == 3) // UR and DL
	{
		if(GetPixel(idx, 1, 1) > imageBuf[idx] || GetPixel(idx, -1, -1) > imageBuf[idx])
			image[idx] = 0;
		else
			image[idx] = imageBuf[idx];
	}
	else if(angle == 0) // Left and Right
	{
		if(GetPixel(idx, 1, 0) > imageBuf[idx] || GetPixel(idx, -1, 0) > imageBuf[idx])
			image[idx] = 0;
		else
			image[idx] = imageBuf[idx];
	}
	else if(angle == 1) // UL and DR
	{
		if(GetPixel(idx, -1, 1) > imageBuf[idx] || GetPixel(idx, 1, -1) > imageBuf[idx])
			image[idx] = 0;
		else
			image[idx] = imageBuf[idx];
	}
	
	if(image[idx] > highThreshold) image[idx] = 0.95f;
	else if (image[idx] > lowThreshold) image[idx] = .5f;
	else image[idx] = 0;
}

__global__ void hysteresis()
{
	int idx = GetIdx();
	image[idx] = 0;
	
	if(imageBuf[idx] <= .8f) return;//Ignore weak thresholds at first
	
	//Make sure it doesn't loop back into an already covered path
	while(imageBuf[idx] != 0 && image[idx] == 0)
	{
		image[idx] = imageBuf[idx];
		if(angles[idx] == 0) idx += 1;
		else if(angles[idx] == 1) idx -= width - 1;
		else if(angles[idx] == 2) idx -= width;
		else if(angles[idx] == 3) idx += width - 1;
	}
	
	if(idx < 0 || idx > width * height) return;
}

int main(int argc, char* argv[])
{
	cudaEvent_t start, stop;
	float timer;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

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
	unsigned char* cImage = LoadBitmapFile("../Images/GT.bmp", &bitmapHeader, &bmpInfo);
	int imageSize = bmpInfo.biSizeImage;

	float *h_image, *d_image, *d_buf, *d_angles, *d_bw, *h_bw;
	h_image = (float*)malloc(sizeof(float) * imageSize);
	for(int i = 0; i < imageSize; i++)
		h_image[i] = (float)cImage[i] / 256.0f;
		
	int sharedSize = sizeof(float) * cuda_threadX * cuda_threadY;
	printf("Share size: %d\n", sharedSize);
	
	cudaEventRecord(start, 0);
	cudaMalloc((void**)&d_image, sizeof(float) * imageSize);
	cudaMalloc((void**)&d_bw, sizeof(float) * imageSize / 3);
	cudaMalloc((void**)&d_buf, sizeof(float) * imageSize / 3);
	cudaMalloc((void**)&d_angles, sizeof(float) * imageSize / 3);
	cudaMemcpy(d_image, h_image, sizeof(float) * imageSize, cudaMemcpyHostToDevice);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);
	printf("Load time: %f ms\n", timer);
	
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	//Convert the image into black and white
	cudaEventRecord(start, 0);
	blockSize = dim3(cuda_threadX, cuda_threadY);
	gridSize = dim3(bmpInfo.biWidth / cuda_threadX, bmpInfo.biHeight / cuda_threadY);
	MakeBW<<<gridSize, blockSize>>>(d_image, d_bw, d_buf, d_angles, bmpInfo.biWidth, bmpInfo.biHeight);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);
	printf("BW time: %f ms\n", timer);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	//Blur the image
	cudaEventRecord(start, 0);
	GaussianBlur<<<gridSize, blockSize, sharedSize>>>();
	cudaThreadSynchronize();
	CopyToBuffer<<<gridSize, blockSize>>>();
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);
	printf("Blur time: %f ms\n", timer);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	//Find the gradients
	cudaEventRecord(start, 0);
	FindGradient<<<gridSize, blockSize, sharedSize>>>();
	cudaThreadSynchronize();
	CopyToBuffer<<<gridSize, blockSize>>>();
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);
	printf("Gradient time: %f ms\n", timer);
	
	//Non-Maximum suppression
	cudaEventRecord(start, 0);
	Suppression<<<gridSize, blockSize, sharedSize>>>();
	cudaThreadSynchronize();
	CopyToBuffer<<<gridSize, blockSize>>>();
	cudaThreadSynchronize();
	
	//hysteresis 
	/*cudaEventRecord(start, 0);
	hysteresis<<<gridSize, blockSize>>>();
	cudaThreadSynchronize();*/
	
	h_bw = (float*)malloc(sizeof(float) * imageSize);
	cudaMemcpy(h_bw, d_bw, sizeof(float) * imageSize / 3, cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < imageSize; i++)
	{
		cImage[i] = (unsigned char)(h_bw[i / 3] * 256);
	}
	
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	SaveBitmapFile("GT.bmp", cImage, &bitmapHeader, &bmpInfo);
}
