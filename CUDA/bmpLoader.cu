//Credit to BeholderOf at:
//http://www.vbforums.com/showthread.php?p=4060558
#include <stdio.h>
#include <stdlib.h>
#ifndef __unix__
#include <windows.h>
#else
typedef unsigned short WORD;
typedef unsigned int DWORD;
typedef int LONG;

//http://www.vbforums.com/showthread.php?p=4060558
typedef struct __attribute__((__packed__)) tagBITMAPFILEHEADER
{
WORD bfType;  //specifies the file type
DWORD bfSize;  //specifies the size in bytes of the bitmap file
WORD bfReserved1;  //reserved; must be 0
WORD bfReserved2;  //reserved; must be 0
DWORD bfOffBits;  //species the offset in bytes from the bitmapfileheader to the bitmap bits
}BITMAPFILEHEADER;

typedef struct __attribute__((__packed__)) tagBITMAPINFOHEADER
{
DWORD biSize;  //specifies the number of bytes required by the struct
LONG biWidth;  //specifies width in pixels
LONG biHeight;  //species height in pixels
WORD biPlanes; //specifies the number of color planes, must be 1
WORD biBitCount; //specifies the number of bit per pixel
DWORD biCompression;//spcifies the type of compression
DWORD biSizeImage;  //size of image in bytes
LONG biXPelsPerMeter;  //number of pixels per meter in x axis
LONG biYPelsPerMeter;  //number of pixels per meter in y axis
DWORD biClrUsed;  //number of colors used by th ebitmap
DWORD biClrImportant;  //number of colors that are important
}BITMAPINFOHEADER;
#endif

unsigned char *LoadBitmapFile(char *filename, BITMAPFILEHEADER *bmpFileHeader, BITMAPINFOHEADER *bitmapInfoHeader)
{
	FILE *filePtr; //our file pointer
	unsigned char *bitmapImage;  //store image data
	int imageIdx=0;  //image index counter
	unsigned char tempRGB;  //our swap variable
	BITMAPFILEHEADER bitmapFileHeader;

	//open filename in read binary mode
	filePtr = fopen(filename,"rb");
	if (filePtr == NULL)
		return NULL;

	//read the bitmap file header
	fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER),1,filePtr);

	//verify that this is a bmp file by check bitmap id
	if (bitmapFileHeader.bfType !=0x4D42)
	{
		fclose(filePtr);
		return NULL;
	}

	//read the bitmap info header
	fread(bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr);

	//move file point to the begging of bitmap data
	fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

	//allocate enough memory for the bitmap image data
	bitmapImage = (unsigned char*)malloc(bitmapInfoHeader->biSizeImage);

	//verify memory allocation
	if (!bitmapImage)
	{
		free(bitmapImage);
		fclose(filePtr);
		return NULL;
	}

	//read in the bitmap image data
	fread(bitmapImage,1,bitmapInfoHeader->biSizeImage,filePtr);

	//make sure bitmap image data was read
	if (bitmapImage == NULL)
	{
		fclose(filePtr);
		return NULL;
	}

	//swap the r and b values to get RGB (bitmap is BGR)
	for (imageIdx = 0; imageIdx < bitmapInfoHeader->biSizeImage; imageIdx+=3)
	{
		tempRGB = bitmapImage[imageIdx];
		bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
		bitmapImage[imageIdx + 2] = tempRGB;
	}

	//close file and return bitmap iamge data
	fclose(filePtr);
	memcpy(bmpFileHeader, &bitmapFileHeader, sizeof(BITMAPFILEHEADER));
	return bitmapImage;
}

void SaveBitmapFile(char* filename, unsigned char *bitmapImage, BITMAPFILEHEADER *bmpFileHeader, BITMAPINFOHEADER *bitmapInfoHeader)
{
	FILE *filePtr; //our file pointer
	int imageIdx=0;  //image index counter
	unsigned char tempRGB;  //our swap variable
	BITMAPFILEHEADER bitmapFileHeader = *bmpFileHeader;

	//open filename in read binary mode
	filePtr = fopen(filename,"r+b");
	if (filePtr == NULL)
		return;

	//move file point to the begging of bitmap data
	fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);
	
	//swap the r and b values to get RGB (bitmap is BGR)
	for (imageIdx = 0; imageIdx < bitmapInfoHeader->biSizeImage; imageIdx+=3)
	{
		tempRGB = bitmapImage[imageIdx];
		bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
		bitmapImage[imageIdx + 2] = tempRGB;
	}

	//read in the bitmap image data
	fwrite(bitmapImage,1,bitmapInfoHeader->biSizeImage,filePtr);

	//close file and return bitmap iamge data
	fclose(filePtr);
}
