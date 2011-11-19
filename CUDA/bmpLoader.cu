//Credit to BeholderOf at:
//http://www.vbforums.com/showthread.php?p=4060558

#include <windows.h>

unsigned char *LoadBitmapFile(char *filename, BITMAPFILEHEADER *bmpFileHeader, BITMAPINFOHEADER *bitmapInfoHeader)
{
	FILE *filePtr; //our file pointer
	unsigned char *bitmapImage;  //store image data
	int imageIdx=0;  //image index counter
	unsigned char tempRGB;  //our swap variable
	BITMAPFILEHEADER bitmapFileHeader;

	//open filename in read binary mode
	filePtr = fopen(filename,"r+b");
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