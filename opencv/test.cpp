#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
 }

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    cout << "Usage error" << endl;
    return 0;
  }

  string filename = argv[1];

  src = imread(filename);

  cout << "Image loaded" << endl << flush;

  dst.create(src.size(), src.type());
  cvtColor(src, src_gray, CV_BGR2GRAY);

  lowThreshold = 0;
  CannyThreshold(0, 0);

  imwrite("output.jpg", dst);

  return 0;
}
