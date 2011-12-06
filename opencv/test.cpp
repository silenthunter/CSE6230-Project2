#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;
Mat thr_detected_edges;

int lowThreshold = 40;
int highThreshold = 80;
int aperature_size = 3;

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    cout << "Usage: " << argv[0] << " " << "IMAGE" << endl;
    return 0;
  }

  string filename = argv[1];

  src = imread(filename);

  cout << "Image loaded" << endl << flush;

  // Convert to grayscale
  cvtColor(src, src_gray, CV_RGB2GRAY);
  dst.create(src_gray.size(), src_gray.type());

  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(5,5) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, highThreshold, aperature_size );

  /// Using Canny's output as a mask, we display our result
  //dst = Scalar::all(0);

  //src_gray.copyTo( dst, detected_edges);



  //imwrite("output.jpg", thr_detected_edges);
  imwrite("output.jpg", detected_edges);

  return 0;
}
