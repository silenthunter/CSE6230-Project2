#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sys/time.h>

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
  struct timeval time_1;
  struct timeval time_2;

  if (argc != 2)
  {
    cout << "Usage: " << argv[0] << " " << "IMAGE" << endl;
    return 0;
  }

  string filename = argv[1];
  string outfilename = string("e-") + filename;

  // Image load

  src = imread(filename);

  gettimeofday(&time_1, NULL);
  cout << "Image loaded" << endl << flush;
  gettimeofday(&time_2, NULL);

  cout <<"Image load time: " << ((time_2.tv_sec - time_1.tv_sec) * 1000000 +
      (time_2.tv_usec - time_1.tv_usec)) << " us" << endl;

  // Convert to grayscale
  gettimeofday(&time_1, NULL);
  cvtColor(src, src_gray, CV_RGB2GRAY);
  dst.create(src_gray.size(), src_gray.type());
  gettimeofday(&time_2, NULL);

  cout <<"Image convert to grayscale time: " << ((time_2.tv_sec - time_1.tv_sec) * 1000000 +
      (time_2.tv_usec - time_1.tv_usec)) << " us" << endl;

  /// Reduce noise with a 3x3 grid
  gettimeofday(&time_1, NULL);
  blur( src_gray, detected_edges, Size(5,5) );
  gettimeofday(&time_2, NULL);

  cout <<"Image blur time: " << ((time_2.tv_sec - time_1.tv_sec) * 1000000 +
      (time_2.tv_usec - time_1.tv_usec)) << " us" << endl;

  /// Canny detector
  gettimeofday(&time_1, NULL);
  Canny( detected_edges, detected_edges, lowThreshold, highThreshold, aperature_size );
  gettimeofday(&time_2, NULL);

  cout <<"Canny filter time: " << ((time_2.tv_sec - time_1.tv_sec) * 1000000 +
      (time_2.tv_usec - time_1.tv_usec)) << " us" << endl;

  imwrite(outfilename, detected_edges);

  return 0;
}
