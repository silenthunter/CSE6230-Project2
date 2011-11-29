#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    cout << "Usage error" << endl;
    return 0;
  }

  string filename = argv[1];
  Mat image;

  //cv::imread(filename);


  return 0;
}
