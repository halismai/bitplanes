#include "bitplanes/test/slomo_data_loader.h"
#include "bitplanes/utils/str2num.h"
#include <opencv2/highgui.hpp>
#include <iostream>

static inline void usage(const char* s)
{
  std::cout << "usage: " << s << " <dir> <video_number>" << std::endl;
}

int main(int argc, char** argv)
{
  if(argc < 3)
  {
    usage(argv[0]);
    return 1;
  }


  std::string dname(argv[1]);
  int video_number = bp::str2num<int>(std::string(argv[2]));
  double scale = argc > 3 ? bp::str2num<double>(std::string(argv[3])) : 1.0;
  bp::SloMoDataLoader data_loader(dname, video_number, scale);

  cv::Mat original_image, gray_image;
  while(data_loader(gray_image, original_image))
  {
    cv::imshow("image", gray_image);
    if( (0xff&cv::waitKey(5)) == 'q' )
      break;
  }

  return 0;
}

