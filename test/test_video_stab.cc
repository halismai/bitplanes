/**
 * Example video stablization using data collected from an UAV
 */
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/affine.h>
#include <bitplanes/core/viz.h>
#include <bitplanes/core/debug.h>
#include <bitplanes/utils/str2num.h>
#include <bitplanes/utils/config_file.h>
#include <bitplanes/utils/error.h>
#include <bitplanes/utils/timer.h>

#include <iostream>
#include <memory>

class Visualizer
{
 public:
  Visualizer(std::string out_video_name = "", cv::Size = cv::Size(),
             cv::Rect bbox = cv::Rect());

  void setTemplate(const cv::Mat& I);

  bool addFrame(const cv::Mat& I, const bp::Matrix33f& H, const char* txt = NULL);

 private:
  cv::VideoWriter _video_writer;
  cv::Rect _bbox;
  cv::Mat _I0; // template
  cv::Mat _Iw; // warped
  cv::Mat _dimg; // display image
  cv::Mat _tmp; // tmp buffer
}; // Visualizer


static const char* USAGE = "%s <config_file> <video_name> <frame_start> [output file]\n";


int main(int argc, char** argv)
{
  if(argc < 4) {
    printf(USAGE, argv[0]);
    return 1;
  }

  std::string config_file(argv[1]);
  std::string vname(argv[2]);
  int frame_start = bp::str2num<int>(argv[3]);

  std::string output_video;
  if(argc > 4)
    output_video = std::string(argv[4]);

  cv::VideoCapture vcap(vname);
  THROW_ERROR_IF(!vcap.isOpened(), ("failed to open " + vname).c_str());

  // skip the few first messed up frames from the video compression
  cv::Mat I0, I;
  for(int i = 0; i < frame_start; ++i) {
    vcap >> I0;
    THROW_ERROR_IF(I0.empty(), "video was cut too short\n");
  }


  bp::AlgorithmParameters params = bp::AlgorithmParameters::FromConfigFile(config_file);
  std::cout << params << std::endl;

  bp::BitPlanesTrackerPyramid<bp::Homography> tracker(params);

  cv::cvtColor(I0, I, cv::COLOR_BGR2GRAY);
  cv::Rect bbox(50, 50, I.cols - 50, I.rows - 50);
  tracker.setTemplate(I, bbox);

  bp::Matrix33f H_init(bp::Matrix33f::Identity());

  std::cout << I0.size() << std::endl;
  Visualizer viz(output_video, I0.size(), bbox);
  viz.setTemplate(I0);

  cv::Mat dimg, dimg_warped, I_orig, I_warped;
  double total_time = 0.0;
  int f_i = 0;


  char text_buf[128];
  vcap >> I_orig;
  while(!I_orig.empty()) {
    cv::cvtColor(I_orig, I, cv::COLOR_BGR2GRAY);
    bp::Timer timer;
    auto result = tracker.track(I, H_init);
    total_time += timer.stop().count() / 1000.0;
    H_init = result.T;

    double t_mag = H_init(0,2)*H_init(0,2) + H_init(1,2)*H_init(1,2);
    if(t_mag > 1500.0) {
      H_init.setIdentity();
      tracker.setTemplate(I, bbox);
      viz.setTemplate(I_orig);
      snprintf(text_buf, sizeof(text_buf), "Frame %05d @ %0.2f Hz [Template update]", f_i, f_i / total_time);
    } else {
      snprintf(text_buf, sizeof(text_buf), "Frame %05d @ %0.2f Hz", f_i, f_i / total_time);
    }

    Info("%s\n", text_buf);
    if(!viz.addFrame(I_orig, H_init, text_buf))
      break;

    vcap >> I_orig;
    ++f_i;

    if(f_i > 1000)
      break;
  }

  return 0;
}


Visualizer::Visualizer(std::string out_video_name, cv::Size image_size, cv::Rect bbox)
{
  _bbox = bbox;

  if(!out_video_name.empty()) {
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    auto v_size = image_size;
    v_size.width *= 2; // we'll have two images side by side
    _video_writer.open(out_video_name, fourcc, 25, v_size);
    if(_video_writer.isOpened()) {
      _video_writer.set(cv::VIDEOWRITER_PROP_QUALITY, 90);
    } else {
      Warn("Failed to open video %s\n", out_video_name.c_str());
    }
  }
}

void Visualizer::setTemplate(const cv::Mat& I)
{
  I.copyTo(_I0);
}

bool Visualizer::addFrame(const cv::Mat& I, const bp::Matrix33f& H, const char* txt)
{
  THROW_ERROR_IF(_I0.empty(), "must setTemplate first");

  cv::Mat_<float> M(3,3);
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
      M(i,j) = H(i,j);

  cv::warpPerspective(I, _Iw, M, cv::Size(), cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);

  const double alpha = 0.5;
  const double beta = 1.0 - alpha;
  const double gamma = 1.0;
  cv::addWeighted(_I0, alpha, _Iw, beta, gamma, _tmp);

  if(txt) {
    cv::putText(_tmp, txt, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                CV_RGB(255, 217, 0), 1, cv::LINE_AA);
  }

  if(_bbox.area()) {
    cv::Mat I_copy(I);
    bp::DrawTrackingResult(I_copy, I_copy, _bbox, H.data(),
                           bp::ColorByName::Yellow, 2);
  }

  cv::hconcat(_tmp, I, _dimg);

  if(_video_writer.isOpened()) {
    _video_writer << _dimg;
  }

  cv::imshow("output", _dimg);
  int k = cv::waitKey(5) & 0xff;
  if(k == ' ')
    k = cv::waitKey(0) & 0xff;

  return k != 'q';
}
