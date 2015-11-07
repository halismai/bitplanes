#include <opencv2/highgui.hpp>
#include <bitplanes/test/slomo_data_loader.h>
#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/viz.h>
#include <bitplanes/core/debug.h>
#include <bitplanes/utils/str2num.h>
#include <bitplanes/utils/config_file.h>
#include <bitplanes/utils/error.h>
#include <bitplanes/utils/timer.h>

static inline double GetScaleFromConfig(std::string fn)
{
  try
  {
    bp::ConfigFile cf(fn);
    return cf.get<double>("Scale", 1.0);
  } catch( const std::exception& ex )
  {
    Warn("Failed to parse scale frm %s\n", fn.c_str());
    return 1.0;
  }
}


cv::Scalar gFontColor;

void Run(bp::SloMoDataLoader&, bp::AlgorithmParameters, std::string);

static const char* USAGE = "%s <config_file> <dirname> <video_number>\n";

int main(int argc, char** argv)
{
  if(argc < 4) {
    printf(USAGE, argv[0]);
    return 1;
  }

  std::string config_file(argv[1]);
  std::string dirname(argv[2]);
  int v_number = bp::str2num<int>(argv[3]);

  if(v_number == 6 || v_number == 8)
    gFontColor = CV_RGB(255,255,255);
  else
    gFontColor = CV_RGB(0,0,0);


  bp::SloMoDataLoader data_loader(dirname, v_number, GetScaleFromConfig(config_file));
  Run(data_loader, bp::AlgorithmParameters::FromConfigFile(config_file),
      argc > 4 ? std::string(argv[4]) : "");

  return 0;
}

void Run(bp::SloMoDataLoader& data_loader, bp::AlgorithmParameters params, std::string out_video)
{
  bp::BitPlanesTrackerPyramid<bp::Homography> tracker(params);

  cv::Mat original_image, gray_image;
  THROW_ERROR_IF( !data_loader.getFrame(gray_image, original_image),
                 "failed to get first frame");
  cv::Rect original_roi = data_loader.getOriginalRoi();

  tracker.setTemplate(gray_image, data_loader.getTemplateRoi());

  cv::VideoWriter video_writer;
  if(!out_video.empty())
  {
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    video_writer.open(out_video, fourcc, 25, original_image.size());
    if(video_writer.isOpened()) {
      video_writer.set(cv::VIDEOWRITER_PROP_QUALITY, 90);
    } else {
      Warn("Failed to open video %s\n", out_video.c_str());
    }
  }


  bp::Matrix33f H_init(bp::Matrix33f::Identity());

  char text_buf[64];
  cv::Mat dimg;
  double total_time = 0.0;
  int num_frames = 1;
  while( data_loader(gray_image, original_image) && ('q' != cv::waitKey(1)) )
  {
    bp::Timer timer;
    auto result = tracker.track(gray_image, H_init);
    total_time += timer.stop().count() / 1000.0;
    ++num_frames;

    H_init = result.T;

    auto H_scaled = bp::Homography::Scale(result.T, 1.0/data_loader.getScale());
    bp::DrawTrackingResult(dimg, original_image, original_roi, H_scaled.data());
    snprintf(text_buf, 64, "Frame %05d @ %3.2f Hz", num_frames, num_frames/total_time);
    cv::putText(dimg, text_buf, cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                gFontColor, 2, cv::LINE_AA);

    cv::imshow("result", dimg);

    if(video_writer.isOpened())
      video_writer << dimg;

    fprintf(stdout, "%s\r", text_buf);
    fflush(stdout);
  }

  printf("\n");
  Info("%05d frames @ %3.2f Hz\r", num_frames, num_frames / total_time);
  printf("\n");
}

