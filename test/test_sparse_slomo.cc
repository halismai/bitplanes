#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bitplanes/test/slomo_data_loader.h>
#include <bitplanes/core/feature_based.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/viz.h>
#include <bitplanes/core/debug.h>
#include <bitplanes/utils/str2num.h>
#include <bitplanes/utils/config_file.h>
#include <bitplanes/utils/error.h>
#include <bitplanes/utils/timer.h>
#include <bitplanes/utils/icompare.h>

static inline double GetScaleFromConfig(std::string fn)
{
  try
  {
    bp::ConfigFile cf(fn);
    return cf.get<double>("Scale", 1.0);
  } catch( const std::exception& ex )
  {
    Warn("Failed to parse scale from %s\n", fn.c_str());
    return 1.0;
  }
}

cv::Scalar gFontColor;

struct FeatureBasedConfig
{
  std::string feature_detector;
  bp::FeatureBasedPlaneTracker::Config config;

  FeatureBasedConfig() :
      feature_detector("ORB"), config() {}

  inline bp::FeatureBasedPlaneTracker create() const
  {
    return bp::FeatureBasedPlaneTracker(createDetector(), createMatcher(), config);
  }

  inline cv::Ptr<cv::Feature2D> createDetector() const
  {
    if(bp::icompare("ORB", feature_detector))
    {
      return cv::ORB::create( config.max_num_pts );
    } else if(bp::icompare("BRISK", feature_detector))
    {
      return cv::BRISK::create(30, 2);
    } else {
      THROW_ERROR("Unknown feature detector\n");
    }
  }

  inline cv::Ptr<cv::DescriptorMatcher> createMatcher() const
  {
    if(bp::icompare("ORB", feature_detector) || bp::icompare("BRISK", feature_detector))
    {
      return cv::Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(cv::NORM_HAMMING2, true));
    } else {
      return cv::Ptr<cv::DescriptorMatcher>(new cv::FlannBasedMatcher);
    }
  }
}; // FeatureBasedConfig

FeatureBasedConfig LoadSparseConfig(std::string);
void Run(bp::SloMoDataLoader&, FeatureBasedConfig, std::string video_output);

static const char* USAGE = "%s <config_file> <dirname> <video_number> [output_video]\n";

int main(int argc,  char** argv)
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
  Run(data_loader, LoadSparseConfig(config_file), argc > 4 ? std::string(argv[4]) : "");

}

static inline void DrawPoints(cv::Mat& dst, const std::vector<cv::Point2f>& pts,
                              double scale)
{
  for(auto& p : pts)
  {
    cv::Point2f pp = cv::Point2f(p.x * scale, p.y * scale);
    cv::circle(dst, pp, 2, CV_RGB(0,0,255), 2, cv::LINE_AA, 0);
  }
}

void Run(bp::SloMoDataLoader& data_loader, FeatureBasedConfig config, std::string out_video)
{
  auto tracker = config.create();

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

  char text_buf[64];
  cv::Mat dimg;
  double total_time = 0.0;
  int num_frames = 1;
  while( data_loader(gray_image, original_image) && ('q' != cv::waitKey(1)) )
  {
    bp::Timer timer;
    auto result = tracker.track(gray_image);
    total_time += timer.stop().count() / 1000.0;
    if(!result.successfull)
      total_time += 50 / 1000.0;
    ++num_frames;

    auto H_scaled = bp::Homography::Scale(result.T, 1.0/data_loader.getScale());
    bp::DrawTrackingResult(dimg, original_image, original_roi, H_scaled.data());
    snprintf(text_buf, 64, "Frame %05d @ %3.2f Hz", num_frames, num_frames/total_time);
    cv::putText(dimg, text_buf, cv::Point(10,40),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, gFontColor, 2, cv::LINE_AA);

    DrawPoints(dimg, tracker.getInliersPoints(), 1.0/data_loader.getScale());

    cv::imshow("result", dimg);

    if(video_writer.isOpened())
      video_writer << dimg;

    //fprintf(stdout, "%s\r", text_buf);
    //fflush(stdout);
  }

  printf("\n");
  Info("%05d frames @ %3.2f Hz\r", num_frames, num_frames / total_time);
  printf("\n");
}

FeatureBasedConfig LoadSparseConfig(std::string fn)
{
  FeatureBasedConfig ret;
  try
  {
    bp::ConfigFile cf(fn);
    ret.feature_detector = cf.get<std::string>("FeatureDetector", "ORB");
    ret.config.max_num_pts = cf.get<int>("MaxNumKeypoints", 256);
    ret.config.ransac_max_iters = cf.get<int>("RansacMaxIterations", 2000);
    ret.config.ransac_reproj_threshold = cf.get<double>("RansacReprojectionError", 2);
  } catch(const std::exception& ex)
  {
    Warn("Failed to load config from %s\n", fn.c_str());
  }

  return ret;
}
