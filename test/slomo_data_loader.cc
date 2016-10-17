#include "bitplanes/test/slomo_data_loader.h"
#include "bitplanes/utils/error.h"
#include <array>

#include <opencv2/imgproc.hpp>

static const std::array<cv::Rect,9> RECTS
{
  cv::Rect(263, 129, 613, 463), // vid1.png
  cv::Rect(314, 205, 511, 392), // vid2.png
  cv::Rect(418, 194, 367, 356), // 3
  cv::Rect(259, 131, 625, 479), // 4
  cv::Rect(314, 206, 516, 386), // 5
  cv::Rect(428, 199, 372, 344), // 6
  cv::Rect(295, 120, 626, 500), // 7
  cv::Rect(349, 71,  493, 494), // 8
  cv::Rect(219, 101, 674, 497)  // 9
};

static inline cv::Rect ScaleRect(cv::Rect r, double s)
{
  return cv::Rect(s*r.x, s*r.y, s*r.width, s*r.height);
}

static inline void PrepareImage(const cv::Mat& src, cv::Mat& dst, double scale)
{
  cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
  if(fabs(1.0 - scale) > 1e-6)
    cv::resize(dst, dst, cv::Size(), scale, scale);
}

namespace bp {

SloMoDataLoader::SloMoDataLoader(std::string dname, int v_number, double scale)
    : _scale(scale)
{
  THROW_ERROR_IF(v_number < 1 || v_number > 9, "video number must be in [1,9]");
  std::string filename = dname + "/v" + std::to_string(v_number) + ".mov";
  _cap.open(filename);
  THROW_ERROR_IF(!_cap.isOpened(), "Failed to open video file");

  _original_roi = RECTS[v_number-1];
  _roi = ScaleRect(RECTS[v_number-1], _scale);
}

bool SloMoDataLoader::getFrame(cv::Mat& dst)
{
  _cap >> _image;
  if(!_image.empty())
  {
    PrepareImage(_image, dst, _scale);

    _original_image_size = _image.size();
    _image_size = dst.size();
  } else
  {
    dst = cv::Mat();
  }

  return !dst.empty();
}

bool SloMoDataLoader::getFrame(cv::Mat& gray, cv::Mat& original)
{
  _cap >> _image;
  if(!_image.empty())
  {
    original = _image.clone();
    PrepareImage(_image, gray, _scale);
  } else
  {
    gray = cv::Mat();
    original = cv::Mat();
  }

  return !gray.empty();
}

cv::Size SloMoDataLoader::originaImageSize() const
{
  return _original_image_size;
}

cv::Size SloMoDataLoader::imageSize() const
{
  return _image_size;
}

cv::Size SloMoDataLoader::roiSize() const
{
  return _roi.size();
}

cv::Rect SloMoDataLoader::getTemplateRoi() const
{
  return _roi;
}

cv::Rect SloMoDataLoader::getOriginalRoi() const
{
  return _original_roi;
}

double SloMoDataLoader::getScale() const
{
  return _scale;
}

}; // bp
