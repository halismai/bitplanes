#ifndef SLOMO_DATA_LOADER_H
#define SLOMO_DATA_LOADER_H

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace bp {

class SloMoDataLoader
{
 public:
  /**
   * \param dname directory name where data is located
   * \param v_number video number (from 1 to 9)
   * \param scale    scaling factor to reduce/increase image size
   *
   */
  SloMoDataLoader(std::string dname, int v_number, double scale = 1.0);

  /**
   * \return the template ROI in the first frame
   */
  cv::Rect getTemplateRoi() const;

  cv::Rect getOriginalRoi() const;

  /**
   * \return true if there are more frames
   */
  bool getFrame(cv::Mat&);
  inline bool operator()(cv::Mat& image) { return getFrame(image); }

  bool getFrame(cv::Mat& gray, cv::Mat& original);
  inline bool operator()(cv::Mat& gray, cv::Mat& original)
  {
    return getFrame(gray, original);
  }

  /**
   * \return original image size
   */
  cv::Size originaImageSize() const;

  /**
   * \return image size after scaling
   */
  cv::Size imageSize() const;

  /**
   * \return roi size after scaling
   */
  cv::Size roiSize() const;

  /**
   * \return the scaling factor
   */
  double getScale() const;

 private:
  cv::VideoCapture _cap;
  cv::Size _original_image_size;
  cv::Size _image_size;
  cv::Rect _roi;
  cv::Rect _original_roi;
  cv::Mat _image;
  double _scale;
}; // SloMoDataLoader

}; // bp

#endif // SLOMO_DATA_LOADER_H
