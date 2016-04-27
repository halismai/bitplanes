#include "bitplanes/demo_cv/demo.h"
#include "bitplanes/core/debug.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/bitplanes_tracker_pyramid.h"
#include "bitplanes/core/viz.h"

#include "bitplanes/utils/config_file.h"
#include "bitplanes/utils/error.h"

#include "bitplanes/demo_cv/bounded_buffer.h"

#include <memory>
#include <string>
#include <thread>
#include <iostream>

#include <opencv2/highgui.hpp>

typedef std::unique_ptr<cv::Mat> ImagePointer;

struct ResultForDisplay
{
  ImagePointer image;           //< the input image
  bp::Result   tracker_result;  //< tracker result
  int time_ms;                  //< time in milliseconds
}; // ResultForDisplay

struct GuiData
{
  ImagePointer image;
  ImagePointer image_gray;
  bp::Result    result;

  GuiData() : image(new cv::Mat), image_gray(new cv::Mat) {}

  void swap(GuiData&& other)
  {
    image.swap(other.image);
    image_gray.swap(other.image_gray);
    std::swap(result, other.result);
  }

  GuiData(GuiData&& other)
  {
    image.swap(other.image);
    image_gray.swap(other.image_gray);
    std::swap(result, other.result);
  }
}; // ImageData



typedef BoundedBuffer<std::unique_ptr<GuiData>> ImageBufferType;

struct DemoLiveCapture::Impl
{
  typedef bp::BitPlanesTrackerPyramid<bp::Homography> TrackerType;

  Impl(std::string config_file)
      : _video_capture()
  {
    if(!_video_capture.isOpened()) {
      _video_capture.open(0);
      _video_capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920 / 2);
      _video_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080 / 2);
    }

    bp::AlgorithmParameters params;
    if(!config_file.empty()) {
      params = bp::AlgorithmParameters::FromConfigFile(config_file);
    } else {
      params.num_levels = 2;
      params.max_iterations = 50;
      params.subsampling = 2;
      params.verbose = false;
    }

    _tracker.reset(new TrackerType(params));

    _main_thread.reset(new std::thread(&DemoLiveCapture::Impl::mainThread, this));
  }

  ~Impl()
  {
    _stop_requested = true;

    if(_main_thread && _main_thread->joinable())
      _main_thread->join();

    if(_display_thread && _display_thread->joinable())
      _display_thread->join();

    if(_data_thread && _data_thread->joinable())
      _data_thread->join();
  }

  inline bool isRunning() const { return _stop_requested == false; }
  inline void stop() { _stop_requested = true; }

  cv::VideoCapture _video_capture;

  std::atomic<bool> _stop_requested{false};
  std::unique_ptr<TrackerType> _tracker;
  cv::Rect _roi;

  std::unique_ptr<std::thread> _main_thread;
  std::unique_ptr<std::thread> _display_thread;
  std::unique_ptr<std::thread> _data_thread;

  std::unique_ptr<ImageBufferType> _data_buffer;
  std::unique_ptr<ImageBufferType> _results_buffer;

  void displayThread();
  void mainThread();
  void dataThread();
}; // DemoLiveCapture::Impl

DemoLiveCapture::DemoLiveCapture(std::string config_file)
  : _impl(new Impl(config_file))  {}

DemoLiveCapture::~DemoLiveCapture()
{
  delete _impl;
}

bool DemoLiveCapture::isRunning() const
{
  return _impl->isRunning();
}

void DemoLiveCapture::stop() { _impl->stop(); }

struct MouseHandleData
{
  std::atomic<bool> start_selection{false};
  std::atomic<bool> has_template{false};
  cv::Point origin;
  cv::Rect roi;
}; // MouseHandleData

void onMouse(int event, int x, int y, int /*flags*/, void* data_)
{
  MouseHandleData* data = reinterpret_cast<MouseHandleData*>(data_);
  THROW_ERROR_IF( data == NULL, "badness" );

  if(data->start_selection) {
    data->roi = cv::Rect(
        std::min(x, data->origin.x),
        std::min(y, data->origin.y),
        std::abs(x - data->origin.x),
        std::abs(y - data->origin.y));
  }

  switch(event)
  {
    case cv::EVENT_LBUTTONDOWN:
      {
        data->origin = cv::Point(x, y);
        data->roi = cv::Rect(x, y, 0, 0);
        data->start_selection = true;
        printf("start\n");
      } break;

    case cv::EVENT_LBUTTONUP:
      {
        data->start_selection = false;
        if(data->roi.area() >0)
          data->has_template = true;
        printf("got template\n");
      } break;
  }
}

void DemoLiveCapture::Impl::mainThread()
{
  //
  // get the template from the user input
  //

  MouseHandleData handle_data;

  const char* window_name = "Select ROI";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(window_name, onMouse, &handle_data);

  cv::Mat image, image_copy;
  int k = 0;
  while(k != 'q' && !handle_data.has_template) {
    _video_capture >> image;
    if(image.empty()) {
      Warn("could not poll camera\n");
      break;
    }

    image.copyTo(image_copy);

    if(handle_data.start_selection && handle_data.roi.area() > 0) {
      cv::Mat roi(image_copy, handle_data.roi);
      cv::bitwise_not(roi, roi);
    }

    cv::imshow(window_name, image_copy);
    k = cv::waitKey(5) & 0xff;
  }

  if(!handle_data.has_template) {
    Warn("Terminated... exiting\n");
    _stop_requested = true;
    return;
  }

  _data_buffer.reset(new ImageBufferType(10));
  _results_buffer.reset(new ImageBufferType(10));

  cv::cvtColor(image, image_copy, cv::COLOR_BGR2GRAY);
  _tracker->setTemplate(image_copy, handle_data.roi);

  cv::destroyWindow(window_name);
  _data_thread.reset(new std::thread(&DemoLiveCapture::Impl::dataThread, this));

  cv::namedWindow("bitplanes");

  _roi = handle_data.roi;
  cv::Mat dimg;
  bp::Matrix33f tform(bp::Matrix33f::Identity());
  while(!_stop_requested) {

    std::unique_ptr<GuiData> data;
    if(_data_buffer->pop(&data)) {
      data->result = _tracker->track(*data->image_gray, tform);

      tform = data->result.T;

      bp::DrawTrackingResult(dimg, *data->image, _roi, tform.data());
      cv::imshow("bitplanes", dimg);

      int k = 0xff & cv::waitKey(5);
      if(k == 'q') {
        _stop_requested = true;
      }
    }
  }
}

void DemoLiveCapture::Impl::dataThread()
{
  std::unique_ptr<GuiData> data(new GuiData);
  while(!_stop_requested) {

    _video_capture >> *data->image;
    if(data->image->empty()) {
      Warn("failed to get image\n");
    }

    cv::cvtColor(*data->image, *data->image_gray, cv::COLOR_BGR2GRAY);
    _data_buffer->push(std::move(data));
    data.reset(new GuiData);
  }

}


