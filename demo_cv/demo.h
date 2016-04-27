#ifndef BITPLANES_DEMO_CV_DEMO_H
#define BITPLANES_DEMO_CV_DEMO_H

#include <string>
#include <atomic>

class DemoLiveCapture
{
 public:
  /**
   * path to configuration file
   */
  DemoLiveCapture(std::string config_file = "");

  ~DemoLiveCapture();

  void stop();
  bool isRunning() const;

 private:
  struct Impl;
  Impl* _impl;
}; // DemoLiveCapture

#endif
