#ifndef BITPLANES_DEMO_GUI_LIVE_CAMERA_H
#define BITPLANES_DEMO_GUI_LIVE_CAMERA_H

#include <QMainWindow>
#include <memory>

namespace cv {
class VideoCapture;
}; // cv

class GuiLiveCamera : public QMainWindow
{
  Q_OBJECT
 public:
  explicit GuiLiveCamera(QWidget* parent = NULL);
  virtual ~GuiLiveCamera();

 private slots:
     void on_action_start();

 protected:
  void timerEvent(QTimerEvent*);

 private:
  std::unique_ptr<cv::VideoCapture> _video_capture;
  struct GuiWidgets;
  std::unique_ptr<GuiWidgets> _widgets;
}; // GuiLiveCamera


#endif // BITPLANES_DEMO_GUI_LIVE_CAMERA_H
