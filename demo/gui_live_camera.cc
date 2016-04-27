#include "bitplanes/demo/gui_live_camera.h"
#include "bitplanes/demo/viewer.h"
#include "bitplanes/core/debug.h"

#include <opencv2/highgui.hpp>

#include <QSplitter>
#include <QStatusBar>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>

struct GuiLiveCamera::GuiWidgets
{
  QLabel* status_label;
  QLabel* info_label;
  QPushButton* start_button;
  QSplitter* splitter;
  Viewer* viewer;
}; // GuiLiveCamera

GuiLiveCamera::GuiLiveCamera(QWidget* parent)
  : QMainWindow(parent),
    _video_capture(new cv::VideoCapture(0)),
    _widgets(new GuiWidgets)
{
  _widgets->info_label = new QLabel();
  _widgets->status_label = new QLabel();
  _widgets->splitter = new QSplitter(Qt::Horizontal);
  setCentralWidget(_widgets->splitter);

  _widgets->viewer = new Viewer(parent);
  _widgets->viewer->setBaseSize(QSize(640, 480));
  _widgets->splitter->addWidget(_widgets->viewer);

  {
    QSplitter* vs = new QSplitter(Qt::Vertical);
    vs->addWidget(_widgets->info_label);
    _widgets->splitter->addWidget(vs);
  }

  statusBar()->insertWidget(0, _widgets->status_label, 0);
  statusBar()->showMessage("status...");

  _widgets->start_button = new QPushButton(QString("Start camera"), this);
  connect(_widgets->start_button, SIGNAL(clicked()), this, SLOT(on_action_start()));

  resize( 640 + 10, 480 + 10 );
  setWindowTitle("BitPlanes");
}

GuiLiveCamera::~GuiLiveCamera() {}

void GuiLiveCamera::on_action_start()
{
  if(!_video_capture->isOpened()) {
    if(!_video_capture->open(0)) {
      Warn("Failed to open camera\n");
      return;
    }
  }

  this->startTimer(30);
}

void GuiLiveCamera::timerEvent(QTimerEvent* /*event*/)
{
  if(!_video_capture->isOpened()) {
    Warn("camera is not opened\n");
    return;
  }

  cv::Mat image;
  *_video_capture >> image;

  cv::flip(image, image, 0);
  _widgets->viewer->showImage(image);
}

