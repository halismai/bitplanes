#ifndef BITPLANES_DEMO_VIEWER_H
#define BITPLANES_DEMO_VIEWER_H

#include <QGLWidget>
#include <QOpenGLFunctions>
#include <memory>

namespace cv {
class Mat;
}; //cv

class Viewer : public QGLWidget, protected QOpenGLFunctions
{
  Q_OBJECT

 public:
  explicit Viewer(QWidget* parent = NULL);

  virtual ~Viewer();

 signals:
  void imageSizeChanged(int w, int h);

 public slots:
  bool showImage(const cv::Mat& image);

 protected:
  virtual void initializeGL();
  virtual void resizeGL(int, int);
  virtual void paintGL();

  void updateScene();
  void renderImage();

 private:
  bool _scene_has_changed;
  QImage _image;
  int _width;
  int _height;
  float _aspect_ratio;
  int _x_pos;
  int _y_pos;
  std::unique_ptr<cv::Mat> _cv_image;
}; // Viewer


#endif // BITPLANES_DEMO_VIEWER_H
