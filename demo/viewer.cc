#include <bitplanes/demo/viewer.h>
#include <bitplanes/core/debug.h>
#include <opencv2/core.hpp>

Viewer::Viewer(QWidget* parent)
  : QGLWidget(parent),
    _scene_has_changed(false), _width(640), _height(480),  _aspect_ratio(4.0/3.0f),
    _x_pos(0), _y_pos(0), _cv_image(new cv::Mat()) {}

Viewer::~Viewer() {}

void Viewer::initializeGL()
{
  this->makeCurrent();
  this->initializeOpenGLFunctions();
  glClearColor(0.0, 0.0, 0.0, 1.0f);
}

void Viewer::resizeGL(int width, int height)
{
  this->makeCurrent();
  glViewport(0, 0, (GLint) width, (GLint) height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, width, 0, height, 0, 1);

  glMatrixMode(GL_MODELVIEW);

  _height = width / _aspect_ratio;
  _width = width;

  if(_height > height) {
    _width = height * _aspect_ratio;
    _height = height;
  }

  emit imageSizeChanged(_width, _height);

  _x_pos = (width - _width) / 2;
  _y_pos = (height - _height) / 2;

  _scene_has_changed = true;

  this->updateScene();
}

void Viewer::updateScene()
{
  if(_scene_has_changed && this->isVisible())
    update();
}

void Viewer::paintGL()
{
  this->makeCurrent();

  if(_scene_has_changed) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderImage();

    _scene_has_changed = false;
  }
}

void Viewer::renderImage()
{
  this->makeCurrent();

  glClear(GL_COLOR_BUFFER_BIT);

  if(!_image.isNull()) {
    glLoadIdentity();

    QImage image;
    glPushMatrix();
    {
      int w = _image.width(),
          h = _image.height();
      if(w != this->size().width() && h != this->size().height()) {
        image = _image.scaled(QSize(_width, _height), Qt::IgnoreAspectRatio,
                              Qt::SmoothTransformation);
      } else {
        image = _image;
      }

      glRasterPos2i(_x_pos, _y_pos);
      w = image.width();
      h = image.height();

      glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, image.bits());
    }

    glPopMatrix();
    glFlush();
  }
}

bool Viewer::showImage(const cv::Mat& image)
{
  image.copyTo( *_cv_image );
  int nc = image.channels();
  int rows = image.rows,
      cols = image.cols,
      step = image.step;
  auto ptr = (const unsigned char*) _cv_image->data;

  switch(nc) {
    case 3:
      {
        _image = QImage(ptr, cols, rows, step, QImage::Format_RGB888);
      } break;

    case 1:
      {
        _image = QImage(ptr, cols, rows, step, QImage::Format_Indexed8);
      }; break;
    default:
      {
        Warn("unsupported number of channales %d\n", nc);
        return false;
      }
  }

  _scene_has_changed = true;
  updateScene();
  return true;
}


