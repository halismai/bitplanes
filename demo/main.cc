#include <QApplication>
#include <QDebug>
#include <QException>

#include <bitplanes/demo/gui_live_camera.h>

int main(int argc, char** argv)
{
  int ret = 0;
  try {
    QApplication app(argc, argv);
    GuiLiveCamera gui;
    gui.show();

    ret = app.exec();
  } catch(const QException& ex) {
    qCritical() << QString("Exception %1").arg(ex.what());
  } catch(...) {
    qCritical() << QString("unmatched exception");
  }

  return ret;
}
