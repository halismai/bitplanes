#include <bitplanes/demo_cv/demo.h>
#include <bitplanes/utils/utils.h>

int main()
{

  DemoLiveCapture demo;

  while(demo.isRunning())
    bp::Sleep(100);

  return 0;
}
