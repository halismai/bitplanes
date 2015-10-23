#include "bitplanes/core/internal/bitplanes_channel_data_packed.h"
#include "bitplanes/core/internal/bitplanes_channel_data_fast.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/utils/timer.h"
#include "bitplanes/core/types.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct Data
{
  typedef Eigen::Matrix<float,8,1> Gradient;
  typedef Eigen::Matrix<float,8,8> JacobianBlock;
  typedef bp::EigenStdVector<JacobianBlock>::type JacobianBlockVector;

  void init(int n)
  {
    _j_data.resize(n);
    for(size_t i = 0; i < _j_data.size(); ++i)
      _j_data[i].setRandom();

    _pixels.setRandom(n,1);
  }

  void run(const cv::Mat& Iw, Gradient& g)
  {
    g.setZero();

    const auto* c0_ptr = _pixels.data();
    const int src_stride = Iw.cols;

    typename bp::EigenStdVector<Gradient>::type G(8);

    Eigen::Matrix<float,8,1> R;
    for(int y = 1, i=0; y < Iw.rows - 1; ++y)
    {
      const uint8_t* srow = Iw.ptr<const uint8_t>(y);

      int x = 1;

      for( ; x < Iw.cols - 1; ++x, ++i)
      {
        const uint8_t* p = srow + x;
        const uint8_t c = *c0_ptr++;

        /*
        R <<
            (*(p - src_stride - 1) >= *p) - (c & (1<<0) >> 0),
            (*(p - src_stride    ) >= *p) - (c & (1<<1) >> 1),
            (*(p - src_stride + 1) >= *p) - (c & (1<<2) >> 2),
            (*(p              - 1) >= *p) - (c & (1<<3) >> 3),
            (*(p              + 1) >= *p) - (c & (1<<4) >> 4),
            (*(p + src_stride - 1) >= *p) - (c & (1<<5) >> 5),
            (*(p + src_stride    ) >= *p) - (c & (1<<6) >> 6),
            (*(p + src_stride + 1) >= *p) - (c & (1<<7) >> 7);*/

        G[0].noalias() += _j_data[i].row(1) *
            (float) ((*(p - src_stride - 1) >= *p) - (c & (1<<0) >> 0));

        G[1].noalias() += _j_data[i].row(2) *
            (float) ((*(p - src_stride    ) >= *p) - (c & (1<<1) >> 1));

        G[2].noalias() += _j_data[i].row(3) *
            (float) ((*(p - src_stride + 1) >= *p) - (c & (1<<2) >> 2));

        G[3].noalias() += _j_data[i].row(4) *
            (float) ((*(p              - 1) >= *p) - (c & (1<<3) >> 3));

        G[4].noalias() += _j_data[i].row(5) *
            (float) ((*(p              + 1) >= *p) - (c & (1<<4) >> 4));

        G[5].noalias() += _j_data[i].row(5) *
            (float) ((*(p + src_stride - 1) >= *p) - (c & (1<<5) >> 5));

        G[6].noalias() += _j_data[i].row(6) *
            (float) ((*(p + src_stride    ) >= *p) - (c & (1<<6) >> 6));

        G[7].noalias() += _j_data[i].row(6) *
            (float) ((*(p + src_stride + 1) >= *p) - (c & (1<<7) >> 7));
      }
    }

    for(size_t i = 0; i < G.size(); ++i)
      g.noalias() += G[i];
  }

  JacobianBlockVector _j_data;
  bp::Vector_<uint8_t> _pixels;
}; // Data

int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  cv::Rect roi(80, 50, 320, 240);

  bp::BitPlanesChannelDataPacked<bp::Homography> cdata1;
  cdata1.set(I, roi);

  const auto J = cdata1.jacobian();
  bp::Vector_<float> residuals(J.rows());

  cv::Mat I1;
  I(roi).copyTo(I1);

  typename bp::Homography::Gradient g;
  {
    auto t_ms = bp::TimeCode(100, [&]()
                             {
                             cdata1.computeResiduals(I1, residuals);
                             g = J.transpose() * residuals; });
    printf("time: %0.2f ms\n", t_ms);
  }

  bp::BitPlanesChannelDataFast<bp::Homography> cdata2;
  cdata2.set(I, roi);

  {
    auto t_ms = bp::TimeCode(100, [&]() { cdata2.linearize(I1, g); } );
    printf("time: %0.2f ms\n", t_ms);
  }


  Data cdata3;
  cdata3.init(cdata1.pixels().size());
  {
    auto t_ms = bp::TimeCode(100, [&]() { cdata3.run(I1, g); });
    printf("time: %0.2f ms\n", t_ms);
  }

  return 0;
}

