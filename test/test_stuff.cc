#include "bitplanes/core/internal/bitplanes_channel_data_subsampled.h"
#include "bitplanes/core/homography.h"

#include <type_traits>

using namespace bp;

int main()
{
  typedef BitPlanesChannelDataSubSampled<Homography> CDataType;
  BitPlanesChannelData<CDataType> cdata;

  return 0;
}


