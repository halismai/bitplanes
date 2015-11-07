/*
  This file is part of bitplanes.

  bitplanes is free software: you can redistribute it and/or modify
  it under the terms of the Lesser GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  bitplanes is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  Lesser GNU General Public License for more details.

  You should have received a copy of the Lesser GNU General Public License
  along with bitplanes.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef BITPLANES_CORE_ALGORITHM_PARAMETERS_H
#define BITPLANES_CORE_ALGORITHM_PARAMETERS_H

#include <iosfwd>
#include <string>

namespace bp {

struct AlgorithmParameters
{
  inline AlgorithmParameters() {}

  /**
   * minimum pixels to attempt alignment. Used for auto pyramid levels
   */
  static const int MIN_NUM_PIXELS_TO_WORK = (100*100) / 16.0;

  /**
   * multi-channel extraction function type
   */
  enum class MultiChannelExtractorType
  {
    IntensityGrayChannel, //< single channel grayscale
    GradientAbsMag,       //< single channel gradient absolute magnitude
    IntensityAndGradient, //< 2 channels, intensity + gradient constraint
    CensusChannel,        //< single channel census signature
    DescriptorFields1,    //< 1-st order descriptor fields
    DescriptorFields2,    //< 2-nd order descriptor fields
    BitPlanes             //< BitPlanes (8-channels)
  }; // MultiChannelExtractorType

  enum class LinearizerType
  {
    InverseCompositional, //< IC algorithm
    ForwardCompositional, //< FC algorithm
  }; // LinearizerType

  /**
   * Type of the motion to estimate
   */
  enum class MotionType
  {
    Translation,
    Affine,
    Homography
  }; // MotionType


  /**
   * number of pyramid levels. A negative value means 'Auto'
   * A value of 1 means a single level (no pyramid)
   */
  int num_levels = -1;

  /**
   * maximum number of iterations
   */
  int max_iterations = 50;

  /**
   * parameter tolerance. If the relative magnitude of parameters falls belows
   * this we converge
   */
  float parameter_tolerance = 5e-6;

  /**
   * function value tolerance. If the the relative function value falls below
   * this, we converge
   */
  float function_tolerance = 5e-5;

  /**
   * std. deviation of an isotropic Gaussian to pre-smooth images prior to
   * computing the channels
   */
  float sigma = 1.2f;

  /**
   * print information
   */
  bool verbose = true;

  /**
   * Process the template by skipping every nth pixel.
   *
   * For example,
   * if 'subsampling' == 2, then we process the template by every other pixel
   * If 'subsampling' == 1, then all pixels will be processed
   *
   */
  int subsampling = 1;

  /**
   * Multi-channel function to use
   *
   * Currently, this is the only one supported. NOTE: other channels have been
   * removed to focus on BitPlanes for iOS
   */
  MultiChannelExtractorType multi_channel_function =
      MultiChannelExtractorType::BitPlanes;

  /**
   * linearization algorithm
   *
   * NOTE: this is the only linearizer supported. In the future, we will add
   * more linearizers (e.g. ForwardCompositional, and ESM)
   */
  LinearizerType linearizer = LinearizerType::InverseCompositional;

  /**
   * loads the configurations from a config file
   */
  static AlgorithmParameters FromConfigFile(std::string filename);

  /**
   * loads parameters from config file
   */
  bool load(std::string filename);

  /**
   * saves the parameters to a file
   */
  bool save(std::string filename);



  friend std::ostream& operator<<(std::ostream&, const AlgorithmParameters&);
}; // AlgorithmParameters


/**
 * converts the MultiChannelExtractorType to a string
 */
std::string ToString(AlgorithmParameters::MultiChannelExtractorType);

/**
 * converts a string to MultiChannelExtractorType
 */
AlgorithmParameters::MultiChannelExtractorType
MultiChannelExtractorTypeFromString(std::string);

/**
 * converts LinearizerType to string
 */
std::string ToString(AlgorithmParameters::LinearizerType);

/**
 */
AlgorithmParameters::LinearizerType
LinearizerTypeFromString(std::string);

}; // bp

#endif // BITPLANES_CORE_ALGORITHM_PARAMETERS_H
