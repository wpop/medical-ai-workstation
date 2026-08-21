#include "maiw/cardiac/CardiacMriPreprocessing.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using maiw::cardiac::Float64VolumeView;
using maiw::cardiac::VolumeDimensions;
using maiw::cardiac::VolumeSpacing;

constexpr double kTolerance = 1e-12;

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void requireNear(double actual, double expected, const std::string& message)
{
  if (std::fabs(actual - expected) > kTolerance)
  {
    throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", actual " +
                             std::to_string(actual));
  }
}

template <typename Exception, typename Function>
void requireThrows(Function&& function, const std::string& message)
{
  try
  {
    function();
  }
  catch (const Exception&)
  {
    return;
  }
  catch (const std::exception& error)
  {
    throw std::runtime_error(message + ": wrong exception type: " + error.what());
  }
  throw std::runtime_error(message + ": expected exception was not thrown");
}

std::vector<double> makeRampVolume(VolumeDimensions dimensions)
{
  std::vector<double> values(dimensions.width * dimensions.height * dimensions.depth);
  for (std::size_t z = 0; z < dimensions.depth; ++z)
  {
    for (std::size_t y = 0; y < dimensions.height; ++y)
    {
      for (std::size_t x = 0; x < dimensions.width; ++x)
      {
        const std::size_t index = (z * dimensions.height * dimensions.width) +
                                  (y * dimensions.width) + x;
        values[index] = static_cast<double>(x + (10U * y) + (100U * z));
      }
    }
  }
  return values;
}

Float64VolumeView viewOf(VolumeDimensions dimensions, const std::vector<double>& values)
{
  return Float64VolumeView{dimensions, values};
}

void testFrozenConfig()
{
  const auto config = maiw::cardiac::frozenCardiacMriPreprocessingConfig();
  requireNear(config.targetSpacingXyz.x, 1.5, "target spacing X mismatch");
  requireNear(config.targetSpacingXyz.y, 1.5, "target spacing Y mismatch");
  requireNear(config.targetSpacingXyz.z, 7.5, "target spacing Z mismatch");
  require(config.finalTensorShapeDhw.depth == 14, "final D mismatch");
  require(config.finalTensorShapeDhw.height == 144, "final H mismatch");
  require(config.finalTensorShapeDhw.width == 144, "final W mismatch");
}

void testTargetSizeCalculation()
{
  require(maiw::cardiac::targetSizeForAxis(216, 1.5625, 1.5) == 225,
          "ordinary target-size calculation mismatch");
  require(maiw::cardiac::roundHalfToEvenNonNegative(2.5) == 2,
          "ties-to-even failed for even lower integer");
  require(maiw::cardiac::roundHalfToEvenNonNegative(3.5) == 4,
          "ties-to-even failed for odd lower integer");
  require(maiw::cardiac::roundHalfToEvenNonNegative(std::nextafter(2.5, 0.0)) == 2,
          "rounding immediately below half should round down");
  require(maiw::cardiac::roundHalfToEvenNonNegative(
              std::nextafter(2.5, std::numeric_limits<double>::infinity())) == 3,
          "rounding immediately above half should round up");
  require(maiw::cardiac::targetSizeForAxis(1, 7.5, 1.5) == 1,
          "target size should never fall below 1");
}

void testSourceCoordinateCalculation()
{
  requireNear(maiw::cardiac::sourceCoordinateForTargetIndex(5, 9, 2.0, 1.0, 0),
              0.0,
              "first target coordinate mismatch");
  requireNear(maiw::cardiac::sourceCoordinateForTargetIndex(5, 9, 2.0, 1.0, 4),
              2.0,
              "center target coordinate mismatch");
  requireNear(maiw::cardiac::sourceCoordinateForTargetIndex(5, 9, 2.0, 1.0, 8),
              4.0,
              "last target coordinate mismatch");
}

void testIdentityGridSampling()
{
  const VolumeDimensions dimensions{3, 2, 2};
  const std::vector<double> values = makeRampVolume(dimensions);
  const auto output = maiw::cardiac::resampleLinearNearestBoundary(
      viewOf(dimensions, values), VolumeSpacing{1.0, 1.0, 1.0}, dimensions, VolumeSpacing{1.0, 1.0, 1.0});
  require(output == values, "identity-grid resampling should reproduce input values");
}

void testVoxelCentersAndLinearInterpolation()
{
  const VolumeDimensions dimensions{3, 3, 3};
  const std::vector<double> values = makeRampVolume(dimensions);
  const auto volume = viewOf(dimensions, values);

  requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(volume, 2.0, 1.0, 0.0),
              12.0,
              "voxel-center interpolation mismatch");
  requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(volume, 0.5, 0.0, 0.0),
              0.5,
              "halfway X interpolation mismatch");
  requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(volume, 0.0, 0.5, 0.0),
              5.0,
              "halfway Y interpolation mismatch");
}

void testTrilinearInterpolation()
{
  const VolumeDimensions dimensions{2, 2, 2};
  const std::vector<double> values = makeRampVolume(dimensions);
  const auto volume = viewOf(dimensions, values);

  requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(volume, 0.5, 0.5, 0.5),
              55.5,
              "center trilinear interpolation mismatch");
  requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(volume, 0.25, 0.5, 0.75),
              80.25,
              "off-center trilinear interpolation mismatch");
}

void testNearestBoundaryExtension()
{
  const VolumeDimensions dimensions{2, 2, 2};
  const std::vector<double> values = makeRampVolume(dimensions);
  const auto volume = viewOf(dimensions, values);

  requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(volume, -1.25, 0.0, 0.0),
              0.0,
              "negative out-of-range coordinate should clamp to lower boundary");
  requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(volume, 8.0, 1.0, 1.0),
              111.0,
              "upper out-of-range coordinate should clamp to upper boundary");
  requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(volume, -2.0, 0.5, 0.0),
              5.0,
              "partly outside X and inside Y coordinate mismatch");
}

void testRejectsNonFiniteCoordinates()
{
  const VolumeDimensions dimensions{2, 2, 2};
  const std::vector<double> values = makeRampVolume(dimensions);
  const auto volume = viewOf(dimensions, values);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  requireThrows<std::invalid_argument>(
      [&volume, nan] {
        static_cast<void>(
            maiw::cardiac::sampleTrilinearNearestBoundary(volume, nan, 0.0, 0.0));
      },
      "NaN X coordinate should be rejected");
  requireThrows<std::invalid_argument>(
      [&volume, nan] {
        static_cast<void>(
            maiw::cardiac::sampleTrilinearNearestBoundary(volume, 0.0, nan, 0.0));
      },
      "NaN Y coordinate should be rejected");
  requireThrows<std::invalid_argument>(
      [&volume, nan] {
        static_cast<void>(
            maiw::cardiac::sampleTrilinearNearestBoundary(volume, 0.0, 0.0, nan));
      },
      "NaN Z coordinate should be rejected");
  requireThrows<std::invalid_argument>(
      [&volume, infinity] {
        static_cast<void>(
            maiw::cardiac::sampleTrilinearNearestBoundary(volume, infinity, 0.0, 0.0));
      },
      "positive infinity coordinate should be rejected");
  requireThrows<std::invalid_argument>(
      [&volume, infinity] {
        static_cast<void>(
            maiw::cardiac::sampleTrilinearNearestBoundary(volume, -infinity, 0.0, 0.0));
      },
      "negative infinity coordinate should be rejected");
}

void testSingleVoxelAxisInterpolation()
{
  {
    const VolumeDimensions dimensions{1, 2, 2};
    const std::vector<double> values = makeRampVolume(dimensions);
    requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(
                    viewOf(dimensions, values), 12.0, 0.5, 0.5),
                55.0,
                "single-X-axis interpolation mismatch");
  }
  {
    const VolumeDimensions dimensions{2, 1, 2};
    const std::vector<double> values = makeRampVolume(dimensions);
    requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(
                    viewOf(dimensions, values), 0.5, 12.0, 0.5),
                50.5,
                "single-Y-axis interpolation mismatch");
  }
  {
    const VolumeDimensions dimensions{2, 2, 1};
    const std::vector<double> values = makeRampVolume(dimensions);
    requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(
                    viewOf(dimensions, values), 0.5, 0.5, 12.0),
                5.5,
                "single-Z-axis interpolation mismatch");
  }
  {
    const VolumeDimensions dimensions{1, 1, 1};
    const std::vector<double> values{42.0};
    requireNear(maiw::cardiac::sampleTrilinearNearestBoundary(
                    viewOf(dimensions, values), 12.0, -8.0, 0.5),
                42.0,
                "single-voxel interpolation mismatch");
  }
}

void testResamplingWithSingleSourceDimension()
{
  const VolumeDimensions sourceDimensions{1, 2, 2};
  const std::vector<double> values = makeRampVolume(sourceDimensions);
  const VolumeDimensions targetDimensions{1, 3, 3};
  const auto output = maiw::cardiac::resampleLinearNearestBoundary(
      viewOf(sourceDimensions, values),
      VolumeSpacing{1.0, 1.0, 1.0},
      targetDimensions,
      VolumeSpacing{1.0, 0.5, 0.5});

  require(output.size() == 9, "single-source-dimension resampling size mismatch");
  requireNear(output[4], 55.0, "single-source-dimension resampling center mismatch");
}

void testInvalidInputs()
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  requireThrows<std::invalid_argument>(
      [] { static_cast<void>(maiw::cardiac::targetSizeForAxis(0, 1.0, 1.0)); },
      "zero source dimension should be rejected");
  requireThrows<std::invalid_argument>(
      [] { static_cast<void>(maiw::cardiac::targetSizeForAxis(2, 0.0, 1.0)); },
      "zero source spacing should be rejected");
  requireThrows<std::invalid_argument>(
      [] { static_cast<void>(maiw::cardiac::targetSizeForAxis(2, 1.0, -1.0)); },
      "negative target spacing should be rejected");
  requireThrows<std::invalid_argument>(
      [] { static_cast<void>(maiw::cardiac::targetSizeForAxis(2, 1.0, 0.0)); },
      "zero target spacing should be rejected");
  requireThrows<std::invalid_argument>(
      [nan] { static_cast<void>(maiw::cardiac::targetSizeForAxis(2, 1.0, nan)); },
      "NaN target spacing should be rejected");
  requireThrows<std::invalid_argument>(
      [infinity] { static_cast<void>(maiw::cardiac::targetSizeForAxis(2, 1.0, infinity)); },
      "infinite target spacing should be rejected");
  requireThrows<std::invalid_argument>(
      [nan] { static_cast<void>(maiw::cardiac::targetSizeForAxis(2, nan, 1.0)); },
      "NaN source spacing should be rejected");
  requireThrows<std::invalid_argument>(
      [infinity] {
        static_cast<void>(maiw::cardiac::sourceCoordinateForTargetIndex(
            2, 2, infinity, 1.0, 0));
      },
      "infinite source spacing should be rejected");
  requireThrows<std::invalid_argument>(
      [] { static_cast<void>(maiw::cardiac::roundHalfToEvenNonNegative(-0.5)); },
      "negative round input should be rejected");
  requireThrows<std::overflow_error>(
      [] {
        static_cast<void>(maiw::cardiac::targetSizeForAxis(
            std::numeric_limits<std::size_t>::max(), 1.0, 1.0));
      },
      "target-size overflow should be rejected");
  requireThrows<std::invalid_argument>(
      [] {
        const std::vector<double> values{1.0};
        static_cast<void>(maiw::cardiac::sampleTrilinearNearestBoundary(
            Float64VolumeView{VolumeDimensions{2, 1, 1}, values}, 0.0, 0.0, 0.0));
      },
      "mismatched voxel count should be rejected");
  requireThrows<std::invalid_argument>(
      [] {
        static_cast<void>(
            maiw::cardiac::sourceCoordinateForTargetIndex(2, 2, 1.0, 1.0, 2));
      },
      "out-of-range target index should be rejected");
}

} // namespace

int main()
{
  try
  {
    testFrozenConfig();
    testTargetSizeCalculation();
    testSourceCoordinateCalculation();
    testIdentityGridSampling();
    testVoxelCentersAndLinearInterpolation();
    testTrilinearInterpolation();
    testNearestBoundaryExtension();
    testRejectsNonFiniteCoordinates();
    testSingleVoxelAxisInterpolation();
    testResamplingWithSingleSourceDimension();
    testInvalidInputs();
    std::cout << "Cardiac MRI preprocessing mathematical tests passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI preprocessing mathematical tests failed: " << error.what() << '\n';
    return 1;
  }
}
