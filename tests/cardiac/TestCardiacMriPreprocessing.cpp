#include "maiw/cardiac/CardiacMriPreprocessing.h"

#include "qtviewerpro/core/VolumeData.h"

#include <array>
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
constexpr std::array<double, 9> kIdentityDirection{1.0, 0.0, 0.0,
                                                   0.0, 1.0, 0.0,
                                                   0.0, 0.0, 1.0};

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

std::size_t linearIndex(std::array<std::size_t, 3> dimensions,
                        std::size_t x,
                        std::size_t y,
                        std::size_t z)
{
  return (z * dimensions[1] * dimensions[0]) + (y * dimensions[0]) + x;
}

std::size_t linearIndexXyz(VolumeDimensions dimensions,
                        std::size_t x,
                        std::size_t y,
                        std::size_t z)
{
  return (z * dimensions.height * dimensions.width) + (y * dimensions.width) + x;
}

std::vector<float> makeRampVolumeFloat(std::array<std::size_t, 3> dimensions)
{
  std::vector<float> values(dimensions[0] * dimensions[1] * dimensions[2]);
  for (std::size_t z = 0; z < dimensions[2]; ++z)
  {
    for (std::size_t y = 0; y < dimensions[1]; ++y)
    {
      for (std::size_t x = 0; x < dimensions[0]; ++x)
      {
        values[linearIndex(dimensions, x, y, z)] =
            static_cast<float>(x + (10U * y) + (100U * z));
      }
    }
  }
  return values;
}

qvp::VolumeData makeQvpVolume(
    std::array<std::size_t, 3> dimensions,
    std::array<float, 3> spacing,
    std::array<double, 3> origin,
    std::array<double, 9> direction,
    qvp::VolumeData::CoordinateSystem coordinateSystem = qvp::VolumeData::CoordinateSystem::LPS)
{
  qvp::VolumeData::SpatialGeometry geometry;
  geometry.origin = origin;
  geometry.direction = direction;
  geometry.coordinateSystem = coordinateSystem;
  geometry.hasOrientation = true;

  return qvp::VolumeData(dimensions[0],
                         dimensions[1],
                         dimensions[2],
                         spacing[0],
                         spacing[1],
                         spacing[2],
                         makeRampVolumeFloat(dimensions),
                         geometry);
}

std::vector<double> makeRampVolumeDouble(VolumeDimensions dimensions, double offset = 0.0)
{
  std::vector<double> values(dimensions.width * dimensions.height * dimensions.depth);
  for (std::size_t z = 0; z < dimensions.depth; ++z)
  {
    for (std::size_t y = 0; y < dimensions.height; ++y)
    {
      for (std::size_t x = 0; x < dimensions.width; ++x)
      {
        values[linearIndexXyz(dimensions, x, y, z)] =
            offset + static_cast<double>(x + (1000U * y) + (1000000U * z));
      }
    }
  }
  return values;
}

maiw::cardiac::CardiacMriXyzVolume makeXyzVolume(
    VolumeDimensions dimensions,
    VolumeSpacing spacing = VolumeSpacing{1.5, 1.5, 7.5},
    std::array<double, 3> origin = {0.0, 0.0, 0.0},
    std::array<double, 9> direction = kIdentityDirection,
    double valueOffset = 0.0)
{
  maiw::cardiac::CardiacMriXyzVolume volume;
  volume.dimensions = dimensions;
  volume.spacing = spacing;
  volume.origin = origin;
  volume.direction = direction;
  volume.voxels = makeRampVolumeDouble(dimensions, valueOffset);
  return volume;
}

std::array<double, 3> physicalPoint(const qvp::VolumeData& volume,
                                    std::size_t x,
                                    std::size_t y,
                                    std::size_t z)
{
  const auto& geometry = volume.spatialGeometry();
  const std::array<double, 3> scaled{
      static_cast<double>(x) * static_cast<double>(volume.spacingX()),
      static_cast<double>(y) * static_cast<double>(volume.spacingY()),
      static_cast<double>(z) * static_cast<double>(volume.spacingZ())};

  return {geometry.origin[0] + (geometry.direction[0] * scaled[0]) +
              (geometry.direction[1] * scaled[1]) + (geometry.direction[2] * scaled[2]),
          geometry.origin[1] + (geometry.direction[3] * scaled[0]) +
              (geometry.direction[4] * scaled[1]) + (geometry.direction[5] * scaled[2]),
          geometry.origin[2] + (geometry.direction[6] * scaled[0]) +
              (geometry.direction[7] * scaled[1]) + (geometry.direction[8] * scaled[2])};
}

std::array<double, 3> physicalPoint(const maiw::cardiac::CardiacMriXyzVolume& volume,
                                    double x,
                                    double y,
                                    double z)
{
  const std::array<double, 3> scaled{x * volume.spacing.x,
                                     y * volume.spacing.y,
                                     z * volume.spacing.z};
  return {volume.origin[0] + (volume.direction[0] * scaled[0]) +
              (volume.direction[1] * scaled[1]) + (volume.direction[2] * scaled[2]),
          volume.origin[1] + (volume.direction[3] * scaled[0]) +
              (volume.direction[4] * scaled[1]) + (volume.direction[5] * scaled[2]),
          volume.origin[2] + (volume.direction[6] * scaled[0]) +
              (volume.direction[7] * scaled[1]) + (volume.direction[8] * scaled[2])};
}

void requirePointNear(const std::array<double, 3>& actual,
                      const std::array<double, 3>& expected,
                      const std::string& message)
{
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    requireNear(actual[axis], expected[axis], message);
  }
}

void requireVolumeSpacingNear(VolumeSpacing actual, VolumeSpacing expected, const std::string& message)
{
  requireNear(actual.x, expected.x, message + " X");
  requireNear(actual.y, expected.y, message + " Y");
  requireNear(actual.z, expected.z, message + " Z");
}

void requireDirectionNear(const qvp::VolumeData& volume,
                          const std::array<double, 9>& expected,
                          const std::string& message)
{
  const auto& actual = volume.spatialGeometry().direction;
  for (std::size_t index = 0; index < expected.size(); ++index)
  {
    requireNear(actual[index], expected[index], message);
  }
}

void requireDirectionNear(const maiw::cardiac::CardiacMriXyzVolume& volume,
                          const std::array<double, 9>& expected,
                          const std::string& message)
{
  for (std::size_t index = 0; index < expected.size(); ++index)
  {
    requireNear(volume.direction[index], expected[index], message);
  }
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

void testAlreadyLpsVolumeRemainsVoxelIdentical()
{
  const auto volume = makeQvpVolume({2, 3, 4},
                                   {1.0F, 2.0F, 3.0F},
                                   {10.0, 20.0, 30.0},
                                   kIdentityDirection);
  const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

  require(oriented.width() == 2 && oriented.height() == 3 && oriented.depth() == 4,
          "already-LPS dimensions changed");
  require(oriented.voxels() == volume.voxels(), "already-LPS voxels changed");
  requireNear(oriented.spacingX(), 1.0, "already-LPS spacing X changed");
  requireNear(oriented.spacingY(), 2.0, "already-LPS spacing Y changed");
  requireNear(oriented.spacingZ(), 3.0, "already-LPS spacing Z changed");
  requireDirectionNear(oriented, kIdentityDirection, "already-LPS direction changed");
  requirePointNear(oriented.spatialGeometry().origin,
                   volume.spatialGeometry().origin,
                   "already-LPS origin changed");
}

void testSingleAxisFlipsToLps()
{
  {
    const std::array<double, 9> direction{-1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0};
    const auto volume = makeQvpVolume({3, 2, 2}, {2.0F, 3.0F, 4.0F}, {10.0, 20.0, 30.0}, direction);
    const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

    require(oriented.voxels()[linearIndex({3, 2, 2}, 0, 1, 1)] ==
                volume.voxels()[linearIndex({3, 2, 2}, 2, 1, 1)],
            "single X flip voxel remapping mismatch");
    requireDirectionNear(oriented, kIdentityDirection, "single X flip direction mismatch");
    requirePointNear(physicalPoint(oriented, 0, 1, 1),
                     physicalPoint(volume, 2, 1, 1),
                     "single X flip physical point mismatch");
  }
  {
    const std::array<double, 9> direction{1.0, 0.0, 0.0,
                                          0.0, -1.0, 0.0,
                                          0.0, 0.0, 1.0};
    const auto volume = makeQvpVolume({2, 3, 2}, {2.0F, 3.0F, 4.0F}, {10.0, 20.0, 30.0}, direction);
    const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

    require(oriented.voxels()[linearIndex({2, 3, 2}, 1, 0, 1)] ==
                volume.voxels()[linearIndex({2, 3, 2}, 1, 2, 1)],
            "single Y flip voxel remapping mismatch");
    requireDirectionNear(oriented, kIdentityDirection, "single Y flip direction mismatch");
    requirePointNear(physicalPoint(oriented, 1, 0, 1),
                     physicalPoint(volume, 1, 2, 1),
                     "single Y flip physical point mismatch");
  }
  {
    const std::array<double, 9> direction{1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, -1.0};
    const auto volume = makeQvpVolume({2, 2, 3}, {2.0F, 3.0F, 4.0F}, {10.0, 20.0, 30.0}, direction);
    const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

    require(oriented.voxels()[linearIndex({2, 2, 3}, 1, 1, 0)] ==
                volume.voxels()[linearIndex({2, 2, 3}, 1, 1, 2)],
            "single Z flip voxel remapping mismatch");
    requireDirectionNear(oriented, kIdentityDirection, "single Z flip direction mismatch");
    requirePointNear(physicalPoint(oriented, 1, 1, 0),
                     physicalPoint(volume, 1, 1, 2),
                     "single Z flip physical point mismatch");
  }
}

void testAxisPermutationWithoutFlip()
{
  const std::array<double, 9> direction{0.0, 1.0, 0.0,
                                        1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0};
  const auto volume = makeQvpVolume({2, 3, 4}, {2.0F, 3.0F, 4.0F}, {10.0, 20.0, 30.0}, direction);
  const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

  require(oriented.width() == 3 && oriented.height() == 2 && oriented.depth() == 4,
          "permutation dimensions mismatch");
  requireNear(oriented.spacingX(), 3.0, "permutation spacing X mismatch");
  requireNear(oriented.spacingY(), 2.0, "permutation spacing Y mismatch");
  requireNear(oriented.spacingZ(), 4.0, "permutation spacing Z mismatch");
  require(oriented.voxels()[linearIndex({3, 2, 4}, 2, 1, 3)] ==
              volume.voxels()[linearIndex({2, 3, 4}, 1, 2, 3)],
          "permutation voxel remapping mismatch");
  requireDirectionNear(oriented, kIdentityDirection, "permutation direction mismatch");
  requirePointNear(physicalPoint(oriented, 2, 1, 3),
                   physicalPoint(volume, 1, 2, 3),
                   "permutation physical point mismatch");
}

void testPermutationPlusFlip()
{
  const std::array<double, 9> direction{0.0, 0.0, -1.0,
                                        -1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0};
  const auto volume = makeQvpVolume({2, 3, 4}, {2.0F, 3.0F, 4.0F}, {10.0, 20.0, 30.0}, direction);
  const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

  require(oriented.width() == 4 && oriented.height() == 2 && oriented.depth() == 3,
          "permutation-plus-flip dimensions mismatch");
  requireNear(oriented.spacingX(), 4.0, "permutation-plus-flip spacing X mismatch");
  requireNear(oriented.spacingY(), 2.0, "permutation-plus-flip spacing Y mismatch");
  requireNear(oriented.spacingZ(), 3.0, "permutation-plus-flip spacing Z mismatch");
  require(oriented.voxels()[linearIndex({4, 2, 3}, 0, 0, 0)] ==
              volume.voxels()[linearIndex({2, 3, 4}, 1, 0, 3)],
          "permutation-plus-flip voxel remapping mismatch");
  requireDirectionNear(oriented, kIdentityDirection, "permutation-plus-flip direction mismatch");
  requirePointNear(physicalPoint(oriented, 0, 0, 0),
                   physicalPoint(volume, 1, 0, 3),
                   "permutation-plus-flip origin physical point mismatch");
  requirePointNear(physicalPoint(oriented, 3, 1, 2),
                   physicalPoint(volume, 0, 2, 0),
                   "permutation-plus-flip corner physical point mismatch");
}

void testXFastestIndexingAfterReorientation()
{
  const std::array<double, 9> direction{0.0, 1.0, 0.0,
                                        1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0};
  const auto volume = makeQvpVolume({3, 2, 1}, {1.0F, 1.0F, 1.0F}, {0.0, 0.0, 0.0}, direction);
  const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

  require(oriented.voxels()[0] == volume.voxels()[0], "X-fastest first voxel mismatch");
  require(oriented.voxels()[1] == volume.voxels()[linearIndex({3, 2, 1}, 0, 1, 0)],
          "X-fastest adjacent-X voxel mismatch");
  require(oriented.voxels()[oriented.width()] == volume.voxels()[linearIndex({3, 2, 1}, 1, 0, 0)],
          "X-fastest next-row voxel mismatch");
}

void testMultipleAxisFlipsPreservePhysicalLocations()
{
  const std::array<double, 9> direction{-1.0, 0.0, 0.0,
                                        0.0, -1.0, 0.0,
                                        0.0, 0.0, 1.0};
  const auto volume = makeQvpVolume({3, 4, 2}, {2.0F, 3.0F, 4.0F}, {10.0, 20.0, 30.0}, direction);
  const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

  requirePointNear(physicalPoint(oriented, 0, 0, 1),
                   physicalPoint(volume, 2, 3, 1),
                   "multiple-axis flip lower corner physical point mismatch");
  requirePointNear(physicalPoint(oriented, 2, 3, 0),
                   physicalPoint(volume, 0, 0, 0),
                   "multiple-axis flip upper corner physical point mismatch");
}

void testRasCoordinateSystemIsConvertedToLps()
{
  const auto volume = makeQvpVolume({3, 4, 2},
                                   {2.0F, 3.0F, 4.0F},
                                   {10.0, 20.0, 30.0},
                                   kIdentityDirection,
                                   qvp::VolumeData::CoordinateSystem::RAS);
  const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

  require(oriented.spatialGeometry().coordinateSystem == qvp::VolumeData::CoordinateSystem::LPS,
          "RAS input should produce LPS geometry");
  requireDirectionNear(oriented, kIdentityDirection, "RAS-to-LPS direction mismatch");
  require(oriented.voxels()[linearIndex({3, 4, 2}, 0, 0, 1)] ==
              volume.voxels()[linearIndex({3, 4, 2}, 2, 3, 1)],
          "RAS-to-LPS voxel remapping mismatch");
  requirePointNear(physicalPoint(oriented, 0, 0, 1),
                   std::array<double, 3>{-14.0, -29.0, 34.0},
                   "RAS-to-LPS physical point mismatch");
}

void testOrientedPairValidation()
{
  const qvp::VolumeData ed = maiw::cardiac::normalizeVolumeDataToLps(
      makeQvpVolume({2, 3, 4}, {1.0F, 2.0F, 3.0F}, {10.0, 20.0, 30.0}, kIdentityDirection));
  qvp::VolumeData es = maiw::cardiac::normalizeVolumeDataToLps(
      makeQvpVolume({2, 3, 4}, {1.0F, 2.0F, 3.0F}, {-1.0, -2.0, -3.0}, kIdentityDirection));

  maiw::cardiac::validateLpsOrientedVolumePair(ed, es);

  requireThrows<std::invalid_argument>(
      [&ed] {
        const qvp::VolumeData mismatched = maiw::cardiac::normalizeVolumeDataToLps(
            makeQvpVolume({2, 3, 5}, {1.0F, 2.0F, 3.0F}, {10.0, 20.0, 30.0}, kIdentityDirection));
        maiw::cardiac::validateLpsOrientedVolumePair(ed, mismatched);
      },
      "oriented pair shape mismatch should be rejected");
  requireThrows<std::invalid_argument>(
      [&ed] {
        const qvp::VolumeData mismatched = maiw::cardiac::normalizeVolumeDataToLps(
            makeQvpVolume({2, 3, 4},
                          {std::nextafter(1.0F, 2.0F), 2.0F, 3.0F},
                          {10.0, 20.0, 30.0},
                          kIdentityDirection));
        maiw::cardiac::validateLpsOrientedVolumePair(ed, mismatched);
      },
      "oriented pair spacing mismatch should be rejected");
  const double angle = 0.01;
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  const std::array<double, 9> rotatedDirection{cosine, -sine, 0.0,
                                               sine, cosine, 0.0,
                                               0.0, 0.0, 1.0};
  const qvp::VolumeData rotated = maiw::cardiac::normalizeVolumeDataToLps(
      makeQvpVolume({2, 3, 4}, {1.0F, 2.0F, 3.0F}, {10.0, 20.0, 30.0}, rotatedDirection));
  maiw::cardiac::validateLpsOrientedVolumePair(rotated, rotated);
  requireThrows<std::invalid_argument>(
      [&ed, &rotated] { maiw::cardiac::validateLpsOrientedVolumePair(ed, rotated); },
      "valid oriented pair direction-spacing matrix mismatch should be rejected");

  qvp::VolumeData::SpatialGeometry shiftedGeometry = es.spatialGeometry();
  shiftedGeometry.origin = {999.0, -999.0, 42.0};
  es = qvp::VolumeData(es.width(),
                       es.height(),
                       es.depth(),
                       es.spacingX(),
                       es.spacingY(),
                       es.spacingZ(),
                       es.voxels(),
                       shiftedGeometry,
                       es.voxelAxisAnatomy());
  maiw::cardiac::validateLpsOrientedVolumePair(ed, es);
}

void testValidMildlyObliqueOrientationIsAccepted()
{
  const double angle = 0.25;
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  const std::array<double, 9> direction{cosine, -sine, 0.0,
                                        sine, cosine, 0.0,
                                        0.0, 0.0, 1.0};
  const auto volume = makeQvpVolume({2, 3, 4}, {1.0F, 2.0F, 3.0F}, {10.0, 20.0, 30.0}, direction);
  const qvp::VolumeData oriented = maiw::cardiac::normalizeVolumeDataToLps(volume);

  require(oriented.width() == 2 && oriented.height() == 3 && oriented.depth() == 4,
          "mildly oblique dimensions changed");
  require(oriented.voxels() == volume.voxels(), "mildly oblique voxels changed");
  requireDirectionNear(oriented, direction, "mildly oblique direction changed");
  requirePointNear(physicalPoint(oriented, 1, 2, 3),
                   physicalPoint(volume, 1, 2, 3),
                   "mildly oblique physical point mismatch");
}

void testInvalidOrientationGeometryIsRejected()
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  qvp::VolumeData::SpatialGeometry unknownGeometry;
  unknownGeometry.direction = kIdentityDirection;
  unknownGeometry.coordinateSystem = qvp::VolumeData::CoordinateSystem::Unknown;
  unknownGeometry.hasOrientation = true;
  const qvp::VolumeData unknownCoordinateSystem(
      2, 2, 2, 1.0F, 1.0F, 1.0F, makeRampVolumeFloat({2, 2, 2}), unknownGeometry);
  requireThrows<std::invalid_argument>(
      [&unknownCoordinateSystem] {
        static_cast<void>(maiw::cardiac::normalizeVolumeDataToLps(unknownCoordinateSystem));
      },
      "unknown coordinate system should be rejected");

  const std::array<std::array<double, 9>, 6> invalidDirections{
      std::array<double, 9>{1.0, 0.25, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0},
      std::array<double, 9>{2.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0},
      std::array<double, 9>{1.0, 0.1, 0.0,
                            0.0, std::sqrt(0.99), 0.0,
                            0.0, 0.0, 1.0},
      std::array<double, 9>{std::sqrt(0.5), -std::sqrt(0.5), 0.0,
                            std::sqrt(0.5), std::sqrt(0.5), 0.0,
                            0.0, 0.0, 1.0},
      std::array<double, 9>{nan, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0},
      std::array<double, 9>{infinity, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0}};
  for (const auto& direction : invalidDirections)
  {
    const qvp::VolumeData invalidVolume =
        makeQvpVolume({2, 2, 2}, {1.0F, 1.0F, 1.0F}, {0.0, 0.0, 0.0}, direction);
    requireThrows<std::invalid_argument>(
        [&invalidVolume] {
          static_cast<void>(maiw::cardiac::normalizeVolumeDataToLps(invalidVolume));
        },
        "invalid orientation direction should be rejected");
  }

  qvp::VolumeData::SpatialGeometry duplicateGeometry;
  duplicateGeometry.direction = {1.0, 1.0, 0.0,
                                 0.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0};
  duplicateGeometry.coordinateSystem = qvp::VolumeData::CoordinateSystem::LPS;
  duplicateGeometry.hasOrientation = true;
  const qvp::VolumeData duplicateAxes(
      2, 2, 2, 1.0F, 1.0F, 1.0F, makeRampVolumeFloat({2, 2, 2}), duplicateGeometry);
  requireThrows<std::invalid_argument>(
      [&duplicateAxes] {
        static_cast<void>(maiw::cardiac::normalizeVolumeDataToLps(duplicateAxes));
      },
      "duplicate anatomy axes should be rejected");

  const qvp::VolumeData validVolume = maiw::cardiac::normalizeVolumeDataToLps(
      makeQvpVolume({2, 2, 2}, {1.0F, 1.0F, 1.0F}, {0.0, 0.0, 0.0}, kIdentityDirection));
  qvp::VolumeData::SpatialGeometry nonFiniteDirectionGeometry = validVolume.spatialGeometry();
  nonFiniteDirectionGeometry.direction[0] = nan;
  const qvp::VolumeData nonFiniteDirectionVolume(validVolume.width(),
                                                 validVolume.height(),
                                                 validVolume.depth(),
                                                 validVolume.spacingX(),
                                                 validVolume.spacingY(),
                                                 validVolume.spacingZ(),
                                                 validVolume.voxels(),
                                                 nonFiniteDirectionGeometry,
                                                 validVolume.voxelAxisAnatomy());
  requireThrows<std::invalid_argument>(
      [&validVolume, &nonFiniteDirectionVolume] {
        maiw::cardiac::validateLpsOrientedVolumePair(validVolume, nonFiniteDirectionVolume);
      },
      "non-finite pair-validation direction should be rejected");
}

void testEdDerivedResamplingUsesSharedDestinationGrid()
{
  const auto edVolume = makeQvpVolume({3, 3, 2},
                                      {1.5F, 1.5F, 7.5F},
                                      {10.0, 20.0, 30.0},
                                      kIdentityDirection);
  const auto esVolume = makeQvpVolume({3, 3, 2},
                                      {1.5F, 1.5F, 7.5F},
                                      {100.0, 200.0, 300.0},
                                      kIdentityDirection);

  const maiw::cardiac::CardiacMriXyzVolumePair pair =
      maiw::cardiac::resampleOrientedPairToEdDerivedGrid(edVolume, esVolume);

  require(pair.ed.dimensions.width == 3 && pair.ed.dimensions.height == 3 &&
              pair.ed.dimensions.depth == 2,
          "ED-derived target dimensions mismatch");
  require(pair.es.dimensions.width == pair.ed.dimensions.width &&
              pair.es.dimensions.height == pair.ed.dimensions.height &&
              pair.es.dimensions.depth == pair.ed.dimensions.depth,
          "ED/ES target dimensions should match");
  requireVolumeSpacingNear(pair.ed.spacing, VolumeSpacing{1.5, 1.5, 7.5}, "ED target spacing");
  requireVolumeSpacingNear(pair.es.spacing, pair.ed.spacing, "ES target spacing");
  requirePointNear(pair.ed.origin, edVolume.spatialGeometry().origin, "identity-grid ED origin");
  requirePointNear(pair.es.origin, pair.ed.origin, "ES should reuse ED-derived destination origin");
  requireDirectionNear(pair.ed, kIdentityDirection, "ED destination direction");
  requireDirectionNear(pair.es, kIdentityDirection, "ES destination direction");
  require(pair.ed.voxels.size() == edVolume.voxels().size(), "ED resampled voxel count mismatch");
  require(pair.ed.voxels[linearIndexXyz(pair.ed.dimensions, 2, 1, 1)] ==
              static_cast<double>(edVolume.voxels()[linearIndex({3, 3, 2}, 2, 1, 1)]),
          "identity-equivalent ED resampling value mismatch");
  require(pair.es.origin != esVolume.spatialGeometry().origin,
          "ES origin should not be independently used as destination origin");
}

void testEdDerivedResamplingMatchesCoordinateMath()
{
  const auto edVolume = makeQvpVolume({3, 3, 1},
                                      {3.0F, 3.0F, 7.5F},
                                      {0.0, 0.0, 0.0},
                                      kIdentityDirection);
  const auto esVolume = makeQvpVolume({3, 3, 1},
                                      {3.0F, 3.0F, 7.5F},
                                      {42.0, 24.0, 12.0},
                                      kIdentityDirection);

  const maiw::cardiac::CardiacMriXyzVolumePair pair =
      maiw::cardiac::resampleOrientedPairToEdDerivedGrid(edVolume, esVolume);

  require(pair.ed.dimensions.width == 5 && pair.ed.dimensions.height == 5 &&
              pair.ed.dimensions.depth == 1,
          "nontrivial ED-derived target dimensions mismatch");
  requireNear(pair.ed.voxels[linearIndexXyz(pair.ed.dimensions, 1, 2, 0)],
              10.5,
              "nontrivial spacing interpolation mismatch");
  requireNear(pair.ed.voxels[linearIndexXyz(pair.ed.dimensions, 1, 0, 0)],
              0.5,
              "X-fastest adjacent-X resampled storage mismatch");
  requireNear(pair.ed.voxels[linearIndexXyz(pair.ed.dimensions, 0, 1, 0)],
              5.0,
              "X-fastest next-row resampled storage mismatch");

  const std::array<double, 3> sourceCenter = physicalPoint(edVolume, 1, 1, 0);
  const std::array<double, 3> targetCenter = physicalPoint(pair.ed, 2.0, 2.0, 0.0);
  requirePointNear(targetCenter, sourceCenter, "destination physical center mismatch");
  requirePointNear(pair.es.origin, pair.ed.origin, "ES should reuse nontrivial ED origin");
  requireDirectionNear(pair.es, pair.ed.direction, "ES should reuse ED destination direction");
}

void testResamplingRejectsIncompatibleOrientedPair()
{
  const qvp::VolumeData ed = maiw::cardiac::normalizeVolumeDataToLps(
      makeQvpVolume({3, 3, 1}, {1.5F, 1.5F, 7.5F}, {0.0, 0.0, 0.0}, kIdentityDirection));
  const qvp::VolumeData es = maiw::cardiac::normalizeVolumeDataToLps(
      makeQvpVolume({3, 3, 1}, {1.5F, 1.6F, 7.5F}, {0.0, 0.0, 0.0}, kIdentityDirection));

  requireThrows<std::invalid_argument>(
      [&ed, &es] {
        static_cast<void>(maiw::cardiac::resampleOrientedPairToEdDerivedGrid(ed, es));
      },
      "resampling should reject incompatible ED/ES geometry before processing");
}

void testFrozenXyCropWindowDerivation()
{
  const maiw::cardiac::XyCropWindow exact =
      maiw::cardiac::deriveFrozenXyCenterCropWindow(makeXyzVolume(VolumeDimensions{144, 144, 3}));
  require(exact.startX == 0 && exact.startY == 0 && exact.endX == 144 && exact.endY == 144,
          "exact-size crop window mismatch");

  const maiw::cardiac::XyCropWindow even =
      maiw::cardiac::deriveFrozenXyCenterCropWindow(makeXyzVolume(VolumeDimensions{148, 150, 2}));
  require(even.startX == 2 && even.endX == 146 && even.startY == 3 && even.endY == 147,
          "even-excess crop window mismatch");

  const maiw::cardiac::XyCropWindow oddX =
      maiw::cardiac::deriveFrozenXyCenterCropWindow(makeXyzVolume(VolumeDimensions{147, 144, 1}));
  require(oddX.startX == 1 && oddX.endX == 145,
          "odd-X crop should remove floor excess from lower side");

  const maiw::cardiac::XyCropWindow oddY =
      maiw::cardiac::deriveFrozenXyCenterCropWindow(makeXyzVolume(VolumeDimensions{144, 147, 1}));
  require(oddY.startY == 1 && oddY.endY == 145,
          "odd-Y crop should remove floor excess from lower side");

  requireThrows<std::invalid_argument>(
      [] {
        static_cast<void>(
            maiw::cardiac::deriveFrozenXyCenterCropWindow(makeXyzVolume(VolumeDimensions{143, 144, 1})));
      },
      "crop should reject source X smaller than frozen width");
  requireThrows<std::invalid_argument>(
      [] {
        static_cast<void>(
            maiw::cardiac::deriveFrozenXyCenterCropWindow(makeXyzVolume(VolumeDimensions{144, 143, 1})));
      },
      "crop should reject source Y smaller than frozen height");
}

void testFrozenXyCropCopiesVoxelsAndUpdatesOrigin()
{
  const maiw::cardiac::CardiacMriXyzVolume source =
      makeXyzVolume(VolumeDimensions{147, 146, 3},
                    VolumeSpacing{1.5, 2.0, 7.5},
                    {10.0, 20.0, 30.0});
  const maiw::cardiac::XyCropWindow window =
      maiw::cardiac::deriveFrozenXyCenterCropWindow(source);
  const maiw::cardiac::CardiacMriXyzVolume cropped =
      maiw::cardiac::cropVolumeToWindow(source, window);

  require(cropped.dimensions.width == 144 && cropped.dimensions.height == 144 &&
              cropped.dimensions.depth == 3,
          "cropped dimensions mismatch");
  requireNear(cropped.voxels[linearIndexXyz(cropped.dimensions, 0, 0, 0)],
              source.voxels[linearIndexXyz(source.dimensions, 1, 1, 0)],
              "cropped first voxel mismatch");
  requireNear(cropped.voxels[linearIndexXyz(cropped.dimensions, 1, 0, 0)],
              source.voxels[linearIndexXyz(source.dimensions, 2, 1, 0)],
              "cropped X-fastest adjacent-X mismatch");
  requireNear(cropped.voxels[linearIndexXyz(cropped.dimensions, 0, 1, 0)],
              source.voxels[linearIndexXyz(source.dimensions, 1, 2, 0)],
              "cropped X-fastest next-row mismatch");
  requireNear(cropped.voxels[linearIndexXyz(cropped.dimensions, 143, 143, 2)],
              source.voxels[linearIndexXyz(source.dimensions, 144, 144, 2)],
              "cropped last retained voxel mismatch");
  requirePointNear(cropped.origin, physicalPoint(source, 1.0, 1.0, 0.0), "cropped origin mismatch");
  requireVolumeSpacingNear(cropped.spacing, source.spacing, "cropped spacing");
  requireDirectionNear(cropped, source.direction, "cropped direction");
}

void testFrozenXyCropPairUsesSharedEdWindow()
{
  const maiw::cardiac::CardiacMriXyzVolume ed =
      makeXyzVolume(VolumeDimensions{147, 146, 2},
                    VolumeSpacing{1.5, 2.0, 7.5},
                    {10.0, 20.0, 30.0});
  const maiw::cardiac::CardiacMriXyzVolume es =
      makeXyzVolume(VolumeDimensions{147, 146, 2},
                    VolumeSpacing{1.5, 2.0, 7.5},
                    {10.0, 20.0, 30.0},
                    kIdentityDirection,
                    5000000.0);

  const maiw::cardiac::CardiacMriXyzVolumePair cropped =
      maiw::cardiac::cropResampledPairToFrozenXy(ed, es);

  requirePointNear(cropped.es.origin, cropped.ed.origin, "cropped ED/ES origin mismatch");
  requireDirectionNear(cropped.es, cropped.ed.direction, "cropped ED/ES direction mismatch");
  requireNear(cropped.es.voxels[linearIndexXyz(cropped.es.dimensions, 0, 0, 0)],
              es.voxels[linearIndexXyz(es.dimensions, 1, 1, 0)],
              "ES crop should use the ED-derived crop start");
}

void testCropRejectsOverflowingVolumeCount()
{
  maiw::cardiac::CardiacMriXyzVolume huge;
  huge.dimensions = VolumeDimensions{144, 144, std::numeric_limits<std::size_t>::max()};
  huge.spacing = VolumeSpacing{1.5, 1.5, 7.5};
  huge.origin = {0.0, 0.0, 0.0};
  huge.direction = kIdentityDirection;

  requireThrows<std::overflow_error>(
      [&huge] {
        static_cast<void>(maiw::cardiac::cropVolumeToWindow(
            huge, maiw::cardiac::XyCropWindow{0, 0, 144, 144}));
      },
      "crop should reject overflowing source element count");
}

void testInvalidInputs()
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  requireThrows<std::invalid_argument>(
      [] { static_cast<void>(maiw::cardiac::targetSizeForAxis(0, 1.0, 1.0)); },
      "zero source dimension should be rejected");
  requireThrows<std::invalid_argument>(
      [nan] {
        static_cast<void>(maiw::cardiac::normalizeVolumeDataToLps(
            makeQvpVolume({2, 2, 2},
                          {1.0F, static_cast<float>(nan), 1.0F},
                          {0.0, 0.0, 0.0},
                          kIdentityDirection)));
      },
      "NaN VolumeData spacing should be rejected");
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
    testAlreadyLpsVolumeRemainsVoxelIdentical();
    testSingleAxisFlipsToLps();
    testAxisPermutationWithoutFlip();
    testPermutationPlusFlip();
    testXFastestIndexingAfterReorientation();
    testMultipleAxisFlipsPreservePhysicalLocations();
    testRasCoordinateSystemIsConvertedToLps();
    testOrientedPairValidation();
    testValidMildlyObliqueOrientationIsAccepted();
    testInvalidOrientationGeometryIsRejected();
    testEdDerivedResamplingUsesSharedDestinationGrid();
    testEdDerivedResamplingMatchesCoordinateMath();
    testResamplingRejectsIncompatibleOrientedPair();
    testFrozenXyCropWindowDerivation();
    testFrozenXyCropCopiesVoxelsAndUpdatesOrigin();
    testFrozenXyCropPairUsesSharedEdWindow();
    testCropRejectsOverflowingVolumeCount();
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
