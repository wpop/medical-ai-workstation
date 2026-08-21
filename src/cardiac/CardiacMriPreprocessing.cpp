#include "maiw/cardiac/CardiacMriPreprocessing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

void requirePositiveDimension(std::size_t value, const char* name)
{
  if (value == 0)
  {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

void requirePositiveFiniteSpacing(double value, const char* name)
{
  if (!std::isfinite(value) || value <= 0.0)
  {
    throw std::invalid_argument(std::string(name) + " must be positive and finite");
  }
}

void requireFiniteCoordinate(double value, const char* name)
{
  if (!std::isfinite(value))
  {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

void requireValidDimensions(maiw::cardiac::VolumeDimensions dimensions)
{
  requirePositiveDimension(dimensions.width, "width");
  requirePositiveDimension(dimensions.height, "height");
  requirePositiveDimension(dimensions.depth, "depth");
}

void requireValidSpacing(maiw::cardiac::VolumeSpacing spacing)
{
  requirePositiveFiniteSpacing(spacing.x, "spacing.x");
  requirePositiveFiniteSpacing(spacing.y, "spacing.y");
  requirePositiveFiniteSpacing(spacing.z, "spacing.z");
}

std::size_t checkedMultiply(std::size_t lhs, std::size_t rhs)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
  {
    throw std::overflow_error("Volume element count exceeds size limits");
  }
  return lhs * rhs;
}

std::size_t checkedElementCount(maiw::cardiac::VolumeDimensions dimensions)
{
  return checkedMultiply(checkedMultiply(dimensions.width, dimensions.height), dimensions.depth);
}

double clampCoordinate(double coordinate, std::size_t size)
{
  const double upper = static_cast<double>(size - 1);
  return std::clamp(coordinate, 0.0, upper);
}

struct AxisInterpolation
{
  std::size_t lower = 0;
  std::size_t upper = 0;
  double weight = 0.0;
};

AxisInterpolation interpolationForAxis(double coordinate, std::size_t size, const char* name)
{
  requirePositiveDimension(size, "axis size");
  requireFiniteCoordinate(coordinate, name);
  const double clamped = clampCoordinate(coordinate, size);
  const double lowerAsDouble = std::floor(clamped);
  const auto lower = static_cast<std::size_t>(lowerAsDouble);
  const std::size_t upper = std::min(lower + 1, size - 1);
  return AxisInterpolation{lower, upper, clamped - lowerAsDouble};
}

std::size_t voxelIndex(maiw::cardiac::VolumeDimensions dimensions,
                       std::size_t x,
                       std::size_t y,
                       std::size_t z)
{
  return (z * dimensions.height * dimensions.width) + (y * dimensions.width) + x;
}

double lerp(double lhs, double rhs, double weight)
{
  return (lhs * (1.0 - weight)) + (rhs * weight);
}

} // namespace

namespace maiw::cardiac
{

CardiacMriPreprocessingConfig frozenCardiacMriPreprocessingConfig() noexcept
{
  return CardiacMriPreprocessingConfig{};
}

std::size_t roundHalfToEvenNonNegative(double value)
{
  if (!std::isfinite(value) || value < 0.0)
  {
    throw std::invalid_argument("Value must be non-negative and finite");
  }

  const double lowerAsDouble = std::floor(value);
  if (lowerAsDouble >= static_cast<double>(std::numeric_limits<std::size_t>::max()))
  {
    throw std::overflow_error("Rounded value exceeds size limits");
  }

  const auto lower = static_cast<std::size_t>(lowerAsDouble);
  const double fraction = value - lowerAsDouble;
  if (fraction < 0.5)
  {
    return lower;
  }
  if (fraction > 0.5)
  {
    if (lower == std::numeric_limits<std::size_t>::max())
    {
      throw std::overflow_error("Rounded value exceeds size limits");
    }
    return lower + 1;
  }

  if ((lower % 2U) == 0U)
  {
    return lower;
  }
  if (lower == std::numeric_limits<std::size_t>::max())
  {
    throw std::overflow_error("Rounded value exceeds size limits");
  }
  return lower + 1;
}

std::size_t targetSizeForAxis(std::size_t sourceSize,
                              double sourceSpacing,
                              double targetSpacing)
{
  requirePositiveDimension(sourceSize, "sourceSize");
  requirePositiveFiniteSpacing(sourceSpacing, "sourceSpacing");
  requirePositiveFiniteSpacing(targetSpacing, "targetSpacing");

  const double centerSpan =
      (static_cast<double>(sourceSize) - 1.0) * static_cast<double>(sourceSpacing);
  const double scaledExtent = centerSpan / targetSpacing;
  const std::size_t roundedExtent = roundHalfToEvenNonNegative(scaledExtent);
  if (roundedExtent == std::numeric_limits<std::size_t>::max())
  {
    throw std::overflow_error("Target size exceeds size limits");
  }
  return std::max<std::size_t>(1, roundedExtent + 1U);
}

double sourceCoordinateForTargetIndex(std::size_t sourceSize,
                                      std::size_t targetSize,
                                      double sourceSpacing,
                                      double targetSpacing,
                                      std::size_t targetIndex)
{
  requirePositiveDimension(sourceSize, "sourceSize");
  requirePositiveDimension(targetSize, "targetSize");
  requirePositiveFiniteSpacing(sourceSpacing, "sourceSpacing");
  requirePositiveFiniteSpacing(targetSpacing, "targetSpacing");
  if (targetIndex >= targetSize)
  {
    throw std::invalid_argument("targetIndex must be inside targetSize");
  }

  const double sourceCenter = (static_cast<double>(sourceSize) - 1.0) / 2.0;
  const double targetCenter = (static_cast<double>(targetSize) - 1.0) / 2.0;
  const double spacingRatio = targetSpacing / sourceSpacing;
  return (static_cast<double>(targetIndex) - targetCenter) * spacingRatio + sourceCenter;
}

double sampleTrilinearNearestBoundary(const Float64VolumeView& volume,
                                      double sourceX,
                                      double sourceY,
                                      double sourceZ)
{
  requireValidDimensions(volume.dimensions);
  if (volume.voxels.size() != checkedElementCount(volume.dimensions))
  {
    throw std::invalid_argument("Volume voxel count does not match dimensions");
  }

  const AxisInterpolation x = interpolationForAxis(sourceX, volume.dimensions.width, "sourceX");
  const AxisInterpolation y = interpolationForAxis(sourceY, volume.dimensions.height, "sourceY");
  const AxisInterpolation z = interpolationForAxis(sourceZ, volume.dimensions.depth, "sourceZ");

  const auto valueAt = [&volume](std::size_t vx, std::size_t vy, std::size_t vz) {
    return volume.voxels[voxelIndex(volume.dimensions, vx, vy, vz)];
  };

  const double c00 = lerp(valueAt(x.lower, y.lower, z.lower),
                         valueAt(x.upper, y.lower, z.lower),
                         x.weight);
  const double c10 = lerp(valueAt(x.lower, y.upper, z.lower),
                         valueAt(x.upper, y.upper, z.lower),
                         x.weight);
  const double c01 = lerp(valueAt(x.lower, y.lower, z.upper),
                         valueAt(x.upper, y.lower, z.upper),
                         x.weight);
  const double c11 = lerp(valueAt(x.lower, y.upper, z.upper),
                         valueAt(x.upper, y.upper, z.upper),
                         x.weight);

  const double c0 = lerp(c00, c10, y.weight);
  const double c1 = lerp(c01, c11, y.weight);
  return lerp(c0, c1, z.weight);
}

std::vector<double> resampleLinearNearestBoundary(const Float64VolumeView& source,
                                                  VolumeSpacing sourceSpacing,
                                                  VolumeDimensions targetDimensions,
                                                  VolumeSpacing targetSpacing)
{
  requireValidDimensions(source.dimensions);
  requireValidDimensions(targetDimensions);
  requireValidSpacing(sourceSpacing);
  requireValidSpacing(targetSpacing);
  if (source.voxels.size() != checkedElementCount(source.dimensions))
  {
    throw std::invalid_argument("Source voxel count does not match dimensions");
  }

  std::vector<double> output(checkedElementCount(targetDimensions));
  for (std::size_t z = 0; z < targetDimensions.depth; ++z)
  {
    const double sourceZ = sourceCoordinateForTargetIndex(
        source.dimensions.depth, targetDimensions.depth, sourceSpacing.z, targetSpacing.z, z);
    for (std::size_t y = 0; y < targetDimensions.height; ++y)
    {
      const double sourceY = sourceCoordinateForTargetIndex(
          source.dimensions.height, targetDimensions.height, sourceSpacing.y, targetSpacing.y, y);
      for (std::size_t x = 0; x < targetDimensions.width; ++x)
      {
        const double sourceX = sourceCoordinateForTargetIndex(
            source.dimensions.width, targetDimensions.width, sourceSpacing.x, targetSpacing.x, x);
        output[voxelIndex(targetDimensions, x, y, z)] =
            sampleTrilinearNearestBoundary(source, sourceX, sourceY, sourceZ);
      }
    }
  }

  return output;
}

} // namespace maiw::cardiac
