#include "maiw/cardiac/CardiacMriPreprocessing.h"

#include "qtviewerpro/core/AnatomicalOrientation.h"
#include "qtviewerpro/core/VolumeData.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

using Matrix3 = std::array<double, 9>;
using Vector3 = std::array<double, 3>;
using maiw::cardiac::CardiacMriPreprocessingConfig;
using maiw::cardiac::Float64VolumeView;
using maiw::cardiac::VolumeDimensions;
using maiw::cardiac::VolumeSpacing;

constexpr double kNumpyAllcloseRtol = 1e-5;
constexpr double kNumpyAllcloseAtol = 1e-8;
constexpr double kDirectionOrthonormalTolerance = 1e-5;
constexpr double kDominantAxisTolerance = 1e-5;

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

std::size_t checkedAdd(std::size_t lhs, std::size_t rhs)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
  {
    throw std::overflow_error("Volume element count exceeds size limits");
  }
  return lhs + rhs;
}

std::size_t checkedElementCount(maiw::cardiac::VolumeDimensions dimensions)
{
  return checkedMultiply(checkedMultiply(dimensions.width, dimensions.height), dimensions.depth);
}

std::array<std::size_t, 3> dimensionsArray(const qvp::VolumeData& volume)
{
  return {volume.width(), volume.height(), volume.depth()};
}

std::array<double, 3> spacingArray(const qvp::VolumeData& volume)
{
  return {static_cast<double>(volume.spacingX()),
          static_cast<double>(volume.spacingY()),
          static_cast<double>(volume.spacingZ())};
}

maiw::cardiac::VolumeDimensions volumeDimensions(const qvp::VolumeData& volume)
{
  return maiw::cardiac::VolumeDimensions{volume.width(), volume.height(), volume.depth()};
}

void requireValidVolumeData(const qvp::VolumeData& volume, const char* name)
{
  if (!volume.isValid())
  {
    throw std::invalid_argument(std::string(name) + " must be non-empty and internally valid");
  }
  requireValidDimensions(volumeDimensions(volume));
  requirePositiveFiniteSpacing(static_cast<double>(volume.spacingX()), "spacing.x");
  requirePositiveFiniteSpacing(static_cast<double>(volume.spacingY()), "spacing.y");
  requirePositiveFiniteSpacing(static_cast<double>(volume.spacingZ()), "spacing.z");
}

std::size_t voxelIndex(std::array<std::size_t, 3> dimensions,
                       std::size_t x,
                       std::size_t y,
                       std::size_t z)
{
  return (z * dimensions[1] * dimensions[0]) + (y * dimensions[0]) + x;
}

struct LpsGeometry
{
  Vector3 origin{};
  Matrix3 direction{};
};

struct DestinationGrid
{
  maiw::cardiac::VolumeDimensions dimensions{};
  maiw::cardiac::VolumeSpacing spacing{};
  Vector3 origin{};
  Matrix3 direction{};
};

LpsGeometry lpsGeometryFromVolume(const qvp::VolumeData& volume)
{
  const auto& geometry = volume.spatialGeometry();
  if (!geometry.hasOrientation)
  {
    throw std::invalid_argument("Volume must have trusted spatial orientation");
  }

  Vector3 rowScale{1.0, 1.0, 1.0};
  switch (geometry.coordinateSystem)
  {
  case qvp::VolumeData::CoordinateSystem::LPS:
    break;
  case qvp::VolumeData::CoordinateSystem::RAS:
    rowScale = {-1.0, -1.0, 1.0};
    break;
  case qvp::VolumeData::CoordinateSystem::Unknown:
    throw std::invalid_argument("Volume coordinate system must be LPS or RAS");
  }

  LpsGeometry lpsGeometry;
  for (std::size_t row = 0; row < 3; ++row)
  {
    const double originValue = geometry.origin[row];
    if (!std::isfinite(originValue))
    {
      throw std::invalid_argument("Volume origin must be finite");
    }
    lpsGeometry.origin[row] = originValue * rowScale[row];

    for (std::size_t column = 0; column < 3; ++column)
    {
      const double directionValue = geometry.direction[(row * 3) + column];
      if (!std::isfinite(directionValue))
      {
        throw std::invalid_argument("Volume direction must be finite");
      }
      lpsGeometry.direction[(row * 3) + column] = directionValue * rowScale[row];
    }
  }

  return lpsGeometry;
}

struct AxisAnatomy
{
  std::size_t worldAxis = 0;
  bool positive = true;
};

double directionValue(const Matrix3& direction, std::size_t row, std::size_t column)
{
  return direction[(row * 3) + column];
}

double columnDotProduct(const Matrix3& direction, std::size_t lhsColumn, std::size_t rhsColumn)
{
  double sum = 0.0;
  for (std::size_t row = 0; row < 3; ++row)
  {
    sum += directionValue(direction, row, lhsColumn) *
           directionValue(direction, row, rhsColumn);
  }
  return sum;
}

void requireSupportedDirectionMatrix(const Matrix3& direction)
{
  for (std::size_t column = 0; column < 3; ++column)
  {
    const double norm = std::sqrt(columnDotProduct(direction, column, column));
    if (std::fabs(norm - 1.0) > kDirectionOrthonormalTolerance)
    {
      throw std::invalid_argument("Volume direction columns must have unit norm");
    }
  }

  for (std::size_t lhsColumn = 0; lhsColumn < 3; ++lhsColumn)
  {
    for (std::size_t rhsColumn = lhsColumn + 1U; rhsColumn < 3; ++rhsColumn)
    {
      if (std::fabs(columnDotProduct(direction, lhsColumn, rhsColumn)) >
          kDirectionOrthonormalTolerance)
      {
        throw std::invalid_argument("Volume direction columns must be mutually orthogonal");
      }
    }
  }
}

AxisAnatomy anatomyForSourceAxis(const Matrix3& direction, std::size_t sourceAxis)
{
  std::size_t bestWorldAxis = 0;
  double bestMagnitude = -1.0;
  double secondBestMagnitude = -1.0;
  for (std::size_t worldAxis = 0; worldAxis < 3; ++worldAxis)
  {
    const double magnitude = std::fabs(directionValue(direction, worldAxis, sourceAxis));
    if (magnitude > bestMagnitude)
    {
      secondBestMagnitude = bestMagnitude;
      bestMagnitude = magnitude;
      bestWorldAxis = worldAxis;
    }
    else if (magnitude > secondBestMagnitude)
    {
      secondBestMagnitude = magnitude;
    }
  }

  if (bestMagnitude <= 0.0)
  {
    throw std::invalid_argument("Volume direction must provide non-zero orientation axes");
  }
  if (bestMagnitude - secondBestMagnitude <= kDominantAxisTolerance)
  {
    throw std::invalid_argument("Volume direction has ambiguous voxel-axis anatomy");
  }

  const double signedDirection = directionValue(direction, bestWorldAxis, sourceAxis);
  return AxisAnatomy{bestWorldAxis, signedDirection > 0.0};
}

struct AxisTransform
{
  std::size_t sourceAxis = 0;
  bool flip = false;
};

std::array<AxisTransform, 3> targetToSourceTransform(const Matrix3& direction)
{
  requireSupportedDirectionMatrix(direction);

  std::array<AxisTransform, 3> transform{};
  std::array<bool, 3> hasTargetAxis{false, false, false};

  for (std::size_t sourceAxis = 0; sourceAxis < 3; ++sourceAxis)
  {
    const AxisAnatomy anatomy = anatomyForSourceAxis(direction, sourceAxis);
    if (hasTargetAxis[anatomy.worldAxis])
    {
      throw std::invalid_argument("Volume direction must map voxel axes to unique anatomy axes");
    }
    hasTargetAxis[anatomy.worldAxis] = true;
    transform[anatomy.worldAxis] = AxisTransform{sourceAxis, !anatomy.positive};
  }

  return transform;
}

bool transformIsAlreadyLps(const std::array<AxisTransform, 3>& transform)
{
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    if (transform[axis].sourceAxis != axis || transform[axis].flip)
    {
      return false;
    }
  }
  return true;
}

std::array<float, 3> transformedSpacing(const std::array<double, 3>& sourceSpacing,
                                        const std::array<AxisTransform, 3>& transform)
{
  return {static_cast<float>(sourceSpacing[transform[0].sourceAxis]),
          static_cast<float>(sourceSpacing[transform[1].sourceAxis]),
          static_cast<float>(sourceSpacing[transform[2].sourceAxis])};
}

std::array<std::size_t, 3> transformedDimensions(const std::array<std::size_t, 3>& sourceDimensions,
                                                 const std::array<AxisTransform, 3>& transform)
{
  return {sourceDimensions[transform[0].sourceAxis],
          sourceDimensions[transform[1].sourceAxis],
          sourceDimensions[transform[2].sourceAxis]};
}

qvp::VolumeData::SpatialGeometry transformedSpatialGeometry(
    const LpsGeometry& sourceGeometry,
    const std::array<double, 3>& sourceSpacing,
    const std::array<std::size_t, 3>& sourceDimensions,
    const std::array<AxisTransform, 3>& transform)
{
  qvp::VolumeData::SpatialGeometry outputGeometry;
  outputGeometry.origin = sourceGeometry.origin;
  outputGeometry.direction = {};
  outputGeometry.coordinateSystem = qvp::VolumeData::CoordinateSystem::LPS;
  outputGeometry.hasOrientation = true;

  for (std::size_t targetAxis = 0; targetAxis < 3; ++targetAxis)
  {
    const AxisTransform axisTransform = transform[targetAxis];
    const double axisSign = axisTransform.flip ? -1.0 : 1.0;

    if (axisTransform.flip)
    {
      const double offset =
          static_cast<double>(sourceDimensions[axisTransform.sourceAxis] - 1U) *
          sourceSpacing[axisTransform.sourceAxis];
      for (std::size_t row = 0; row < 3; ++row)
      {
        outputGeometry.origin[row] +=
            sourceGeometry.direction[(row * 3) + axisTransform.sourceAxis] * offset;
      }
    }

    for (std::size_t row = 0; row < 3; ++row)
    {
      outputGeometry.direction[(row * 3) + targetAxis] =
          sourceGeometry.direction[(row * 3) + axisTransform.sourceAxis] * axisSign;
    }
  }

  return outputGeometry;
}

std::array<std::size_t, 3> sourceIndexForTargetIndex(
    std::array<std::size_t, 3> targetIndex,
    std::array<std::size_t, 3> sourceDimensions,
    const std::array<AxisTransform, 3>& transform)
{
  std::array<std::size_t, 3> sourceIndex{};
  for (std::size_t targetAxis = 0; targetAxis < 3; ++targetAxis)
  {
    const AxisTransform axisTransform = transform[targetAxis];
    std::size_t coordinate = targetIndex[targetAxis];
    if (axisTransform.flip)
    {
      coordinate = sourceDimensions[axisTransform.sourceAxis] - 1U - coordinate;
    }
    sourceIndex[axisTransform.sourceAxis] = coordinate;
  }
  return sourceIndex;
}

qvp::VoxelAxisAnatomy lpsVoxelAxisAnatomy()
{
  return qvp::VoxelAxisAnatomy{
      qvp::AnatomicalDirection::Left,
      qvp::AnatomicalDirection::Posterior,
      qvp::AnatomicalDirection::Superior};
}

bool spacingEqualExactly(const qvp::VolumeData& lhs, const qvp::VolumeData& rhs, std::size_t axis)
{
  switch (axis)
  {
  case 0:
    return lhs.spacingX() == rhs.spacingX();
  case 1:
    return lhs.spacingY() == rhs.spacingY();
  case 2:
    return lhs.spacingZ() == rhs.spacingZ();
  default:
    throw std::invalid_argument("Spacing axis must be 0, 1, or 2");
  }
}

Matrix3 physicalAxisMatrix(const qvp::VolumeData& volume)
{
  const auto& direction = volume.spatialGeometry().direction;
  const auto spacing = spacingArray(volume);
  Matrix3 matrix{};
  for (std::size_t column = 0; column < 3; ++column)
  {
    for (std::size_t row = 0; row < 3; ++row)
    {
      matrix[(row * 3) + column] = directionValue(direction, row, column) * spacing[column];
    }
  }
  return matrix;
}

void requireLpsOrientedVolume(const qvp::VolumeData& volume, const char* name);

Vector3 physicalPoint(Vector3 origin,
                      const Matrix3& direction,
                      maiw::cardiac::VolumeSpacing spacing,
                      double x,
                      double y,
                      double z)
{
  const Vector3 scaled{x * spacing.x, y * spacing.y, z * spacing.z};
  return {origin[0] + (directionValue(direction, 0, 0) * scaled[0]) +
              (directionValue(direction, 0, 1) * scaled[1]) +
              (directionValue(direction, 0, 2) * scaled[2]),
          origin[1] + (directionValue(direction, 1, 0) * scaled[0]) +
              (directionValue(direction, 1, 1) * scaled[1]) +
              (directionValue(direction, 1, 2) * scaled[2]),
          origin[2] + (directionValue(direction, 2, 0) * scaled[0]) +
              (directionValue(direction, 2, 1) * scaled[1]) +
              (directionValue(direction, 2, 2) * scaled[2])};
}

DestinationGrid deriveEdDestinationGrid(const qvp::VolumeData& edVolume)
{
  requireLpsOrientedVolume(edVolume, "ED volume");

  const VolumeDimensions sourceDimensions = volumeDimensions(edVolume);
  const VolumeSpacing sourceSpacing{static_cast<double>(edVolume.spacingX()),
                                    static_cast<double>(edVolume.spacingY()),
                                    static_cast<double>(edVolume.spacingZ())};
  const CardiacMriPreprocessingConfig config = maiw::cardiac::frozenCardiacMriPreprocessingConfig();
  const VolumeSpacing targetSpacing = config.targetSpacingXyz;
  const VolumeDimensions targetDimensions{
      maiw::cardiac::targetSizeForAxis(sourceDimensions.width, sourceSpacing.x, targetSpacing.x),
      maiw::cardiac::targetSizeForAxis(sourceDimensions.height, sourceSpacing.y, targetSpacing.y),
      maiw::cardiac::targetSizeForAxis(sourceDimensions.depth, sourceSpacing.z, targetSpacing.z)};

  const auto& edGeometry = edVolume.spatialGeometry();
  const Vector3 sourceIndexCenter{(static_cast<double>(sourceDimensions.width) - 1.0) / 2.0,
                                  (static_cast<double>(sourceDimensions.height) - 1.0) / 2.0,
                                  (static_cast<double>(sourceDimensions.depth) - 1.0) / 2.0};
  const Vector3 targetIndexCenter{(static_cast<double>(targetDimensions.width) - 1.0) / 2.0,
                                  (static_cast<double>(targetDimensions.height) - 1.0) / 2.0,
                                  (static_cast<double>(targetDimensions.depth) - 1.0) / 2.0};
  const Vector3 sourcePhysicalCenter = physicalPoint(edGeometry.origin,
                                                     edGeometry.direction,
                                                     sourceSpacing,
                                                     sourceIndexCenter[0],
                                                     sourceIndexCenter[1],
                                                     sourceIndexCenter[2]);
  const Vector3 targetCenterOffset = physicalPoint(Vector3{0.0, 0.0, 0.0},
                                                   edGeometry.direction,
                                                   targetSpacing,
                                                   targetIndexCenter[0],
                                                   targetIndexCenter[1],
                                                   targetIndexCenter[2]);

  DestinationGrid grid;
  grid.dimensions = targetDimensions;
  grid.spacing = targetSpacing;
  grid.direction = edGeometry.direction;
  grid.origin = {sourcePhysicalCenter[0] - targetCenterOffset[0],
                 sourcePhysicalCenter[1] - targetCenterOffset[1],
                 sourcePhysicalCenter[2] - targetCenterOffset[2]};
  return grid;
}

std::vector<double> volumeVoxelsAsDouble(const qvp::VolumeData& volume)
{
  std::vector<double> values;
  values.reserve(volume.voxels().size());
  for (const float value : volume.voxels())
  {
    values.push_back(static_cast<double>(value));
  }
  return values;
}

maiw::cardiac::CardiacMriXyzVolume resampleToGrid(const qvp::VolumeData& volume,
                                                  const DestinationGrid& grid,
                                                  VolumeDimensions sourceDimensions,
                                                  VolumeSpacing sourceSpacing)
{
  const std::vector<double> sourceVoxels = volumeVoxelsAsDouble(volume);
  std::vector<double> outputVoxels = maiw::cardiac::resampleLinearNearestBoundary(
      Float64VolumeView{sourceDimensions, sourceVoxels},
      sourceSpacing,
      grid.dimensions,
      grid.spacing);

  maiw::cardiac::CardiacMriXyzVolume output;
  output.dimensions = grid.dimensions;
  output.spacing = grid.spacing;
  output.origin = grid.origin;
  output.direction = grid.direction;
  output.voxels = std::move(outputVoxels);
  return output;
}

void requireValidXyzVolume(const maiw::cardiac::CardiacMriXyzVolume& volume, const char* name)
{
  requireValidDimensions(volume.dimensions);
  requireValidSpacing(volume.spacing);
  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    if (!std::isfinite(volume.origin[axis]))
    {
      throw std::invalid_argument(std::string(name) + " origin must be finite");
    }
  }
  for (const double value : volume.direction)
  {
    if (!std::isfinite(value))
    {
      throw std::invalid_argument(std::string(name) + " direction must be finite");
    }
  }
  if (volume.voxels.size() != checkedElementCount(volume.dimensions))
  {
    throw std::invalid_argument(std::string(name) + " voxel count does not match dimensions");
  }
}

bool equalVolumeSpacing(maiw::cardiac::VolumeSpacing lhs, maiw::cardiac::VolumeSpacing rhs)
{
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

bool equalDirection(const Matrix3& lhs, const Matrix3& rhs)
{
  for (std::size_t index = 0; index < lhs.size(); ++index)
  {
    if (lhs[index] != rhs[index])
    {
      return false;
    }
  }
  return true;
}

void requireMatchingXyzGeometry(const maiw::cardiac::CardiacMriXyzVolume& edVolume,
                                const maiw::cardiac::CardiacMriXyzVolume& esVolume)
{
  if (edVolume.dimensions.width != esVolume.dimensions.width ||
      edVolume.dimensions.height != esVolume.dimensions.height ||
      edVolume.dimensions.depth != esVolume.dimensions.depth)
  {
    throw std::invalid_argument("ED/ES resampled shape mismatch");
  }
  if (!equalVolumeSpacing(edVolume.spacing, esVolume.spacing) ||
      edVolume.origin != esVolume.origin || !equalDirection(edVolume.direction, esVolume.direction))
  {
    throw std::invalid_argument("ED/ES resampled destination geometry mismatch");
  }
}

std::size_t checkedJointVoxelCount(const maiw::cardiac::CardiacMriXyzVolume& edVolume,
                                   const maiw::cardiac::CardiacMriXyzVolume& esVolume)
{
  return checkedAdd(checkedElementCount(edVolume.dimensions),
                    checkedElementCount(esVolume.dimensions));
}

void requireFiniteVoxels(const std::vector<double>& voxels, const char* name)
{
  for (const double value : voxels)
  {
    if (!std::isfinite(value))
    {
      throw std::invalid_argument(std::string(name) + " voxels must be finite");
    }
  }
}

void requireFiniteFloatVoxels(const std::vector<float>& voxels, const char* name)
{
  for (const float value : voxels)
  {
    if (!std::isfinite(value))
    {
      throw std::invalid_argument(std::string(name) + " voxels must be finite");
    }
  }
}

void appendVoxelsInPythonXyzCOrder(std::vector<double>& destination,
                                   maiw::cardiac::VolumeDimensions dimensions,
                                   const std::vector<double>& voxels)
{
  if (voxels.size() != checkedElementCount(dimensions))
  {
    throw std::invalid_argument("Voxel count does not match dimensions");
  }

  for (std::size_t x = 0; x < dimensions.width; ++x)
  {
    for (std::size_t y = 0; y < dimensions.height; ++y)
    {
      for (std::size_t z = 0; z < dimensions.depth; ++z)
      {
        destination.push_back(voxels[voxelIndex({dimensions.width, dimensions.height, dimensions.depth},
                                                x,
                                                y,
                                                z)]);
      }
    }
  }
}

std::vector<double> jointVoxelsEdThenEs(const maiw::cardiac::CardiacMriXyzVolume& edVolume,
                                        const maiw::cardiac::CardiacMriXyzVolume& esVolume)
{
  const std::size_t jointCount = checkedJointVoxelCount(edVolume, esVolume);
  std::vector<double> values;
  values.reserve(jointCount);
  appendVoxelsInPythonXyzCOrder(values, edVolume.dimensions, edVolume.voxels);
  appendVoxelsInPythonXyzCOrder(values, esVolume.dimensions, esVolume.voxels);
  return values;
}

double percentileLinear(std::vector<double> values, double percentile)
{
  if (values.empty())
  {
    throw std::invalid_argument("Percentile input must not be empty");
  }
  if (!std::isfinite(percentile) || percentile < 0.0 || percentile > 100.0)
  {
    throw std::invalid_argument("Percentile must be finite and inside [0, 100]");
  }
  for (const double value : values)
  {
    if (!std::isfinite(value))
    {
      throw std::invalid_argument("Percentile input values must be finite");
    }
  }

  std::sort(values.begin(), values.end());
  if (values.size() == 1U)
  {
    return values.front();
  }

  const double virtualIndex = (static_cast<double>(values.size()) - 1.0) * percentile / 100.0;
  const double lowerAsDouble = std::floor(virtualIndex);
  const auto lower = static_cast<std::size_t>(lowerAsDouble);
  const std::size_t upper = std::min(lower + 1U, values.size() - 1U);
  const double weight = virtualIndex - lowerAsDouble;
  return (values[lower] * (1.0 - weight)) + (values[upper] * weight);
}

std::vector<double> clippedVoxels(const std::vector<double>& values, double lower, double upper)
{
  if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper)
  {
    throw std::invalid_argument("Clip bounds must be finite and ordered");
  }

  std::vector<double> clipped;
  clipped.reserve(values.size());
  for (const double value : values)
  {
    if (!std::isfinite(value))
    {
      throw std::invalid_argument("Clip input values must be finite");
    }
    clipped.push_back(std::min(std::max(value, lower), upper));
  }
  return clipped;
}

class NumpyPairwiseReducer
{
public:
  [[nodiscard]] static double sum(std::span<const double> values)
  {
    return sumRecursive(values);
  }

private:
  static constexpr std::size_t kScalarCount = 8;
  static constexpr std::size_t kBlockSize = 128;

  [[nodiscard]] static double sumRecursive(std::span<const double> values)
  {
    if (values.size() < kScalarCount)
    {
      return sumScalar(values);
    }

    if (values.size() <= kBlockSize)
    {
      return sumSmallBlock(values);
    }

    std::size_t split = values.size() / 2U;
    split -= split % kScalarCount;
    return sumRecursive(values.first(split)) + sumRecursive(values.subspan(split));
  }

  [[nodiscard]] static double sumScalar(std::span<const double> values)
  {
    double sum = 0.0;
    for (const double value : values)
    {
      sum += value;
    }
    return sum;
  }

  [[nodiscard]] static double sumSmallBlock(std::span<const double> values)
  {
    std::array<double, kScalarCount> partials{};
    for (std::size_t index = 0; index < kScalarCount; ++index)
    {
      partials[index] = values[index];
    }

    std::size_t index = kScalarCount;
    const std::size_t vectorizedEnd = values.size() - (values.size() % kScalarCount);
    while (index < vectorizedEnd)
    {
      for (std::size_t lane = 0; lane < kScalarCount; ++lane)
      {
        partials[lane] += values[index + lane];
      }
      index += kScalarCount;
    }

    // NumPy's double ufunc reduction folds the eight partial sums as a fixed
    // pairwise tree before adding any scalar tail elements.
    double sum = ((partials[0] + partials[1]) + (partials[2] + partials[3])) +
                 ((partials[4] + partials[5]) + (partials[6] + partials[7]));
    while (index < values.size())
    {
      sum += values[index];
      ++index;
    }
    return sum;
  }
};

double meanOf(const std::vector<double>& values)
{
  if (values.empty())
  {
    throw std::invalid_argument("Mean input must not be empty");
  }

  // np.mean(float64) delegates to NumPy's ufunc pairwise reduction before the
  // final division. The reduction tree is part of the golden parity contract.
  const double sum = NumpyPairwiseReducer::sum(values);
  return sum / static_cast<double>(values.size());
}

double populationStdOf(const std::vector<double>& values, double mean)
{
  if (values.empty())
  {
    throw std::invalid_argument("Standard-deviation input must not be empty");
  }

  std::vector<double> squaredDeltas;
  squaredDeltas.reserve(values.size());
  for (const double value : values)
  {
    const double delta = value - mean;
    squaredDeltas.push_back(delta * delta);
  }

  const double squaredSum = NumpyPairwiseReducer::sum(squaredDeltas);
  return std::sqrt(squaredSum / static_cast<double>(values.size()));
}

maiw::cardiac::NormalizedCardiacMriXyzVolume normalizedVolumeFrom(
    const maiw::cardiac::CardiacMriXyzVolume& source,
    const std::vector<double>& clipped,
    double mean,
    double denominator)
{
  if (clipped.size() != source.voxels.size())
  {
    throw std::invalid_argument("Clipped voxel count must match source voxel count");
  }
  if (!std::isfinite(mean) || !std::isfinite(denominator) || denominator <= 0.0)
  {
    throw std::invalid_argument("Normalization parameters must be finite and valid");
  }

  maiw::cardiac::NormalizedCardiacMriXyzVolume normalized;
  normalized.dimensions = source.dimensions;
  normalized.spacing = source.spacing;
  normalized.origin = source.origin;
  normalized.direction = source.direction;
  normalized.voxels.reserve(clipped.size());
  for (const double value : clipped)
  {
    normalized.voxels.push_back(static_cast<float>((value - mean) / denominator));
  }
  return normalized;
}

void requireValidNormalizedXyzVolume(const maiw::cardiac::NormalizedCardiacMriXyzVolume& volume,
                                     const char* name)
{
  requireValidDimensions(volume.dimensions);
  requireValidSpacing(volume.spacing);
  for (const double value : volume.origin)
  {
    if (!std::isfinite(value))
    {
      throw std::invalid_argument(std::string(name) + " origin must be finite");
    }
  }
  for (const double value : volume.direction)
  {
    if (!std::isfinite(value))
    {
      throw std::invalid_argument(std::string(name) + " direction must be finite");
    }
  }
  if (volume.voxels.size() != checkedElementCount(volume.dimensions))
  {
    throw std::invalid_argument(std::string(name) + " voxel count does not match dimensions");
  }
  requireFiniteFloatVoxels(volume.voxels, name);
}

std::size_t cdhwIndex(std::size_t channel,
                      std::size_t depth,
                      std::size_t height,
                      std::size_t width)
{
  const CardiacMriPreprocessingConfig config = maiw::cardiac::frozenCardiacMriPreprocessingConfig();
  return (((channel * config.finalTensorShapeDhw.depth) + depth) *
              config.finalTensorShapeDhw.height +
          height) *
             config.finalTensorShapeDhw.width +
         width;
}

bool allclose(double lhs, double rhs)
{
  if (!std::isfinite(lhs) || !std::isfinite(rhs))
  {
    return false;
  }
  return std::fabs(lhs - rhs) <= (kNumpyAllcloseAtol + (kNumpyAllcloseRtol * std::fabs(rhs)));
}

void requireLpsOrientedVolume(const qvp::VolumeData& volume, const char* name)
{
  requireValidVolumeData(volume, name);
  const LpsGeometry lpsGeometry = lpsGeometryFromVolume(volume);
  if (volume.spatialGeometry().coordinateSystem != qvp::VolumeData::CoordinateSystem::LPS)
  {
    throw std::invalid_argument(std::string(name) + " must use LPS world coordinates");
  }
  if (!transformIsAlreadyLps(targetToSourceTransform(lpsGeometry.direction)))
  {
    throw std::invalid_argument(std::string(name) + " voxel axes must be LPS oriented");
  }
}

std::size_t mapNearestBoundaryIndex(long long index, std::size_t size)
{
  requirePositiveDimension(size, "axis size");
  if (index < 0)
  {
    return 0;
  }

  const auto unsignedIndex = static_cast<unsigned long long>(index);
  const auto upper = static_cast<unsigned long long>(size - 1U);
  if (unsignedIndex > upper)
  {
    return size - 1U;
  }

  return static_cast<std::size_t>(index);
}

struct AxisInterpolation
{
  std::size_t lower = 0;
  std::size_t upper = 0;
  double lowerWeight = 1.0;
  double upperWeight = 0.0;
};

AxisInterpolation interpolationForAxis(double coordinate, std::size_t size, const char* name)
{
  requirePositiveDimension(size, "axis size");
  requireFiniteCoordinate(coordinate, name);
  const double lowerAsDouble = std::floor(coordinate);
  const auto lowerRaw = static_cast<long long>(lowerAsDouble);

  // SciPy computes linear interpolation weights from the raw coordinate and
  // applies nearest-boundary mapping to each filter-footprint index afterward.
  const double fractional = coordinate - lowerAsDouble;
  const double lowerWeight = 1.0 - fractional;
  double upperWeight = 1.0;
  upperWeight -= lowerWeight;

  return AxisInterpolation{
      mapNearestBoundaryIndex(lowerRaw, size),
      mapNearestBoundaryIndex(lowerRaw + 1LL, size),
      lowerWeight,
      upperWeight};
}

std::size_t voxelIndex(maiw::cardiac::VolumeDimensions dimensions,
                       std::size_t x,
                       std::size_t y,
                       std::size_t z)
{
  return (z * dimensions.height * dimensions.width) + (y * dimensions.width) + x;
}

double scipyOrder1Contribution(double value,
                               double xWeight,
                               double yWeight,
                               double zWeight)
{
  // Match scipy.ndimage's order=1 accumulation: source value first, then one
  // rounded multiplication per axis before adding the corner contribution.
  double coefficient = value;
  coefficient *= xWeight;
  coefficient *= yWeight;
  coefficient *= zWeight;
  return coefficient;
}

} // namespace

namespace maiw::cardiac
{

CardiacMriPreprocessingConfig frozenCardiacMriPreprocessingConfig() noexcept
{
  return CardiacMriPreprocessingConfig{};
}

CardiacMriInputTensor::CardiacMriInputTensor(std::vector<float> values)
    : values_(std::move(values))
{
  if (values_.size() != elementCount())
  {
    throw std::invalid_argument("Cardiac MRI input tensor value count mismatch");
  }
  requireFiniteFloatVoxels(values_, "Cardiac MRI input tensor");
}

const std::vector<float>& CardiacMriInputTensor::values() const noexcept
{
  return values_;
}

std::span<const float> CardiacMriInputTensor::span() const noexcept
{
  return values_;
}

std::array<std::size_t, 4> CardiacMriInputTensor::shapeCdhw() const noexcept
{
  return {kChannels, kDepth, kHeight, kWidth};
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

  double output = 0.0;
  output += scipyOrder1Contribution(valueAt(x.lower, y.lower, z.lower),
                                    x.lowerWeight,
                                    y.lowerWeight,
                                    z.lowerWeight);
  output += scipyOrder1Contribution(valueAt(x.lower, y.lower, z.upper),
                                    x.lowerWeight,
                                    y.lowerWeight,
                                    z.upperWeight);
  output += scipyOrder1Contribution(valueAt(x.lower, y.upper, z.lower),
                                    x.lowerWeight,
                                    y.upperWeight,
                                    z.lowerWeight);
  output += scipyOrder1Contribution(valueAt(x.lower, y.upper, z.upper),
                                    x.lowerWeight,
                                    y.upperWeight,
                                    z.upperWeight);
  output += scipyOrder1Contribution(valueAt(x.upper, y.lower, z.lower),
                                    x.upperWeight,
                                    y.lowerWeight,
                                    z.lowerWeight);
  output += scipyOrder1Contribution(valueAt(x.upper, y.lower, z.upper),
                                    x.upperWeight,
                                    y.lowerWeight,
                                    z.upperWeight);
  output += scipyOrder1Contribution(valueAt(x.upper, y.upper, z.lower),
                                    x.upperWeight,
                                    y.upperWeight,
                                    z.lowerWeight);
  output += scipyOrder1Contribution(valueAt(x.upper, y.upper, z.upper),
                                    x.upperWeight,
                                    y.upperWeight,
                                    z.upperWeight);

  return output;
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

double percentileLinearFloat64(std::span<const double> values, double percentile)
{
  return percentileLinear(std::vector<double>(values.begin(), values.end()), percentile);
}

qvp::VolumeData normalizeVolumeDataToLps(const qvp::VolumeData& volume)
{
  requireValidVolumeData(volume, "volume");
  const std::array<std::size_t, 3> sourceDimensions = dimensionsArray(volume);
  const std::array<double, 3> sourceSpacing = spacingArray(volume);
  const LpsGeometry lpsGeometry = lpsGeometryFromVolume(volume);
  const std::array<AxisTransform, 3> transform = targetToSourceTransform(lpsGeometry.direction);
  const std::array<std::size_t, 3> targetDimensions =
      transformedDimensions(sourceDimensions, transform);
  const std::array<float, 3> targetSpacing = transformedSpacing(sourceSpacing, transform);
  const qvp::VolumeData::SpatialGeometry targetGeometry =
      transformedSpatialGeometry(lpsGeometry, sourceSpacing, sourceDimensions, transform);

  std::vector<float> orientedVoxels(checkedMultiply(
      checkedMultiply(targetDimensions[0], targetDimensions[1]), targetDimensions[2]));
  const auto& sourceVoxels = volume.voxels();
  for (std::size_t z = 0; z < targetDimensions[2]; ++z)
  {
    for (std::size_t y = 0; y < targetDimensions[1]; ++y)
    {
      for (std::size_t x = 0; x < targetDimensions[0]; ++x)
      {
        const std::array<std::size_t, 3> sourceIndex =
            sourceIndexForTargetIndex({x, y, z}, sourceDimensions, transform);
        orientedVoxels[voxelIndex(targetDimensions, x, y, z)] =
            sourceVoxels[voxelIndex(sourceDimensions, sourceIndex[0], sourceIndex[1], sourceIndex[2])];
      }
    }
  }

  return qvp::VolumeData(targetDimensions[0],
                         targetDimensions[1],
                         targetDimensions[2],
                         targetSpacing[0],
                         targetSpacing[1],
                         targetSpacing[2],
                         std::move(orientedVoxels),
                         targetGeometry,
                         lpsVoxelAxisAnatomy());
}

void validateLpsOrientedVolumePair(const qvp::VolumeData& edVolume,
                                   const qvp::VolumeData& esVolume)
{
  requireLpsOrientedVolume(edVolume, "ED volume");
  requireLpsOrientedVolume(esVolume, "ES volume");

  if (edVolume.width() != esVolume.width() || edVolume.height() != esVolume.height() ||
      edVolume.depth() != esVolume.depth())
  {
    throw std::invalid_argument("ED/ES oriented shape mismatch");
  }

  for (std::size_t axis = 0; axis < 3; ++axis)
  {
    if (!spacingEqualExactly(edVolume, esVolume, axis))
    {
      throw std::invalid_argument("ED/ES oriented spacing mismatch");
    }
  }

  const Matrix3 edPhysicalAxisMatrix = physicalAxisMatrix(edVolume);
  const Matrix3 esPhysicalAxisMatrix = physicalAxisMatrix(esVolume);
  for (std::size_t index = 0; index < edPhysicalAxisMatrix.size(); ++index)
  {
    if (!allclose(edPhysicalAxisMatrix[index], esPhysicalAxisMatrix[index]))
    {
      throw std::invalid_argument("ED/ES oriented affine matrix mismatch");
    }
  }
}

CardiacMriXyzVolumePair resampleOrientedPairToEdDerivedGrid(
    const qvp::VolumeData& edVolume,
    const qvp::VolumeData& esVolume)
{
  validateLpsOrientedVolumePair(edVolume, esVolume);

  const DestinationGrid grid = deriveEdDestinationGrid(edVolume);
  const VolumeDimensions sourceDimensions = volumeDimensions(edVolume);
  const VolumeSpacing sourceSpacing{static_cast<double>(edVolume.spacingX()),
                                    static_cast<double>(edVolume.spacingY()),
                                    static_cast<double>(edVolume.spacingZ())};

  return CardiacMriXyzVolumePair{resampleToGrid(edVolume, grid, sourceDimensions, sourceSpacing),
                                 resampleToGrid(esVolume, grid, sourceDimensions, sourceSpacing)};
}

XyCropWindow deriveFrozenXyCenterCropWindow(const CardiacMriXyzVolume& volume)
{
  requireValidXyzVolume(volume, "volume");
  const CardiacMriPreprocessingConfig config = frozenCardiacMriPreprocessingConfig();
  const std::size_t targetWidth = config.finalTensorShapeDhw.width;
  const std::size_t targetHeight = config.finalTensorShapeDhw.height;

  if (volume.dimensions.width < targetWidth || volume.dimensions.height < targetHeight)
  {
    throw std::invalid_argument("XY padding would be required for frozen crop");
  }

  const std::size_t excessX = volume.dimensions.width - targetWidth;
  const std::size_t excessY = volume.dimensions.height - targetHeight;
  const std::size_t startX = excessX / 2U;
  const std::size_t startY = excessY / 2U;
  return XyCropWindow{startX, startY, startX + targetWidth, startY + targetHeight};
}

CardiacMriXyzVolume cropVolumeToWindow(const CardiacMriXyzVolume& volume,
                                       XyCropWindow window)
{
  requireValidXyzVolume(volume, "volume");
  if (window.endX <= window.startX || window.endY <= window.startY ||
      window.endX > volume.dimensions.width || window.endY > volume.dimensions.height)
  {
    throw std::invalid_argument("Crop window must be non-empty and inside the source volume");
  }

  const VolumeDimensions croppedDimensions{
      window.endX - window.startX,
      window.endY - window.startY,
      volume.dimensions.depth};
  std::vector<double> croppedVoxels(checkedElementCount(croppedDimensions));

  for (std::size_t z = 0; z < croppedDimensions.depth; ++z)
  {
    for (std::size_t y = 0; y < croppedDimensions.height; ++y)
    {
      for (std::size_t x = 0; x < croppedDimensions.width; ++x)
      {
        croppedVoxels[voxelIndex(croppedDimensions, x, y, z)] =
            volume.voxels[voxelIndex(volume.dimensions,
                                     window.startX + x,
                                     window.startY + y,
                                     z)];
      }
    }
  }

  CardiacMriXyzVolume cropped;
  cropped.dimensions = croppedDimensions;
  cropped.spacing = volume.spacing;
  cropped.origin = physicalPoint(volume.origin,
                                 volume.direction,
                                 volume.spacing,
                                 static_cast<double>(window.startX),
                                 static_cast<double>(window.startY),
                                 0.0);
  cropped.direction = volume.direction;
  cropped.voxels = std::move(croppedVoxels);
  return cropped;
}

CardiacMriXyzVolumePair cropResampledPairToFrozenXy(
    const CardiacMriXyzVolume& edVolume,
    const CardiacMriXyzVolume& esVolume)
{
  requireValidXyzVolume(edVolume, "ED volume");
  requireValidXyzVolume(esVolume, "ES volume");
  requireMatchingXyzGeometry(edVolume, esVolume);

  const XyCropWindow window = deriveFrozenXyCenterCropWindow(edVolume);
  return CardiacMriXyzVolumePair{cropVolumeToWindow(edVolume, window),
                                 cropVolumeToWindow(esVolume, window)};
}

CardiacMriNormalizationResult normalizeCroppedPairIntensities(
    const CardiacMriXyzVolume& edVolume,
    const CardiacMriXyzVolume& esVolume)
{
  requireValidXyzVolume(edVolume, "ED volume");
  requireValidXyzVolume(esVolume, "ES volume");
  requireMatchingXyzGeometry(edVolume, esVolume);
  requireFiniteVoxels(edVolume.voxels, "ED volume");
  requireFiniteVoxels(esVolume.voxels, "ES volume");

  const CardiacMriPreprocessingConfig config = frozenCardiacMriPreprocessingConfig();
  const std::vector<double> jointValues = jointVoxelsEdThenEs(edVolume, esVolume);
  const double clipLower = percentileLinear(jointValues, config.lowerPercentile);
  const double clipUpper = percentileLinear(jointValues, config.upperPercentile);
  std::vector<double> edClipped = clippedVoxels(edVolume.voxels, clipLower, clipUpper);
  std::vector<double> esClipped = clippedVoxels(esVolume.voxels, clipLower, clipUpper);

  std::vector<double> clippedJoint;
  clippedJoint.reserve(checkedJointVoxelCount(edVolume, esVolume));
  appendVoxelsInPythonXyzCOrder(clippedJoint, edVolume.dimensions, edClipped);
  appendVoxelsInPythonXyzCOrder(clippedJoint, esVolume.dimensions, esClipped);

  const double mean = meanOf(clippedJoint);
  const double std = populationStdOf(clippedJoint, mean);
  if (!std::isfinite(std) || std <= config.epsilon)
  {
    throw std::invalid_argument("Normalization std must be finite and greater than epsilon");
  }
  const double denominator = std > config.epsilon ? std : config.epsilon;

  CardiacMriNormalizationResult result;
  result.metadata = CardiacMriNormalizationMetadata{clipLower, clipUpper, mean, std};
  result.volumes.ed = normalizedVolumeFrom(edVolume, edClipped, mean, denominator);
  result.volumes.es = normalizedVolumeFrom(esVolume, esClipped, mean, denominator);
  return result;
}

NormalizedCardiacMriXyzVolume padNormalizedVolumeZToFrozenDepth(
    const NormalizedCardiacMriXyzVolume& volume)
{
  requireValidNormalizedXyzVolume(volume, "volume");

  const CardiacMriPreprocessingConfig config = frozenCardiacMriPreprocessingConfig();
  const std::size_t targetDepth = config.finalTensorShapeDhw.depth;
  if (volume.dimensions.depth > targetDepth)
  {
    throw std::invalid_argument("Z cropping would be required for frozen depth");
  }

  const std::size_t totalPadding = targetDepth - volume.dimensions.depth;
  const std::size_t lowerPadding = totalPadding / 2U;
  const VolumeDimensions paddedDimensions{
      volume.dimensions.width,
      volume.dimensions.height,
      targetDepth};
  std::vector<float> paddedVoxels(checkedElementCount(paddedDimensions), 0.0F);

  for (std::size_t z = 0; z < volume.dimensions.depth; ++z)
  {
    for (std::size_t y = 0; y < volume.dimensions.height; ++y)
    {
      for (std::size_t x = 0; x < volume.dimensions.width; ++x)
      {
        paddedVoxels[voxelIndex(paddedDimensions, x, y, lowerPadding + z)] =
            volume.voxels[voxelIndex(volume.dimensions, x, y, z)];
      }
    }
  }

  NormalizedCardiacMriXyzVolume padded;
  padded.dimensions = paddedDimensions;
  padded.spacing = volume.spacing;
  padded.origin = volume.origin;
  padded.direction = volume.direction;
  padded.voxels = std::move(paddedVoxels);
  return padded;
}

std::vector<float> normalizedXyzVolumeToDhwContiguous(
    const NormalizedCardiacMriXyzVolume& volume)
{
  requireValidNormalizedXyzVolume(volume, "volume");

  const CardiacMriPreprocessingConfig config = frozenCardiacMriPreprocessingConfig();
  if (volume.dimensions.width != config.finalTensorShapeDhw.width ||
      volume.dimensions.height != config.finalTensorShapeDhw.height ||
      volume.dimensions.depth != config.finalTensorShapeDhw.depth)
  {
    throw std::invalid_argument("Normalized volume must match frozen DHW tensor shape");
  }

  std::vector<float> output(checkedElementCount(volume.dimensions));
  for (std::size_t z = 0; z < volume.dimensions.depth; ++z)
  {
    for (std::size_t y = 0; y < volume.dimensions.height; ++y)
    {
      for (std::size_t x = 0; x < volume.dimensions.width; ++x)
      {
        const std::size_t outputIndex =
            ((z * config.finalTensorShapeDhw.height) + y) * config.finalTensorShapeDhw.width + x;
        output[outputIndex] = volume.voxels[voxelIndex(volume.dimensions, x, y, z)];
      }
    }
  }
  return output;
}

CardiacMriInputTensor stackNormalizedPairToInputTensor(
    const NormalizedCardiacMriXyzVolume& edVolume,
    const NormalizedCardiacMriXyzVolume& esVolume)
{
  const std::vector<float> edDhw = normalizedXyzVolumeToDhwContiguous(edVolume);
  const std::vector<float> esDhw = normalizedXyzVolumeToDhwContiguous(esVolume);
  if (edDhw.size() != esDhw.size())
  {
    throw std::invalid_argument("ED/ES DHW tensor value count mismatch");
  }

  std::vector<float> values(CardiacMriInputTensor::elementCount());
  const CardiacMriPreprocessingConfig config = frozenCardiacMriPreprocessingConfig();
  for (std::size_t z = 0; z < config.finalTensorShapeDhw.depth; ++z)
  {
    for (std::size_t y = 0; y < config.finalTensorShapeDhw.height; ++y)
    {
      for (std::size_t x = 0; x < config.finalTensorShapeDhw.width; ++x)
      {
        const std::size_t dhwIndex =
            ((z * config.finalTensorShapeDhw.height) + y) * config.finalTensorShapeDhw.width + x;
        values[cdhwIndex(0, z, y, x)] = edDhw[dhwIndex];
        values[cdhwIndex(1, z, y, x)] = esDhw[dhwIndex];
      }
    }
  }
  return CardiacMriInputTensor(std::move(values));
}

CardiacMriInputTensor CardiacMriPreprocessor::preprocess(const qvp::VolumeData& edVolume,
                                                         const qvp::VolumeData& esVolume) const
{
  const qvp::VolumeData edLps = normalizeVolumeDataToLps(edVolume);
  const qvp::VolumeData esLps = normalizeVolumeDataToLps(esVolume);
  validateLpsOrientedVolumePair(edLps, esLps);

  const CardiacMriXyzVolumePair resampled = resampleOrientedPairToEdDerivedGrid(edLps, esLps);
  const CardiacMriXyzVolumePair cropped = cropResampledPairToFrozenXy(resampled.ed, resampled.es);
  const CardiacMriNormalizationResult normalized =
      normalizeCroppedPairIntensities(cropped.ed, cropped.es);
  const NormalizedCardiacMriXyzVolume edPadded =
      padNormalizedVolumeZToFrozenDepth(normalized.volumes.ed);
  const NormalizedCardiacMriXyzVolume esPadded =
      padNormalizedVolumeZToFrozenDepth(normalized.volumes.es);
  return stackNormalizedPairToInputTensor(edPadded, esPadded);
}

} // namespace maiw::cardiac
