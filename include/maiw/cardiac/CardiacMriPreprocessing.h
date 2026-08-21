#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace maiw::cardiac
{

/**
 * @brief Three-dimensional voxel dimensions in source XYZ axis order.
 */
struct VolumeDimensions
{
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t depth = 0;
};

/**
 * @brief Three-dimensional physical spacing in source XYZ axis order.
 */
struct VolumeSpacing
{
  double x = 1.0;
  double y = 1.0;
  double z = 1.0;
};

/**
 * @brief Frozen model tensor spatial shape in DHW axis order.
 */
struct TensorShapeDhw
{
  std::size_t depth = 14;
  std::size_t height = 144;
  std::size_t width = 144;
};

/**
 * @brief Frozen cardiac MRI preprocessing constants needed by resampling.
 */
struct CardiacMriPreprocessingConfig
{
  VolumeSpacing targetSpacingXyz{1.5, 1.5, 7.5};
  TensorShapeDhw finalTensorShapeDhw{};
};

/**
 * @brief Non-owning float64 view of a dense XYZ volume with X as the fastest axis.
 */
struct Float64VolumeView
{
  VolumeDimensions dimensions;
  std::span<const double> voxels;
};

/**
 * @brief Return the frozen cardiac MRI preprocessing constants.
 */
CardiacMriPreprocessingConfig frozenCardiacMriPreprocessingConfig() noexcept;

/**
 * @brief Round a non-negative finite value like Python round(), using ties to even.
 * @throws std::invalid_argument If value is negative or not finite.
 * @throws std::overflow_error If the rounded value cannot fit in std::size_t.
 */
std::size_t roundHalfToEvenNonNegative(double value);

/**
 * @brief Calculate the Python-compatible target size for one resampled axis.
 * @throws std::invalid_argument If dimensions or spacing are invalid.
 * @throws std::overflow_error If the result cannot fit in std::size_t.
 */
std::size_t targetSizeForAxis(std::size_t sourceSize,
                              double sourceSpacing,
                              double targetSpacing);

/**
 * @brief Map one target voxel index to its center-aligned source coordinate.
 * @throws std::invalid_argument If dimensions, spacing, or the target index are invalid.
 */
double sourceCoordinateForTargetIndex(std::size_t sourceSize,
                                      std::size_t targetSize,
                                      double sourceSpacing,
                                      double targetSpacing,
                                      std::size_t targetIndex);

/**
 * @brief Sample one XYZ float64 volume using trilinear interpolation and nearest-boundary extension.
 * @throws std::invalid_argument If the volume view is invalid.
 */
double sampleTrilinearNearestBoundary(const Float64VolumeView& volume,
                                      double sourceX,
                                      double sourceY,
                                      double sourceZ);

/**
 * @brief Resample a float64 XYZ volume on a center-aligned grid using trilinear interpolation.
 * @throws std::invalid_argument If dimensions or spacing are invalid.
 * @throws std::overflow_error If the output element count cannot fit in memory sizes.
 */
std::vector<double> resampleLinearNearestBoundary(const Float64VolumeView& source,
                                                  VolumeSpacing sourceSpacing,
                                                  VolumeDimensions targetDimensions,
                                                  VolumeSpacing targetSpacing);

} // namespace maiw::cardiac
