#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace qvp
{
class VolumeData;
}

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
 * @brief Dense double-precision XYZ cardiac MRI volume with LPS physical geometry.
 *
 * The voxel buffer is X-fastest and intentionally remains float64 between
 * resampling and later intensity normalization to match the Python
 * preprocessing pipeline.
 */
struct CardiacMriXyzVolume
{
  VolumeDimensions dimensions;
  VolumeSpacing spacing;
  std::array<double, 3> origin{0.0, 0.0, 0.0};
  std::array<double, 9> direction{1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0};
  std::vector<double> voxels;
};

/**
 * @brief ED/ES pair of cardiac MRI XYZ volumes.
 */
struct CardiacMriXyzVolumePair
{
  CardiacMriXyzVolume ed;
  CardiacMriXyzVolume es;
};

/**
 * @brief Shared XY crop window in XYZ coordinates.
 */
struct XyCropWindow
{
  std::size_t startX = 0;
  std::size_t startY = 0;
  std::size_t endX = 0;
  std::size_t endY = 0;
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

/**
 * @brief Return a copy of a qtviewerpro volume reoriented to voxel-axis LPS order.
 *
 * The input coordinate system describes the patient-world coordinate convention
 * used by origin/direction (for example LPS or RAS). The voxel-axis orientation
 * describes which anatomical direction is reached when each voxel index
 * increases. This function first converts trusted RAS geometry into LPS world
 * coordinates, then applies only voxel permutation and axis reversal so the
 * output voxel axes increase toward Left, Posterior, and Superior.
 *
 * @throws std::invalid_argument If the volume, spacing, or trusted orientation
 * metadata are invalid or cannot provide a unique 3D anatomical orientation.
 * @throws std::overflow_error If the output voxel count cannot fit in memory sizes.
 */
qvp::VolumeData normalizeVolumeDataToLps(const qvp::VolumeData& volume);

/**
 * @brief Validate Python-equivalent ED/ES geometry after LPS orientation normalization.
 *
 * Both volumes must be voxel-axis LPS in LPS world coordinates and have
 * exactly matching dimensions and spacing values. Their 3x3 direction-spacing
 * physical axis matrices are compared with NumPy np.allclose default
 * semantics. Origins are intentionally not compared because the Python
 * production preprocessing validates affine[:3, :3] only.
 *
 * @throws std::invalid_argument If any Python-equivalent oriented-pair condition fails.
 */
void validateLpsOrientedVolumePair(const qvp::VolumeData& edVolume,
                                   const qvp::VolumeData& esVolume);

/**
 * @brief Resample oriented ED/ES volumes onto one ED-derived frozen physical grid.
 *
 * The pair is validated with the Python-equivalent oriented-pair contract before
 * any resampling. Target dimensions, target spacing, origin, and direction are
 * derived from ED only and reused for ES. Voxel values are converted from
 * VolumeData float storage to double before interpolation and the outputs keep
 * double storage for parity with Python's float64 resampling stage.
 *
 * @throws std::invalid_argument If oriented-pair validation or volume metadata validation fails.
 * @throws std::overflow_error If an intermediate or output element count cannot fit in memory sizes.
 */
CardiacMriXyzVolumePair resampleOrientedPairToEdDerivedGrid(
    const qvp::VolumeData& edVolume,
    const qvp::VolumeData& esVolume);

/**
 * @brief Derive the frozen 144 x 144 geometric FOV-center crop window from one XYZ volume.
 *
 * The crop is centered with integer floor division: odd excess removes
 * floor(diff / 2) voxels from the lower-index side and the remainder from the
 * upper-index side. Z is not cropped.
 *
 * @throws std::invalid_argument If the source XY extent is smaller than the frozen target.
 */
XyCropWindow deriveFrozenXyCenterCropWindow(const CardiacMriXyzVolume& volume);

/**
 * @brief Apply an XY crop window to one double-precision XYZ volume.
 *
 * Spacing and direction are preserved. Origin is updated to the physical
 * location of the retained source voxel at (window.startX, window.startY, 0).
 *
 * @throws std::invalid_argument If the volume or crop window is invalid.
 * @throws std::overflow_error If the cropped output element count cannot fit in memory sizes.
 */
CardiacMriXyzVolume cropVolumeToWindow(const CardiacMriXyzVolume& volume,
                                       XyCropWindow window);

/**
 * @brief Derive one frozen ED XY crop window and apply it to both ED and ES.
 *
 * ED and ES must already share the same resampled dimensions and physical
 * destination geometry. The returned cropped volumes share the same updated
 * geometry.
 *
 * @throws std::invalid_argument If the pair geometry or crop window is invalid.
 * @throws std::overflow_error If a cropped output element count cannot fit in memory sizes.
 */
CardiacMriXyzVolumePair cropResampledPairToFrozenXy(
    const CardiacMriXyzVolume& edVolume,
    const CardiacMriXyzVolume& esVolume);

} // namespace maiw::cardiac
