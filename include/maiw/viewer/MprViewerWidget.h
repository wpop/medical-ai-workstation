#pragma once

#include "maiw/viewer/SliceViewerWidget.h"

#include "qtviewerpro/core/VolumePhysicalCoordinateMapper.h"

#include <QPointF>
#include <QWidget>

#include <optional>

namespace maiw::viewer
{

/**
 * @brief Presents synchronized axial, sagittal, and coronal volume slices.
 *
 * The widget observes one externally owned medical volume and owns only its
 * voxel navigation position. Three child SliceViewerWidget instances are
 * managed through Qt parent-child ownership and remain permanently assigned to
 * their respective orthogonal orientations.
 *
 * The caller must ensure that an assigned volume outlives this widget or clear
 * the assignment before destroying the volume.
 */
class MprViewerWidget final : public QWidget
{
public:
  /**
   * @brief Construct an empty three-view MPR widget.
   *
   * @param parent Optional parent widget.
   */
  explicit MprViewerWidget(QWidget* parent = nullptr);

  /**
   * @brief Assign an externally owned medical volume.
   *
   * A valid, non-empty volume initializes navigation to the center voxel. Each
   * coordinate uses integer division by two; for an even dimension this selects
   * the higher of the two central zero-based indices. Passing nullptr or an
   * unusable volume resets all views to the empty state.
   *
   * @param volume Non-owning pointer to a volume that must outlive its assignment.
   */
  void setVolume(const qvp::VolumeData* volume);

  /**
   * @brief Clear the volume assignment and reset navigation to the origin.
   */
  void clearVolume();

  /**
   * @brief Return whether a volume is currently assigned.
   */
  [[nodiscard]] bool hasVolume() const noexcept;

  /**
   * @brief Select a voxel navigation position.
   *
   * Each coordinate is clamped independently to the assigned volume bounds. If
   * no usable volume is assigned, the position resets to the origin.
   *
   * @param position Requested zero-based voxel position.
   */
  void setVoxelPosition(qvp::VoxelIndex3D position);

  /**
   * @brief Return the current clamped voxel navigation position.
   */
  [[nodiscard]] qvp::VoxelIndex3D voxelPosition() const noexcept;

  /**
   * @brief Navigate from a point in one orthogonal slice image.
   *
   * The argument uses MAIW/qvp MPR slice pixel-space coordinates, with pixel
   * centers at `(index + 0.5)`. Positive fractional coordinates identify the
   * containing pixel. These are not the normalized image-local coordinates
   * emitted by qvp::OpenGLSliceViewer::crosshairPositionChanged. A caller using
   * that signal must first perform an explicit normalized-to-pixel conversion.
   *
   * The two in-plane voxel coordinates are updated and clamped while the
   * coordinate normal to the slice is preserved. With no usable volume
   * assigned, this operation has no effect.
   *
   * @param orientation Orientation of the source slice image.
   * @param imagePoint Point in source image coordinates.
   */
  void setPositionFromImagePoint(qvp::SliceOrientation orientation,
                                 QPointF imagePoint);

  /**
   * @brief Return the current patient-space physical coordinate.
   *
   * @return Physical coordinate for the selected voxel, or std::nullopt when no
   * usable volume is assigned.
   */
  [[nodiscard]] std::optional<qvp::PhysicalPoint3D> physicalPosition() const;

  /**
   * @brief Return the fixed axial slice viewer.
   */
  [[nodiscard]] const SliceViewerWidget& axialViewer() const noexcept;

  /**
   * @brief Return the fixed sagittal slice viewer.
   */
  [[nodiscard]] const SliceViewerWidget& sagittalViewer() const noexcept;

  /**
   * @brief Return the fixed coronal slice viewer.
   */
  [[nodiscard]] const SliceViewerWidget& coronalViewer() const noexcept;

private:
  [[nodiscard]] bool hasUsableVolume() const noexcept;
  void clampVoxelPosition() noexcept;
  void synchronizeViews();

  /**
   * @brief Externally owned volume observed by this widget.
   */
  const qvp::VolumeData* volume_ = nullptr;
  qvp::VoxelIndex3D voxelPosition_{};

  // Qt parent ownership manages the viewers; these pointers are non-owning.
  SliceViewerWidget* axialViewer_ = nullptr;
  SliceViewerWidget* sagittalViewer_ = nullptr;
  SliceViewerWidget* coronalViewer_ = nullptr;
};

} // namespace maiw::viewer
