#pragma once

#include "qtviewerpro/core/SliceOrientation.h"

#include <QImage>
#include <QPointF>
#include <QSize>
#include <QWidget>

#include <cstddef>
#include <optional>

namespace qvp
{

class OpenGLSliceViewer;
class VolumeData;

} // namespace qvp

class QEvent;
class QObject;

namespace maiw::viewer
{

/**
 * @brief Presents one orthogonal 2D slice from an externally owned volume.
 *
 * The widget composes qt-viewer-pro extraction, image conversion, and OpenGL
 * rendering primitives. It does not own or modify the assigned medical volume.
 * The caller must ensure that the volume outlives the assignment or clear the
 * widget before destroying the volume.
 */
class SliceViewerWidget final : public QWidget
{
  Q_OBJECT

public:
  /**
   * @brief Construct an empty axial slice viewer.
   *
   * @param parent Optional parent widget.
   */
  explicit SliceViewerWidget(QWidget* parent = nullptr);

  /**
   * @brief Assign an externally owned medical volume.
   *
   * Passing nullptr clears the assignment. The current slice index is clamped
   * to the valid range of the new volume and orientation.
   *
   * @param volume Non-owning pointer to a volume that must outlive its assignment.
   */
  void setVolume(const qvp::VolumeData* volume);

  /**
   * @brief Clear the volume assignment and return to the empty state.
   */
  void clearVolume();

  /**
   * @brief Return whether a volume is currently assigned.
   */
  [[nodiscard]] bool hasVolume() const noexcept;

  /**
   * @brief Select the orthogonal orientation presented by the widget.
   *
   * The current slice index is clamped to the valid range for the selected
   * orientation.
   *
   * @param orientation Orthogonal slice orientation to present.
   */
  void setOrientation(qvp::SliceOrientation orientation);

  /**
   * @brief Return the currently selected slice orientation.
   */
  [[nodiscard]] qvp::SliceOrientation orientation() const noexcept;

  /**
   * @brief Select the zero-based slice index.
   *
   * The requested index is clamped to the current valid range. With no usable
   * volume assigned, the index remains zero.
   *
   * @param sliceIndex Requested zero-based slice index.
   */
  void setSliceIndex(std::size_t sliceIndex);

  /**
   * @brief Return the current zero-based slice index.
   */
  [[nodiscard]] std::size_t sliceIndex() const noexcept;

  /**
   * @brief Return the valid slice count for the current volume and orientation.
   *
   * Returns zero when no valid, non-empty volume is assigned.
   */
  [[nodiscard]] std::size_t sliceCount() const noexcept;

  /**
   * @brief Set or hide the persistent crosshair in image coordinates.
   *
   * The persisted value uses slice-image pixel space, with pixel centers at
   * `(index + 0.5)`. It is not the normalized image-local coordinate space
   * emitted by qvp::OpenGLSliceViewer::crosshairPositionChanged. A position is
   * clamped to the displayed image bounds. Passing std::nullopt hides the
   * crosshair; a position is ignored while no slice image is available.
   *
   * @param position Crosshair position, or std::nullopt to hide it.
   */
  void setCrosshairPosition(std::optional<QPointF> position);

  /**
   * @brief Return the current crosshair position in slice-image pixel space.
   *
   * Pixel centers are represented as `(index + 0.5)`; the returned value is not
   * an OpenGL normalized image-local coordinate.
   *
   * @return Clamped pixel-space crosshair position, or std::nullopt when hidden.
   */
  [[nodiscard]] std::optional<QPointF> crosshairPosition() const noexcept;

  /**
   * @brief Return the dimensions of the currently displayed slice image.
   *
   * @return Image dimensions, or an empty size when no slice is available.
   */
  [[nodiscard]] QSize imageSize() const noexcept;

  /**
   * @brief Re-extract and present the current slice.
   *
   * The empty state clears the renderer without attempting extraction.
   */
  void refresh();

signals:
  /**
   * @brief Notify that the user moved the qtvp crosshair over this slice.
   *
   * The position is converted from qtvp normalized image-local coordinates to
   * this class's slice-image pixel-space contract, with centers at
   * `(index + 0.5)`.
   *
   * @param imagePoint Crosshair position in slice-image pixel space.
   */
  void imagePointChanged(QPointF imagePoint);

private:
  /** @brief Keep the overlay aligned after qtvp viewport geometry changes. */
  bool eventFilter(QObject* watched, QEvent* event) override;
  void handleViewerCrosshairPositionChanged(QPointF normalizedPosition);
  void clampSliceIndex() noexcept;
  void clampCrosshairPosition() noexcept;
  void presentCurrentImage();

  /** @brief Map the pixel-space crosshair into the fitted display rectangle. */
  void updateCrosshairOverlay();

  /**
   * @brief Externally owned volume observed by this widget.
   */
  const qvp::VolumeData* volume_ = nullptr;
  qvp::SliceOrientation orientation_ = qvp::SliceOrientation::Axial;
  std::size_t sliceIndex_ = 0;
  QImage sliceImage_;
  float sliceSpacingX_ = 1.0F;
  float sliceSpacingY_ = 1.0F;
  std::optional<QPointF> crosshairPosition_;

  // Qt parent ownership manages these widgets; the pointers are non-owning.
  qvp::OpenGLSliceViewer* viewer_ = nullptr;
  QWidget* verticalCrosshairLine_ = nullptr;
  QWidget* horizontalCrosshairLine_ = nullptr;
};

} // namespace maiw::viewer
