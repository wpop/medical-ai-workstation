#pragma once

#include "qtviewerpro/core/SliceOrientation.h"

#include <QWidget>

#include <cstddef>

namespace qvp
{

class OpenGLSliceViewer;
class VolumeData;

} // namespace qvp

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
   * @brief Re-extract and present the current slice.
   *
   * The empty state clears the renderer without attempting extraction.
   */
  void refresh();

private:
  void clampSliceIndex() noexcept;

  /**
   * @brief Externally owned volume observed by this widget.
   */
  const qvp::VolumeData* volume_ = nullptr;
  qvp::SliceOrientation orientation_ = qvp::SliceOrientation::Axial;
  std::size_t sliceIndex_ = 0;

  // Qt parent ownership manages the viewer; this pointer is non-owning.
  qvp::OpenGLSliceViewer* viewer_ = nullptr;
};

} // namespace maiw::viewer
