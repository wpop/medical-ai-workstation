#pragma once

#include "maiw/viewer/MprViewerWidget.h"
#include "maiw/viewer/VolumeRenderingWidget.h"

#include <QWidget>

namespace maiw::viewer
{

/**
 * @brief Presents synchronized MPR and 3D views of one immutable medical volume.
 *
 * The workspace owns the canonical SharedVolume. Its MPR child observes that
 * instance without ownership, while its 3D child shares ownership of the same
 * instance. Both child widgets are managed through Qt parent-child ownership.
 */
class ViewerWorkspaceWidget final : public QWidget
{
public:
  /**
   * @brief Construct an empty viewer workspace.
   *
   * @param parent Optional parent widget.
   */
  explicit ViewerWorkspaceWidget(QWidget* parent = nullptr);

  /**
   * @brief Clear child assignments before releasing the canonical volume.
   */
  ~ViewerWorkspaceWidget() override;

  /**
   * @brief Assign one shared immutable volume to both viewer modes.
   *
   * A null, invalid, or empty volume clears the workspace. Valid assignments
   * retain one canonical shared handle, give its non-owning pointer to MPR, and
   * give the same shared instance to 3D rendering.
   *
   * @param volume Shared immutable medical volume.
   */
  void setVolume(SharedVolume volume);

  /**
   * @brief Clear MPR first, 3D rendering second, and ownership last.
   *
   * This order guarantees that the canonical SharedVolume outlives the MPR
   * child's non-owning observation. The operation is idempotent.
   */
  void clearVolume();

  /**
   * @brief Return whether a valid, non-empty volume is assigned.
   */
  [[nodiscard]] bool hasVolume() const noexcept;

  /**
   * @brief Return the stable MPR child widget.
   */
  [[nodiscard]] const MprViewerWidget& mprViewer() const noexcept;

  /**
   * @brief Return the stable 3D volume-rendering child widget.
   */
  [[nodiscard]] const VolumeRenderingWidget& volumeRenderingWidget() const noexcept;

private:
  SharedVolume volume_;

  // Qt parent ownership manages the child viewers; these pointers are non-owning.
  MprViewerWidget* mprViewer_ = nullptr;
  VolumeRenderingWidget* volumeRenderingWidget_ = nullptr;
};

} // namespace maiw::viewer
