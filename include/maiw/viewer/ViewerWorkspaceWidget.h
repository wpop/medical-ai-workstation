#pragma once

#include "maiw/viewer/MprViewerWidget.h"
#include "maiw/viewer/VolumeLoadWorkflow.h"
#include "maiw/viewer/VolumeRenderingWidget.h"

#include <QString>
#include <QWidget>

#include <memory>

namespace maiw::viewer
{

/**
 * @brief Presents synchronized MPR and 3D views of one immutable medical volume.
 *
 * The workspace owns the canonical SharedVolume. Its MPR child observes that
 * instance without ownership, while its 3D child shares ownership of the same
 * instance. An owned VolumeLoadWorkflow coordinates asynchronous replacement.
 * All composed objects are managed through Qt parent-child ownership.
 */
class ViewerWorkspaceWidget final : public QWidget
{
  Q_OBJECT

public:
  /**
   * @brief Construct an empty viewer workspace.
   *
   * @param parent Optional parent widget.
   */
  explicit ViewerWorkspaceWidget(QWidget* parent = nullptr);

  /**
   * @brief Clear viewer assignments and safely destroy the owned workflow.
   *
   * Any active load is awaited by the workflow's destruction contract.
   */
  ~ViewerWorkspaceWidget() override;

  /**
   * @brief Start asynchronous loading of one medical volume.
   *
   * The currently displayed volume remains assigned while loading. Empty paths
   * and overlapping requests are rejected through volumeLoadingFailed(),
   * consistently with VolumeLoadWorkflow.
   *
   * @param path Path to the medical volume or supported series directory.
   */
  void loadVolume(const QString& path);

  /**
   * @brief Return true while an asynchronous volume load is active.
   */
  [[nodiscard]] bool isLoading() const noexcept;

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
   * child's non-owning observation. The operation is idempotent and does not
   * cancel an active asynchronous load.
   */
  void clearVolume();

  /**
   * @brief Return whether a valid, non-empty volume is assigned.
   */
  [[nodiscard]] bool hasVolume() const noexcept;

  /**
   * @brief Return a non-owning observer of the canonical immutable volume.
   *
   * The observer expires after replacement, clearing, or workspace destruction
   * unless another owner independently retains the same volume.
   */
  [[nodiscard]] std::weak_ptr<const qvp::VolumeData> volumeObserver() const noexcept;

  /**
   * @brief Return the stable MPR child widget.
   */
  [[nodiscard]] const MprViewerWidget& mprViewer() const noexcept;

  /**
   * @brief Return the stable 3D volume-rendering child widget.
   */
  [[nodiscard]] const VolumeRenderingWidget& volumeRenderingWidget() const noexcept;

signals:
  /**
   * @brief Emitted when asynchronous medical-volume loading starts.
   */
  void volumeLoadingStarted();

  /**
   * @brief Emitted after a loaded volume has been assigned to both viewer modes.
   */
  void volumeLoadingSucceeded();

  /**
   * @brief Emitted when a load request cannot replace the displayed volume.
   *
   * @param message User-presentable error description.
   */
  void volumeLoadingFailed(const QString& message);

private:
  void handleVolumeLoaded(SharedVolume volume);

  SharedVolume volume_;

  // Qt parent ownership manages these objects; the pointers are non-owning.
  VolumeLoadWorkflow* loadWorkflow_ = nullptr;
  MprViewerWidget* mprViewer_ = nullptr;
  VolumeRenderingWidget* volumeRenderingWidget_ = nullptr;
};

} // namespace maiw::viewer
