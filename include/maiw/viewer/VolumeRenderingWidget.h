#pragma once

#include "maiw/viewer/VolumeLoadWorkflow.h"

#include "qtviewerpro/render/VolumeTransferFunction.h"

#include <QString>
#include <QWidget>

namespace qvp
{

class OpenGLVolumeRendererWidget;

} // namespace qvp

namespace maiw::viewer
{

/**
 * @brief Presents one immutable medical volume through the public qtvp 3D renderer.
 *
 * The widget stores shared ownership of the assigned volume and gives the same
 * shared immutable instance to qtvp. Only shared_ptr handles are copied; one
 * control block and one voxel store remain shared. The renderer child is owned
 * by Qt.
 */
class VolumeRenderingWidget final : public QWidget
{
  Q_OBJECT

public:
  /**
   * @brief Construct an empty 3D volume-rendering widget.
   *
   * @param parent Optional parent widget.
   */
  explicit VolumeRenderingWidget(QWidget* parent = nullptr);

  /**
   * @brief Assign shared immutable ownership of a medical volume.
   *
   * A null or invalid volume clears the current assignment. The same shared
   * instance is retained by this wrapper and passed to the qtvp renderer.
   *
   * @param volume Shared immutable volume to render.
   */
  void setVolume(SharedVolume volume);

  /**
   * @brief Clear the volume assignment and renderer state.
   *
   * Both MAIW and qtvp release their CPU-side shared ownership immediately.
   * qtvp deletes an existing GPU texture during its next paint or destruction.
   * The operation is idempotent.
   */
  void clearVolume();

  /**
   * @brief Return whether a valid medical volume is currently assigned.
   */
  [[nodiscard]] bool hasVolume() const noexcept;

  /**
   * @brief Select a public qtvp volume-rendering preset.
   *
   * @param preset Rendering preset to apply.
   */
  void setRenderPreset(qvp::VolumeRenderPreset preset);

  /**
   * @brief Set global volume opacity through the qtvp transfer function.
   *
   * qtvp clamps the value to its supported range.
   *
   * @param opacity Requested global opacity.
   */
  void setGlobalOpacity(float opacity);

  /**
   * @brief Set a manual source-intensity interval for volume rendering.
   *
   * qtvp orders and clamps the interval according to the active volume and
   * selects its Custom rendering preset.
   *
   * @param minimum First requested intensity bound.
   * @param maximum Second requested intensity bound.
   */
  void setManualIntensityRange(float minimum, float maximum);

  /**
   * @brief Return the public qtvp transfer-function state.
   *
   * @return Current renderer preset, opacity, and intensity interval.
   */
  [[nodiscard]] qvp::VolumeTransferFunctionState transferFunctionState() const;

  /**
   * @brief Reset qtvp camera rotation and distance to their defaults.
   */
  void resetView();

signals:
  /**
   * @brief Forward successful qtvp 3D texture upload notification.
   *
   * @param width Uploaded texture width.
   * @param height Uploaded texture height.
   * @param depth Uploaded texture depth.
   */
  void volumeUploadSucceeded(int width, int height, int depth);

  /**
   * @brief Forward a qtvp 3D texture upload failure.
   *
   * @param message Renderer-provided failure description.
   */
  void volumeUploadFailed(const QString& message);

private:
  SharedVolume volume_;

  // Qt parent ownership manages the renderer; this pointer is non-owning.
  qvp::OpenGLVolumeRendererWidget* renderer_ = nullptr;
};

} // namespace maiw::viewer
