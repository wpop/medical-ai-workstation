#pragma once

#include "maiw/qt/CardiacMriClassificationWindow.h"
#include "maiw/viewer/ViewerWorkspaceWidget.h"

#include <QMainWindow>
#include <QString>

namespace maiw::qt
{

/**
 * @brief Composes medical-image viewing and cardiac classification in one window.
 *
 * The window owns one viewer workspace and one cardiac classification widget
 * through Qt parent-child ownership. The classification workflow is injected
 * and remains externally owned; this class performs no loading, preprocessing,
 * inference, or numerical postprocessing itself.
 */
class MedicalAiWorkstationWindow final : public QMainWindow
{
public:
  /**
   * @brief Construct the unified Medical AI Workstation window.
   *
   * The supplied workflow must outlive this window and every classification
   * operation started through its classification child.
   *
   * @param classificationWorkflow Externally owned cardiac workflow.
   * @param classNames Validated deployment class names in model-output order.
   * @param parent Optional parent widget.
   */
  explicit MedicalAiWorkstationWindow(
      CardiacMriClassificationWorkflow& classificationWorkflow,
      cardiac::CardiacMriDeploymentMetadata::ClassNames classNames,
      QWidget* parent = nullptr);

  /**
   * @brief Return the stable viewer workspace owned by this window.
   */
  [[nodiscard]] const viewer::ViewerWorkspaceWidget& viewerWorkspace() const noexcept;

  /**
   * @brief Return the stable cardiac classification widget owned by this window.
   */
  [[nodiscard]] const CardiacMriClassificationWindow&
  classificationWindow() const noexcept;

private:
  /**
   * @brief Request viewer loading for a committed non-empty volume path.
   *
   * Whitespace-only paths are ignored so they cannot replace the currently
   * displayed volume or start a viewer workflow.
   *
   * @param path Path committed by the cardiac study input widget.
   */
  void loadCommittedVolumePath(const QString& path);

  // Qt parent ownership manages both children; these pointers are non-owning.
  viewer::ViewerWorkspaceWidget* viewerWorkspace_ = nullptr;
  CardiacMriClassificationWindow* classificationWindow_ = nullptr;
};

} // namespace maiw::qt
