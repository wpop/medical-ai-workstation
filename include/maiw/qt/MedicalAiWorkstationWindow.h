#pragma once

#include "maiw/qt/CardiacMriClassificationWindow.h"
#include "maiw/viewer/ViewerWorkspaceWidget.h"

#include <QMainWindow>
#include <QString>

class QCloseEvent;

namespace maiw::qt
{

class CardiacMriClassificationWorkflow;

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

protected:
  /**
   * @brief Reject close requests while asynchronous workstation work is active.
   *
   * Close handling remains non-blocking. Active work continues under the
   * existing workflow lifetime contracts, and the workstation may be closed
   * after the operation finishes.
   *
   * @param event Qt close event.
   */
  void closeEvent(QCloseEvent* event) override;

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

  /**
   * @brief Externally owned cardiac workflow used for lifecycle state queries.
   *
   * The application composition root guarantees that this workflow outlives
   * the workstation window.
   */
  CardiacMriClassificationWorkflow& classificationWorkflow_;

  // Qt parent ownership manages both children; these pointers are non-owning.
  viewer::ViewerWorkspaceWidget* viewerWorkspace_ = nullptr;
  CardiacMriClassificationWindow* classificationWindow_ = nullptr;
};

} // namespace maiw::qt
