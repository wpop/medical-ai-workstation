#pragma once

#include "maiw/cardiac/CardiacMriClassificationResult.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"

#include <QString>
#include <QWidget>

class QLabel;
class QLineEdit;
class QPushButton;
class QEvent;

namespace maiw::qt
{

class CardiacMriClassificationResultWidget;
class CardiacMriClassificationWorkflow;

/**
 * @brief Qt presentation widget for the cardiac MRI classification workflow.
 *
 * This class is a presentation-layer coordinator. It collects the paths to the
 * end-diastolic (ED) and end-systolic (ES) volumes from the user, starts the
 * asynchronous cardiac classification workflow, reflects the current operation
 * state in the controls, and forwards completed results to the dedicated result
 * presentation widget.
 *
 * The window deliberately contains no medical-image preprocessing, ONNX Runtime
 * inference, softmax calculation, prediction selection, or deployment-class
 * mapping logic. Those responsibilities remain in the validated headless
 * application-service layer.
 *
 * The CardiacMriClassificationWorkflow instance is externally owned. The caller
 * must guarantee that the workflow outlives this window.
 */
class CardiacMriClassificationWindow final : public QWidget
{
  Q_OBJECT

public:
  /**
   * @brief Construct the cardiac MRI classification presentation widget.
   *
   * The supplied workflow is used as the asynchronous execution boundary for
   * all classification requests initiated by this window.
   *
   * The supplied class names are copied into the result presentation widget and
   * must already represent the validated deployment output order.
   *
   * @param workflow Externally owned asynchronous classification workflow.
   * @param classNames Validated deployment class names in model-output order.
   * @param parent Optional Qt parent widget.
   */
  explicit CardiacMriClassificationWindow(
      CardiacMriClassificationWorkflow& workflow,
      cardiac::CardiacMriDeploymentMetadata::ClassNames classNames,
      QWidget* parent = nullptr);

  /**
   * @brief Return whether the ED path editor is enabled.
   */
  [[nodiscard]] bool isEdPathEditEnabled() const;

  /**
   * @brief Return whether the ED browse button is enabled.
   */
  [[nodiscard]] bool isEdBrowseButtonEnabled() const;

  /**
   * @brief Return whether the ES path editor is enabled.
   */
  [[nodiscard]] bool isEsPathEditEnabled() const;

  /**
   * @brief Return whether the ES browse button is enabled.
   */
  [[nodiscard]] bool isEsBrowseButtonEnabled() const;

  /**
   * @brief Return whether the classify button is enabled.
   */
  [[nodiscard]] bool isClassifyButtonEnabled() const;

  /**
   * @brief Return the result presentation widget owned by this window.
   */
  [[nodiscard]] const CardiacMriClassificationResultWidget* resultWidget() const noexcept;

signals:
  /**
   * @brief Notify that the user committed the ED medical-volume path.
   *
   * The path is emitted after an accepted file selection or manual editing is
   * finished. Classification is not started by this signal.
   *
   * @param path Committed ED volume path exactly as entered by the user.
   */
  void edVolumePathCommitted(const QString& path);

  /**
   * @brief Notify that the user committed the ES medical-volume path.
   *
   * The path is emitted after an accepted file selection or manual editing is
   * finished. Classification is not started by this signal.
   *
   * @param path Committed ES volume path exactly as entered by the user.
   */
  void esVolumePathCommitted(const QString& path);

private slots:
  /**
   * @brief Open a file-selection dialog for the ED medical volume.
   *
   * If the user selects a file, the selected path replaces the current ED path.
   * Cancelling the dialog leaves the existing value unchanged.
   */
  void browseEdVolume();

  /**
   * @brief Open a file-selection dialog for the ES medical volume.
   *
   * If the user selects a file, the selected path replaces the current ES path.
   * Cancelling the dialog leaves the existing value unchanged.
   */
  void browseEsVolume();

  /**
   * @brief Publish a manually completed ED path edit.
   *
   * An editingFinished notification caused by the same Browse activation is
   * ignored because the Browse result independently determines whether a path
   * is committed. Ordinary keyboard focus navigation remains a manual commit.
   */
  void handleEdPathEditingFinished();

  /**
   * @brief Publish a manually completed ES path edit.
   *
   * An editingFinished notification caused by the same Browse activation is
   * ignored because the Browse result independently determines whether a path
   * is committed. Ordinary keyboard focus navigation remains a manual commit.
   */
  void handleEsPathEditingFinished();

  /**
   * @brief Apply and publish an ED path selected through Browse.
   *
   * An empty selection represents cancellation and has no effect.
   *
   * @param selectedPath Path returned by the ED file-selection dialog.
   */
  void publishEdBrowseSelection(const QString& selectedPath);

  /**
   * @brief Apply and publish an ES path selected through Browse.
   *
   * An empty selection represents cancellation and has no effect.
   *
   * @param selectedPath Path returned by the ES file-selection dialog.
   */
  void publishEsBrowseSelection(const QString& selectedPath);

  /**
   * @brief Start classification using the currently selected ED and ES paths.
   *
   * This method delegates execution to CardiacMriClassificationWorkflow and
   * performs no medical-image loading or inference on the Qt main thread.
   */
  void startClassification();

  /**
   * @brief Handle notification that asynchronous classification has started.
   *
   * The window updates its controls and status presentation so that another
   * classification cannot be started while the current request is active.
   */
  void handleClassificationStarted();

  /**
   * @brief Present a successfully completed classification result.
   *
   * The result is forwarded unchanged to CardiacMriClassificationResultWidget.
   * This window does not recompute probabilities, prediction indices, or class
   * names.
   *
   * @param result Validated result produced by CardiacMriClassificationService.
   */
  void handleClassificationSucceeded(
      const cardiac::CardiacMriClassificationResult& result);

  /**
   * @brief Present a controlled asynchronous classification failure.
   *
   * @param message User-presentable error description supplied by the workflow.
   */
  void handleClassificationFailed(const QString& message);

private:
  /**
   * @brief Mark mouse activation of a Browse button before focus changes.
   *
   * This prevents the corresponding line edit from publishing its old value
   * when the same mouse action transfers focus and opens Browse.
   */
  bool eventFilter(QObject* watched, QEvent* event) override;

  /**
   * @brief Create widgets, layouts, and signal/slot connections.
   *
   * All child widgets created here are owned through Qt parent-child ownership.
   * Raw widget pointers stored by this class are non-owning convenience handles.
   */
  void initializeUi();

  /**
   * @brief Synchronize all input controls with the workflow running state.
   */
  void updateControls();

  /**
   * @brief Externally owned asynchronous classification workflow.
   *
   * This reference is non-owning. The application composition root must destroy
   * this window before destroying the workflow.
   */
  CardiacMriClassificationWorkflow& workflow_;

  /**
   * @brief ED volume path editor.
   *
   * Owned by the Qt parent-child hierarchy.
   */
  QLineEdit* edPathEdit_ = nullptr;

  /**
   * @brief Button that opens the ED volume file-selection dialog.
   *
   * The pointer is non-owning. Qt parent ownership controls its lifetime.
   */
  QPushButton* edBrowseButton_ = nullptr;

  /**
   * @brief ES volume path editor.
   *
   * Owned by the Qt parent-child hierarchy.
   */
  QLineEdit* esPathEdit_ = nullptr;

  /**
   * @brief Button that opens the ES volume file-selection dialog.
   *
   * The pointer is non-owning. Qt parent ownership controls its lifetime.
   */
  QPushButton* esBrowseButton_ = nullptr;

  /**
   * @brief Button that starts asynchronous classification.
   *
   * The button is disabled while the workflow reports an active operation.
   * Owned by the Qt parent-child hierarchy.
   */
  QPushButton* classifyButton_ = nullptr;

  /**
   * @brief Text label used for transient workflow state information.
   *
   * Examples include the active classification state. Persistent result and
   * error presentation remains delegated to the result widget.
   *
   * Owned by the Qt parent-child hierarchy.
   */
  QLabel* statusLabel_ = nullptr;

  /**
   * @brief Dedicated presentation widget for classification output.
   *
   * The pointer is non-owning. Qt parent ownership controls its lifetime.
   */
  CardiacMriClassificationResultWidget* resultWidget_ = nullptr;

  bool suppressNextEdEditingCommit_ = false;
  bool suppressNextEsEditingCommit_ = false;
};

} // namespace maiw::qt
