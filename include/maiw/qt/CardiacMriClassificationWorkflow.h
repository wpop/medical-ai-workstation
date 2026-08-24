#pragma once

#include "maiw/cardiac/CardiacMriClassificationResult.h"
#include "maiw/cardiac/CardiacMriClassificationService.h"

#include <QFutureWatcher>
#include <QObject>
#include <QString>

#include <optional>

namespace maiw::qt
{

/**
 * @brief Executes cardiac MRI classification asynchronously for Qt applications.
 *
 * The workflow keeps the synchronous classification service outside the Qt event
 * loop and delivers completion notifications in the workflow object's thread.
 *
 * The referenced classification service is externally owned and must outlive
 * this workflow and any classification operation in progress.
 */
class CardiacMriClassificationWorkflow final : public QObject
{
  Q_OBJECT

public:
  /**
   * @brief Construct a Qt asynchronous workflow around an existing service.
   *
   * @param service Externally owned classification service that must outlive
   * this workflow and all work started through it.
   * @param parent Optional QObject parent.
   */
  explicit CardiacMriClassificationWorkflow(
      cardiac::CardiacMriClassificationService& service,
      QObject* parent = nullptr);

  /**
   * @brief Wait for any active classification operation before destruction.
   */
  ~CardiacMriClassificationWorkflow() override;

  /**
   * @brief Start asynchronous classification of the selected ED and ES volumes.
   *
   * Only one classification operation may be active at a time. Invalid input,
   * medical-volume loading failures, and inference failures are reported through
   * classificationFailed().
   *
   * @param edPath Path to the end-diastolic medical volume.
   * @param esPath Path to the end-systolic medical volume.
   */
  void startClassification(QString edPath, QString esPath);

  /**
   * @brief Return true while a classification operation is active.
   */
  [[nodiscard]] bool isRunning() const noexcept;

signals:
  /**
   * @brief Emitted when a classification operation has started.
   */
  void classificationStarted();

  /**
   * @brief Emitted after successful classification.
   *
   * @param result Validated cardiac classification result.
   */
  void classificationSucceeded(
      const cardiac::CardiacMriClassificationResult& result);

  /**
   * @brief Emitted when classification cannot complete successfully.
   *
   * @param message User-presentable error description.
   */
  void classificationFailed(const QString& message);

private:
  struct AsyncResult
  {
    std::optional<cardiac::CardiacMriClassificationResult> classification;
    QString errorMessage;
  };

  void handleFinished();

  cardiac::CardiacMriClassificationService& service_;
  QFutureWatcher<AsyncResult> watcher_;
  bool running_ = false;
};

} // namespace maiw::qt
