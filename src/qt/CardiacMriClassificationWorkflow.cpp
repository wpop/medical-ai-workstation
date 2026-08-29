#include "maiw/qt/CardiacMriClassificationWorkflow.h"

#include "qtviewerpro/io/MedicalVolumeLoaderRegistry.h"

#include <QFileInfo>
#include <QtConcurrent/QtConcurrentRun>

#include <exception>
#include <filesystem>
#include <system_error>
#include <utility>

namespace maiw::qt
{

namespace
{

bool hasValidatedCardiacInputFormat(const QString& path)
{
  const QString fileName = QFileInfo(path).fileName();
  return fileName.endsWith(QStringLiteral(".nii"), Qt::CaseInsensitive) ||
         fileName.endsWith(QStringLiteral(".nii.gz"), Qt::CaseInsensitive);
}

} // namespace

CardiacMriClassificationWorkflow::CardiacMriClassificationWorkflow(
    cardiac::CardiacMriClassificationService& service,
    QObject* parent)
    : QObject(parent),
      service_(service)
{
  connect(&watcher_,
          &QFutureWatcher<AsyncResult>::finished,
          this,
          &CardiacMriClassificationWorkflow::handleFinished);
}

CardiacMriClassificationWorkflow::~CardiacMriClassificationWorkflow()
{
  if (watcher_.isRunning())
  {
    watcher_.future().waitForFinished();
  }
}

void CardiacMriClassificationWorkflow::startClassification(
    QString edPath,
    QString esPath)
{
  if (running_)
  {
    emit classificationFailed(
        QStringLiteral("A cardiac MRI classification is already in progress."));
    return;
  }

  if (edPath.trimmed().isEmpty() || esPath.trimmed().isEmpty())
  {
    emit classificationFailed(
        QStringLiteral("Both ED and ES medical volume paths are required."));
    return;
  }

  std::error_code equivalenceError;
  const bool pathsAreEquivalent = std::filesystem::equivalent(
      std::filesystem::path{edPath.toStdString()},
      std::filesystem::path{esPath.toStdString()},
      equivalenceError);
  if (!equivalenceError && pathsAreEquivalent)
  {
    emit classificationFailed(
        QStringLiteral("ED and ES medical volume paths refer to the same file."));
    return;
  }

  if (!hasValidatedCardiacInputFormat(edPath) ||
      !hasValidatedCardiacInputFormat(esPath))
  {
    emit classificationFailed(
        QStringLiteral("Cardiac MRI classification accepts only validated "
                       "NIfTI inputs (.nii or .nii.gz)."));
    return;
  }

  running_ = true;
  emit classificationStarted();

  watcher_.setFuture(
      QtConcurrent::run(
          [this,
           edPath = std::move(edPath),
           esPath = std::move(esPath)]() -> AsyncResult
          {
            try
            {
              const qvp::VolumeLoadResult edLoadResult =
                  qvp::loadMedicalVolume(edPath);
              if (!edLoadResult.success)
              {
                return AsyncResult{
                    std::nullopt,
                    QStringLiteral("Failed to load the ED volume: %1")
                        .arg(edLoadResult.errorMessage)};
              }

              const qvp::VolumeLoadResult esLoadResult =
                  qvp::loadMedicalVolume(esPath);
              if (!esLoadResult.success)
              {
                return AsyncResult{
                    std::nullopt,
                    QStringLiteral("Failed to load the ES volume: %1")
                        .arg(esLoadResult.errorMessage)};
              }

              return AsyncResult{
                  service_.classify(edLoadResult.volume, esLoadResult.volume),
                  QString()};
            }
            catch (const std::exception& exception)
            {
              return AsyncResult{
                  std::nullopt,
                  QStringLiteral("Cardiac MRI classification failed: %1")
                      .arg(QString::fromUtf8(exception.what()))};
            }
            catch (...)
            {
              return AsyncResult{
                  std::nullopt,
                  QStringLiteral("Cardiac MRI classification failed.")};
            }
          }));
}

bool CardiacMriClassificationWorkflow::isRunning() const noexcept
{
  return running_;
}

void CardiacMriClassificationWorkflow::handleFinished()
{
  const AsyncResult result = watcher_.result();
  running_ = false;

  if (!result.classification.has_value())
  {
    emit classificationFailed(result.errorMessage);
    return;
  }

  emit classificationSucceeded(*result.classification);
}

} // namespace maiw::qt
