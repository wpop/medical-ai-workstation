#include "maiw/viewer/VolumeLoadWorkflow.h"

#include "qtviewerpro/io/MedicalVolumeLoaderRegistry.h"
#include "qtviewerpro/io/VolumeLoadResult.h"

#include <QtConcurrent/QtConcurrentRun>

#include <exception>
#include <utility>

namespace maiw::viewer
{

VolumeLoadWorkflow::VolumeLoadWorkflow(QObject* parent)
    : QObject(parent)
{
  qRegisterMetaType<SharedVolume>("maiw::viewer::SharedVolume");

  connect(&watcher_,
          &QFutureWatcher<AsyncResult>::finished,
          this,
          &VolumeLoadWorkflow::handleFinished);
}

VolumeLoadWorkflow::~VolumeLoadWorkflow()
{
  if (watcher_.isRunning())
  {
    watcher_.future().waitForFinished();
  }
}

void VolumeLoadWorkflow::startLoading(QString path)
{
  if (running_)
  {
    emit loadingFailed(
        QStringLiteral("A medical volume load is already in progress."));
    return;
  }

  path = path.trimmed();
  if (path.isEmpty())
  {
    emit loadingFailed(
        QStringLiteral("A medical volume path is required."));
    return;
  }

  running_ = true;
  emit loadingStarted();

  watcher_.setFuture(
      QtConcurrent::run(
          [path = std::move(path)]() -> AsyncResult
          {
            try
            {
              qvp::VolumeLoadResult loadResult = qvp::loadMedicalVolume(path);
              if (!loadResult.success)
              {
                return AsyncResult{
                    nullptr,
                    QStringLiteral("Failed to load the medical volume: %1")
                        .arg(loadResult.errorMessage)};
              }

              if (!loadResult.volume.isValid() || loadResult.volume.isEmpty())
              {
                return AsyncResult{
                    nullptr,
                    QStringLiteral("The loaded medical volume is invalid or empty.")};
              }

              return AsyncResult{
                  std::make_shared<const qvp::VolumeData>(
                      std::move(loadResult.volume)),
                  QString()};
            }
            catch (const std::exception& exception)
            {
              return AsyncResult{
                  nullptr,
                  QStringLiteral("Medical volume loading failed: %1")
                      .arg(QString::fromUtf8(exception.what()))};
            }
            catch (...)
            {
              return AsyncResult{
                  nullptr,
                  QStringLiteral("Medical volume loading failed.")};
            }
          }));
}

bool VolumeLoadWorkflow::isRunning() const noexcept
{
  return running_;
}

void VolumeLoadWorkflow::handleFinished()
{
  // Consume the future result so a persistent workflow does not retain the
  // loaded volume after signal delivery.
  AsyncResult result = watcher_.future().takeResult();
  running_ = false;

  if (!result.volume)
  {
    emit loadingFailed(result.errorMessage);
    return;
  }

  emit loadingSucceeded(std::move(result.volume));
}

} // namespace maiw::viewer
