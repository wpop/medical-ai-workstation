#include "maiw/viewer/VolumeLoadWorkflow.h"

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFileInfo>
#include <QThread>
#include <QTimer>
#include <QUuid>

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

constexpr int kTestTimeoutMilliseconds = 10000;
const QString kRequiredPathError =
    QStringLiteral("A medical volume path is required.");
const QString kConcurrentLoadError =
    QStringLiteral("A medical volume load is already in progress.");

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

QString missingVolumePath()
{
  const QString path =
      QDir::temp().filePath(QStringLiteral("maiw-missing-volume-%1.nii.gz")
                                .arg(QUuid::createUuid().toString(
                                    QUuid::WithoutBraces)));
  require(!QFileInfo::exists(path),
          "generated missing medical-volume path unexpectedly exists");
  return path;
}

void waitForAsynchronousFailure(QCoreApplication& application,
                                maiw::viewer::VolumeLoadWorkflow& workflow,
                                const QString& path,
                                std::size_t& asynchronousFailureCount,
                                bool& deliveredOnMainThread)
{
  QEventLoop eventLoop;
  QTimer watchdog;
  watchdog.setSingleShot(true);
  watchdog.setInterval(kTestTimeoutMilliseconds);
  std::string failure;

  const auto failureConnection = QObject::connect(
      &workflow,
      &maiw::viewer::VolumeLoadWorkflow::loadingFailed,
      &eventLoop,
      [&application,
       &workflow,
       &eventLoop,
       &watchdog,
       &asynchronousFailureCount,
       &deliveredOnMainThread,
       &failure](const QString& message)
      {
        if (workflow.isRunning())
        {
          return;
        }

        if (!message.startsWith(
                QStringLiteral("Failed to load the medical volume: ")))
        {
          failure = "missing volume produced an unexpected error message";
        }
        deliveredOnMainThread =
            deliveredOnMainThread &&
            QThread::currentThread() == application.thread();
        ++asynchronousFailureCount;
        watchdog.stop();
        eventLoop.quit();
      });
  QObject::connect(&watchdog,
                   &QTimer::timeout,
                   &eventLoop,
                   [&eventLoop, &failure]()
                   {
                     failure = "timed out waiting for medical-volume load failure";
                     eventLoop.quit();
                   });

  workflow.startLoading(path);
  require(workflow.isRunning(), "workflow did not enter the running state");
  watchdog.start();
  eventLoop.exec();
  QObject::disconnect(failureConnection);

  require(failure.empty(), failure);
  require(!workflow.isRunning(),
          "workflow remained running after asynchronous failure");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QCoreApplication application(argc, argv);
    maiw::viewer::VolumeLoadWorkflow workflow;

    require(!workflow.isRunning(), "workflow is initially running");

    std::size_t startedCount = 0;
    std::size_t synchronousFailureCount = 0;
    std::size_t asynchronousFailureCount = 0;
    bool deliveredOnMainThread = true;
    bool unexpectedSuccess = false;

    QObject::connect(
        &workflow,
        &maiw::viewer::VolumeLoadWorkflow::loadingStarted,
        &application,
        [&startedCount]()
        {
          ++startedCount;
        });
    QObject::connect(
        &workflow,
        &maiw::viewer::VolumeLoadWorkflow::loadingSucceeded,
        &application,
        [&unexpectedSuccess](maiw::viewer::SharedVolume)
        {
          unexpectedSuccess = true;
        });
    QObject::connect(
        &workflow,
        &maiw::viewer::VolumeLoadWorkflow::loadingFailed,
        &application,
        [&workflow, &synchronousFailureCount](const QString& message)
        {
          if (workflow.isRunning() && message == kConcurrentLoadError)
          {
            ++synchronousFailureCount;
          }
          else if (!workflow.isRunning() && message == kRequiredPathError)
          {
            ++synchronousFailureCount;
          }
        });

    workflow.startLoading(QStringLiteral("   "));
    require(!workflow.isRunning(),
            "empty path unexpectedly started the workflow");
    require(synchronousFailureCount == 1,
            "empty path did not emit the required-path failure");

    const QString firstMissingPath = missingVolumePath();
    QEventLoop firstFailureLoop;
    QTimer firstFailureWatchdog;
    firstFailureWatchdog.setSingleShot(true);
    firstFailureWatchdog.setInterval(kTestTimeoutMilliseconds);
    QString firstAsynchronousFailure;
    bool firstFailureTimedOut = false;
    const auto firstFailureConnection = QObject::connect(
        &workflow,
        &maiw::viewer::VolumeLoadWorkflow::loadingFailed,
        &firstFailureLoop,
        [&application,
         &workflow,
         &firstFailureLoop,
         &firstFailureWatchdog,
         &firstAsynchronousFailure,
         &asynchronousFailureCount,
         &deliveredOnMainThread](const QString& message)
        {
          if (!workflow.isRunning())
          {
            firstAsynchronousFailure = message;
            deliveredOnMainThread =
                deliveredOnMainThread &&
                QThread::currentThread() == application.thread();
            ++asynchronousFailureCount;
            firstFailureWatchdog.stop();
            firstFailureLoop.quit();
          }
        });
    QObject::connect(&firstFailureWatchdog,
                     &QTimer::timeout,
                     &firstFailureLoop,
                     [&firstFailureLoop, &firstFailureTimedOut]()
                     {
                       firstFailureTimedOut = true;
                       firstFailureLoop.quit();
                     });

    workflow.startLoading(firstMissingPath);
    require(workflow.isRunning(), "first load did not enter the running state");
    workflow.startLoading(missingVolumePath());
    require(synchronousFailureCount == 2,
            "overlapping load was not rejected synchronously");
    require(workflow.isRunning(),
            "overlapping rejection stopped the active load");

    firstFailureWatchdog.start();
    firstFailureLoop.exec();
    QObject::disconnect(firstFailureConnection);
    require(!firstFailureTimedOut,
            "timed out waiting for the first medical-volume load failure");
    require(!workflow.isRunning(),
            "workflow remained running after the first load failure");
    require(firstAsynchronousFailure.startsWith(
                QStringLiteral("Failed to load the medical volume: ")),
            "first missing volume produced an unexpected error message");

    waitForAsynchronousFailure(application,
                               workflow,
                               missingVolumePath(),
                               asynchronousFailureCount,
                               deliveredOnMainThread);

    require(startedCount == 2,
            "workflow started an unexpected number of load operations");
    require(asynchronousFailureCount == 2,
            "both asynchronous load failures were not delivered");
    require(deliveredOnMainThread,
            "asynchronous failure was not delivered on the main Qt thread");
    require(!unexpectedSuccess,
            "missing medical volume unexpectedly loaded successfully");

    auto activeWorkflow =
        std::make_unique<maiw::viewer::VolumeLoadWorkflow>();
    activeWorkflow->startLoading(missingVolumePath());
    require(activeWorkflow->isRunning(),
            "shutdown workflow did not enter the running state");
    activeWorkflow.reset();

    std::cout << "Volume load workflow test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Volume load workflow test failed: "
              << error.what() << '\n';
    return 1;
  }
}
