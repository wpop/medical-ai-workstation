#include "maiw/viewer/SliceViewerWidget.h"
#include "maiw/viewer/VolumeLoadWorkflow.h"

#include "qtviewerpro/core/VolumeData.h"

#include <QApplication>
#include <QEventLoop>
#include <QString>
#include <QThread>
#include <QTimer>

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

constexpr int kLoadTimeoutMilliseconds = 30000;

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void validateOrientation(maiw::viewer::SliceViewerWidget& widget,
                         qvp::SliceOrientation orientation,
                         std::size_t expectedSliceCount,
                         const std::string& name)
{
  require(expectedSliceCount > 0, name + " volume dimension is zero");

  widget.setOrientation(orientation);
  require(widget.orientation() == orientation,
          name + " orientation was not retained");
  require(widget.sliceCount() == expectedSliceCount,
          name + " slice count does not match the real volume dimension");

  widget.setSliceIndex(0);
  require(widget.sliceIndex() == 0,
          name + " first slice was not selected");

  widget.setSliceIndex(expectedSliceCount - 1);
  require(widget.sliceIndex() == expectedSliceCount - 1,
          name + " last slice was not selected");

  widget.setSliceIndex(std::numeric_limits<std::size_t>::max());
  require(widget.sliceIndex() == expectedSliceCount - 1,
          name + " out-of-range slice index was not clamped");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);
    maiw::viewer::VolumeLoadWorkflow workflow;

    std::size_t startedCount = 0;
    std::size_t successCount = 0;
    std::size_t failureCount = 0;
    bool successDeliveredOnMainThread = false;
    bool workflowRunningDuringSuccess = true;
    bool timedOut = false;
    QString failureMessage;
    maiw::viewer::SharedVolume volume;

    QEventLoop loadEventLoop;
    QTimer loadWatchdog;
    loadWatchdog.setSingleShot(true);
    loadWatchdog.setInterval(kLoadTimeoutMilliseconds);

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
        &loadEventLoop,
        [&application,
         &workflow,
         &loadEventLoop,
         &loadWatchdog,
         &successCount,
         &successDeliveredOnMainThread,
         &workflowRunningDuringSuccess,
         &volume](maiw::viewer::SharedVolume loadedVolume)
        {
          ++successCount;
          successDeliveredOnMainThread =
              QThread::currentThread() == application.thread();
          workflowRunningDuringSuccess = workflow.isRunning();
          volume = std::move(loadedVolume);
          loadWatchdog.stop();
          loadEventLoop.quit();
        },
        Qt::QueuedConnection);
    QObject::connect(
        &workflow,
        &maiw::viewer::VolumeLoadWorkflow::loadingFailed,
        &loadEventLoop,
        [&loadEventLoop,
         &loadWatchdog,
         &failureCount,
         &failureMessage](const QString& message)
        {
          ++failureCount;
          failureMessage = message;
          loadWatchdog.stop();
          loadEventLoop.quit();
        });
    QObject::connect(
        &loadWatchdog,
        &QTimer::timeout,
        &loadEventLoop,
        [&loadEventLoop, &timedOut]()
        {
          timedOut = true;
          loadEventLoop.quit();
        });

    loadWatchdog.start();
    workflow.startLoading(
        QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH));
    require(workflow.isRunning(),
            "real ACDC load did not enter the running state");
    loadEventLoop.exec();

    require(!timedOut, "timed out loading the real ACDC ED volume");
    require(startedCount == 1,
            "real ACDC load did not start exactly once");
    require(successCount == 1,
            "real ACDC load did not succeed exactly once");
    require(failureCount == 0,
            "real ACDC load failed: " + failureMessage.toStdString());
    require(successDeliveredOnMainThread,
            "real ACDC load success was not delivered on the main Qt thread");
    require(!workflowRunningDuringSuccess,
            "workflow remained running while success was handled");
    require(!workflow.isRunning(),
            "workflow remained running after real ACDC load success");
    require(volume != nullptr,
            "real ACDC load returned a null shared volume");
    require(volume->isValid(),
            "real ACDC ED volume is invalid");
    require(!volume->isEmpty(),
            "real ACDC ED volume is empty");

    {
      maiw::viewer::SliceViewerWidget widget;
      widget.setVolume(volume.get());

      require(widget.hasVolume(),
              "widget did not retain the real volume assignment");
      validateOrientation(widget,
                          qvp::SliceOrientation::Axial,
                          volume->depth(),
                          "axial");
      validateOrientation(widget,
                          qvp::SliceOrientation::Sagittal,
                          volume->width(),
                          "sagittal");
      validateOrientation(widget,
                          qvp::SliceOrientation::Coronal,
                          volume->height(),
                          "coronal");

      widget.clearVolume();
      require(!widget.hasVolume(),
              "widget retained the volume after clear");
      require(widget.sliceCount() == 0,
              "cleared widget retained a slice count");
      require(widget.sliceIndex() == 0,
              "cleared widget retained a slice index");

      widget.setVolume(volume.get());
      require(widget.hasVolume(),
              "widget did not accept the reassigned real volume");
      require(widget.sliceCount() == volume->height(),
              "reassigned coronal slice count is incorrect");
      widget.setSliceIndex(volume->height() - 1);
      require(widget.sliceIndex() == volume->height() - 1,
              "reassigned real volume did not select the last coronal slice");
    }

    volume.reset();

    std::cout << "Real-volume slice viewer widget test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Real-volume slice viewer widget test failed: "
              << error.what() << '\n';
    return 1;
  }
}
