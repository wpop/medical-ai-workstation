#include "maiw/viewer/MprViewerWidget.h"
#include "maiw/viewer/VolumeLoadWorkflow.h"

#include "qtviewerpro/core/VolumeData.h"
#include "qtviewerpro/core/VolumePhysicalCoordinateMapper.h"

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

maiw::viewer::SharedVolume loadRealVolume(QApplication& application)
{
  maiw::viewer::VolumeLoadWorkflow workflow;
  QEventLoop eventLoop;
  QTimer watchdog;
  watchdog.setSingleShot(true);
  watchdog.setInterval(kLoadTimeoutMilliseconds);

  bool timedOut = false;
  bool deliveredOnMainThread = false;
  QString failureMessage;
  maiw::viewer::SharedVolume volume;

  QObject::connect(
      &workflow,
      &maiw::viewer::VolumeLoadWorkflow::loadingSucceeded,
      &eventLoop,
      [&application,
       &eventLoop,
       &watchdog,
       &deliveredOnMainThread,
       &volume](maiw::viewer::SharedVolume loadedVolume)
      {
        deliveredOnMainThread =
            QThread::currentThread() == application.thread();
        volume = std::move(loadedVolume);
        watchdog.stop();
        eventLoop.quit();
      },
      Qt::QueuedConnection);
  QObject::connect(
      &workflow,
      &maiw::viewer::VolumeLoadWorkflow::loadingFailed,
      &eventLoop,
      [&eventLoop, &watchdog, &failureMessage](const QString& message)
      {
        failureMessage = message;
        watchdog.stop();
        eventLoop.quit();
      });
  QObject::connect(
      &watchdog,
      &QTimer::timeout,
      &eventLoop,
      [&eventLoop, &timedOut]()
      {
        timedOut = true;
        eventLoop.quit();
      });

  watchdog.start();
  workflow.startLoading(QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH));
  require(workflow.isRunning(),
          "real ACDC MPR load did not enter the running state");
  eventLoop.exec();

  require(!timedOut, "timed out loading the real ACDC MPR volume");
  require(failureMessage.isEmpty(),
          "real ACDC MPR load failed: " + failureMessage.toStdString());
  require(!workflow.isRunning(),
          "workflow remained running after real ACDC MPR load");
  require(deliveredOnMainThread,
          "real ACDC MPR load was not delivered on the main Qt thread");
  require(volume != nullptr,
          "real ACDC MPR load returned a null shared volume");
  require(volume->isValid() && !volume->isEmpty(),
          "real ACDC MPR volume is invalid or empty");

  return volume;
}

void requirePosition(const qvp::VoxelIndex3D& actual,
                     const qvp::VoxelIndex3D& expected,
                     const std::string& context)
{
  require(actual.x == expected.x &&
              actual.y == expected.y &&
              actual.z == expected.z,
          context + ": voxel position mismatch");
}

void requireSynchronizedSlices(const maiw::viewer::MprViewerWidget& widget,
                               const qvp::VoxelIndex3D& position,
                               const std::string& context)
{
  require(widget.axialViewer().sliceIndex() == position.z,
          context + ": axial slice does not match voxel Z");
  require(widget.sagittalViewer().sliceIndex() == position.x,
          context + ": sagittal slice does not match voxel X");
  require(widget.coronalViewer().sliceIndex() == position.y,
          context + ": coronal slice does not match voxel Y");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);
    maiw::viewer::SharedVolume volume = loadRealVolume(application);

    {
      maiw::viewer::MprViewerWidget widget;
      widget.setVolume(volume.get());

      require(widget.hasVolume(),
              "MPR widget did not retain the real volume assignment");
      require(widget.axialViewer().sliceCount() == volume->depth(),
              "axial slice count does not match volume depth");
      require(widget.sagittalViewer().sliceCount() == volume->width(),
              "sagittal slice count does not match volume width");
      require(widget.coronalViewer().sliceCount() == volume->height(),
              "coronal slice count does not match volume height");

      const qvp::VoxelIndex3D center{
          volume->width() / 2,
          volume->height() / 2,
          volume->depth() / 2};
      require(center.x < volume->width() &&
                  center.y < volume->height() &&
                  center.z < volume->depth(),
              "initial center voxel is outside the real volume bounds");
      requirePosition(widget.voxelPosition(), center, "initial center");
      requireSynchronizedSlices(widget, center, "initial center");

      const qvp::VoxelIndex3D origin{};
      widget.setVoxelPosition(origin);
      requirePosition(widget.voxelPosition(), origin, "origin selection");
      requireSynchronizedSlices(widget, origin, "origin selection");

      const qvp::VoxelIndex3D last{
          volume->width() - 1,
          volume->height() - 1,
          volume->depth() - 1};
      widget.setVoxelPosition(last);
      requirePosition(widget.voxelPosition(), last, "last voxel selection");
      requireSynchronizedSlices(widget, last, "last voxel selection");

      const auto maximum = std::numeric_limits<std::size_t>::max();
      widget.setVoxelPosition(
          qvp::VoxelIndex3D{maximum, maximum, maximum});
      requirePosition(widget.voxelPosition(), last, "clamped selection");
      requireSynchronizedSlices(widget, last, "clamped selection");

      const auto physicalPosition = widget.physicalPosition();
      require(physicalPosition.has_value(),
              "MPR physical position is unavailable for the real volume");
      const qvp::PhysicalPoint3D expectedPhysicalPosition =
          qvp::VolumePhysicalCoordinateMapper::voxelToPhysical(*volume, last);
      require(physicalPosition->x == expectedPhysicalPosition.x &&
                  physicalPosition->y == expectedPhysicalPosition.y &&
                  physicalPosition->z == expectedPhysicalPosition.z,
              "MPR physical position does not match the public mapper");

      widget.clearVolume();
      require(!widget.hasVolume(),
              "MPR widget retained the volume after clear");
      requirePosition(widget.voxelPosition(), {}, "cleared state");
      require(!widget.physicalPosition().has_value(),
              "cleared MPR widget retained a physical position");
      require(!widget.axialViewer().hasVolume() &&
                  !widget.sagittalViewer().hasVolume() &&
                  !widget.coronalViewer().hasVolume(),
              "cleared MPR widget retained child volume assignments");

      widget.setVolume(volume.get());
      require(widget.hasVolume(),
              "MPR widget did not accept the reassigned real volume");
      requirePosition(widget.voxelPosition(), center, "reassigned center");
      requireSynchronizedSlices(widget, center, "reassigned center");
    }

    volume.reset();

    std::cout << "Real-volume MPR viewer widget test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Real-volume MPR viewer widget test failed: "
              << error.what() << '\n';
    return 1;
  }
}
