#include "maiw/viewer/MprViewerWidget.h"
#include "maiw/viewer/VolumeLoadWorkflow.h"

#include "qtviewerpro/core/VolumeData.h"
#include "qtviewerpro/core/VolumePhysicalCoordinateMapper.h"

#include <QApplication>
#include <QEventLoop>
#include <QMetaObject>
#include <QPointF>
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

void requireCrosshair(const maiw::viewer::SliceViewerWidget& viewer,
                      std::size_t imageX,
                      std::size_t imageY,
                      const std::string& context)
{
  const auto position = viewer.crosshairPosition();
  require(position.has_value(), context + ": crosshair is hidden");
  require(position->x() == static_cast<double>(imageX) + 0.5 &&
              position->y() == static_cast<double>(imageY) + 0.5,
          context + ": crosshair position mismatch");
}

void requireSynchronizedCrosshairs(const maiw::viewer::MprViewerWidget& widget,
                                   const qvp::VoxelIndex3D& position,
                                   const std::string& context)
{
  requireCrosshair(widget.axialViewer(),
                   position.x,
                   position.y,
                   context + " axial");
  requireCrosshair(widget.sagittalViewer(),
                   position.y,
                   position.z,
                   context + " sagittal");
  requireCrosshair(widget.coronalViewer(),
                   position.x,
                   position.z,
                   context + " coronal");
}

void requireSynchronizedState(const maiw::viewer::MprViewerWidget& widget,
                              const qvp::VoxelIndex3D& position,
                              const std::string& context)
{
  requirePosition(widget.voxelPosition(), position, context);
  requireSynchronizedSlices(widget, position, context);
  requireSynchronizedCrosshairs(widget, position, context);
}

QPointF normalizedPositionForPixel(std::size_t imageX,
                                   std::size_t imageY,
                                   const QSize& imageSize)
{
  require(imageSize.width() > 1 && imageSize.height() > 1,
          "real MPR slice image is too small for normalized navigation");

  return QPointF(
      (2.0 * static_cast<double>(imageX) /
       static_cast<double>(imageSize.width() - 1)) -
          1.0,
      1.0 -
          (2.0 * static_cast<double>(imageY) /
           static_cast<double>(imageSize.height() - 1)));
}

void emitNormalizedCrosshair(
    const maiw::viewer::SliceViewerWidget& viewer,
    std::size_t imageX,
    std::size_t imageY)
{
  auto* const qtvpViewer =
      viewer.findChild<QObject*>(QStringLiteral("maiwSliceOpenGLViewer"));
  require(qtvpViewer != nullptr,
          "qtvp slice viewer testing hook is missing");

  const QPointF normalizedPosition =
      normalizedPositionForPixel(imageX, imageY, viewer.imageSize());
  require(QMetaObject::invokeMethod(qtvpViewer,
                                    "crosshairPositionChanged",
                                    Qt::DirectConnection,
                                    Q_ARG(QPointF, normalizedPosition)),
          "failed to emit a normalized qtvp crosshair position");
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
      requireSynchronizedState(widget, center, "initial center");

      const qvp::VoxelIndex3D origin{};
      widget.setVoxelPosition(origin);
      requireSynchronizedState(widget, origin, "origin selection");

      const qvp::VoxelIndex3D last{
          volume->width() - 1,
          volume->height() - 1,
          volume->depth() - 1};
      widget.setVoxelPosition(last);
      requireSynchronizedState(widget, last, "last voxel selection");

      const qvp::VoxelIndex3D axialStart{
          volume->width() - 1,
          volume->height() - 1,
          volume->depth() / 2};
      widget.setVoxelPosition(axialStart);
      const qvp::VoxelIndex3D axialResult{
          volume->width() / 3,
          volume->height() / 3,
          axialStart.z};
      widget.setPositionFromImagePoint(
          qvp::SliceOrientation::Axial,
          QPointF(static_cast<double>(axialResult.x) + 0.5,
                  static_cast<double>(axialResult.y) + 0.5));
      requireSynchronizedState(widget, axialResult, "axial image-point mapping");

      const qvp::VoxelIndex3D sagittalResult{
          axialResult.x,
          volume->height() / 4,
          volume->depth() / 3};
      widget.setPositionFromImagePoint(
          qvp::SliceOrientation::Sagittal,
          QPointF(static_cast<double>(sagittalResult.y) + 0.5,
                  static_cast<double>(sagittalResult.z) + 0.5));
      requireSynchronizedState(widget,
                               sagittalResult,
                               "sagittal image-point mapping");

      const qvp::VoxelIndex3D coronalResult{
          volume->width() / 4,
          sagittalResult.y,
          volume->depth() / 4};
      widget.setPositionFromImagePoint(
          qvp::SliceOrientation::Coronal,
          QPointF(static_cast<double>(coronalResult.x) + 0.5,
                  static_cast<double>(coronalResult.z) + 0.5));
      requireSynchronizedState(widget,
                               coronalResult,
                               "coronal image-point mapping");

      const qvp::VoxelIndex3D interactiveAxialResult{
          volume->width() / 5,
          volume->height() / 5,
          coronalResult.z};
      emitNormalizedCrosshair(widget.axialViewer(),
                              interactiveAxialResult.x,
                              interactiveAxialResult.y);
      requireSynchronizedState(widget,
                               interactiveAxialResult,
                               "interactive axial qtvp mapping");

      const qvp::VoxelIndex3D interactiveSagittalResult{
          interactiveAxialResult.x,
          volume->height() / 6,
          volume->depth() / 5};
      emitNormalizedCrosshair(widget.sagittalViewer(),
                              interactiveSagittalResult.y,
                              interactiveSagittalResult.z);
      requireSynchronizedState(widget,
                               interactiveSagittalResult,
                               "interactive sagittal qtvp mapping");

      const qvp::VoxelIndex3D interactiveCoronalResult{
          volume->width() / 6,
          interactiveSagittalResult.y,
          volume->depth() / 6};
      emitNormalizedCrosshair(widget.coronalViewer(),
                              interactiveCoronalResult.x,
                              interactiveCoronalResult.z);
      requireSynchronizedState(widget,
                               interactiveCoronalResult,
                               "interactive coronal qtvp mapping");

      const auto maximum = std::numeric_limits<std::size_t>::max();
      widget.setPositionFromImagePoint(
          qvp::SliceOrientation::Axial,
          QPointF(-100.0, static_cast<double>(maximum)));
      const qvp::VoxelIndex3D clamped{
          0,
          volume->height() - 1,
          interactiveCoronalResult.z};
      requireSynchronizedState(widget, clamped, "clamped image-point mapping");

      const auto physicalPosition = widget.physicalPosition();
      require(physicalPosition.has_value(),
              "MPR physical position is unavailable for the real volume");
      const qvp::PhysicalPoint3D expectedPhysicalPosition =
          qvp::VolumePhysicalCoordinateMapper::voxelToPhysical(*volume, clamped);
      require(physicalPosition->x == expectedPhysicalPosition.x &&
                  physicalPosition->y == expectedPhysicalPosition.y &&
                  physicalPosition->z == expectedPhysicalPosition.z,
              "MPR physical position does not match the public mapper");

      widget.setVoxelPosition(last);
      widget.setPositionFromImagePoint(
          qvp::SliceOrientation::Axial,
          QPointF(std::numeric_limits<double>::quiet_NaN(),
                  std::numeric_limits<double>::infinity()));
      const qvp::VoxelIndex3D nonFiniteResult{0, 0, last.z};
      requireSynchronizedState(widget,
                               nonFiniteResult,
                               "non-finite axial image-point mapping");

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
      require(!widget.axialViewer().crosshairPosition().has_value() &&
                  !widget.sagittalViewer().crosshairPosition().has_value() &&
                  !widget.coronalViewer().crosshairPosition().has_value(),
              "cleared MPR widget retained crosshair state");

      widget.setVolume(volume.get());
      require(widget.hasVolume(),
              "MPR widget did not accept the reassigned real volume");
      requireSynchronizedState(widget, center, "reassigned center");
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
