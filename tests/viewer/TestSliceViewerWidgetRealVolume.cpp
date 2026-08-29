#include "maiw/viewer/SliceViewerWidget.h"
#include "maiw/viewer/VolumeLoadWorkflow.h"

#include "qtviewerpro/core/VolumeData.h"

#include <QApplication>
#include <QEventLoop>
#include <QLayout>
#include <QPointF>
#include <QRectF>
#include <QString>
#include <QThread>
#include <QTimer>
#include <QWidget>

#include <cmath>
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

struct SliceDisplayGeometry
{
  qvp::SliceOrientation orientation;
  float spacingX;
  float spacingY;
  const char* name;
};

QRectF fittedImageRectangle(const QSize& viewportSize,
                            const QSize& imageSize,
                            float spacingX,
                            float spacingY)
{
  const double viewportWidth = static_cast<double>(viewportSize.width());
  const double viewportHeight = static_cast<double>(viewportSize.height());
  const double viewportAspect = viewportWidth / viewportHeight;
  const double imageAspect =
      (static_cast<double>(imageSize.width()) * spacingX) /
      (static_cast<double>(imageSize.height()) * spacingY);

  double displayedWidth = viewportWidth;
  double displayedHeight = viewportHeight;
  if (imageAspect > viewportAspect)
  {
    displayedHeight = viewportWidth / imageAspect;
  }
  else
  {
    displayedWidth = viewportHeight * imageAspect;
  }

  return QRectF((viewportWidth - displayedWidth) / 2.0,
                (viewportHeight - displayedHeight) / 2.0,
                displayedWidth,
                displayedHeight);
}

void requireOverlayInsideImage(const QWidget& overlay,
                               const QRectF& imageRectangle,
                               const std::string& context)
{
  const QRect geometry = overlay.geometry();
  constexpr double roundingTolerance = 1.0;
  require(static_cast<double>(geometry.left()) >=
              imageRectangle.left() - roundingTolerance &&
              static_cast<double>(geometry.top()) >=
                  imageRectangle.top() - roundingTolerance &&
              static_cast<double>(geometry.x() + geometry.width()) <=
                  imageRectangle.right() + roundingTolerance &&
              static_cast<double>(geometry.y() + geometry.height()) <=
                  imageRectangle.bottom() + roundingTolerance,
          context + ": overlay extends outside the fitted image rectangle");
}

void validateCrosshairOverlay(maiw::viewer::SliceViewerWidget& widget,
                              const SliceDisplayGeometry& displayGeometry,
                              const QSize& viewportSize,
                              const std::string& context)
{
  widget.setOrientation(displayGeometry.orientation);
  widget.resize(viewportSize);
  widget.layout()->activate();
  widget.setCrosshairPosition(
      QPointF(static_cast<double>(widget.imageSize().width()) / 2.0,
              static_cast<double>(widget.imageSize().height()) / 2.0));
  QApplication::processEvents();

  auto* const viewer =
      widget.findChild<QWidget*>(QStringLiteral("maiwSliceOpenGLViewer"));
  auto* const verticalLine =
      widget.findChild<QWidget*>(QStringLiteral("maiwVerticalCrosshairOverlay"));
  auto* const horizontalLine =
      widget.findChild<QWidget*>(QStringLiteral("maiwHorizontalCrosshairOverlay"));
  require(viewer != nullptr, context + ": qtvp viewer is missing");
  require(verticalLine != nullptr && horizontalLine != nullptr,
          context + ": crosshair overlays are missing");
  require(verticalLine->parentWidget() == viewer &&
              horizontalLine->parentWidget() == viewer,
          context + ": crosshair overlays are not owned by the qtvp viewer");
  require(verticalLine->testAttribute(Qt::WA_TransparentForMouseEvents) &&
              horizontalLine->testAttribute(Qt::WA_TransparentForMouseEvents),
          context + ": crosshair overlays intercept mouse events");
  require(!verticalLine->isHidden() && !horizontalLine->isHidden(),
          context + ": crosshair overlays are hidden");
  require(verticalLine->width() <= 2 && horizontalLine->height() <= 2,
          context + ": crosshair overlay exceeds two logical pixels");
  require(verticalLine->width() == 1 && horizontalLine->height() == 1,
          context + ": crosshair overlay is not one logical pixel");

  const QRectF imageRectangle =
      fittedImageRectangle(viewer->size(),
                           widget.imageSize(),
                           displayGeometry.spacingX,
                           displayGeometry.spacingY);
  requireOverlayInsideImage(*verticalLine,
                            imageRectangle,
                            context + " vertical line");
  requireOverlayInsideImage(*horizontalLine,
                            imageRectangle,
                            context + " horizontal line");
  constexpr double geometryTolerance = 2.0;
  require(std::abs(static_cast<double>(verticalLine->x()) -
                       imageRectangle.center().x()) <= geometryTolerance &&
              std::abs(static_cast<double>(horizontalLine->y()) -
                       imageRectangle.center().y()) <= geometryTolerance,
          context + ": crosshair overlay is not mapped to the image center");
  require(std::abs(static_cast<double>(verticalLine->height()) -
                       imageRectangle.height()) <= geometryTolerance &&
              std::abs(static_cast<double>(horizontalLine->width()) -
                       imageRectangle.width()) <= geometryTolerance,
          context + ": crosshair lines do not match the fitted image extent");
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

      const SliceDisplayGeometry axialDisplay{
          qvp::SliceOrientation::Axial,
          volume->spacingX(),
          volume->spacingY(),
          "axial"};
      const SliceDisplayGeometry sagittalDisplay{
          qvp::SliceOrientation::Sagittal,
          volume->spacingY(),
          volume->spacingZ(),
          "sagittal"};
      const SliceDisplayGeometry coronalDisplay{
          qvp::SliceOrientation::Coronal,
          volume->spacingX(),
          volume->spacingZ(),
          "coronal"};

      for (const auto& display :
           {axialDisplay, sagittalDisplay, coronalDisplay})
      {
        validateCrosshairOverlay(widget,
                                 display,
                                 QSize(640, 480),
                                 std::string(display.name) + " initial size");
        validateCrosshairOverlay(widget,
                                 display,
                                 QSize(960, 540),
                                 std::string(display.name) + " resized");
      }

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
      auto* const verticalLine = widget.findChild<QWidget*>(
          QStringLiteral("maiwVerticalCrosshairOverlay"));
      auto* const horizontalLine = widget.findChild<QWidget*>(
          QStringLiteral("maiwHorizontalCrosshairOverlay"));
      require(!widget.hasVolume(),
              "widget retained the volume after clear");
      require(widget.sliceCount() == 0,
              "cleared widget retained a slice count");
      require(widget.sliceIndex() == 0,
              "cleared widget retained a slice index");
      require(verticalLine != nullptr && horizontalLine != nullptr &&
                  verticalLine->isHidden() && horizontalLine->isHidden(),
              "cleared widget retained visible crosshair overlays");

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
