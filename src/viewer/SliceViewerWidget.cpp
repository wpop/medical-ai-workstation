#include "maiw/viewer/SliceViewerWidget.h"

#include "qtviewerpro/core/SliceData.h"
#include "qtviewerpro/core/SliceExtractor.h"
#include "qtviewerpro/core/VolumeData.h"
#include "qtviewerpro/processing/SliceImageConverter.h"
#include "qtviewerpro/render/OpenGLSliceViewer.h"

#include <QEvent>
#include <QImage>
#include <QPalette>
#include <QRectF>
#include <QSignalBlocker>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <utility>

namespace maiw::viewer
{

namespace
{

std::optional<QPointF> normalizedToPixelCenter(
    QPointF normalizedPosition,
    const QSize& imageSize) noexcept
{
  if (imageSize.width() <= 0 || imageSize.height() <= 0 ||
      !std::isfinite(normalizedPosition.x()) ||
      !std::isfinite(normalizedPosition.y()))
  {
    return std::nullopt;
  }

  const double normalizedX = std::clamp(normalizedPosition.x(), -1.0, 1.0);
  const double normalizedY = std::clamp(normalizedPosition.y(), -1.0, 1.0);
  const double pixelX =
      ((normalizedX + 1.0) * 0.5) *
          static_cast<double>(imageSize.width() - 1) +
      0.5;
  const double pixelY =
      ((1.0 - normalizedY) * 0.5) *
          static_cast<double>(imageSize.height() - 1) +
      0.5;
  return QPointF(pixelX, pixelY);
}

QWidget* createCrosshairLine(QWidget* parent, const QString& objectName)
{
  auto* line = new QWidget(parent);
  line->setObjectName(objectName);
  line->setAttribute(Qt::WA_TransparentForMouseEvents);
  line->setFocusPolicy(Qt::NoFocus);

  QPalette palette = line->palette();
  palette.setColor(QPalette::Window, Qt::white);
  line->setPalette(palette);
  line->setAutoFillBackground(true);
  line->hide();
  return line;
}

double pixelCenterRatio(double coordinate, int pixelCount) noexcept
{
  if (pixelCount <= 1)
  {
    return 0.5;
  }

  return (coordinate - 0.5) / static_cast<double>(pixelCount - 1);
}

} // namespace

SliceViewerWidget::SliceViewerWidget(QWidget* parent)
    : QWidget(parent)
{
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);

  viewer_ = new qvp::OpenGLSliceViewer(this);
  viewer_->setObjectName(QStringLiteral("maiwSliceOpenGLViewer"));
  viewer_->setCrosshairVisible(false);
  viewer_->setOrientation(orientation_);
  layout->addWidget(viewer_);

  verticalCrosshairLine_ = createCrosshairLine(
      viewer_, QStringLiteral("maiwVerticalCrosshairOverlay"));
  horizontalCrosshairLine_ = createCrosshairLine(
      viewer_, QStringLiteral("maiwHorizontalCrosshairOverlay"));
  viewer_->installEventFilter(this);

  connect(viewer_,
          &qvp::OpenGLSliceViewer::crosshairPositionChanged,
          this,
          &SliceViewerWidget::handleViewerCrosshairPositionChanged);
}

bool SliceViewerWidget::eventFilter(QObject* watched, QEvent* event)
{
  if (watched == viewer_)
  {
    switch (event->type())
    {
    case QEvent::Resize:
    case QEvent::Show:
      updateCrosshairOverlay();
      break;
    case QEvent::MouseMove:
    case QEvent::Wheel:
      QTimer::singleShot(0, this, &SliceViewerWidget::updateCrosshairOverlay);
      break;
    default:
      break;
    }
  }

  return QWidget::eventFilter(watched, event);
}

void SliceViewerWidget::setVolume(const qvp::VolumeData* volume)
{
  volume_ = volume;
  clampSliceIndex();
  refresh();
}

void SliceViewerWidget::clearVolume()
{
  setVolume(nullptr);
}

bool SliceViewerWidget::hasVolume() const noexcept
{
  return volume_ != nullptr;
}

void SliceViewerWidget::setOrientation(qvp::SliceOrientation orientation)
{
  orientation_ = orientation;
  clampSliceIndex();
  refresh();
}

qvp::SliceOrientation SliceViewerWidget::orientation() const noexcept
{
  return orientation_;
}

void SliceViewerWidget::setSliceIndex(std::size_t sliceIndex)
{
  sliceIndex_ = sliceIndex;
  clampSliceIndex();
  refresh();
}

std::size_t SliceViewerWidget::sliceIndex() const noexcept
{
  return sliceIndex_;
}

std::size_t SliceViewerWidget::sliceCount() const noexcept
{
  if (volume_ == nullptr || !volume_->isValid() || volume_->isEmpty())
  {
    return 0;
  }

  switch (orientation_)
  {
  case qvp::SliceOrientation::Axial:
    return volume_->depth();
  case qvp::SliceOrientation::Coronal:
    return volume_->height();
  case qvp::SliceOrientation::Sagittal:
    return volume_->width();
  }

  return 0;
}

void SliceViewerWidget::setCrosshairPosition(std::optional<QPointF> position)
{
  crosshairPosition_ = std::move(position);
  clampCrosshairPosition();
  presentCurrentImage();
}

std::optional<QPointF> SliceViewerWidget::crosshairPosition() const noexcept
{
  return crosshairPosition_;
}

QSize SliceViewerWidget::imageSize() const noexcept
{
  return sliceImage_.size();
}

void SliceViewerWidget::refresh()
{
  viewer_->setOrientation(orientation_);

  if (sliceCount() == 0)
  {
    sliceImage_ = {};
    sliceSpacingX_ = 1.0F;
    sliceSpacingY_ = 1.0F;
    crosshairPosition_.reset();
    presentCurrentImage();
    return;
  }

  const qvp::SliceData slice =
      qvp::SliceExtractor::extract(*volume_, orientation_, sliceIndex_);
  const float minimum = volume_->intensityMinimum();
  const float maximum = volume_->intensityMaximum();
  const float intensityRange = maximum - minimum;
  const float level = minimum + (intensityRange / 2.0F);
  const float window = intensityRange > 0.0F ? intensityRange : 1.0F;
  sliceImage_ = qvp::SliceImageConverter::toGrayscaleImage(slice, window, level);
  sliceSpacingX_ = slice.spacingX();
  sliceSpacingY_ = slice.spacingY();
  clampCrosshairPosition();
  presentCurrentImage();
}

void SliceViewerWidget::handleViewerCrosshairPositionChanged(
    QPointF normalizedPosition)
{
  const auto imagePoint =
      normalizedToPixelCenter(normalizedPosition, sliceImage_.size());
  if (imagePoint.has_value())
  {
    emit imagePointChanged(*imagePoint);
  }
}

void SliceViewerWidget::clampCrosshairPosition() noexcept
{
  if (!crosshairPosition_.has_value() || sliceImage_.isNull() ||
      !std::isfinite(crosshairPosition_->x()) ||
      !std::isfinite(crosshairPosition_->y()))
  {
    crosshairPosition_.reset();
    return;
  }

  const double minimumPixelCenter = 0.5;
  const double maximumX = static_cast<double>(sliceImage_.width()) - minimumPixelCenter;
  const double maximumY = static_cast<double>(sliceImage_.height()) - minimumPixelCenter;
  crosshairPosition_->setX(
      std::clamp(crosshairPosition_->x(), minimumPixelCenter, maximumX));
  crosshairPosition_->setY(
      std::clamp(crosshairPosition_->y(), minimumPixelCenter, maximumY));
}

void SliceViewerWidget::presentCurrentImage()
{
  if (sliceImage_.isNull())
  {
    const QSignalBlocker blocker(viewer_);
    viewer_->setSliceImage(QImage{});
    updateCrosshairOverlay();
    return;
  }

  const QSignalBlocker blocker(viewer_);
  viewer_->setSliceImage(sliceImage_, sliceSpacingX_, sliceSpacingY_);
  updateCrosshairOverlay();
}

void SliceViewerWidget::updateCrosshairOverlay()
{
  if (!crosshairPosition_.has_value() || sliceImage_.isNull() ||
      viewer_->width() <= 0 || viewer_->height() <= 0)
  {
    verticalCrosshairLine_->hide();
    horizontalCrosshairLine_->hide();
    return;
  }

  const double viewportWidth = static_cast<double>(viewer_->width());
  const double viewportHeight = static_cast<double>(viewer_->height());
  const double widgetAspect = viewportWidth / viewportHeight;
  const double safeSpacingX = sliceSpacingX_ > 0.0F ? sliceSpacingX_ : 1.0F;
  const double safeSpacingY = sliceSpacingY_ > 0.0F ? sliceSpacingY_ : 1.0F;
  const double imageAspect =
      (static_cast<double>(sliceImage_.width()) *
       safeSpacingX) /
      (static_cast<double>(sliceImage_.height()) *
       safeSpacingY);

  double displayedWidth = viewportWidth;
  double displayedHeight = viewportHeight;
  if (imageAspect > widgetAspect)
  {
    displayedHeight = viewportWidth / imageAspect;
  }
  else
  {
    displayedWidth = viewportHeight * imageAspect;
  }

  const double zoomFactor = static_cast<double>(viewer_->zoomFactor());
  displayedWidth *= zoomFactor;
  displayedHeight *= zoomFactor;

  const QPointF panOffset = viewer_->panOffset();
  const QPointF imageCenter(
      ((panOffset.x() + 1.0) * viewportWidth) / 2.0,
      ((1.0 - panOffset.y()) * viewportHeight) / 2.0);
  const QRectF imageRectangle(
      imageCenter.x() - (displayedWidth / 2.0),
      imageCenter.y() - (displayedHeight / 2.0),
      displayedWidth,
      displayedHeight);
  const QRectF visibleImageRectangle =
      imageRectangle.intersected(QRectF(viewer_->rect()));

  const int left = static_cast<int>(std::ceil(visibleImageRectangle.left()));
  const int top = static_cast<int>(std::ceil(visibleImageRectangle.top()));
  const int right = static_cast<int>(std::floor(visibleImageRectangle.right()));
  const int bottom = static_cast<int>(std::floor(visibleImageRectangle.bottom()));
  if (right <= left || bottom <= top)
  {
    verticalCrosshairLine_->hide();
    horizontalCrosshairLine_->hide();
    return;
  }

  const double crosshairX =
      imageRectangle.left() +
      (pixelCenterRatio(crosshairPosition_->x(), sliceImage_.width()) *
       imageRectangle.width());
  const double crosshairY =
      imageRectangle.top() +
      (pixelCenterRatio(crosshairPosition_->y(), sliceImage_.height()) *
       imageRectangle.height());

  if (crosshairX >= visibleImageRectangle.left() &&
      crosshairX <= visibleImageRectangle.right())
  {
    const int x = std::clamp(static_cast<int>(std::lround(crosshairX)),
                             left,
                             right - 1);
    verticalCrosshairLine_->setGeometry(x, top, 1, bottom - top);
    verticalCrosshairLine_->show();
    verticalCrosshairLine_->raise();
  }
  else
  {
    verticalCrosshairLine_->hide();
  }

  if (crosshairY >= visibleImageRectangle.top() &&
      crosshairY <= visibleImageRectangle.bottom())
  {
    const int y = std::clamp(static_cast<int>(std::lround(crosshairY)),
                             top,
                             bottom - 1);
    horizontalCrosshairLine_->setGeometry(left, y, right - left, 1);
    horizontalCrosshairLine_->show();
    horizontalCrosshairLine_->raise();
  }
  else
  {
    horizontalCrosshairLine_->hide();
  }
}

void SliceViewerWidget::clampSliceIndex() noexcept
{
  const std::size_t count = sliceCount();
  sliceIndex_ = count == 0
                    ? 0
                    : std::min(sliceIndex_, count - 1);
}

} // namespace maiw::viewer
