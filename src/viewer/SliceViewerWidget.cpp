#include "maiw/viewer/SliceViewerWidget.h"

#include "qtviewerpro/core/SliceData.h"
#include "qtviewerpro/core/SliceExtractor.h"
#include "qtviewerpro/core/VolumeData.h"
#include "qtviewerpro/processing/SliceImageConverter.h"
#include "qtviewerpro/render/OpenGLSliceViewer.h"

#include <QImage>
#include <QLineF>
#include <QPainter>
#include <QPen>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <utility>

namespace maiw::viewer
{

SliceViewerWidget::SliceViewerWidget(QWidget* parent)
    : QWidget(parent)
{
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);

  viewer_ = new qvp::OpenGLSliceViewer(this);
  viewer_->setCrosshairVisible(false);
  viewer_->setOrientation(orientation_);
  layout->addWidget(viewer_);
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
    viewer_->setSliceImage(QImage{});
    return;
  }

  QImage presentationImage = sliceImage_;
  if (crosshairPosition_.has_value())
  {
    presentationImage = sliceImage_.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QPainter painter(&presentationImage);
    painter.setRenderHint(QPainter::Antialiasing, false);

    const QPointF position = *crosshairPosition_;
    const QLineF vertical(position.x(), 0.0, position.x(), presentationImage.height());
    const QLineF horizontal(0.0, position.y(), presentationImage.width(), position.y());

    QPen outlinePen(Qt::black);
    outlinePen.setWidth(2);
    painter.setPen(outlinePen);
    painter.drawLine(vertical);
    painter.drawLine(horizontal);

    QPen innerPen(Qt::white);
    innerPen.setWidth(1);
    painter.setPen(innerPen);
    painter.drawLine(vertical);
    painter.drawLine(horizontal);
  }

  viewer_->setSliceImage(presentationImage, sliceSpacingX_, sliceSpacingY_);
}

void SliceViewerWidget::clampSliceIndex() noexcept
{
  const std::size_t count = sliceCount();
  sliceIndex_ = count == 0
                    ? 0
                    : std::min(sliceIndex_, count - 1);
}

} // namespace maiw::viewer
