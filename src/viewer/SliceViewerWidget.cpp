#include "maiw/viewer/SliceViewerWidget.h"

#include "qtviewerpro/core/SliceData.h"
#include "qtviewerpro/core/SliceExtractor.h"
#include "qtviewerpro/core/VolumeData.h"
#include "qtviewerpro/processing/SliceImageConverter.h"
#include "qtviewerpro/render/OpenGLSliceViewer.h"

#include <QImage>
#include <QVBoxLayout>

#include <algorithm>

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

void SliceViewerWidget::refresh()
{
  viewer_->setOrientation(orientation_);

  if (sliceCount() == 0)
  {
    viewer_->setSliceImage(QImage{});
    return;
  }

  const qvp::SliceData slice =
      qvp::SliceExtractor::extract(*volume_, orientation_, sliceIndex_);
  const float minimum = volume_->intensityMinimum();
  const float maximum = volume_->intensityMaximum();
  const float intensityRange = maximum - minimum;
  const float level = minimum + (intensityRange / 2.0F);
  const float window = intensityRange > 0.0F ? intensityRange : 1.0F;
  const QImage image =
      qvp::SliceImageConverter::toGrayscaleImage(slice, window, level);

  viewer_->setSliceImage(image, slice.spacingX(), slice.spacingY());
}

void SliceViewerWidget::clampSliceIndex() noexcept
{
  const std::size_t count = sliceCount();
  sliceIndex_ = count == 0
                    ? 0
                    : std::min(sliceIndex_, count - 1);
}

} // namespace maiw::viewer
