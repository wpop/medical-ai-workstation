#include "maiw/viewer/MprViewerWidget.h"

#include "qtviewerpro/core/SliceOrientation.h"
#include "qtviewerpro/core/VolumeData.h"

#include <QGridLayout>

#include <algorithm>
#include <cmath>

namespace
{

std::size_t clampedImageCoordinate(double coordinate, std::size_t count) noexcept
{
  if (count == 0 || !std::isfinite(coordinate) || coordinate <= 0.0)
  {
    return 0;
  }

  const double lastIndex = static_cast<double>(count - 1);
  if (coordinate >= lastIndex)
  {
    return count - 1;
  }

  return static_cast<std::size_t>(coordinate);
}

QPointF pixelCenter(std::size_t x, std::size_t y)
{
  return QPointF(static_cast<double>(x) + 0.5,
                 static_cast<double>(y) + 0.5);
}

} // namespace

namespace maiw::viewer
{

MprViewerWidget::MprViewerWidget(QWidget* parent)
    : QWidget(parent)
{
  auto* layout = new QGridLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);

  axialViewer_ = new SliceViewerWidget(this);
  sagittalViewer_ = new SliceViewerWidget(this);
  coronalViewer_ = new SliceViewerWidget(this);

  axialViewer_->setOrientation(qvp::SliceOrientation::Axial);
  sagittalViewer_->setOrientation(qvp::SliceOrientation::Sagittal);
  coronalViewer_->setOrientation(qvp::SliceOrientation::Coronal);

  connect(axialViewer_,
          &SliceViewerWidget::imagePointChanged,
          this,
          [this](QPointF imagePoint)
          {
            setPositionFromImagePoint(qvp::SliceOrientation::Axial, imagePoint);
          });
  connect(sagittalViewer_,
          &SliceViewerWidget::imagePointChanged,
          this,
          [this](QPointF imagePoint)
          {
            setPositionFromImagePoint(qvp::SliceOrientation::Sagittal, imagePoint);
          });
  connect(coronalViewer_,
          &SliceViewerWidget::imagePointChanged,
          this,
          [this](QPointF imagePoint)
          {
            setPositionFromImagePoint(qvp::SliceOrientation::Coronal, imagePoint);
          });

  layout->addWidget(axialViewer_, 0, 0);
  layout->addWidget(sagittalViewer_, 0, 1);
  layout->addWidget(coronalViewer_, 1, 0, 1, 2);
  layout->setColumnStretch(0, 1);
  layout->setColumnStretch(1, 1);
  layout->setRowStretch(0, 1);
  layout->setRowStretch(1, 1);
}

void MprViewerWidget::setVolume(const qvp::VolumeData* volume)
{
  volume_ = volume;

  if (!hasUsableVolume())
  {
    clearVolume();
    return;
  }

  voxelPosition_ = qvp::VoxelIndex3D{
      volume_->width() / 2,
      volume_->height() / 2,
      volume_->depth() / 2};

  axialViewer_->setVolume(volume_);
  sagittalViewer_->setVolume(volume_);
  coronalViewer_->setVolume(volume_);
  synchronizeViews();
}

void MprViewerWidget::clearVolume()
{
  volume_ = nullptr;
  voxelPosition_ = {};
  axialViewer_->clearVolume();
  sagittalViewer_->clearVolume();
  coronalViewer_->clearVolume();
}

bool MprViewerWidget::hasVolume() const noexcept
{
  return volume_ != nullptr;
}

void MprViewerWidget::setVoxelPosition(qvp::VoxelIndex3D position)
{
  voxelPosition_ = position;
  clampVoxelPosition();
  synchronizeViews();
}

qvp::VoxelIndex3D MprViewerWidget::voxelPosition() const noexcept
{
  return voxelPosition_;
}

void MprViewerWidget::setPositionFromImagePoint(qvp::SliceOrientation orientation,
                                                QPointF imagePoint)
{
  if (!hasUsableVolume())
  {
    return;
  }

  switch (orientation)
  {
  case qvp::SliceOrientation::Axial:
    voxelPosition_.x = clampedImageCoordinate(imagePoint.x(), volume_->width());
    voxelPosition_.y = clampedImageCoordinate(imagePoint.y(), volume_->height());
    break;
  case qvp::SliceOrientation::Sagittal:
    voxelPosition_.y = clampedImageCoordinate(imagePoint.x(), volume_->height());
    voxelPosition_.z = clampedImageCoordinate(imagePoint.y(), volume_->depth());
    break;
  case qvp::SliceOrientation::Coronal:
    voxelPosition_.x = clampedImageCoordinate(imagePoint.x(), volume_->width());
    voxelPosition_.z = clampedImageCoordinate(imagePoint.y(), volume_->depth());
    break;
  }

  synchronizeViews();
}

std::optional<qvp::PhysicalPoint3D> MprViewerWidget::physicalPosition() const
{
  if (!hasUsableVolume())
  {
    return std::nullopt;
  }

  return qvp::VolumePhysicalCoordinateMapper::voxelToPhysical(
      *volume_, voxelPosition_);
}

const SliceViewerWidget& MprViewerWidget::axialViewer() const noexcept
{
  return *axialViewer_;
}

const SliceViewerWidget& MprViewerWidget::sagittalViewer() const noexcept
{
  return *sagittalViewer_;
}

const SliceViewerWidget& MprViewerWidget::coronalViewer() const noexcept
{
  return *coronalViewer_;
}

bool MprViewerWidget::hasUsableVolume() const noexcept
{
  return volume_ != nullptr && volume_->isValid() && !volume_->isEmpty();
}

void MprViewerWidget::clampVoxelPosition() noexcept
{
  if (!hasUsableVolume())
  {
    voxelPosition_ = {};
    return;
  }

  voxelPosition_.x = std::min(voxelPosition_.x, volume_->width() - 1);
  voxelPosition_.y = std::min(voxelPosition_.y, volume_->height() - 1);
  voxelPosition_.z = std::min(voxelPosition_.z, volume_->depth() - 1);
}

void MprViewerWidget::synchronizeViews()
{
  if (!hasUsableVolume())
  {
    axialViewer_->setCrosshairPosition(std::nullopt);
    sagittalViewer_->setCrosshairPosition(std::nullopt);
    coronalViewer_->setCrosshairPosition(std::nullopt);
    axialViewer_->setSliceIndex(0);
    sagittalViewer_->setSliceIndex(0);
    coronalViewer_->setSliceIndex(0);
    return;
  }

  axialViewer_->setSliceIndex(voxelPosition_.z);
  sagittalViewer_->setSliceIndex(voxelPosition_.x);
  coronalViewer_->setSliceIndex(voxelPosition_.y);
  axialViewer_->setCrosshairPosition(
      pixelCenter(voxelPosition_.x, voxelPosition_.y));
  sagittalViewer_->setCrosshairPosition(
      pixelCenter(voxelPosition_.y, voxelPosition_.z));
  coronalViewer_->setCrosshairPosition(
      pixelCenter(voxelPosition_.x, voxelPosition_.z));
}

} // namespace maiw::viewer
