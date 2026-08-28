#include "maiw/viewer/MprViewerWidget.h"

#include "qtviewerpro/core/SliceOrientation.h"
#include "qtviewerpro/core/VolumeData.h"

#include <QHBoxLayout>

#include <algorithm>

namespace maiw::viewer
{

MprViewerWidget::MprViewerWidget(QWidget* parent)
    : QWidget(parent)
{
  auto* layout = new QHBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);

  axialViewer_ = new SliceViewerWidget(this);
  sagittalViewer_ = new SliceViewerWidget(this);
  coronalViewer_ = new SliceViewerWidget(this);

  axialViewer_->setOrientation(qvp::SliceOrientation::Axial);
  sagittalViewer_->setOrientation(qvp::SliceOrientation::Sagittal);
  coronalViewer_->setOrientation(qvp::SliceOrientation::Coronal);

  layout->addWidget(axialViewer_);
  layout->addWidget(sagittalViewer_);
  layout->addWidget(coronalViewer_);
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
  synchronizeSliceIndices();
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
  synchronizeSliceIndices();
}

qvp::VoxelIndex3D MprViewerWidget::voxelPosition() const noexcept
{
  return voxelPosition_;
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

void MprViewerWidget::synchronizeSliceIndices()
{
  if (!hasUsableVolume())
  {
    axialViewer_->setSliceIndex(0);
    sagittalViewer_->setSliceIndex(0);
    coronalViewer_->setSliceIndex(0);
    return;
  }

  axialViewer_->setSliceIndex(voxelPosition_.z);
  sagittalViewer_->setSliceIndex(voxelPosition_.x);
  coronalViewer_->setSliceIndex(voxelPosition_.y);
}

} // namespace maiw::viewer
