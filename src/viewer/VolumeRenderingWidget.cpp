#include "maiw/viewer/VolumeRenderingWidget.h"

#include "qtviewerpro/core/VolumeData.h"
#include "qtviewerpro/render/OpenGLVolumeRendererWidget.h"

#include <QVBoxLayout>

#include <utility>

namespace maiw::viewer
{

VolumeRenderingWidget::VolumeRenderingWidget(QWidget* parent)
    : QWidget(parent)
{
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);

  renderer_ = new qvp::OpenGLVolumeRendererWidget(this);
  layout->addWidget(renderer_);

  connect(renderer_,
          &qvp::OpenGLVolumeRendererWidget::volumeTextureUploaded,
          this,
          &VolumeRenderingWidget::volumeUploadSucceeded);
  connect(renderer_,
          &qvp::OpenGLVolumeRendererWidget::volumeTextureUploadFailed,
          this,
          &VolumeRenderingWidget::volumeUploadFailed);
}

void VolumeRenderingWidget::setVolume(SharedVolume volume)
{
  if (!volume || !volume->isValid())
  {
    clearVolume();
    return;
  }

  volume_ = std::move(volume);
  renderer_->setVolume(volume_);
}

void VolumeRenderingWidget::clearVolume()
{
  renderer_->setVolume(nullptr);
  volume_.reset();
}

bool VolumeRenderingWidget::hasVolume() const noexcept
{
  return volume_ != nullptr;
}

void VolumeRenderingWidget::setRenderPreset(qvp::VolumeRenderPreset preset)
{
  renderer_->setRenderPreset(preset);
}

void VolumeRenderingWidget::setGlobalOpacity(float opacity)
{
  renderer_->setGlobalOpacity(opacity);
}

void VolumeRenderingWidget::setManualIntensityRange(float minimum, float maximum)
{
  renderer_->setManualIntensityRange(minimum, maximum);
}

qvp::VolumeTransferFunctionState VolumeRenderingWidget::transferFunctionState() const
{
  return renderer_->transferFunctionState();
}

void VolumeRenderingWidget::resetView()
{
  renderer_->resetView();
}

} // namespace maiw::viewer
