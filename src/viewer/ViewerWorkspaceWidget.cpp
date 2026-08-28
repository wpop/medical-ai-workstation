#include "maiw/viewer/ViewerWorkspaceWidget.h"

#include "qtviewerpro/core/VolumeData.h"

#include <QHBoxLayout>
#include <QSplitter>

#include <utility>

namespace maiw::viewer
{

ViewerWorkspaceWidget::ViewerWorkspaceWidget(QWidget* parent)
    : QWidget(parent)
{
  auto* layout = new QHBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);

  auto* splitter = new QSplitter(Qt::Horizontal, this);
  mprViewer_ = new MprViewerWidget;
  splitter->addWidget(mprViewer_);
  volumeRenderingWidget_ = new VolumeRenderingWidget;
  splitter->addWidget(volumeRenderingWidget_);
  splitter->setStretchFactor(0, 3);
  splitter->setStretchFactor(1, 1);
  layout->addWidget(splitter);
}

ViewerWorkspaceWidget::~ViewerWorkspaceWidget()
{
  clearVolume();
}

void ViewerWorkspaceWidget::setVolume(SharedVolume volume)
{
  if (!volume || !volume->isValid() || volume->isEmpty())
  {
    clearVolume();
    return;
  }

  clearVolume();
  volume_ = std::move(volume);
  mprViewer_->setVolume(volume_.get());
  volumeRenderingWidget_->setVolume(volume_);
}

void ViewerWorkspaceWidget::clearVolume()
{
  mprViewer_->clearVolume();
  volumeRenderingWidget_->clearVolume();
  volume_.reset();
}

bool ViewerWorkspaceWidget::hasVolume() const noexcept
{
  return volume_ != nullptr;
}

const MprViewerWidget& ViewerWorkspaceWidget::mprViewer() const noexcept
{
  return *mprViewer_;
}

const VolumeRenderingWidget&
ViewerWorkspaceWidget::volumeRenderingWidget() const noexcept
{
  return *volumeRenderingWidget_;
}

} // namespace maiw::viewer
