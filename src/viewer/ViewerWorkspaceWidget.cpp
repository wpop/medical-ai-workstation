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
  loadWorkflow_ = new VolumeLoadWorkflow(this);

  connect(loadWorkflow_,
          &VolumeLoadWorkflow::loadingStarted,
          this,
          &ViewerWorkspaceWidget::volumeLoadingStarted);
  connect(loadWorkflow_,
          &VolumeLoadWorkflow::loadingSucceeded,
          this,
          &ViewerWorkspaceWidget::handleVolumeLoaded);
  connect(loadWorkflow_,
          &VolumeLoadWorkflow::loadingFailed,
          this,
          &ViewerWorkspaceWidget::volumeLoadingFailed);

  auto* layout = new QHBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);

  auto* splitter = new QSplitter(Qt::Horizontal, this);
  mprViewer_ = new MprViewerWidget;
  splitter->addWidget(mprViewer_);
  volumeRenderingWidget_ = new VolumeRenderingWidget;
  splitter->addWidget(volumeRenderingWidget_);
  splitter->setStretchFactor(0, 2);
  splitter->setStretchFactor(1, 1);
  splitter->setChildrenCollapsible(false);

  const int initialMprWidth = width() * 2 / 3;
  splitter->setSizes({initialMprWidth, width() - initialMprWidth});
  layout->addWidget(splitter);
}

ViewerWorkspaceWidget::~ViewerWorkspaceWidget()
{
  clearVolume();
}

void ViewerWorkspaceWidget::loadVolume(const QString& path)
{
  loadWorkflow_->startLoading(path);
}

bool ViewerWorkspaceWidget::isLoading() const noexcept
{
  return loadWorkflow_->isRunning();
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

std::weak_ptr<const qvp::VolumeData>
ViewerWorkspaceWidget::volumeObserver() const noexcept
{
  return volume_;
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

void ViewerWorkspaceWidget::handleVolumeLoaded(SharedVolume volume)
{
  if (!volume || !volume->isValid() || volume->isEmpty())
  {
    emit volumeLoadingFailed(
        QStringLiteral("The loaded medical volume is invalid or empty."));
    return;
  }

  setVolume(std::move(volume));
  emit volumeLoadingSucceeded();
}

} // namespace maiw::viewer
