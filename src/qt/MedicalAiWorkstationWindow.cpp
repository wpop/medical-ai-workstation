#include "maiw/qt/MedicalAiWorkstationWindow.h"

#include <QSplitter>
#include <QString>

#include <utility>

namespace maiw::qt
{

MedicalAiWorkstationWindow::MedicalAiWorkstationWindow(
    CardiacMriClassificationWorkflow& classificationWorkflow,
    cardiac::CardiacMriDeploymentMetadata::ClassNames classNames,
    QWidget* parent)
    : QMainWindow(parent)
{
  auto* splitter = new QSplitter(Qt::Horizontal, this);

  viewerWorkspace_ = new viewer::ViewerWorkspaceWidget;
  splitter->addWidget(viewerWorkspace_);

  classificationWindow_ = new CardiacMriClassificationWindow(
      classificationWorkflow,
      std::move(classNames));
  splitter->addWidget(classificationWindow_);

  connect(classificationWindow_,
          &CardiacMriClassificationWindow::edVolumePathCommitted,
          this,
          &MedicalAiWorkstationWindow::loadCommittedVolumePath);

  connect(classificationWindow_,
          &CardiacMriClassificationWindow::esVolumePathCommitted,
          this,
          &MedicalAiWorkstationWindow::loadCommittedVolumePath);

  splitter->setStretchFactor(0, 4);
  splitter->setStretchFactor(1, 1);
  splitter->setChildrenCollapsible(false);

  setCentralWidget(splitter);
  setWindowTitle(QStringLiteral("Medical AI Workstation"));
  resize(1600, 900);
}

const viewer::ViewerWorkspaceWidget&
MedicalAiWorkstationWindow::viewerWorkspace() const noexcept
{
  return *viewerWorkspace_;
}

const CardiacMriClassificationWindow&
MedicalAiWorkstationWindow::classificationWindow() const noexcept
{
  return *classificationWindow_;
}

void MedicalAiWorkstationWindow::loadCommittedVolumePath(const QString& path)
{
  if (path.trimmed().isEmpty())
  {
    return;
  }

  viewerWorkspace_->loadVolume(path);
}

} // namespace maiw::qt
