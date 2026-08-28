#include "maiw/viewer/ViewerWorkspaceWidget.h"
#include "maiw/viewer/VolumeLoadWorkflow.h"

#include "qtviewerpro/core/VolumeData.h"

#include <QApplication>
#include <QEventLoop>
#include <QString>
#include <QThread>
#include <QTimer>

#include <cstddef>
#include <iostream>
#include <memory>
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

maiw::viewer::SharedVolume loadRealVolume(QApplication& application)
{
  maiw::viewer::VolumeLoadWorkflow workflow;
  QEventLoop eventLoop;
  QTimer watchdog;
  watchdog.setSingleShot(true);
  watchdog.setInterval(kLoadTimeoutMilliseconds);

  std::size_t startedCount = 0;
  std::size_t successCount = 0;
  std::size_t failureCount = 0;
  bool timedOut = false;
  bool deliveredOnMainThread = false;
  QString failureMessage;
  maiw::viewer::SharedVolume volume;

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
      &eventLoop,
      [&application,
       &eventLoop,
       &watchdog,
       &successCount,
       &deliveredOnMainThread,
       &volume](maiw::viewer::SharedVolume loadedVolume)
      {
        ++successCount;
        deliveredOnMainThread =
            QThread::currentThread() == application.thread();
        volume = std::move(loadedVolume);
        watchdog.stop();
        eventLoop.quit();
      },
      Qt::QueuedConnection);
  QObject::connect(
      &workflow,
      &maiw::viewer::VolumeLoadWorkflow::loadingFailed,
      &eventLoop,
      [&eventLoop,
       &watchdog,
       &failureCount,
       &failureMessage](const QString& message)
      {
        ++failureCount;
        failureMessage = message;
        watchdog.stop();
        eventLoop.quit();
      });
  QObject::connect(
      &watchdog,
      &QTimer::timeout,
      &eventLoop,
      [&eventLoop, &timedOut]()
      {
        timedOut = true;
        eventLoop.quit();
      });

  watchdog.start();
  workflow.startLoading(QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH));
  require(workflow.isRunning(),
          "real ACDC workspace load did not enter the running state");
  eventLoop.exec();

  require(!timedOut, "timed out loading the real ACDC workspace volume");
  require(startedCount == 1,
          "real ACDC workspace load did not start exactly once");
  require(successCount == 1,
          "real ACDC workspace load did not succeed exactly once");
  require(failureCount == 0,
          "real ACDC workspace load failed: " + failureMessage.toStdString());
  require(deliveredOnMainThread,
          "real ACDC workspace load was not delivered on the main Qt thread");
  require(!workflow.isRunning(),
          "workflow remained running after real ACDC workspace load");
  require(volume != nullptr,
          "real ACDC workspace load returned a null shared volume");
  require(volume->isValid() && !volume->isEmpty(),
          "real ACDC workspace volume is invalid or empty");

  return volume;
}

void requireCenterState(const maiw::viewer::ViewerWorkspaceWidget& workspace,
                        const qvp::VolumeData& volume,
                        const std::string& context)
{
  const qvp::VoxelIndex3D center{
      volume.width() / 2,
      volume.height() / 2,
      volume.depth() / 2};
  const qvp::VoxelIndex3D actual = workspace.mprViewer().voxelPosition();

  require(actual.x == center.x &&
              actual.y == center.y &&
              actual.z == center.z,
          context + ": MPR position is not centered");
  require(workspace.mprViewer().axialViewer().sliceCount() == volume.depth(),
          context + ": axial count does not match volume depth");
  require(workspace.mprViewer().sagittalViewer().sliceCount() == volume.width(),
          context + ": sagittal count does not match volume width");
  require(workspace.mprViewer().coronalViewer().sliceCount() == volume.height(),
          context + ": coronal count does not match volume height");
  require(workspace.mprViewer().axialViewer().sliceIndex() == center.z,
          context + ": axial slice is not synchronized to center Z");
  require(workspace.mprViewer().sagittalViewer().sliceIndex() == center.x,
          context + ": sagittal slice is not synchronized to center X");
  require(workspace.mprViewer().coronalViewer().sliceIndex() == center.y,
          context + ": coronal slice is not synchronized to center Y");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);
    maiw::viewer::SharedVolume callerVolume = loadRealVolume(application);
    maiw::viewer::SharedVolume retainedExternalVolume = callerVolume;
    const qvp::VolumeData* const volumeIdentity = callerVolume.get();
    std::weak_ptr<const qvp::VolumeData> volumeObserver = callerVolume;

    require(callerVolume.use_count() == 2,
            "unexpected shared ownership before workspace assignment");

    {
      maiw::viewer::ViewerWorkspaceWidget workspace;
      const auto* const mprIdentity = &workspace.mprViewer();
      const auto* const volumeRenderingIdentity =
          &workspace.volumeRenderingWidget();

      workspace.setVolume(callerVolume);
      require(workspace.hasVolume(),
              "workspace did not accept the real ACDC volume");
      require(workspace.mprViewer().hasVolume(),
              "MPR child did not receive the real ACDC volume");
      require(workspace.volumeRenderingWidget().hasVolume(),
              "3D child did not receive the real ACDC volume");
      require(callerVolume.use_count() == 5,
              "workspace children did not share the existing volume instance");
      requireCenterState(workspace, *callerVolume, "initial assignment");

      callerVolume.reset();
      require(workspace.hasVolume() &&
                  workspace.mprViewer().hasVolume() &&
                  workspace.volumeRenderingWidget().hasVolume(),
              "workspace lost the volume after the caller released its copy");
      require(!volumeObserver.expired() &&
                  volumeObserver.lock().get() == volumeIdentity,
              "workspace did not preserve the assigned volume identity");
      require(retainedExternalVolume.use_count() == 4,
              "caller release produced unexpected workspace ownership");
      requireCenterState(workspace,
                         *retainedExternalVolume,
                         "after caller release");

      workspace.clearVolume();
      require(!workspace.hasVolume(),
              "workspace retained the volume after clear");
      require(!workspace.mprViewer().hasVolume(),
              "MPR child retained the volume after clear");
      require(!workspace.volumeRenderingWidget().hasVolume(),
              "3D child retained the volume after clear");
      require(retainedExternalVolume.use_count() == 1,
              "workspace clear retained shared volume ownership");
      require(retainedExternalVolume.get() == volumeIdentity &&
                  retainedExternalVolume->isValid() &&
                  !retainedExternalVolume->isEmpty(),
              "external volume became invalid after workspace clear");

      workspace.setVolume(retainedExternalVolume);
      require(workspace.hasVolume() &&
                  workspace.mprViewer().hasVolume() &&
                  workspace.volumeRenderingWidget().hasVolume(),
              "workspace did not accept a reassigned real volume");
      require(retainedExternalVolume.use_count() == 4,
              "workspace reassignment did not share the existing volume");
      require(&workspace.mprViewer() == mprIdentity,
              "MPR child changed during workspace assignment lifecycle");
      require(&workspace.volumeRenderingWidget() == volumeRenderingIdentity,
              "3D child changed during workspace assignment lifecycle");
      requireCenterState(workspace,
                         *retainedExternalVolume,
                         "reassignment");
    }

    require(retainedExternalVolume != nullptr &&
                retainedExternalVolume.get() == volumeIdentity &&
                retainedExternalVolume->isValid() &&
                !retainedExternalVolume->isEmpty(),
            "external volume became invalid after workspace destruction");
    require(retainedExternalVolume.use_count() == 1,
            "workspace destruction retained shared volume ownership");

    retainedExternalVolume.reset();
    require(volumeObserver.expired(),
            "real ACDC volume remained owned after all copies were released");

    std::cout << "Real-volume viewer workspace widget test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Real-volume viewer workspace widget test failed: "
              << error.what() << '\n';
    return 1;
  }
}
