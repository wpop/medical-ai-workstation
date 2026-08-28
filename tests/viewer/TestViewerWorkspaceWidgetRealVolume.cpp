#include "maiw/viewer/ViewerWorkspaceWidget.h"

#include "qtviewerpro/core/VolumeData.h"

#include <QApplication>
#include <QDir>
#include <QEventLoop>
#include <QFileInfo>
#include <QString>
#include <QThread>
#include <QTimer>
#include <QUuid>

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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

bool hasSameOwner(const std::weak_ptr<const qvp::VolumeData>& left,
                  const std::weak_ptr<const qvp::VolumeData>& right) noexcept
{
  return !left.owner_before(right) && !right.owner_before(left);
}

struct LoadingSignals
{
  std::size_t startedCount = 0;
  std::size_t successCount = 0;
  std::size_t failureCount = 0;
  bool allCompletionsOnMainThread = true;
  bool loadingAtLastCompletion = true;
  bool hadVolumeAtLastStart = false;
  QString lastFailureMessage;
};

struct LoadAttempt
{
  bool runningAfterRequest = false;
  bool timedOut = false;
};

LoadAttempt runLoadAttempt(maiw::viewer::ViewerWorkspaceWidget& workspace,
                           const QString& path,
                           const LoadingSignals& signalState)
{
  QEventLoop eventLoop;
  QTimer watchdog;
  watchdog.setSingleShot(true);
  watchdog.setInterval(kLoadTimeoutMilliseconds);

  LoadAttempt attempt;
  const std::size_t terminalCountBefore =
      signalState.successCount + signalState.failureCount;

  QObject::connect(
      &workspace,
      &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingSucceeded,
      &eventLoop,
      &QEventLoop::quit);
  QObject::connect(
      &workspace,
      &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingFailed,
      &eventLoop,
      [&eventLoop](const QString&)
      {
        eventLoop.quit();
      });
  QObject::connect(
      &watchdog,
      &QTimer::timeout,
      &eventLoop,
      [&attempt, &eventLoop]()
      {
        attempt.timedOut = true;
        eventLoop.quit();
      });

  watchdog.start();
  workspace.loadVolume(path);
  attempt.runningAfterRequest = workspace.isLoading();

  if (signalState.successCount + signalState.failureCount == terminalCountBefore)
  {
    eventLoop.exec();
  }

  watchdog.stop();
  return attempt;
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

void requireAssignedState(const maiw::viewer::ViewerWorkspaceWidget& workspace,
                          const std::string& context)
{
  require(workspace.hasVolume(), context + ": workspace has no volume");
  require(workspace.mprViewer().hasVolume(),
          context + ": MPR child has no volume");
  require(workspace.volumeRenderingWidget().hasVolume(),
          context + ": 3D child has no volume");
}

void requireEmptyState(const maiw::viewer::ViewerWorkspaceWidget& workspace,
                       const std::string& context)
{
  require(!workspace.hasVolume(), context + ": workspace has a volume");
  require(!workspace.mprViewer().hasVolume(),
          context + ": MPR child has a volume");
  require(!workspace.volumeRenderingWidget().hasVolume(),
          context + ": 3D child has a volume");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);
    LoadingSignals signalState;
    std::weak_ptr<const qvp::VolumeData> observerAfterDestruction;

    {
      maiw::viewer::ViewerWorkspaceWidget workspace;
      requireEmptyState(workspace, "initial state");
      require(!workspace.isLoading(),
              "workspace is unexpectedly loading in its initial state");
      require(workspace.volumeObserver().expired(),
              "empty workspace exposes a live volume observer");

      QObject::connect(
          &workspace,
          &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingStarted,
          &application,
          [&signalState, &workspace]()
          {
            ++signalState.startedCount;
            signalState.hadVolumeAtLastStart = workspace.hasVolume();
          });
      QObject::connect(
          &workspace,
          &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingSucceeded,
          &application,
          [&application, &signalState, &workspace]()
          {
            ++signalState.successCount;
            signalState.allCompletionsOnMainThread =
                signalState.allCompletionsOnMainThread &&
                QThread::currentThread() == application.thread();
            signalState.loadingAtLastCompletion = workspace.isLoading();
          });
      QObject::connect(
          &workspace,
          &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingFailed,
          &application,
          [&application,
           &signalState,
           &workspace](const QString& message)
          {
            ++signalState.failureCount;
            signalState.allCompletionsOnMainThread =
                signalState.allCompletionsOnMainThread &&
                QThread::currentThread() == application.thread();
            signalState.loadingAtLastCompletion = workspace.isLoading();
            signalState.lastFailureMessage = message;
          });

      const QString realPath =
          QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH);
      const LoadAttempt firstAttempt =
          runLoadAttempt(workspace, realPath, signalState);
      require(!firstAttempt.timedOut,
              "initial real ACDC workspace load timed out");
      require(firstAttempt.runningAfterRequest,
              "initial real ACDC load did not enter the running state");
      require(signalState.startedCount == 1,
              "initial real ACDC load did not start exactly once");
      require(!signalState.hadVolumeAtLastStart,
              "initial real ACDC load started with an unexpected volume");
      require(signalState.successCount == 1,
              "initial real ACDC load did not succeed exactly once");
      require(signalState.failureCount == 0,
              "initial real ACDC load unexpectedly failed: " +
                  signalState.lastFailureMessage.toStdString());
      require(signalState.allCompletionsOnMainThread,
              "initial real ACDC load was not delivered on the main Qt thread");
      require(!signalState.loadingAtLastCompletion && !workspace.isLoading(),
              "workspace remained loading when initial success was delivered");
      requireAssignedState(workspace, "initial successful load");

      std::weak_ptr<const qvp::VolumeData> firstObserver =
          workspace.volumeObserver();
      {
        const auto firstVolume = firstObserver.lock();
        require(firstVolume != nullptr,
                "initial load did not expose the canonical volume observer");
        require(firstVolume->isValid() && !firstVolume->isEmpty(),
                "initial canonical volume is invalid or empty");
        require(firstVolume.use_count() == 4,
                "initial load retained unexpected shared volume ownership");
        requireCenterState(workspace, *firstVolume, "initial successful load");
      }
      require(!firstObserver.expired(),
              "workspace lost ownership after the observer released its lock");

      workspace.clearVolume();
      requireEmptyState(workspace, "cleared state");
      require(firstObserver.expired(),
              "clear retained ownership of the initial loaded volume");

      const LoadAttempt reloadAttempt =
          runLoadAttempt(workspace, realPath, signalState);
      require(!reloadAttempt.timedOut,
              "real ACDC workspace reload timed out");
      require(reloadAttempt.runningAfterRequest,
              "real ACDC reload did not enter the running state");
      require(signalState.startedCount == 2,
              "real ACDC reload did not start exactly once");
      require(!signalState.hadVolumeAtLastStart,
              "reload after clear started with an unexpected volume");
      require(signalState.successCount == 2,
              "real ACDC reload did not succeed exactly once");
      require(signalState.failureCount == 0,
              "real ACDC reload unexpectedly failed: " +
                  signalState.lastFailureMessage.toStdString());
      require(signalState.allCompletionsOnMainThread,
              "real ACDC reload was not delivered on the main Qt thread");
      require(!signalState.loadingAtLastCompletion && !workspace.isLoading(),
              "workspace remained loading when reload success was delivered");
      requireAssignedState(workspace, "successful reload");

      const std::weak_ptr<const qvp::VolumeData> reloadedObserver =
          workspace.volumeObserver();
      require(!hasSameOwner(firstObserver, reloadedObserver),
              "reload reused the released initial ownership control block");

      const LoadAttempt replacementAttempt =
          runLoadAttempt(workspace, realPath, signalState);
      require(!replacementAttempt.timedOut,
              "successful real ACDC replacement timed out");
      require(replacementAttempt.runningAfterRequest,
              "successful replacement did not enter the running state");
      require(signalState.startedCount == 3,
              "successful replacement did not start exactly once");
      require(signalState.hadVolumeAtLastStart,
              "successful replacement cleared the displayed volume at start");
      require(signalState.successCount == 3,
              "successful replacement did not succeed exactly once");
      require(signalState.failureCount == 0,
              "successful replacement unexpectedly failed: " +
                  signalState.lastFailureMessage.toStdString());
      require(signalState.allCompletionsOnMainThread,
              "successful replacement was not delivered on the main thread");
      require(!signalState.loadingAtLastCompletion && !workspace.isLoading(),
              "workspace remained loading when replacement succeeded");
      requireAssignedState(workspace, "successful replacement");
      require(reloadedObserver.expired(),
              "successful replacement retained the previous volume");

      const std::weak_ptr<const qvp::VolumeData> replacementObserver =
          workspace.volumeObserver();
      require(!hasSameOwner(reloadedObserver, replacementObserver),
              "successful replacement reused the prior ownership control block");

      qvp::VoxelIndex3D positionBeforeFailure;
      const qvp::VolumeData* identityBeforeFailure = nullptr;
      {
        const auto replacementVolume = replacementObserver.lock();
        require(replacementVolume != nullptr,
                "replacement did not expose the canonical volume observer");
        require(replacementVolume.use_count() == 4,
                "replacement retained unexpected shared volume ownership");
        requireCenterState(workspace,
                           *replacementVolume,
                           "successful replacement");
        positionBeforeFailure = workspace.mprViewer().voxelPosition();
        identityBeforeFailure = replacementVolume.get();
      }

      const QString invalidPath =
          QDir::temp().filePath(
              QStringLiteral("maiw-missing-workspace-%1.nii.gz")
                  .arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
      require(!QFileInfo::exists(invalidPath),
              "generated invalid replacement path unexpectedly exists");

      const LoadAttempt failedAttempt =
          runLoadAttempt(workspace, invalidPath, signalState);
      require(!failedAttempt.timedOut,
              "invalid replacement load timed out");
      require(failedAttempt.runningAfterRequest,
              "invalid replacement did not enter the running state");
      require(signalState.startedCount == 4,
              "invalid replacement did not start exactly once");
      require(signalState.hadVolumeAtLastStart,
              "invalid replacement cleared the displayed volume at start");
      require(signalState.successCount == 3,
              "invalid replacement unexpectedly reported success");
      require(signalState.failureCount == 1,
              "invalid replacement did not fail exactly once");
      require(!signalState.lastFailureMessage.isEmpty(),
              "invalid replacement did not provide a diagnostic");
      require(signalState.allCompletionsOnMainThread,
              "invalid replacement failure was not delivered on the main thread");
      require(!signalState.loadingAtLastCompletion && !workspace.isLoading(),
              "workspace remained loading when replacement failure was delivered");
      requireAssignedState(workspace, "failed replacement state");

      const std::weak_ptr<const qvp::VolumeData> preservedObserver =
          workspace.volumeObserver();
      require(hasSameOwner(replacementObserver, preservedObserver),
              "failed replacement changed canonical volume ownership");
      {
        const auto preservedVolume = preservedObserver.lock();
        require(preservedVolume != nullptr &&
                    preservedVolume.get() == identityBeforeFailure,
                "failed replacement changed the displayed volume identity");
        requireCenterState(workspace,
                           *preservedVolume,
                           "failed replacement state");
        const qvp::VoxelIndex3D positionAfterFailure =
            workspace.mprViewer().voxelPosition();
        require(positionAfterFailure.x == positionBeforeFailure.x &&
                    positionAfterFailure.y == positionBeforeFailure.y &&
                    positionAfterFailure.z == positionBeforeFailure.z,
                "failed replacement changed MPR navigation");
      }

      observerAfterDestruction = workspace.volumeObserver();
      require(!observerAfterDestruction.expired(),
              "workspace lost the reloaded volume before destruction");
    }

    require(observerAfterDestruction.expired(),
            "workspace destruction retained loaded volume ownership");

    std::cout << "Real-volume asynchronous viewer workspace test passed."
              << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Real-volume asynchronous viewer workspace test failed: "
              << error.what() << '\n';
    return 1;
  }
}
