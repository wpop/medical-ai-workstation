#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationWindow.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"
#include "maiw/qt/MedicalAiWorkstationWindow.h"
#include "maiw/viewer/ViewerWorkspaceWidget.h"

#include "qtviewerpro/core/VolumeData.h"

#include <QApplication>
#include <QDir>
#include <QEventLoop>
#include <QFileInfo>
#include <QLineEdit>
#include <QMetaObject>
#include <QString>
#include <QThread>
#include <QTimer>
#include <QUuid>

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

constexpr int kLoadTimeoutMilliseconds = 30000;
const QString kEdPathEditObjectName =
    QStringLiteral("cardiacEdVolumePathEdit");
const QString kEsPathEditObjectName =
    QStringLiteral("cardiacEsVolumePathEdit");

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

void commitManualPath(
    const maiw::qt::CardiacMriClassificationWindow& classificationWindow,
    const QString& objectName,
    const QString& path)
{
  auto* const pathEdit = classificationWindow.findChild<QLineEdit*>(objectName);
  require(pathEdit != nullptr,
          "cardiac path editor testing hook is missing");

  pathEdit->setText(path);
  require(QMetaObject::invokeMethod(pathEdit,
                                    "editingFinished",
                                    Qt::DirectConnection),
          "failed to commit a manually edited cardiac volume path");
}

void requireClassificationIdle(
    const maiw::qt::CardiacMriClassificationWorkflow& workflow,
    const maiw::qt::CardiacMriClassificationWindow& window,
    const std::string& context)
{
  require(!workflow.isRunning(),
          context + ": classification workflow is running");
  require(window.isEdPathEditEnabled(),
          context + ": ED path editor is disabled");
  require(window.isEdBrowseButtonEnabled(),
          context + ": ED browse button is disabled");
  require(window.isEsPathEditEnabled(),
          context + ": ES path editor is disabled");
  require(window.isEsBrowseButtonEnabled(),
          context + ": ES browse button is disabled");
  require(window.isClassifyButtonEnabled(),
          context + ": classify button is disabled");
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
  bool runningAfterCommit = false;
  bool timedOut = false;
};

LoadAttempt runCommittedLoadAttempt(
    const maiw::qt::CardiacMriClassificationWindow& classificationWindow,
    const maiw::viewer::ViewerWorkspaceWidget& workspace,
    const QString& pathEditObjectName,
    const QString& path,
    const LoadingSignals& signalState,
    const std::function<void()>& verifyWhileLoading)
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
  commitManualPath(classificationWindow, pathEditObjectName, path);
  attempt.runningAfterCommit = workspace.isLoading();
  verifyWhileLoading();

  if (signalState.successCount + signalState.failureCount == terminalCountBefore)
  {
    eventLoop.exec();
  }

  watchdog.stop();
  return attempt;
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);

    const QString realEdPath = QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH);
    const QString realEsPath = QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ES_PATH);
    require(QFileInfo::exists(realEdPath),
            "configured real ACDC ED path does not exist");
    require(QFileInfo::exists(realEsPath),
            "configured real ACDC ES path does not exist");

    const auto metadata = maiw::cardiac::CardiacMriDeploymentMetadata::load(
        std::filesystem::path{MAIW_CARDIAC_MRI_PACKAGE_DIR});
    Ort::Env environment(
        ORT_LOGGING_LEVEL_ERROR,
        "medical-ai-workstation-study-coordination-test");
    maiw::cardiac::CardiacMriClassificationService service(environment, metadata);
    maiw::qt::CardiacMriClassificationWorkflow classificationWorkflow(service);
    LoadingSignals signalState;

    maiw::qt::MedicalAiWorkstationWindow window(
        classificationWorkflow,
        metadata.classNames());
    const auto& workspace = window.viewerWorkspace();
    const auto& classificationWindow = window.classificationWindow();

    require(!workspace.isLoading(),
            "viewer workspace is unexpectedly loading initially");
    require(!workspace.hasVolume(),
            "viewer workspace unexpectedly has an initial volume");
    require(workspace.volumeObserver().expired(),
            "viewer workspace exposes an initial volume observer");
    require(!workspace.mprViewer().hasVolume(),
            "MPR child unexpectedly has an initial volume");
    require(!workspace.volumeRenderingWidget().hasVolume(),
            "3D child unexpectedly has an initial volume");
    requireClassificationIdle(classificationWorkflow,
                              classificationWindow,
                              "initial state");

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
        [&application, &signalState, &workspace](const QString& message)
        {
          ++signalState.failureCount;
          signalState.allCompletionsOnMainThread =
              signalState.allCompletionsOnMainThread &&
              QThread::currentThread() == application.thread();
          signalState.loadingAtLastCompletion = workspace.isLoading();
          signalState.lastFailureMessage = message;
        });

    const LoadAttempt edAttempt = runCommittedLoadAttempt(
        classificationWindow,
        workspace,
        kEdPathEditObjectName,
        realEdPath,
        signalState,
        [&workspace]()
        {
          require(!workspace.hasVolume(),
                  "ED commit assigned a volume before loading completed");
        });
    require(!edAttempt.timedOut,
            "real ACDC ED coordination timed out");
    require(edAttempt.runningAfterCommit,
            "real ACDC ED commit did not start viewer loading");
    require(signalState.startedCount == 1,
            "real ACDC ED commit did not start exactly one load");
    require(!signalState.hadVolumeAtLastStart,
            "real ACDC ED load started with an unexpected volume");
    require(signalState.successCount == 1 && signalState.failureCount == 0,
            "real ACDC ED load did not succeed exactly once: " +
                signalState.lastFailureMessage.toStdString());
    require(signalState.allCompletionsOnMainThread,
            "real ACDC ED completion was not delivered on the main Qt thread");
    require(!signalState.loadingAtLastCompletion && !workspace.isLoading(),
            "viewer remained loading when real ACDC ED success was delivered");
    requireAssignedState(workspace, "real ACDC ED assignment");
    requireClassificationIdle(classificationWorkflow,
                              classificationWindow,
                              "after ED coordination");

    std::weak_ptr<const qvp::VolumeData> edObserver = workspace.volumeObserver();
    {
      const auto edVolume = edObserver.lock();
      require(edVolume != nullptr && edVolume->isValid() && !edVolume->isEmpty(),
              "real ACDC ED assignment is invalid or empty");
      requireCenterState(workspace, *edVolume, "real ACDC ED assignment");
    }

    const LoadAttempt esAttempt = runCommittedLoadAttempt(
        classificationWindow,
        workspace,
        kEsPathEditObjectName,
        realEsPath,
        signalState,
        [&workspace, &edObserver]()
        {
          require(workspace.hasVolume(),
                  "ES load cleared the current ED volume");
          require(hasSameOwner(edObserver, workspace.volumeObserver()),
                  "ES load replaced ED before loading completed");
        });
    require(!esAttempt.timedOut,
            "real ACDC ES coordination timed out");
    require(esAttempt.runningAfterCommit,
            "real ACDC ES commit did not start viewer loading");
    require(signalState.startedCount == 2,
            "real ACDC ES commit did not start exactly one additional load");
    require(signalState.hadVolumeAtLastStart,
            "real ACDC ES load did not preserve ED at start");
    require(signalState.successCount == 2 && signalState.failureCount == 0,
            "real ACDC ES load did not succeed exactly once: " +
                signalState.lastFailureMessage.toStdString());
    require(signalState.allCompletionsOnMainThread,
            "real ACDC ES completion was not delivered on the main Qt thread");
    require(!signalState.loadingAtLastCompletion && !workspace.isLoading(),
            "viewer remained loading when real ACDC ES success was delivered");
    require(edObserver.expired(),
            "successful ES replacement retained the previous ED volume");
    requireAssignedState(workspace, "real ACDC ES assignment");
    requireClassificationIdle(classificationWorkflow,
                              classificationWindow,
                              "after ES coordination");

    const std::weak_ptr<const qvp::VolumeData> esObserver =
        workspace.volumeObserver();
    require(!hasSameOwner(edObserver, esObserver),
            "ES replacement reused the ED ownership identity");
    const qvp::VolumeData* esIdentity = nullptr;
    qvp::VoxelIndex3D esPosition{};
    {
      const auto esVolume = esObserver.lock();
      require(esVolume != nullptr && esVolume->isValid() && !esVolume->isEmpty(),
              "real ACDC ES assignment is invalid or empty");
      esIdentity = esVolume.get();
      esPosition = workspace.mprViewer().voxelPosition();
      requireCenterState(workspace, *esVolume, "real ACDC ES assignment");
    }

    const QString missingPath =
        QDir::temp().filePath(
            QStringLiteral("maiw-missing-study-volume-%1.nii.gz")
                .arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
    require(!QFileInfo::exists(missingPath),
            "generated missing study path unexpectedly exists");

    const LoadAttempt failedAttempt = runCommittedLoadAttempt(
        classificationWindow,
        workspace,
        kEdPathEditObjectName,
        missingPath,
        signalState,
        [&workspace, &esObserver]()
        {
          require(workspace.hasVolume(),
                  "failed replacement cleared ES while loading");
          require(hasSameOwner(esObserver, workspace.volumeObserver()),
                  "failed replacement changed ES ownership while loading");
        });
    require(!failedAttempt.timedOut,
            "missing study-volume replacement timed out");
    require(failedAttempt.runningAfterCommit,
            "missing study-volume commit did not start viewer loading");
    require(signalState.startedCount == 3,
            "missing study-volume commit did not start exactly one additional load");
    require(signalState.hadVolumeAtLastStart,
            "failed replacement did not preserve ES at start");
    require(signalState.successCount == 2 && signalState.failureCount == 1,
            "missing study-volume replacement did not fail exactly once");
    require(!signalState.lastFailureMessage.isEmpty(),
            "missing study-volume replacement produced no diagnostic");
    require(signalState.allCompletionsOnMainThread,
            "replacement failure was not delivered on the main Qt thread");
    require(!signalState.loadingAtLastCompletion && !workspace.isLoading(),
            "viewer remained loading when replacement failure was delivered");
    requireAssignedState(workspace, "failed replacement state");
    require(hasSameOwner(esObserver, workspace.volumeObserver()),
            "failed replacement changed the displayed ES ownership");
    {
      const auto preservedEsVolume = workspace.volumeObserver().lock();
      require(preservedEsVolume != nullptr &&
                  preservedEsVolume.get() == esIdentity,
              "failed replacement changed the displayed ES identity");
      const qvp::VoxelIndex3D positionAfterFailure =
          workspace.mprViewer().voxelPosition();
      require(positionAfterFailure.x == esPosition.x &&
                  positionAfterFailure.y == esPosition.y &&
                  positionAfterFailure.z == esPosition.z,
              "failed replacement changed MPR navigation");
      requireCenterState(workspace,
                         *preservedEsVolume,
                         "failed replacement state");
    }
    requireClassificationIdle(classificationWorkflow,
                              classificationWindow,
                              "after failed viewer replacement");

    const std::size_t startedBeforeWhitespace = signalState.startedCount;
    const std::size_t successBeforeWhitespace = signalState.successCount;
    const std::size_t failureBeforeWhitespace = signalState.failureCount;
    commitManualPath(classificationWindow,
                     kEsPathEditObjectName,
                     QStringLiteral(" \t "));
    QApplication::processEvents();

    require(signalState.startedCount == startedBeforeWhitespace &&
                signalState.successCount == successBeforeWhitespace &&
                signalState.failureCount == failureBeforeWhitespace,
            "whitespace path commit changed viewer loading signals");
    require(!workspace.isLoading(),
            "whitespace path commit started viewer loading");
    requireAssignedState(workspace, "whitespace path commit state");
    require(hasSameOwner(esObserver, workspace.volumeObserver()),
            "whitespace path commit changed the displayed ES ownership");
    const qvp::VoxelIndex3D positionAfterWhitespace =
        workspace.mprViewer().voxelPosition();
    require(positionAfterWhitespace.x == esPosition.x &&
                positionAfterWhitespace.y == esPosition.y &&
                positionAfterWhitespace.z == esPosition.z,
            "whitespace path commit changed MPR navigation");
    requireClassificationIdle(classificationWorkflow,
                              classificationWindow,
                              "after whitespace path commit");

    std::cout << "Real ACDC cardiac study coordination test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Real ACDC cardiac study coordination test failed: "
              << error.what() << '\n';
    return 1;
  }
}
