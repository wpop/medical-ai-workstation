#include "maiw/cardiac/CardiacMriClassificationResult.h"
#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"
#include "maiw/qt/MedicalAiWorkstationWindow.h"

#include <QApplication>
#include <QEventLoop>
#include <QLineEdit>
#include <QMetaObject>
#include <QStatusBar>
#include <QString>

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

const QString kCloseBlockedMessage =
    QStringLiteral("An operation is still in progress. "
                   "Please wait for it to finish before closing.");

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void requireClassificationCloseGuard(
    maiw::qt::CardiacMriClassificationWorkflow& workflow,
    maiw::cardiac::CardiacMriDeploymentMetadata::ClassNames classNames)
{
  maiw::qt::MedicalAiWorkstationWindow window(
      workflow,
      std::move(classNames));

  window.show();
  QApplication::processEvents();

  bool closeAttempted = false;
  bool closeAccepted = true;
  bool classificationSucceeded = false;
  QString classificationFailure;

  QEventLoop completionLoop;

  QObject::connect(
      &workflow,
      &maiw::qt::CardiacMriClassificationWorkflow::classificationStarted,
      &window,
      [&]()
      {
        closeAttempted = true;
        closeAccepted = window.close();
      });

  QObject::connect(
      &workflow,
      &maiw::qt::CardiacMriClassificationWorkflow::classificationSucceeded,
      &completionLoop,
      [&](const maiw::cardiac::CardiacMriClassificationResult&)
      {
        classificationSucceeded = true;
        completionLoop.quit();
      });

  QObject::connect(
      &workflow,
      &maiw::qt::CardiacMriClassificationWorkflow::classificationFailed,
      &completionLoop,
      [&](const QString& message)
      {
        classificationFailure = message;
        completionLoop.quit();
      });

  workflow.startClassification(
      QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH),
      QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ES_PATH));

  require(closeAttempted,
          "classification close guard did not attempt close from classificationStarted");
  require(!closeAccepted,
          "workstation accepted close while cardiac classification was active");
  require(workflow.isRunning(),
          "classification workflow was not active after rejected close");
  require(window.isVisible(),
          "workstation became hidden after rejecting active classification close");
  require(window.statusBar()->currentMessage() == kCloseBlockedMessage,
          "workstation did not present the active-operation close status");

  if (workflow.isRunning())
  {
    completionLoop.exec();
  }

  require(classificationSucceeded,
          classificationFailure.isEmpty()
              ? "cardiac classification did not complete successfully"
              : "cardiac classification failed: " +
                    classificationFailure.toStdString());
  require(!workflow.isRunning(),
          "classification workflow remained active after completion");
  require(window.isVisible(),
          "workstation closed automatically after classification completion");

  require(window.close(),
          "workstation rejected close after classification completed");
  QApplication::processEvents();
  require(!window.isVisible(),
          "workstation remained visible after idle close");
}

void requireViewerLoadCloseGuard(
    maiw::qt::CardiacMriClassificationWorkflow& workflow,
    maiw::cardiac::CardiacMriDeploymentMetadata::ClassNames classNames)
{
  maiw::qt::MedicalAiWorkstationWindow window(
      workflow,
      std::move(classNames));

  window.show();
  QApplication::processEvents();

  const auto& workspace = window.viewerWorkspace();

  auto* const edPathEdit = window.findChild<QLineEdit*>(
      QStringLiteral("cardiacEdVolumePathEdit"));
  require(edPathEdit != nullptr,
          "viewer close guard could not find the cardiac ED path editor");

  bool closeAttempted = false;
  bool closeAccepted = true;
  bool loadingSucceeded = false;
  QString loadingFailure;

  QEventLoop completionLoop;

  QObject::connect(
      &workspace,
      &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingStarted,
      &window,
      [&]()
      {
        closeAttempted = true;
        closeAccepted = window.close();
      });

  QObject::connect(
      &workspace,
      &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingSucceeded,
      &completionLoop,
      [&]()
      {
        loadingSucceeded = true;
        completionLoop.quit();
      });

  QObject::connect(
      &workspace,
      &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingFailed,
      &completionLoop,
      [&](const QString& message)
      {
        loadingFailure = message;
        completionLoop.quit();
      });

  edPathEdit->setText(QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH));
  const bool commitInvoked = QMetaObject::invokeMethod(
      edPathEdit,
      "editingFinished",
      Qt::DirectConnection);

  require(commitInvoked,
          "viewer close guard could not commit the cardiac ED path");
  require(closeAttempted,
          "viewer close guard did not attempt close from volumeLoadingStarted");
  require(!closeAccepted,
          "workstation accepted close while viewer loading was active");
  require(workspace.isLoading(),
          "viewer workflow was not active after rejected close");
  require(window.isVisible(),
          "workstation became hidden after rejecting active viewer close");
  require(window.statusBar()->currentMessage() == kCloseBlockedMessage,
          "workstation did not present the active-operation close status");

  if (workspace.isLoading())
  {
    completionLoop.exec();
  }

  require(loadingSucceeded,
          loadingFailure.isEmpty()
              ? "viewer volume loading did not complete successfully"
              : "viewer volume loading failed: " +
                    loadingFailure.toStdString());
  require(!workspace.isLoading(),
          "viewer workflow remained active after completion");
  require(window.isVisible(),
          "workstation closed automatically after viewer loading completion");

  require(window.close(),
          "workstation rejected close after viewer loading completed");
  QApplication::processEvents();
  require(!window.isVisible(),
          "workstation remained visible after idle close");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);
    application.setQuitOnLastWindowClosed(false);

    const auto metadata =
        maiw::cardiac::CardiacMriDeploymentMetadata::load(
            std::filesystem::path{MAIW_CARDIAC_MRI_PACKAGE_DIR});

    Ort::Env environment(
        ORT_LOGGING_LEVEL_ERROR,
        "medical-ai-workstation-lifecycle-test");

    maiw::cardiac::CardiacMriClassificationService service(
        environment,
        metadata);

    maiw::qt::CardiacMriClassificationWorkflow workflow(service);

    requireClassificationCloseGuard(
        workflow,
        metadata.classNames());

    requireViewerLoadCloseGuard(
        workflow,
        metadata.classNames());

    std::cout << "Medical AI Workstation lifecycle close-guard test passed."
              << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Medical AI Workstation lifecycle close-guard test failed: "
              << error.what() << '\n';
    return 1;
  }
}
