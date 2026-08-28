#include "maiw/cardiac/CardiacMriClassificationResult.h"
#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationResultWidget.h"
#include "maiw/qt/CardiacMriClassificationWindow.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"

#include <QApplication>
#include <QDir>
#include <QEventLoop>
#include <QFileInfo>
#include <QString>
#include <QTimer>
#include <QUuid>
#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

using maiw::cardiac::CardiacMriDeploymentMetadata;
using maiw::qt::CardiacMriClassificationWindow;

constexpr int kRealClassificationTimeoutMilliseconds = 30000;
const QString kRequiredPathsError =
    QStringLiteral("Both ED and ES medical volume paths are required.");

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void requireControlsEnabled(const CardiacMriClassificationWindow& window,
                            bool expectedEnabled,
                            const std::string& context)
{
  require(window.isEdPathEditEnabled() == expectedEnabled,
          context + ": unexpected ED path editor state");
  require(window.isEdBrowseButtonEnabled() == expectedEnabled,
          context + ": unexpected ED browse button state");
  require(window.isEsPathEditEnabled() == expectedEnabled,
          context + ": unexpected ES path editor state");
  require(window.isEsBrowseButtonEnabled() == expectedEnabled,
          context + ": unexpected ES browse button state");
  require(window.isClassifyButtonEnabled() == expectedEnabled,
          context + ": unexpected classify button state");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);

    const CardiacMriDeploymentMetadata metadata =
        CardiacMriDeploymentMetadata::load(
            std::filesystem::path{MAIW_CARDIAC_MRI_PACKAGE_DIR});
    Ort::Env environment(ORT_LOGGING_LEVEL_ERROR,
                         "medical-ai-workstation-classification-window-test");
    maiw::cardiac::CardiacMriClassificationService service(environment, metadata);
    maiw::qt::CardiacMriClassificationWorkflow workflow(service);
    CardiacMriClassificationWindow window(workflow, metadata.classNames());

    requireControlsEnabled(window, true, "initial state");
    require(window.resultWidget() != nullptr,
            "classification result widget is missing");

    const QString invalidEdPath =
        QDir::temp().filePath(QStringLiteral("maiw-missing-ed-%1.nii.gz")
                                  .arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
    const QString invalidEsPath =
        QDir::temp().filePath(QStringLiteral("maiw-missing-es-%1.nii.gz")
                                  .arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
    require(!QFileInfo::exists(invalidEdPath),
            "generated invalid ED path unexpectedly exists");
    require(!QFileInfo::exists(invalidEsPath),
            "generated invalid ES path unexpectedly exists");

    bool synchronousFailureReceived = false;
    QString synchronousFailureMessage;
    const auto synchronousFailureConnection = QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationFailed,
        &application,
        [&synchronousFailureReceived, &synchronousFailureMessage](const QString& message)
        {
          synchronousFailureReceived = true;
          synchronousFailureMessage = message;
        });

    workflow.startClassification(QString(), invalidEsPath);
    require(synchronousFailureReceived,
            "empty required path did not emit a synchronous validation failure");
    require(synchronousFailureMessage == kRequiredPathsError,
            "empty required path produced an unexpected validation failure");
    require(!workflow.isRunning(),
            "empty required path left the workflow running");
    requireControlsEnabled(window, true, "synchronous validation failure state");
    require(window.resultWidget()->statusText() == synchronousFailureMessage,
            "synchronous validation failure was not forwarded to the result widget");
    QObject::disconnect(synchronousFailureConnection);

    QEventLoop failureEventLoop;
    bool failureReceived = false;
    QString failureMessage;
    QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationFailed,
        &failureEventLoop,
        [&failureEventLoop, &failureReceived, &failureMessage](const QString& message)
        {
          failureReceived = true;
          failureMessage = message;
          failureEventLoop.quit();
        });

    workflow.startClassification(invalidEdPath, invalidEsPath);
    require(workflow.isRunning(),
            "workflow did not enter the running state");
    requireControlsEnabled(window, false, "classification started state");

    QTimer::singleShot(10000, &failureEventLoop, &QEventLoop::quit);
    failureEventLoop.exec();

    require(failureReceived, "timed out waiting for controlled load failure");
    require(!workflow.isRunning(),
            "workflow remained running after controlled load failure");
    requireControlsEnabled(window, true, "classification failure state");
    require(failureMessage.startsWith(
                QStringLiteral("Failed to load the ED volume: ")),
            "invalid ED path produced an unexpected failure message");
    require(window.resultWidget()->predictedClassText() == QStringLiteral("—"),
            "failure did not clear the predicted class presentation");
    require(window.resultWidget()->statusText() == failureMessage,
            "controlled failure was not forwarded to the result widget");

#if defined(MAIW_CARDIAC_MRI_REAL_ED_PATH) && defined(MAIW_CARDIAC_MRI_REAL_ES_PATH)
    const QString realEdPath = QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH);
    const QString realEsPath = QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ES_PATH);
    QEventLoop successEventLoop;
    bool successReceived = false;
    QString successfulClassName;
    QString unexpectedFailure;

    const auto successConnection = QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationSucceeded,
        &successEventLoop,
        [&successEventLoop, &successReceived, &successfulClassName](
            const maiw::cardiac::CardiacMriClassificationResult& result)
        {
          successReceived = true;
          successfulClassName = QString::fromStdString(result.predictedClassName());
          successEventLoop.quit();
        });
    const auto unexpectedFailureConnection = QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationFailed,
        &successEventLoop,
        [&successEventLoop, &unexpectedFailure](const QString& message)
        {
          unexpectedFailure = message;
          successEventLoop.quit();
        });

    workflow.startClassification(realEdPath, realEsPath);
    require(workflow.isRunning(),
            "real window classification did not enter the running state");
    requireControlsEnabled(window, false, "real classification started state");

    QTimer successWatchdog;
    successWatchdog.setSingleShot(true);
    successWatchdog.setInterval(kRealClassificationTimeoutMilliseconds);
    QObject::connect(&successWatchdog,
                     &QTimer::timeout,
                     &successEventLoop,
                     &QEventLoop::quit);
    successWatchdog.start();
    successEventLoop.exec();

    QObject::disconnect(successConnection);
    QObject::disconnect(unexpectedFailureConnection);
    require(unexpectedFailure.isEmpty(),
            "real window classification failed: " + unexpectedFailure.toStdString());
    require(successReceived,
            "timed out waiting for real window classification success");
    require(!workflow.isRunning(),
            "workflow remained running after real window classification success");
    requireControlsEnabled(window, true, "real classification success state");
    require(window.resultWidget()->predictedClassText() == successfulClassName,
            "real successful result was not forwarded to the result widget");
    require(window.resultWidget()->statusText().isEmpty(),
            "result widget retained an unexpected status after real success");
#endif

    std::cout << "Cardiac MRI classification window test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI classification window test failed: "
              << error.what() << '\n';
    return 1;
  }
}
