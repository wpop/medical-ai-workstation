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

using maiw::cardiac::CardiacMriClassificationResult;
using maiw::cardiac::CardiacMriDeploymentMetadata;
using maiw::qt::CardiacMriClassificationWindow;

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

    const CardiacMriClassificationResult::RawLogits logits{
        -1.0F, 0.0F, 2.0F, 1.0F, -0.5F};
    const CardiacMriClassificationResult result =
        CardiacMriClassificationResult::fromLogits(logits, metadata.classNames());

    workflow.classificationSucceeded(result);
    requireControlsEnabled(window, true, "classification success state");
    require(window.resultWidget()->predictedClassText() ==
                QString::fromStdString(result.predictedClassName()),
            "successful result was not forwarded to the result widget");
    require(window.resultWidget()->statusText().isEmpty(),
            "result widget retained an unexpected success status");

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
