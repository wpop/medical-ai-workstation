#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationWindow.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"

#include <QApplication>
#include <QString>
#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);

    const auto metadata = maiw::cardiac::CardiacMriDeploymentMetadata::load(
        std::filesystem::path{MAIW_CARDIAC_MRI_PACKAGE_DIR});
    Ort::Env environment(ORT_LOGGING_LEVEL_ERROR,
                         "medical-ai-workstation-classification-shutdown-test");
    maiw::cardiac::CardiacMriClassificationService service(environment, metadata);

    auto workflow =
        std::make_unique<maiw::qt::CardiacMriClassificationWorkflow>(service);
    auto window = std::make_unique<maiw::qt::CardiacMriClassificationWindow>(
        *workflow,
        metadata.classNames());

    bool classificationStarted = false;
    QObject::connect(
        workflow.get(),
        &maiw::qt::CardiacMriClassificationWorkflow::classificationStarted,
        &application,
        [&classificationStarted]()
        {
          classificationStarted = true;
        });

    workflow->startClassification(
        QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH),
        QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ES_PATH));
    require(classificationStarted,
            "shutdown test did not observe classificationStarted");
    require(workflow->isRunning(),
            "shutdown test did not enter the active workflow state");

    window.reset();
    workflow.reset();
    std::cout << "Cardiac MRI active-work shutdown lifetime test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI active-work shutdown lifetime test failed: "
              << error.what() << '\n';
    return 1;
  }
}
