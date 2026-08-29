#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationResultWidget.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"
#include "maiw/qt/MedicalAiWorkstationWindow.h"

#include <QApplication>
#include <QString>

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <iostream>
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

void requireInitialViewerState(
    const maiw::viewer::ViewerWorkspaceWidget& workspace)
{
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
}

void requireInitialClassificationState(
    const maiw::qt::CardiacMriClassificationWindow& classificationWindow)
{
  require(classificationWindow.isEdPathEditEnabled(),
          "ED path editor is initially disabled");
  require(classificationWindow.isEdBrowseButtonEnabled(),
          "ED browse button is initially disabled");
  require(classificationWindow.isEsPathEditEnabled(),
          "ES path editor is initially disabled");
  require(classificationWindow.isEsBrowseButtonEnabled(),
          "ES browse button is initially disabled");
  require(classificationWindow.isClassifyButtonEnabled(),
          "classification button is initially disabled");

  const auto* const resultWidget = classificationWindow.resultWidget();
  require(resultWidget != nullptr,
          "classification result widget is missing");
  require(resultWidget->predictedClassText() == QStringLiteral("—"),
          "classification result is not initially cleared");
  for (const QString& probabilityText : resultWidget->probabilityTexts())
  {
    require(probabilityText == QStringLiteral("—"),
            "classification probability is not initially cleared");
  }
  require(resultWidget->statusText().isEmpty(),
          "classification result has an unexpected initial status");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);

    const auto metadata = maiw::cardiac::CardiacMriDeploymentMetadata::load(
        std::filesystem::path{MAIW_CARDIAC_MRI_PACKAGE_DIR});
    Ort::Env environment(
        ORT_LOGGING_LEVEL_ERROR,
        "medical-ai-workstation-unified-window-test");
    maiw::cardiac::CardiacMriClassificationService service(environment, metadata);
    maiw::qt::CardiacMriClassificationWorkflow workflow(service);

    {
      maiw::qt::MedicalAiWorkstationWindow window(
          workflow,
          metadata.classNames());
      const auto* const viewerIdentity = &window.viewerWorkspace();
      const auto* const classificationIdentity = &window.classificationWindow();

      require(!window.isVisible(),
              "unified window is unexpectedly visible after construction");
      requireInitialViewerState(window.viewerWorkspace());
      requireInitialClassificationState(window.classificationWindow());

      window.show();
      QApplication::processEvents();
      require(window.isVisible(),
              "unified window did not become visible offscreen");
      require(window.viewerWorkspace().isVisible(),
              "viewer workspace is not visible with the unified window");
      require(window.classificationWindow().isVisible(),
              "classification UI is not visible with the unified window");

      window.hide();
      QApplication::processEvents();
      require(!window.isVisible(),
              "unified window remained visible after hide");
      require(&window.viewerWorkspace() == viewerIdentity,
              "viewer workspace identity changed during show/hide");
      require(&window.classificationWindow() == classificationIdentity,
              "classification UI identity changed during show/hide");
      requireInitialViewerState(window.viewerWorkspace());
      requireInitialClassificationState(window.classificationWindow());
    }

    require(!workflow.isRunning(),
            "unified window destruction changed workflow state");

    std::cout << "Medical AI Workstation unified window test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Medical AI Workstation unified window test failed: "
              << error.what() << '\n';
    return 1;
  }
}
