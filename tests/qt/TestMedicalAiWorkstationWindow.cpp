#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationResultWidget.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"
#include "maiw/qt/MedicalAiWorkstationWindow.h"

#include <QApplication>
#include <QGroupBox>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QRect>
#include <QSize>
#include <QSplitter>
#include <QString>

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cmath>
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

void requireUsableWorkstationGeometry(
    maiw::qt::MedicalAiWorkstationWindow& window,
    const QSize& requestedSize)
{
  const std::string context =
      std::to_string(requestedSize.width()) + "x" +
      std::to_string(requestedSize.height());
  auto* const topLevelSplitter =
      qobject_cast<QSplitter*>(window.centralWidget());
  require(topLevelSplitter != nullptr && topLevelSplitter->count() == 2,
          context + ": unified window is missing its two-pane splitter");
  require(!topLevelSplitter->childrenCollapsible(),
          context + ": top-level workstation panes are collapsible");

  const QList<int> topLevelSizes = topLevelSplitter->sizes();
  const int viewerWidth = topLevelSizes[0];
  const int cardiacWidth = topLevelSizes[1];
  const int availableWidth = viewerWidth + cardiacWidth;
  require(availableWidth > 0,
          context + ": top-level splitter has no usable width");
  require(viewerWidth * 100 >= availableWidth * 70 &&
              viewerWidth * 100 <= availableWidth * 80,
          context + ": viewer workspace is outside the expected 70-80% range");
  require(cardiacWidth * 100 >= availableWidth * 20 &&
              cardiacWidth * 100 <= availableWidth * 30,
          context + ": cardiac side panel is outside the expected 20-30% range");
  require(cardiacWidth >= window.classificationWindow().minimumSizeHint().width(),
          context + ": cardiac side panel is narrower than its usable minimum hint");

  auto* const viewerSplitter =
      window.viewerWorkspace().findChild<QSplitter*>(
          QString{}, Qt::FindDirectChildrenOnly);
  require(viewerSplitter != nullptr && viewerSplitter->count() == 2,
          context + ": viewer workspace is missing its MPR/3D splitter");
  require(!viewerSplitter->childrenCollapsible(),
          context + ": MPR and 3D viewer panes are collapsible");

  const QList<int> viewerSizes = viewerSplitter->sizes();
  const int mprWidth = viewerSizes[0];
  const int volumeRenderingWidth = viewerSizes[1];
  const int availableViewerWidth = mprWidth + volumeRenderingWidth;
  require(availableViewerWidth > 0,
          context + ": internal viewer splitter has no usable width");
  require(mprWidth * 100 >= availableViewerWidth * 62 &&
              mprWidth * 100 <= availableViewerWidth * 70,
          context + ": MPR pane is outside the expected 62-70% range");
  require(volumeRenderingWidth * 100 >= availableViewerWidth * 30 &&
              volumeRenderingWidth * 100 <= availableViewerWidth * 38,
          context + ": 3D pane is outside the expected 30-38% range");

  const auto& mprViewer = window.viewerWorkspace().mprViewer();
  const QRect axial = mprViewer.axialViewer().geometry();
  const QRect sagittal = mprViewer.sagittalViewer().geometry();
  const QRect coronal = mprViewer.coronalViewer().geometry();
  require(axial.width() * 100 >= mprWidth * 45 && axial.height() > 0 &&
              sagittal.width() * 100 >= mprWidth * 45 &&
              sagittal.height() > 0 &&
              coronal.width() * 100 >= mprWidth * 95 &&
              coronal.height() > 0,
          context + ": one or more MPR grid viewports are not meaningfully sized");
  require(std::abs(axial.width() - sagittal.width()) * 100 <= mprWidth * 2,
          context + ": axial and sagittal widths are not balanced");
  require(std::abs(axial.y() - sagittal.y()) * 100 <=
              mprViewer.height() * 2,
          context + ": axial and sagittal do not share the top row");
  require(coronal.y() > axial.y() && coronal.y() > sagittal.y(),
          context + ": coronal viewport is not below the top MPR row");
  require(std::abs(axial.height() - coronal.height()) * 100 <=
              mprViewer.height() * 2,
          context + ": MPR grid row heights are not balanced");

  const auto& classificationWindow = window.classificationWindow();
  QLayout* const cardiacLayout = classificationWindow.layout();
  require(cardiacLayout != nullptr && cardiacLayout->count() > 0,
          context + ": cardiac side panel has no presentation layout");
  QLayoutItem* const trailingItem =
      cardiacLayout->itemAt(cardiacLayout->count() - 1);
  require(trailingItem != nullptr && trailingItem->spacerItem() != nullptr &&
              trailingItem->expandingDirections().testFlag(Qt::Vertical),
          context + ": cardiac content has no trailing vertical stretch");
  require(trailingItem->geometry().height() > 0,
          context + ": cardiac trailing stretch did not absorb excess height");
  auto* const header = classificationWindow.findChild<QLabel*>(
      QStringLiteral("cardiacClassificationHeader"));
  auto* const studyVolumesGroup = classificationWindow.findChild<QGroupBox*>(
      QStringLiteral("cardiacStudyVolumesGroup"));
  auto* const resultGroup = classificationWindow.findChild<QGroupBox*>(
      QStringLiteral("cardiacAiResultGroup"));
  auto* const edPathEdit = classificationWindow.findChild<QLineEdit*>(
      QStringLiteral("cardiacEdVolumePathEdit"));
  auto* const esPathEdit = classificationWindow.findChild<QLineEdit*>(
      QStringLiteral("cardiacEsVolumePathEdit"));
  auto* const classifyButton = classificationWindow.findChild<QPushButton*>(
      QStringLiteral("cardiacClassifyButton"));
  require(header != nullptr && header->isVisible() &&
              studyVolumesGroup != nullptr && studyVolumesGroup->isVisible() &&
              resultGroup != nullptr && resultGroup->isVisible(),
          context + ": cardiac presentation hierarchy is incomplete");
  require(edPathEdit != nullptr && edPathEdit->isVisible() &&
              edPathEdit->width() * 100 >= cardiacWidth * 40 &&
              esPathEdit != nullptr && esPathEdit->isVisible() &&
              esPathEdit->width() * 100 >= cardiacWidth * 40,
          context + ": cardiac path editors are not usable");
  require(classifyButton != nullptr && classifyButton->isVisible() &&
              classifyButton->width() * 100 >= cardiacWidth * 85,
          context + ": primary classification action is not usable");
  require(resultGroup->geometry().bottom() < trailingItem->geometry().top(),
          context + ": cardiac result content is not packed above the trailing stretch");

  std::cout << "Workstation geometry " << context
            << ": top=[" << viewerWidth << ',' << cardiacWidth << ']'
            << " workspace=" << window.viewerWorkspace().width() << 'x'
            << window.viewerWorkspace().height()
            << " cardiac=" << classificationWindow.width() << 'x'
            << classificationWindow.height()
            << " internal=[" << mprWidth << ',' << volumeRenderingWidth << ']'
            << " axial=" << axial.x() << ',' << axial.y() << ' '
            << axial.width() << 'x' << axial.height()
            << " sagittal=" << sagittal.x() << ',' << sagittal.y() << ' '
            << sagittal.width() << 'x' << sagittal.height()
            << " coronal=" << coronal.x() << ',' << coronal.y() << ' '
            << coronal.width() << 'x' << coronal.height()
            << " 3D=" << window.viewerWorkspace().volumeRenderingWidget().width()
            << 'x' << window.viewerWorkspace().volumeRenderingWidget().height()
            << '\n';
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

      const std::array<QSize, 3> requiredWindowSizes{
          QSize{1280, 800},
          QSize{1600, 900},
          QSize{1920, 1080}};
      for (const QSize& requiredWindowSize : requiredWindowSizes)
      {
        window.resize(requiredWindowSize);
        QApplication::processEvents();
        requireUsableWorkstationGeometry(window, requiredWindowSize);
      }

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
