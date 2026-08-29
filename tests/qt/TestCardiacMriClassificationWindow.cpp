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
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QMouseEvent>
#include <QPointF>
#include <QPushButton>
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
const QString kSuccessfulClassificationStatus =
    QStringLiteral("Classification completed. Geometry compatibility validated; "
                   "patient/study identity not independently verified.");
const QString kFailedClassificationStatus =
    QStringLiteral("Classification failed.");
const QString kEdPathEditObjectName =
    QStringLiteral("cardiacEdVolumePathEdit");
const QString kEsPathEditObjectName =
    QStringLiteral("cardiacEsVolumePathEdit");
const QString kEdBrowseButtonObjectName =
    QStringLiteral("cardiacEdVolumeBrowseButton");
const QString kEsBrowseButtonObjectName =
    QStringLiteral("cardiacEsVolumeBrowseButton");
const QString kClassifyButtonObjectName =
    QStringLiteral("cardiacClassifyButton");

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

void commitManualPath(CardiacMriClassificationWindow& window,
                      const QString& objectName,
                      const QString& path)
{
  auto* const pathEdit = window.findChild<QLineEdit*>(objectName);
  require(pathEdit != nullptr,
          "cardiac path editor testing hook is missing");

  pathEdit->setText(path);
  require(QMetaObject::invokeMethod(pathEdit,
                                    "editingFinished",
                                    Qt::DirectConnection),
          "failed to commit a manually edited cardiac volume path");
}

void focusBrowseAfterEditing(CardiacMriClassificationWindow& window,
                             const QString& pathEditObjectName,
                             const QString& browseButtonObjectName,
                             const QString& pendingPath)
{
  auto* const pathEdit = window.findChild<QLineEdit*>(pathEditObjectName);
  auto* const browseButton =
      window.findChild<QPushButton*>(browseButtonObjectName);
  require(pathEdit != nullptr && browseButton != nullptr,
          "cardiac Browse focus testing hooks are missing");

  pathEdit->setFocus(Qt::OtherFocusReason);
  QApplication::processEvents();
  pathEdit->setText(pendingPath);

  QMouseEvent browsePress(QEvent::MouseButtonPress,
                          QPointF(1.0, 1.0),
                          QPointF(1.0, 1.0),
                          Qt::LeftButton,
                          Qt::LeftButton,
                          Qt::NoModifier);
  QApplication::sendEvent(browseButton, &browsePress);
  if (!browseButton->hasFocus())
  {
    browseButton->setFocus(Qt::MouseFocusReason);
  }
  QApplication::processEvents();
  browseButton->setDown(false);
}

void publishBrowseSelection(CardiacMriClassificationWindow& window,
                            const char* slotName,
                            const QString& selectedPath)
{
  require(QMetaObject::invokeMethod(&window,
                                    slotName,
                                    Qt::DirectConnection,
                                    Q_ARG(QString, selectedPath)),
          "failed to publish a simulated Browse selection");
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
    auto* const header = window.findChild<QLabel*>(
        QStringLiteral("cardiacClassificationHeader"));
    auto* const helperText = window.findChild<QLabel*>(
        QStringLiteral("cardiacClassificationHelperText"));
    auto* const classificationStatusLabel = window.findChild<QLabel*>(
        QStringLiteral("cardiacClassificationStatusLabel"));
    auto* const studyVolumesGroup = window.findChild<QGroupBox*>(
        QStringLiteral("cardiacStudyVolumesGroup"));
    auto* const aiResultGroup = window.findChild<QGroupBox*>(
        QStringLiteral("cardiacAiResultGroup"));
    auto* const classifyButton =
        window.findChild<QPushButton*>(kClassifyButtonObjectName);
    auto* const edBrowseButton =
        window.findChild<QPushButton*>(kEdBrowseButtonObjectName);
    auto* const esBrowseButton =
        window.findChild<QPushButton*>(kEsBrowseButtonObjectName);
    require(header != nullptr &&
                header->text() == QStringLiteral("Cardiac MRI Classification") &&
                header->font().bold(),
            "cardiac presentation header is missing or not emphasized");
    require(helperText != nullptr && !helperText->text().isEmpty() &&
                helperText->wordWrap(),
            "cardiac presentation helper text is missing");
    require(classificationStatusLabel != nullptr &&
                classificationStatusLabel->text().isEmpty(),
            "cardiac classification status is not initially empty");
    require(studyVolumesGroup != nullptr &&
                studyVolumesGroup->title() == QStringLiteral("Study volumes"),
            "cardiac study-volumes group is missing");
    require(aiResultGroup != nullptr &&
                aiResultGroup->title() == QStringLiteral("AI result"),
            "cardiac AI-result group is missing");
    require(classifyButton != nullptr && classifyButton->font().bold() &&
                classifyButton->isDefault() &&
                classifyButton->minimumHeight() > classifyButton->sizeHint().height(),
            "cardiac classification action is not visually primary");

    int edPathCommitCount = 0;
    int esPathCommitCount = 0;
    QString committedEdPath;
    QString committedEsPath;
    QObject::connect(
        &window,
        &CardiacMriClassificationWindow::edVolumePathCommitted,
        &application,
        [&edPathCommitCount, &committedEdPath](const QString& path)
        {
          ++edPathCommitCount;
          committedEdPath = path;
        });
    QObject::connect(
        &window,
        &CardiacMriClassificationWindow::esVolumePathCommitted,
        &application,
        [&esPathCommitCount, &committedEsPath](const QString& path)
        {
          ++esPathCommitCount;
          committedEsPath = path;
        });

    QApplication::processEvents();
    require(edPathCommitCount == 0 && esPathCommitCount == 0,
            "classification window construction emitted a path commit");

    const QString manuallyCommittedEdPath =
        QStringLiteral("/test/manual-ed-volume.nii.gz");
    commitManualPath(window, kEdPathEditObjectName, manuallyCommittedEdPath);
    require(edPathCommitCount == 1,
            "manual ED editing did not emit exactly one ED path commit");
    require(esPathCommitCount == 0,
            "manual ED editing emitted an ES path commit");
    require(committedEdPath == manuallyCommittedEdPath,
            "manual ED editing emitted an unexpected path");
    auto* const edPathEdit =
        window.findChild<QLineEdit*>(kEdPathEditObjectName);
    require(edPathEdit != nullptr &&
                edPathEdit->toolTip() == manuallyCommittedEdPath,
            "manual ED path is not available as a tooltip");

    const QString manuallyCommittedEsPath =
        QStringLiteral("/test/manual-es-volume.nii.gz");
    commitManualPath(window, kEsPathEditObjectName, manuallyCommittedEsPath);
    require(edPathCommitCount == 1,
            "manual ES editing emitted an ED path commit");
    require(esPathCommitCount == 1,
            "manual ES editing did not emit exactly one ES path commit");
    require(committedEsPath == manuallyCommittedEsPath,
            "manual ES editing emitted an unexpected path");
    auto* const esPathEdit =
        window.findChild<QLineEdit*>(kEsPathEditObjectName);
    require(esPathEdit != nullptr &&
                esPathEdit->toolTip() == manuallyCommittedEsPath,
            "manual ES path is not available as a tooltip");
    require(!workflow.isRunning(),
            "committing cardiac paths unexpectedly started classification");
    requireControlsEnabled(window, true, "manual path commit state");
    require(window.resultWidget()->predictedClassText() == QStringLiteral("—"),
            "committing cardiac paths changed the classification result");
    require(window.resultWidget()->statusText().isEmpty(),
            "committing cardiac paths changed the classification status");

    window.show();
    QApplication::processEvents();
    require(header->isVisible() && helperText->isVisible() &&
                studyVolumesGroup->isVisible() && aiResultGroup->isVisible() &&
                classifyButton->isVisible(),
            "cardiac presentation hierarchy is not visible");
    require(edBrowseButton != nullptr &&
                edPathEdit->width() > edBrowseButton->width(),
            "ED path editor is not wider than its Browse action");
    require(esBrowseButton != nullptr &&
                esPathEdit->width() > esBrowseButton->width(),
            "ES path editor is not wider than its Browse action");

    const QString pendingEdPath =
        QStringLiteral("/test/pending-ed-before-browse.nii.gz");
    focusBrowseAfterEditing(window,
                            kEdPathEditObjectName,
                            kEdBrowseButtonObjectName,
                            pendingEdPath);
    require(edPathCommitCount == 1,
            "focusing ED Browse committed the pre-Browse ED path");
    require(esPathCommitCount == 1,
            "focusing ED Browse emitted an ES path commit");

    const QString browsedEdPath =
        QStringLiteral("/test/browsed-ed-volume.nii.gz");
    publishBrowseSelection(window,
                           "publishEdBrowseSelection",
                           browsedEdPath);
    require(edPathCommitCount == 2,
            "accepted ED Browse selection did not emit exactly one commit");
    require(esPathCommitCount == 1,
            "accepted ED Browse selection emitted an ES path commit");
    require(committedEdPath == browsedEdPath,
            "accepted ED Browse selection emitted an unexpected path");
    require(edPathEdit->toolTip() == browsedEdPath &&
                edPathEdit->cursorPosition() ==
                    static_cast<int>(browsedEdPath.size()),
            "accepted ED Browse path is not presented in full");

    const QString pendingCancelledEdPath =
        QStringLiteral("/test/pending-cancelled-ed-browse.nii.gz");
    focusBrowseAfterEditing(window,
                            kEdPathEditObjectName,
                            kEdBrowseButtonObjectName,
                            pendingCancelledEdPath);
    publishBrowseSelection(window,
                           "publishEdBrowseSelection",
                           QString());
    require(edPathCommitCount == 2 && esPathCommitCount == 1,
            "cancelled ED Browse selection emitted a path commit");
    require(edPathEdit != nullptr && edPathEdit->text() == pendingCancelledEdPath,
            "cancelled ED Browse selection changed the pending ED path");

    const QString pendingEsPath =
        QStringLiteral("/test/pending-es-before-browse.nii.gz");
    focusBrowseAfterEditing(window,
                            kEsPathEditObjectName,
                            kEsBrowseButtonObjectName,
                            pendingEsPath);
    require(edPathCommitCount == 2,
            "focusing ES Browse emitted an ED path commit");
    require(esPathCommitCount == 1,
            "focusing ES Browse committed the pre-Browse ES path");

    const QString browsedEsPath =
        QStringLiteral("/test/browsed-es-volume.nii.gz");
    publishBrowseSelection(window,
                           "publishEsBrowseSelection",
                           browsedEsPath);
    require(edPathCommitCount == 2,
            "accepted ES Browse selection emitted an ED path commit");
    require(esPathCommitCount == 2,
            "accepted ES Browse selection did not emit exactly one commit");
    require(committedEsPath == browsedEsPath,
            "accepted ES Browse selection emitted an unexpected path");
    require(esPathEdit->toolTip() == browsedEsPath &&
                esPathEdit->cursorPosition() ==
                    static_cast<int>(browsedEsPath.size()),
            "accepted ES Browse path is not presented in full");

    const QString pendingCancelledEsPath =
        QStringLiteral("/test/pending-cancelled-es-browse.nii.gz");
    focusBrowseAfterEditing(window,
                            kEsPathEditObjectName,
                            kEsBrowseButtonObjectName,
                            pendingCancelledEsPath);
    publishBrowseSelection(window,
                           "publishEsBrowseSelection",
                           QString());
    require(edPathCommitCount == 2 && esPathCommitCount == 2,
            "cancelled ES Browse selection emitted a path commit");
    require(esPathEdit != nullptr && esPathEdit->text() == pendingCancelledEsPath,
            "cancelled ES Browse selection changed the pending ES path");

    require(edBrowseButton != nullptr && esBrowseButton != nullptr,
            "cardiac Browse buttons are missing");
    require((edBrowseButton->focusPolicy() & Qt::TabFocus) != 0 &&
                (esBrowseButton->focusPolicy() & Qt::TabFocus) != 0,
            "cardiac Browse buttons are not keyboard-focusable");

    edPathEdit->setFocus(Qt::OtherFocusReason);
    QApplication::processEvents();
    const QString keyboardCommittedEdPath =
        QStringLiteral("/test/keyboard-committed-ed-volume.nii.gz");
    edPathEdit->setText(keyboardCommittedEdPath);
    edBrowseButton->setFocus(Qt::TabFocusReason);
    QApplication::processEvents();
    require(edPathCommitCount == 3 && esPathCommitCount == 2,
            "ordinary keyboard focus navigation did not commit ED exactly once");
    require(committedEdPath == keyboardCommittedEdPath,
            "keyboard focus navigation emitted an unexpected ED path");

    window.hide();
    QApplication::processEvents();
    require(!workflow.isRunning(),
            "Browse path publication unexpectedly started classification");
    requireControlsEnabled(window, true, "Browse path publication state");

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
    require(classificationStatusLabel->text() == kFailedClassificationStatus,
            "synchronous validation failure changed the window failure status");
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
    for (const QString& probabilityText :
         window.resultWidget()->probabilityTexts())
    {
      require(probabilityText == QStringLiteral("—"),
              "failure did not clear a probability presentation");
    }
    require(window.resultWidget()->statusText() == failureMessage,
            "controlled failure was not forwarded to the result widget");
    require(classificationStatusLabel->text() == kFailedClassificationStatus,
            "controlled failure changed the window failure status");

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
    for (const QString& probabilityText :
         window.resultWidget()->probabilityTexts())
    {
      require(probabilityText != QStringLiteral("—") &&
                  probabilityText.endsWith(QStringLiteral(" %")),
              "real successful probability presentation changed unexpectedly");
    }
    require(window.resultWidget()->statusText().isEmpty(),
            "result widget retained an unexpected status after real success");
    require(classificationStatusLabel->text() ==
                kSuccessfulClassificationStatus,
            "real success status omitted the cardiac input identity limitation");
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
