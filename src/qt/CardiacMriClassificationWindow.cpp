#include "maiw/qt/CardiacMriClassificationWindow.h"

#include "maiw/qt/CardiacMriClassificationResultWidget.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"

#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QString>
#include <QVBoxLayout>

#include <utility>

namespace maiw::qt
{

namespace
{

/**
 * @brief Return the file-dialog filter used for supported medical volume inputs.
 *
 * The workflow delegates format detection and loading to qt-viewer-pro medical
 * IO. The filter therefore only improves file-selection usability and does not
 * define the authoritative set of supported formats.
 */
QString medicalVolumeFileFilter()
{
  return QStringLiteral(
      "Medical Volumes (*.nii *.nii.gz *.mhd *.mha *.nrrd *.nhdr *.dcm *.dicom *.json);;"
      "All Files (*)");
}

} // namespace

CardiacMriClassificationWindow::CardiacMriClassificationWindow(
    CardiacMriClassificationWorkflow& workflow,
    cardiac::CardiacMriDeploymentMetadata::ClassNames classNames,
    QWidget* parent)
    : QWidget(parent),
      workflow_(workflow)
{
  setWindowTitle(QStringLiteral("Medical AI Workstation — Cardiac MRI Classification"));

  initializeUi();

  resultWidget_ =
      new CardiacMriClassificationResultWidget(std::move(classNames), this);

  auto* mainLayout = qobject_cast<QVBoxLayout*>(layout());
  mainLayout->addWidget(resultWidget_);

  connect(&workflow_,
          &CardiacMriClassificationWorkflow::classificationStarted,
          this,
          &CardiacMriClassificationWindow::handleClassificationStarted);

  connect(&workflow_,
          &CardiacMriClassificationWorkflow::classificationSucceeded,
          this,
          &CardiacMriClassificationWindow::handleClassificationSucceeded);

  connect(&workflow_,
          &CardiacMriClassificationWorkflow::classificationFailed,
          this,
          &CardiacMriClassificationWindow::handleClassificationFailed);

  updateControls();
}

bool CardiacMriClassificationWindow::isEdPathEditEnabled() const
{
  return edPathEdit_->isEnabled();
}

bool CardiacMriClassificationWindow::isEdBrowseButtonEnabled() const
{
  return edBrowseButton_->isEnabled();
}

bool CardiacMriClassificationWindow::isEsPathEditEnabled() const
{
  return esPathEdit_->isEnabled();
}

bool CardiacMriClassificationWindow::isEsBrowseButtonEnabled() const
{
  return esBrowseButton_->isEnabled();
}

bool CardiacMriClassificationWindow::isClassifyButtonEnabled() const
{
  return classifyButton_->isEnabled();
}

const CardiacMriClassificationResultWidget*
CardiacMriClassificationWindow::resultWidget() const noexcept
{
  return resultWidget_;
}

void CardiacMriClassificationWindow::browseEdVolume()
{
  const QString selectedPath =
      QFileDialog::getOpenFileName(this,
                                   QStringLiteral("Select End-Diastolic Volume"),
                                   edPathEdit_->text(),
                                   medicalVolumeFileFilter());

  if (!selectedPath.isEmpty())
  {
    edPathEdit_->setText(selectedPath);
  }
}

void CardiacMriClassificationWindow::browseEsVolume()
{
  const QString selectedPath =
      QFileDialog::getOpenFileName(this,
                                   QStringLiteral("Select End-Systolic Volume"),
                                   esPathEdit_->text(),
                                   medicalVolumeFileFilter());

  if (!selectedPath.isEmpty())
  {
    esPathEdit_->setText(selectedPath);
  }
}

void CardiacMriClassificationWindow::startClassification()
{
  /*
   * The workflow owns the asynchronous execution boundary. This method must not
   * load medical volumes, preprocess data, or invoke ONNX Runtime directly.
   */
  resultWidget_->clear();
  statusLabel_->clear();

  workflow_.startClassification(edPathEdit_->text(), esPathEdit_->text());
}

void CardiacMriClassificationWindow::handleClassificationStarted()
{
  statusLabel_->setText(QStringLiteral("Classifying cardiac MRI volumes..."));
  updateControls();
}

void CardiacMriClassificationWindow::handleClassificationSucceeded(
    const cardiac::CardiacMriClassificationResult& result)
{
  /*
   * The result widget consumes the validated service result directly. No
   * probability calculation, argmax operation, or class remapping belongs here.
   */
  resultWidget_->setResult(result);

  statusLabel_->setText(QStringLiteral("Classification completed."));
  updateControls();
}

void CardiacMriClassificationWindow::handleClassificationFailed(
    const QString& message)
{
  /*
   * Async execution failures arrive through the workflow's controlled Qt
   * boundary. The window only presents the supplied error state.
   */
  resultWidget_->showError(message);

  statusLabel_->setText(QStringLiteral("Classification failed."));
  updateControls();
}

void CardiacMriClassificationWindow::initializeUi()
{
  auto* mainLayout = new QVBoxLayout(this);

  auto* inputLayout = new QFormLayout();

  /*
   * Each path row contains a text editor and a browse button. All widgets are
   * parent-owned by Qt through the layout/widget hierarchy; stored pointers are
   * non-owning convenience handles only.
   */
  auto* edRowWidget = new QWidget(this);
  auto* edRowLayout = new QHBoxLayout(edRowWidget);
  edRowLayout->setContentsMargins(0, 0, 0, 0);

  edPathEdit_ = new QLineEdit(edRowWidget);
  edPathEdit_->setPlaceholderText(
      QStringLiteral("Select the end-diastolic medical volume"));

  edBrowseButton_ = new QPushButton(QStringLiteral("Browse..."), edRowWidget);

  edRowLayout->addWidget(edPathEdit_);
  edRowLayout->addWidget(edBrowseButton_);

  inputLayout->addRow(QStringLiteral("ED volume:"), edRowWidget);

  auto* esRowWidget = new QWidget(this);
  auto* esRowLayout = new QHBoxLayout(esRowWidget);
  esRowLayout->setContentsMargins(0, 0, 0, 0);

  esPathEdit_ = new QLineEdit(esRowWidget);
  esPathEdit_->setPlaceholderText(
      QStringLiteral("Select the end-systolic medical volume"));

  esBrowseButton_ = new QPushButton(QStringLiteral("Browse..."), esRowWidget);

  esRowLayout->addWidget(esPathEdit_);
  esRowLayout->addWidget(esBrowseButton_);

  inputLayout->addRow(QStringLiteral("ES volume:"), esRowWidget);

  mainLayout->addLayout(inputLayout);

  classifyButton_ =
      new QPushButton(QStringLiteral("Classify Cardiac MRI"), this);
  mainLayout->addWidget(classifyButton_);

  statusLabel_ = new QLabel(this);
  statusLabel_->setWordWrap(true);
  mainLayout->addWidget(statusLabel_);

  connect(edBrowseButton_,
          &QPushButton::clicked,
          this,
          &CardiacMriClassificationWindow::browseEdVolume);

  connect(esBrowseButton_,
          &QPushButton::clicked,
          this,
          &CardiacMriClassificationWindow::browseEsVolume);

  connect(classifyButton_,
          &QPushButton::clicked,
          this,
          &CardiacMriClassificationWindow::startClassification);

  /*
   * Keep the initial window deliberately compact. Phase 8 requires a standalone
   * classification workflow, not a replacement for the existing medical-image
   * viewer.
   */
  resize(680, 420);
}

void CardiacMriClassificationWindow::updateControls()
{
  const bool enabled = !workflow_.isRunning();

  edPathEdit_->setEnabled(enabled);
  edBrowseButton_->setEnabled(enabled);
  esPathEdit_->setEnabled(enabled);
  esBrowseButton_->setEnabled(enabled);
  classifyButton_->setEnabled(enabled);
}

} // namespace maiw::qt
