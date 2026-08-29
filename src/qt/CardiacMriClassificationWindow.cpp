#include "maiw/qt/CardiacMriClassificationWindow.h"

#include "maiw/qt/CardiacMriClassificationResultWidget.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"

#include <QEvent>
#include <QFileDialog>
#include <QFont>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPushButton>
#include <QSizePolicy>
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

  auto* resultGroup = new QGroupBox(QStringLiteral("AI result"), this);
  resultGroup->setObjectName(QStringLiteral("cardiacAiResultGroup"));
  resultWidget_ = new CardiacMriClassificationResultWidget(
      std::move(classNames), resultGroup);
  resultWidget_->setObjectName(
      QStringLiteral("cardiacClassificationResultWidget"));
  auto* resultLayout = new QVBoxLayout(resultGroup);
  resultLayout->addWidget(resultWidget_);

  auto* mainLayout = qobject_cast<QVBoxLayout*>(layout());
  mainLayout->addWidget(resultGroup);
  mainLayout->addStretch();

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
  if (edPathEdit_->hasFocus())
  {
    suppressNextEdEditingCommit_ = true;
  }

  const QString selectedPath =
      QFileDialog::getOpenFileName(this,
                                   QStringLiteral("Select End-Diastolic Volume"),
                                   edPathEdit_->text(),
                                   medicalVolumeFileFilter());
  suppressNextEdEditingCommit_ = false;
  publishEdBrowseSelection(selectedPath);
}

void CardiacMriClassificationWindow::browseEsVolume()
{
  if (esPathEdit_->hasFocus())
  {
    suppressNextEsEditingCommit_ = true;
  }

  const QString selectedPath =
      QFileDialog::getOpenFileName(this,
                                   QStringLiteral("Select End-Systolic Volume"),
                                   esPathEdit_->text(),
                                   medicalVolumeFileFilter());
  suppressNextEsEditingCommit_ = false;
  publishEsBrowseSelection(selectedPath);
}

void CardiacMriClassificationWindow::handleEdPathEditingFinished()
{
  if (std::exchange(suppressNextEdEditingCommit_, false))
  {
    return;
  }

  emit edVolumePathCommitted(edPathEdit_->text());
}

void CardiacMriClassificationWindow::handleEsPathEditingFinished()
{
  if (std::exchange(suppressNextEsEditingCommit_, false))
  {
    return;
  }

  emit esVolumePathCommitted(esPathEdit_->text());
}

void CardiacMriClassificationWindow::publishEdBrowseSelection(
    const QString& selectedPath)
{
  if (selectedPath.isEmpty())
  {
    return;
  }

  edPathEdit_->setText(selectedPath);
  edPathEdit_->setCursorPosition(static_cast<int>(selectedPath.size()));
  emit edVolumePathCommitted(selectedPath);
}

void CardiacMriClassificationWindow::publishEsBrowseSelection(
    const QString& selectedPath)
{
  if (selectedPath.isEmpty())
  {
    return;
  }

  esPathEdit_->setText(selectedPath);
  esPathEdit_->setCursorPosition(static_cast<int>(selectedPath.size()));
  emit esVolumePathCommitted(selectedPath);
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
  mainLayout->setSpacing(10);

  auto* headerLabel =
      new QLabel(QStringLiteral("Cardiac MRI Classification"), this);
  headerLabel->setObjectName(QStringLiteral("cardiacClassificationHeader"));
  QFont headerFont = headerLabel->font();
  headerFont.setBold(true);
  if (headerFont.pointSizeF() > 0.0)
  {
    headerFont.setPointSizeF(headerFont.pointSizeF() + 2.0);
  }
  headerLabel->setFont(headerFont);
  mainLayout->addWidget(headerLabel);

  auto* helperLabel = new QLabel(
      QStringLiteral("Select end-diastolic and end-systolic volumes, then run "
                     "the validated cardiac MRI classifier."),
      this);
  helperLabel->setObjectName(
      QStringLiteral("cardiacClassificationHelperText"));
  helperLabel->setWordWrap(true);
  mainLayout->addWidget(helperLabel);

  auto* studyVolumesGroup =
      new QGroupBox(QStringLiteral("Study volumes"), this);
  studyVolumesGroup->setObjectName(
      QStringLiteral("cardiacStudyVolumesGroup"));
  auto* inputLayout = new QVBoxLayout(studyVolumesGroup);

  /*
   * Each path row contains a text editor and a browse button. All widgets are
   * parent-owned by Qt through the layout/widget hierarchy; stored pointers are
   * non-owning convenience handles only.
   */
  auto* edRowWidget = new QWidget(this);
  auto* edRowLayout = new QHBoxLayout(edRowWidget);
  edRowLayout->setContentsMargins(0, 0, 0, 0);

  edPathEdit_ = new QLineEdit(edRowWidget);
  edPathEdit_->setObjectName(QStringLiteral("cardiacEdVolumePathEdit"));
  edPathEdit_->setPlaceholderText(
      QStringLiteral("Select the end-diastolic medical volume"));

  edBrowseButton_ = new QPushButton(QStringLiteral("Browse..."), edRowWidget);
  edBrowseButton_->setObjectName(QStringLiteral("cardiacEdVolumeBrowseButton"));
  edBrowseButton_->installEventFilter(this);

  edRowLayout->addWidget(edPathEdit_);
  edRowLayout->addWidget(edBrowseButton_);

  inputLayout->addWidget(
      new QLabel(QStringLiteral("End-diastolic (ED) volume"),
                 studyVolumesGroup));
  inputLayout->addWidget(edRowWidget);

  auto* esRowWidget = new QWidget(this);
  auto* esRowLayout = new QHBoxLayout(esRowWidget);
  esRowLayout->setContentsMargins(0, 0, 0, 0);

  esPathEdit_ = new QLineEdit(esRowWidget);
  esPathEdit_->setObjectName(QStringLiteral("cardiacEsVolumePathEdit"));
  esPathEdit_->setPlaceholderText(
      QStringLiteral("Select the end-systolic medical volume"));

  esBrowseButton_ = new QPushButton(QStringLiteral("Browse..."), esRowWidget);
  esBrowseButton_->setObjectName(QStringLiteral("cardiacEsVolumeBrowseButton"));
  esBrowseButton_->installEventFilter(this);

  esRowLayout->addWidget(esPathEdit_);
  esRowLayout->addWidget(esBrowseButton_);

  inputLayout->addWidget(
      new QLabel(QStringLiteral("End-systolic (ES) volume"),
                 studyVolumesGroup));
  inputLayout->addWidget(esRowWidget);

  mainLayout->addWidget(studyVolumesGroup);

  classifyButton_ =
      new QPushButton(QStringLiteral("Classify Cardiac MRI"), this);
  classifyButton_->setObjectName(QStringLiteral("cardiacClassifyButton"));
  QFont classifyFont = classifyButton_->font();
  classifyFont.setBold(true);
  classifyButton_->setFont(classifyFont);
  classifyButton_->setDefault(true);
  classifyButton_->setMinimumHeight(classifyButton_->sizeHint().height() + 8);
  classifyButton_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  mainLayout->addWidget(classifyButton_);

  statusLabel_ = new QLabel(this);
  statusLabel_->setObjectName(
      QStringLiteral("cardiacClassificationStatusLabel"));
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

  connect(edPathEdit_,
          &QLineEdit::editingFinished,
          this,
          &CardiacMriClassificationWindow::handleEdPathEditingFinished);

  connect(esPathEdit_,
          &QLineEdit::editingFinished,
          this,
          &CardiacMriClassificationWindow::handleEsPathEditingFinished);

  connect(edPathEdit_,
          &QLineEdit::textChanged,
          edPathEdit_,
          &QWidget::setToolTip);

  connect(esPathEdit_,
          &QLineEdit::textChanged,
          esPathEdit_,
          &QWidget::setToolTip);

  connect(classifyButton_,
          &QPushButton::clicked,
          this,
          &CardiacMriClassificationWindow::startClassification);
}

bool CardiacMriClassificationWindow::eventFilter(QObject* watched, QEvent* event)
{
  if (event->type() == QEvent::MouseButtonPress &&
      static_cast<QMouseEvent*>(event)->button() == Qt::LeftButton)
  {
    if (watched == edBrowseButton_ && edPathEdit_->hasFocus())
    {
      suppressNextEdEditingCommit_ = true;
    }
    else if (watched == esBrowseButton_ && esPathEdit_->hasFocus())
    {
      suppressNextEsEditingCommit_ = true;
    }
  }
  else if (event->type() == QEvent::MouseButtonRelease &&
           static_cast<QMouseEvent*>(event)->button() == Qt::LeftButton)
  {
    if (watched == edBrowseButton_ && edPathEdit_->hasFocus())
    {
      suppressNextEdEditingCommit_ = false;
    }
    else if (watched == esBrowseButton_ && esPathEdit_->hasFocus())
    {
      suppressNextEsEditingCommit_ = false;
    }
  }

  return QWidget::eventFilter(watched, event);
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
