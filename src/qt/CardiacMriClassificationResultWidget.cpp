#include "maiw/qt/CardiacMriClassificationResultWidget.h"

#include <QFormLayout>
#include <QLabel>
#include <QString>
#include <QVBoxLayout>

#include <cstddef>
#include <utility>

namespace maiw::qt
{

namespace
{

QString formatProbability(double probability)
{
  return QStringLiteral("%1 %").arg(probability * 100.0, 0, 'f', 1);
}

} // namespace

CardiacMriClassificationResultWidget::CardiacMriClassificationResultWidget(
    cardiac::CardiacMriDeploymentMetadata::ClassNames classNames,
    QWidget* parent)
    : QWidget(parent),
      classNames_(std::move(classNames))
{
  initializeUi();
  clear();
}

void CardiacMriClassificationResultWidget::setResult(
    const cardiac::CardiacMriClassificationResult& result)
{
  predictedClassValueLabel_->setText(
      QString::fromStdString(result.predictedClassName()));

  const auto& probabilities = result.probabilities();

  for (std::size_t index = 0; index < probabilityValueLabels_.size(); ++index)
  {
    probabilityValueLabels_[index]->setText(
        formatProbability(probabilities[index]));
  }

  statusLabel_->clear();
}

void CardiacMriClassificationResultWidget::showError(const QString& message)
{
  predictedClassValueLabel_->setText(QStringLiteral("—"));
  clearProbabilities();

  statusLabel_->setText(
      message.isEmpty()
          ? QStringLiteral("Classification failed.")
          : message);
}

void CardiacMriClassificationResultWidget::clear()
{
  predictedClassValueLabel_->setText(QStringLiteral("—"));
  clearProbabilities();
  statusLabel_->clear();
}

QString CardiacMriClassificationResultWidget::predictedClassText() const
{
  return predictedClassValueLabel_->text();
}

std::array<QString, cardiac::CardiacMriDeploymentMetadata::kClassCount>
CardiacMriClassificationResultWidget::probabilityTexts() const
{
  std::array<QString, cardiac::CardiacMriDeploymentMetadata::kClassCount> texts;
  for (std::size_t index = 0; index < texts.size(); ++index)
  {
    texts[index] = probabilityValueLabels_[index]->text();
  }
  return texts;
}

QString CardiacMriClassificationResultWidget::statusText() const
{
  return statusLabel_->text();
}

void CardiacMriClassificationResultWidget::initializeUi()
{
  auto* mainLayout = new QVBoxLayout(this);

  auto* predictionLayout = new QFormLayout();
  predictedClassValueLabel_ = new QLabel(this);

  predictionLayout->addRow(
      QStringLiteral("Predicted class:"),
      predictedClassValueLabel_);

  mainLayout->addLayout(predictionLayout);

  auto* probabilityLayout = new QFormLayout();

  for (std::size_t index = 0; index < classNames_.size(); ++index)
  {
    auto* probabilityLabel = new QLabel(this);
    probabilityValueLabels_[index] = probabilityLabel;

    probabilityLayout->addRow(
        QString::fromStdString(classNames_[index]) + QStringLiteral(":"),
        probabilityLabel);
  }

  mainLayout->addLayout(probabilityLayout);

  statusLabel_ = new QLabel(this);
  statusLabel_->setWordWrap(true);
  mainLayout->addWidget(statusLabel_);
}

void CardiacMriClassificationResultWidget::clearProbabilities()
{
  for (QLabel* label : probabilityValueLabels_)
  {
    label->setText(QStringLiteral("—"));
  }
}

} // namespace maiw::qt
