#include "maiw/cardiac/CardiacMriClassificationResult.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationResultWidget.h"

#include <QApplication>
#include <QString>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

using maiw::cardiac::CardiacMriClassificationResult;
using maiw::cardiac::CardiacMriDeploymentMetadata;
using maiw::qt::CardiacMriClassificationResultWidget;

const CardiacMriDeploymentMetadata::ClassNames kClassNames{
    "Normal",
    "Dilated Cardiomyopathy",
    "Hypertrophic Cardiomyopathy",
    "Myocardial Infarction",
    "Abnormal Right Ventricle"};

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void requireIdlePresentation(const CardiacMriClassificationResultWidget& widget,
                             const std::string& context)
{
  require(widget.predictedClassText() == QStringLiteral("—"),
          context + ": predicted class is not cleared");
  for (const QString& probabilityText : widget.probabilityTexts())
  {
    require(probabilityText == QStringLiteral("—"),
            context + ": probability is not cleared");
  }
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);

    CardiacMriClassificationResultWidget widget(kClassNames);
    requireIdlePresentation(widget, "initial state");
    require(widget.statusText().isEmpty(), "initial status is not empty");

    const CardiacMriClassificationResult::RawLogits logits{
        0.0F, 0.69314718F, 1.0986123F, 1.3862944F, 1.6094379F};
    const CardiacMriClassificationResult result =
        CardiacMriClassificationResult::fromLogits(logits, kClassNames);

    widget.setResult(result);

    require(widget.predictedClassText() ==
                QString::fromStdString(kClassNames[4]),
            "predicted class text mismatch");
    const std::array<QString, CardiacMriDeploymentMetadata::kClassCount>
        expectedProbabilityTexts{
            QStringLiteral("6.7 %"),
            QStringLiteral("13.3 %"),
            QStringLiteral("20.0 %"),
            QStringLiteral("26.7 %"),
            QStringLiteral("33.3 %")};
    require(widget.probabilityTexts() == expectedProbabilityTexts,
            "displayed probability values mismatch");
    require(widget.statusText().isEmpty(),
            "successful result status is not empty");

    widget.clear();
    requireIdlePresentation(widget, "clear state");
    require(widget.statusText().isEmpty(), "clear state status is not empty");

    const QString controlledError =
        QStringLiteral("The selected cardiac MRI volumes could not be classified.");
    widget.showError(controlledError);
    requireIdlePresentation(widget, "error state");
    require(widget.statusText() == controlledError,
            "controlled error text mismatch");

    std::cout << "Cardiac MRI classification result widget test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI classification result widget test failed: "
              << error.what() << '\n';
    return 1;
  }
}
