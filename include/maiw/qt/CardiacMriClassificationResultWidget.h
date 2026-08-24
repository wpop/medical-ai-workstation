#pragma once

#include "maiw/cardiac/CardiacMriClassificationResult.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"

#include <QString>
#include <QWidget>

#include <array>

class QLabel;

namespace maiw::qt
{

/**
 * @brief Presents a validated cardiac MRI classification result in Qt.
 *
 * The widget displays the predicted class and per-class probabilities in the
 * validated deployment order. It performs no inference, probability
 * computation, class selection, or class mapping.
 */
class CardiacMriClassificationResultWidget final : public QWidget
{
public:
  /**
   * @brief Construct the result widget using validated deployment class names.
   *
   * @param classNames Class names in the deployed model output order.
   * @param parent Optional parent widget.
   */
  explicit CardiacMriClassificationResultWidget(
      cardiac::CardiacMriDeploymentMetadata::ClassNames classNames,
      QWidget* parent = nullptr);

  /**
   * @brief Display a completed cardiac MRI classification result.
   *
   * Probabilities are read directly from the validated classification result.
   *
   * @param result Classification result produced by the application service.
   */
  void setResult(const cardiac::CardiacMriClassificationResult& result);

  /**
   * @brief Display a controlled classification error.
   *
   * @param message User-presentable error message.
   */
  void showError(const QString& message);

  /**
   * @brief Clear the current result or error and return to the idle state.
   */
  void clear();

  /**
   * @brief Return the currently displayed predicted-class text.
   */
  [[nodiscard]] QString predictedClassText() const;

  /**
   * @brief Return the displayed probability texts in deployment class order.
   */
  [[nodiscard]] std::array<QString, cardiac::CardiacMriDeploymentMetadata::kClassCount>
  probabilityTexts() const;

  /**
   * @brief Return the currently displayed status or error text.
   */
  [[nodiscard]] QString statusText() const;

private:
  void initializeUi();
  void clearProbabilities();

  cardiac::CardiacMriDeploymentMetadata::ClassNames classNames_;

  // Qt parent ownership manages these widgets; the pointers are non-owning.
  QLabel* predictedClassValueLabel_ = nullptr;
  std::array<QLabel*, cardiac::CardiacMriDeploymentMetadata::kClassCount>
      probabilityValueLabels_{};
  QLabel* statusLabel_ = nullptr;
};

} // namespace maiw::qt
