#pragma once

#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"

#include <array>
#include <cstddef>
#include <span>
#include <string>

namespace maiw::cardiac
{

/**
 * @brief Immutable cardiac MRI classification output with validated probabilities.
 */
class CardiacMriClassificationResult final
{
public:
  using RawLogits = std::array<float, CardiacMriDeploymentMetadata::kClassCount>;
  using Probabilities = std::array<double, CardiacMriDeploymentMetadata::kClassCount>;

  /**
   * @brief Create a validated result and compute stable softmax probabilities.
   * @throws std::invalid_argument If logits or class names violate the result contract.
   * @throws std::runtime_error If a finite normalized probability distribution cannot be computed.
   */
  [[nodiscard]] static CardiacMriClassificationResult fromLogits(
      std::span<const float> logits,
      const CardiacMriDeploymentMetadata::ClassNames& classNames);

  /**
   * @brief Return the original float logits without numerical conversion.
   */
  [[nodiscard]] const RawLogits& rawLogits() const noexcept;

  /**
   * @brief Return double-precision softmax probabilities in deployment class order.
   */
  [[nodiscard]] const Probabilities& probabilities() const noexcept;

  /**
   * @brief Return the deterministic first-maximum predicted class index.
   */
  [[nodiscard]] std::size_t predictedClassIndex() const noexcept;

  /**
   * @brief Return the predicted class name supplied in deployment class order.
   */
  [[nodiscard]] const std::string& predictedClassName() const noexcept;

private:
  CardiacMriClassificationResult(RawLogits rawLogits,
                                 Probabilities probabilities,
                                 std::size_t predictedClassIndex,
                                 std::string predictedClassName);

  RawLogits rawLogits_;
  Probabilities probabilities_;
  std::size_t predictedClassIndex_;
  std::string predictedClassName_;
};

} // namespace maiw::cardiac
