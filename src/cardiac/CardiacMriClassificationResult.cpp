#include "maiw/cardiac/CardiacMriClassificationResult.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace
{

constexpr double kNormalizationTolerance = 1e-12;

void validateClassNames(const maiw::cardiac::CardiacMriDeploymentMetadata::ClassNames& classNames)
{
  std::unordered_set<std::string> uniqueNames;
  for (const std::string& name : classNames)
  {
    if (name.empty())
    {
      throw std::invalid_argument("Cardiac classification class names must not be empty");
    }
    if (!uniqueNames.insert(name).second)
    {
      throw std::invalid_argument("Cardiac classification class names must be unique");
    }
  }
}

template <typename Values>
std::size_t firstMaximumIndex(const Values& values)
{
  return static_cast<std::size_t>(std::distance(values.begin(),
                                                std::max_element(values.begin(), values.end())));
}

} // namespace

namespace maiw::cardiac
{

CardiacMriClassificationResult CardiacMriClassificationResult::fromLogits(
    std::span<const float> logits,
    const CardiacMriDeploymentMetadata::ClassNames& classNames)
{
  if (logits.size() != CardiacMriDeploymentMetadata::kClassCount)
  {
    throw std::invalid_argument("Cardiac classification requires exactly five logits");
  }
  validateClassNames(classNames);

  RawLogits rawLogits;
  std::copy(logits.begin(), logits.end(), rawLogits.begin());
  for (const float logit : rawLogits)
  {
    if (!std::isfinite(logit))
    {
      throw std::invalid_argument("Cardiac classification logits must be finite");
    }
  }

  const float maximumLogit = *std::max_element(rawLogits.begin(), rawLogits.end());
  Probabilities probabilities;
  double denominator = 0.0;
  for (std::size_t index = 0; index < rawLogits.size(); ++index)
  {
    const double exponent = std::exp(static_cast<double>(rawLogits[index]) -
                                     static_cast<double>(maximumLogit));
    if (!std::isfinite(exponent) || exponent < 0.0)
    {
      throw std::runtime_error("Stable softmax produced an invalid exponential term");
    }
    probabilities[index] = exponent;
    denominator += exponent;
  }
  if (!std::isfinite(denominator) || denominator <= 0.0)
  {
    throw std::runtime_error("Stable softmax produced an invalid denominator");
  }

  for (double& probability : probabilities)
  {
    probability /= denominator;
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
    {
      throw std::runtime_error("Stable softmax produced an invalid probability");
    }
  }

  const double probabilitySum =
      std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
  if (!std::isfinite(probabilitySum) ||
      std::fabs(probabilitySum - 1.0) > kNormalizationTolerance)
  {
    throw std::runtime_error("Stable softmax probabilities are not normalized");
  }

  const std::size_t rawArgmax = firstMaximumIndex(rawLogits);
  const std::size_t probabilityArgmax = firstMaximumIndex(probabilities);
  if (rawArgmax != probabilityArgmax)
  {
    throw std::runtime_error("Raw-logit and probability argmax values disagree");
  }

  return CardiacMriClassificationResult(std::move(rawLogits),
                                        std::move(probabilities),
                                        rawArgmax,
                                        classNames[rawArgmax]);
}

const CardiacMriClassificationResult::RawLogits&
CardiacMriClassificationResult::rawLogits() const noexcept
{
  return rawLogits_;
}

const CardiacMriClassificationResult::Probabilities&
CardiacMriClassificationResult::probabilities() const noexcept
{
  return probabilities_;
}

std::size_t CardiacMriClassificationResult::predictedClassIndex() const noexcept
{
  return predictedClassIndex_;
}

const std::string& CardiacMriClassificationResult::predictedClassName() const noexcept
{
  return predictedClassName_;
}

CardiacMriClassificationResult::CardiacMriClassificationResult(
    RawLogits rawLogits,
    Probabilities probabilities,
    std::size_t predictedClassIndex,
    std::string predictedClassName)
  : rawLogits_(std::move(rawLogits)),
    probabilities_(std::move(probabilities)),
    predictedClassIndex_(predictedClassIndex),
    predictedClassName_(std::move(predictedClassName))
{
}

} // namespace maiw::cardiac
