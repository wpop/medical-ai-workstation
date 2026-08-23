#include "maiw/cardiac/CardiacMriClassificationService.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace
{

bool acceptsSingleSample(std::int64_t batchDimension) noexcept
{
  return batchDimension == 1 || batchDimension < 0;
}

void validateInputContract(
    const maiw::onnx::TensorInfo& input,
    const maiw::cardiac::CardiacMriDeploymentMetadata& metadata)
{
  if (input.name != metadata.inputName())
  {
    throw std::runtime_error("ONNX model input name does not match deployment metadata");
  }
  if (input.elementType != maiw::onnx::TensorElementType::Float32)
  {
    throw std::runtime_error("Cardiac ONNX model input must use float32 elements");
  }
  if (input.shape.size() != 5)
  {
    throw std::runtime_error("Cardiac ONNX model input must have rank 5");
  }
  if (!acceptsSingleSample(input.shape[0]))
  {
    throw std::runtime_error("Cardiac ONNX model input batch dimension must accept one sample");
  }

  const std::array<std::int64_t, 4> expectedCdhw{
      static_cast<std::int64_t>(maiw::cardiac::CardiacMriInputTensor::kChannels),
      static_cast<std::int64_t>(maiw::cardiac::CardiacMriInputTensor::kDepth),
      static_cast<std::int64_t>(maiw::cardiac::CardiacMriInputTensor::kHeight),
      static_cast<std::int64_t>(maiw::cardiac::CardiacMriInputTensor::kWidth)};
  for (std::size_t index = 0; index < expectedCdhw.size(); ++index)
  {
    if (input.shape[index + 1] != expectedCdhw[index])
    {
      throw std::runtime_error("Cardiac ONNX model input CDHW shape is incompatible with "
                               "production preprocessing");
    }
  }
}

void validateOutputContract(
    const maiw::onnx::TensorInfo& output,
    const maiw::cardiac::CardiacMriDeploymentMetadata& metadata)
{
  if (output.name != metadata.outputName())
  {
    throw std::runtime_error("ONNX model output name does not match deployment metadata");
  }
  if (output.elementType != maiw::onnx::TensorElementType::Float32)
  {
    throw std::runtime_error("Cardiac ONNX model output must use float32 elements");
  }
  if (output.shape.size() != 2)
  {
    throw std::runtime_error("Cardiac ONNX model output must have rank 2");
  }
  if (!acceptsSingleSample(output.shape[0]))
  {
    throw std::runtime_error("Cardiac ONNX model output batch dimension must accept one sample");
  }
  if (output.shape[1] != static_cast<std::int64_t>(
                             maiw::cardiac::CardiacMriDeploymentMetadata::kClassCount))
  {
    throw std::runtime_error("Cardiac ONNX model output class dimension is invalid");
  }
}

} // namespace

namespace maiw::cardiac
{

CardiacMriClassificationService::CardiacMriClassificationService(
    Ort::Env& environment,
    CardiacMriDeploymentMetadata metadata)
  : metadata_(std::move(metadata)),
    preprocessor_(),
    session_(environment, metadata_.modelPath())
{
  validateModelContract();
}

CardiacMriClassificationResult CardiacMriClassificationService::classify(
    const qvp::VolumeData& edVolume,
    const qvp::VolumeData& esVolume)
{
  const CardiacMriInputTensor inputTensor = preprocessor_.preprocess(edVolume, esVolume);
  const std::array<std::int64_t, 5> inputShape{
      1,
      static_cast<std::int64_t>(CardiacMriInputTensor::kChannels),
      static_cast<std::int64_t>(CardiacMriInputTensor::kDepth),
      static_cast<std::int64_t>(CardiacMriInputTensor::kHeight),
      static_cast<std::int64_t>(CardiacMriInputTensor::kWidth)};

  const maiw::onnx::FloatTensor output = session_.run(metadata_.inputName(),
                                                      inputTensor.span(),
                                                      inputShape,
                                                      metadata_.outputName());
  const std::array<std::int64_t, 2> expectedOutputShape{
      1, static_cast<std::int64_t>(CardiacMriDeploymentMetadata::kClassCount)};
  if (output.shape.size() != expectedOutputShape.size() ||
      !std::equal(output.shape.begin(), output.shape.end(), expectedOutputShape.begin()))
  {
    throw std::runtime_error("ONNX Runtime returned an unexpected cardiac logits shape");
  }
  if (output.data.size() != CardiacMriDeploymentMetadata::kClassCount)
  {
    throw std::runtime_error("ONNX Runtime returned an unexpected cardiac logits element count");
  }

  return CardiacMriClassificationResult::fromLogits(output.data, metadata_.classNames());
}

void CardiacMriClassificationService::validateModelContract() const
{
  if (session_.inputs().size() != 1)
  {
    throw std::runtime_error("Cardiac ONNX model must expose exactly one input");
  }
  if (session_.outputs().size() != 1)
  {
    throw std::runtime_error("Cardiac ONNX model must expose exactly one output");
  }

  validateInputContract(session_.inputs().front(), metadata_);
  validateOutputContract(session_.outputs().front(), metadata_);
}

} // namespace maiw::cardiac
