#include "maiw/onnx/OnnxRuntimeSession.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace
{

Ort::SessionOptions makeDefaultSessionOptions()
{
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  return options;
}

maiw::onnx::TensorElementType toElementType(ONNXTensorElementDataType type)
{
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
  {
    return maiw::onnx::TensorElementType::Float32;
  }
  return maiw::onnx::TensorElementType::Unknown;
}

std::vector<maiw::onnx::TensorInfo> inspectInputs(Ort::Session& session)
{
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<maiw::onnx::TensorInfo> result;
  result.reserve(session.GetInputCount());

  for (std::size_t index = 0; index < session.GetInputCount(); ++index)
  {
    auto name = session.GetInputNameAllocated(index, allocator);
    auto typeInfo = session.GetInputTypeInfo(index);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    result.push_back(maiw::onnx::TensorInfo{
        name.get(),
        toElementType(tensorInfo.GetElementType()),
        tensorInfo.GetShape()});
  }

  return result;
}

std::vector<maiw::onnx::TensorInfo> inspectOutputs(Ort::Session& session)
{
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<maiw::onnx::TensorInfo> result;
  result.reserve(session.GetOutputCount());

  for (std::size_t index = 0; index < session.GetOutputCount(); ++index)
  {
    auto name = session.GetOutputNameAllocated(index, allocator);
    auto typeInfo = session.GetOutputTypeInfo(index);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    result.push_back(maiw::onnx::TensorInfo{
        name.get(),
        toElementType(tensorInfo.GetElementType()),
        tensorInfo.GetShape()});
  }

  return result;
}

std::size_t checkedElementCount(std::span<const std::int64_t> shape)
{
  if (shape.empty())
  {
    throw std::invalid_argument("Tensor shape must not be empty");
  }

  std::size_t count = 1;
  for (const std::int64_t dimension : shape)
  {
    if (dimension <= 0)
    {
      throw std::invalid_argument("Runtime tensor shape contains a non-positive dimension");
    }
    const auto value = static_cast<std::size_t>(dimension);
    if (count > std::numeric_limits<std::size_t>::max() / value)
    {
      throw std::overflow_error("Tensor element count exceeds size limits");
    }
    count *= value;
  }
  return count;
}

const maiw::onnx::TensorInfo& requireTensor(const std::vector<maiw::onnx::TensorInfo>& tensors,
                                            std::string_view name,
                                            const char* tensorKind)
{
  const auto found = std::find_if(tensors.begin(), tensors.end(), [name](const auto& tensor) {
    return tensor.name == name;
  });
  if (found == tensors.end())
  {
    throw std::invalid_argument(std::string("Requested ONNX ") + tensorKind +
                                " name is not present in the model");
  }
  return *found;
}

void validateFloatTensorCompatibility(const maiw::onnx::TensorInfo& modelTensor,
                                      std::span<const std::int64_t> runtimeShape,
                                      const char* tensorKind)
{
  if (modelTensor.elementType != maiw::onnx::TensorElementType::Float32)
  {
    throw std::invalid_argument(std::string("Requested ONNX ") + tensorKind +
                                " is not a float32 tensor");
  }
  if (modelTensor.shape.size() != runtimeShape.size())
  {
    throw std::invalid_argument(std::string("Runtime ") + tensorKind +
                                " rank does not match the model metadata");
  }

  for (std::size_t index = 0; index < runtimeShape.size(); ++index)
  {
    if (runtimeShape[index] <= 0)
    {
      throw std::invalid_argument(std::string("Runtime ") + tensorKind +
                                  " shape contains a non-positive dimension");
    }
    const std::int64_t modelDimension = modelTensor.shape[index];
    if (modelDimension > 0 && modelDimension != runtimeShape[index])
    {
      throw std::invalid_argument(std::string("Runtime ") + tensorKind +
                                  " shape is incompatible with the model metadata");
    }
  }
}

} // namespace

namespace maiw::onnx
{

OnnxRuntimeSession::OnnxRuntimeSession(Ort::Env& environment,
                                       const std::filesystem::path& modelPath)
    : sessionOptions_(makeDefaultSessionOptions()),
      session_(environment, modelPath.string().c_str(), sessionOptions_)
{
  inputs_ = inspectInputs(session_);
  outputs_ = inspectOutputs(session_);
}

const std::vector<TensorInfo>& OnnxRuntimeSession::inputs() const noexcept
{
  return inputs_;
}

const std::vector<TensorInfo>& OnnxRuntimeSession::outputs() const noexcept
{
  return outputs_;
}

FloatTensor OnnxRuntimeSession::run(std::string_view inputName,
                                    std::span<const float> inputData,
                                    std::span<const std::int64_t> inputShape,
                                    std::string_view outputName)
{
  const TensorInfo& inputInfo = requireTensor(inputs_, inputName, "input");
  const TensorInfo& outputInfo = requireTensor(outputs_, outputName, "output");
  validateFloatTensorCompatibility(inputInfo, inputShape, "input");
  if (outputInfo.elementType != TensorElementType::Float32)
  {
    throw std::invalid_argument("Requested ONNX output is not a float32 tensor");
  }

  const std::size_t expectedElements = checkedElementCount(inputShape);
  if (inputData.size() != expectedElements)
  {
    throw std::invalid_argument("Input tensor data size does not match its shape");
  }

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo,
      const_cast<float*>(inputData.data()),
      inputData.size(),
      inputShape.data(),
      inputShape.size());

  if (!inputTensor.IsTensor())
  {
    throw std::runtime_error("Failed to create ONNX Runtime input tensor");
  }

  const std::string inputNameString(inputName);
  const std::string outputNameString(outputName);
  const char* inputNames[] = {inputNameString.c_str()};
  const char* outputNames[] = {outputNameString.c_str()};

  auto outputTensors =
      session_.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
  if (outputTensors.size() != 1 || !outputTensors.front().IsTensor())
  {
    throw std::runtime_error("ONNX Runtime returned an unexpected output tensor set");
  }

  auto runtimeOutputInfo = outputTensors.front().GetTensorTypeAndShapeInfo();
  if (runtimeOutputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
  {
    throw std::runtime_error("ONNX Runtime output tensor is not float32");
  }

  const std::vector<std::int64_t> outputShape = runtimeOutputInfo.GetShape();
  const std::size_t outputElements = checkedElementCount(outputShape);
  const float* outputData = outputTensors.front().GetTensorData<float>();

  return FloatTensor{
      std::vector<float>(outputData, outputData + outputElements),
      outputShape};
}

} // namespace maiw::onnx
