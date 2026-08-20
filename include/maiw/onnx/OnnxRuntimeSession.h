#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace maiw::onnx
{

enum class TensorElementType
{
  Unknown,
  Float32
};

struct TensorInfo
{
  std::string name;
  TensorElementType elementType = TensorElementType::Unknown;
  std::vector<std::int64_t> shape;
};

struct FloatTensor
{
  std::vector<float> data;
  std::vector<std::int64_t> shape;
};

class OnnxRuntimeSession
{
public:
  OnnxRuntimeSession(Ort::Env& environment, const std::filesystem::path& modelPath);

  const std::vector<TensorInfo>& inputs() const noexcept;
  const std::vector<TensorInfo>& outputs() const noexcept;

  FloatTensor run(std::string_view inputName,
                  std::span<const float> inputData,
                  std::span<const std::int64_t> inputShape,
                  std::string_view outputName);

private:
  Ort::SessionOptions sessionOptions_;
  Ort::Session session_;
  std::vector<TensorInfo> inputs_;
  std::vector<TensorInfo> outputs_;
};

} // namespace maiw::onnx
