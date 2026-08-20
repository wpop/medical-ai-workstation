#include "NpyReader.h"
#include "Sha256.h"
#include "maiw/onnx/OnnxRuntimeSession.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Json = nlohmann::json;

constexpr const char* kExpectedInputName = "cine_mri";
constexpr const char* kExpectedOutputName = "logits";
constexpr double kExpectedTolerance = 1e-5;

const std::vector<std::size_t> kFixtureInputShape{1, 2, 14, 144, 144};
const std::vector<std::size_t> kFixtureLogitsShape{1, 5};
const std::vector<std::int64_t> kRuntimeInputShape{1, 2, 14, 144, 144};

Json readJson(const std::filesystem::path& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::runtime_error("Failed to open JSON file: " + path.string());
  }

  Json json;
  input >> json;
  return json;
}

const Json& requireObjectMember(const Json& object, const char* key)
{
  if (!object.is_object() || !object.contains(key))
  {
    throw std::runtime_error(std::string("JSON object is missing key: ") + key);
  }
  return object.at(key);
}

std::string requireString(const Json& object, const char* key)
{
  const Json& value = requireObjectMember(object, key);
  if (!value.is_string())
  {
    throw std::runtime_error(std::string("JSON value must be a string: ") + key);
  }
  return value.get<std::string>();
}

int requireInt(const Json& object, const char* key)
{
  const Json& value = requireObjectMember(object, key);
  if (!value.is_number_integer())
  {
    throw std::runtime_error(std::string("JSON value must be an integer: ") + key);
  }
  return value.get<int>();
}

double requireDouble(const Json& object, const char* key)
{
  const Json& value = requireObjectMember(object, key);
  if (!value.is_number())
  {
    throw std::runtime_error(std::string("JSON value must be numeric: ") + key);
  }
  return value.get<double>();
}

std::vector<std::size_t> requireShape(const Json& object, const char* key)
{
  const Json& value = requireObjectMember(object, key);
  if (!value.is_array())
  {
    throw std::runtime_error(std::string("JSON shape value must be an array: ") + key);
  }

  std::vector<std::size_t> shape;
  shape.reserve(value.size());
  for (const auto& dimension : value)
  {
    if (!dimension.is_number_unsigned() && !dimension.is_number_integer())
    {
      throw std::runtime_error(std::string("JSON shape dimension must be an integer: ") + key);
    }
    const auto parsed = dimension.get<long long>();
    if (parsed <= 0)
    {
      throw std::runtime_error(std::string("JSON shape dimension must be positive: ") + key);
    }
    shape.push_back(static_cast<std::size_t>(parsed));
  }
  return shape;
}

std::size_t checkedElementCount(const std::vector<std::size_t>& shape)
{
  std::size_t count = 1;
  for (const std::size_t dimension : shape)
  {
    if (count > std::numeric_limits<std::size_t>::max() / dimension)
    {
      throw std::overflow_error("Shape element count exceeds size limits");
    }
    count *= dimension;
  }
  return count;
}

void requireEqualShape(const std::vector<std::size_t>& actual,
                       const std::vector<std::size_t>& expected,
                       const std::string& context)
{
  if (actual != expected)
  {
    throw std::runtime_error(context + " shape mismatch");
  }
}

bool isCompatibleBatchDimension(std::int64_t dimension)
{
  return dimension < 0 || dimension == 1;
}

void validateModelInput(const maiw::onnx::TensorInfo& input)
{
  if (input.name != kExpectedInputName)
  {
    throw std::runtime_error("Model input name mismatch: " + input.name);
  }
  if (input.elementType != maiw::onnx::TensorElementType::Float32)
  {
    throw std::runtime_error("Model input element type is not float32");
  }
  if (input.shape.size() != 5)
  {
    throw std::runtime_error("Model input rank must be 5");
  }
  if (!isCompatibleBatchDimension(input.shape[0]) || input.shape[1] != 2 ||
      input.shape[2] != 14 || input.shape[3] != 144 || input.shape[4] != 144)
  {
    throw std::runtime_error("Model input shape is not compatible with [N,2,14,144,144]");
  }
}

void validateModelOutput(const maiw::onnx::TensorInfo& output)
{
  if (output.name != kExpectedOutputName)
  {
    throw std::runtime_error("Model output name mismatch: " + output.name);
  }
  if (output.elementType != maiw::onnx::TensorElementType::Float32)
  {
    throw std::runtime_error("Model output element type is not float32");
  }
  if (output.shape.size() != 2)
  {
    throw std::runtime_error("Model output rank must be 2");
  }
  if (!isCompatibleBatchDimension(output.shape[0]) || output.shape[1] != 5)
  {
    throw std::runtime_error("Model output shape is not compatible with [N,5]");
  }
}

int argmax(std::span<const float> values)
{
  if (values.empty())
  {
    throw std::runtime_error("Cannot compute argmax of an empty tensor");
  }
  return static_cast<int>(
      std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

struct ErrorSummary
{
  double maximumAbsoluteError = 0.0;
  std::size_t flatIndex = 0;
};

ErrorSummary compareLogits(std::span<const float> actual,
                           std::span<const float> expected,
                           double tolerance,
                           const std::string& patientId)
{
  if (actual.size() != expected.size())
  {
    throw std::runtime_error(patientId + ": actual/expected logits size mismatch");
  }

  ErrorSummary summary;
  for (std::size_t index = 0; index < actual.size(); ++index)
  {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
    {
      throw std::runtime_error(patientId + ": logits contain non-finite values");
    }
    const double absoluteError =
        std::fabs(static_cast<double>(actual[index]) - static_cast<double>(expected[index]));
    if (absoluteError > summary.maximumAbsoluteError)
    {
      summary.maximumAbsoluteError = absoluteError;
      summary.flatIndex = index;
    }
    if (absoluteError > tolerance)
    {
      throw std::runtime_error(patientId + ": logit absolute error exceeds tolerance at index " +
                               std::to_string(index));
    }
  }
  return summary;
}

void validateDeploymentContract(const Json& deployment)
{
  const Json& onnx = requireObjectMember(deployment, "onnx");
  const Json& input = requireObjectMember(onnx, "input");
  const Json& output = requireObjectMember(onnx, "output");
  if (requireString(input, "name") != kExpectedInputName)
  {
    throw std::runtime_error("deployment.json input name is not cine_mri");
  }
  if (requireString(input, "dtype") != "float32")
  {
    throw std::runtime_error("deployment.json input dtype is not float32");
  }
  if (requireString(output, "name") != kExpectedOutputName)
  {
    throw std::runtime_error("deployment.json output name is not logits");
  }
  if (requireString(output, "dtype") != "float32")
  {
    throw std::runtime_error("deployment.json output dtype is not float32");
  }
}

void requireFileSha256(const std::filesystem::path& path,
                       const std::string& expectedSha256,
                       const std::string& context)
{
  const std::string actualSha256 = maiw::test::sha256FileHex(path);
  if (actualSha256 != expectedSha256)
  {
    throw std::runtime_error(context + " SHA-256 mismatch: expected " + expectedSha256 +
                             ", actual " + actualSha256);
  }
}

} // namespace

int main()
{
  try
  {
    const std::filesystem::path packageRoot{MAIW_CARDIAC_MRI_PACKAGE_DIR};
    if (!std::filesystem::is_directory(packageRoot))
    {
      throw std::runtime_error("Package root does not exist: " + packageRoot.string());
    }

    const Json deployment = readJson(packageRoot / "deployment.json");
    const Json manifest = readJson(packageRoot / "golden/inference/manifest.json");
    validateDeploymentContract(deployment);

    const Json& onnx = requireObjectMember(deployment, "onnx");
    const std::filesystem::path modelPath = packageRoot / requireString(onnx, "filename");
    if (!std::filesystem::is_regular_file(modelPath))
    {
      throw std::runtime_error("ONNX model does not exist: " + modelPath.string());
    }
    requireFileSha256(modelPath, requireString(onnx, "sha256"), "classifier.onnx");

    const double tolerance = requireDouble(manifest, "absolute_tolerance");
    if (tolerance != kExpectedTolerance)
    {
      throw std::runtime_error("Manifest absolute tolerance is not 1e-5");
    }
    if (!requireObjectMember(manifest, "relative_tolerance").is_null())
    {
      throw std::runtime_error("Manifest relative tolerance must be null/disabled");
    }
    if (requireString(manifest, "input_name") != kExpectedInputName ||
        requireString(manifest, "output_name") != kExpectedOutputName)
    {
      throw std::runtime_error("Manifest input/output names do not match the ONNX contract");
    }

    Ort::Env environment(ORT_LOGGING_LEVEL_ERROR, "medical-ai-workstation-golden-parity");
    maiw::onnx::OnnxRuntimeSession session(environment, modelPath);
    if (session.inputs().size() != 1)
    {
      throw std::runtime_error("Model must expose exactly one input");
    }
    if (session.outputs().size() != 1)
    {
      throw std::runtime_error("Model must expose exactly one output");
    }
    validateModelInput(session.inputs().front());
    validateModelOutput(session.outputs().front());

    const Json& patients = requireObjectMember(manifest, "patients");
    if (!patients.is_array() || patients.size() != 7)
    {
      throw std::runtime_error("Golden inference manifest must contain exactly seven patients");
    }

    bool allPredictionsMatched = true;
    for (const auto& patient : patients)
    {
      const std::string patientId = requireString(patient, "patient_id");
      const auto manifestInputShape = requireShape(patient, "input_shape");
      const auto manifestLogitsShape = requireShape(patient, "logits_shape");
      requireEqualShape(manifestInputShape, kFixtureInputShape, patientId + " manifest input");
      requireEqualShape(manifestLogitsShape, kFixtureLogitsShape, patientId + " manifest logits");
      if (requireString(patient, "input_dtype") != "float32" ||
          requireString(patient, "logits_dtype") != "float32")
      {
        throw std::runtime_error(patientId + ": manifest dtype is not float32");
      }

      const std::filesystem::path inputPath =
          packageRoot / "golden/inference" / requireString(patient, "input_filename");
      const std::filesystem::path logitsPath =
          packageRoot / "golden/inference" / requireString(patient, "logits_filename");
      requireFileSha256(inputPath, requireString(patient, "input_sha256"), patientId + " input");
      requireFileSha256(logitsPath, requireString(patient, "logits_sha256"),
                        patientId + " logits");
      const auto inputArray = maiw::test::readNpyFloat32(inputPath);
      const auto expectedLogits = maiw::test::readNpyFloat32(logitsPath);
      requireEqualShape(inputArray.shape, kFixtureInputShape, patientId + " input fixture");
      requireEqualShape(expectedLogits.shape, kFixtureLogitsShape, patientId + " logits fixture");

      const auto actualLogits =
          session.run(kExpectedInputName, inputArray.data, kRuntimeInputShape, kExpectedOutputName);
      std::vector<std::size_t> actualLogitsShape;
      actualLogitsShape.reserve(actualLogits.shape.size());
      for (const auto dimension : actualLogits.shape)
      {
        if (dimension <= 0)
        {
          throw std::runtime_error(patientId + ": runtime output contains invalid dimension");
        }
        actualLogitsShape.push_back(static_cast<std::size_t>(dimension));
      }
      requireEqualShape(actualLogitsShape, kFixtureLogitsShape, patientId + " runtime logits");
      if (actualLogits.data.size() != checkedElementCount(kFixtureLogitsShape))
      {
        throw std::runtime_error(patientId + ": runtime logits element count mismatch");
      }

      const ErrorSummary error =
          compareLogits(actualLogits.data, expectedLogits.data, tolerance, patientId);
      const int actualPrediction = argmax(actualLogits.data);
      const int expectedPrediction = requireInt(patient, "predicted_class_index");
      const bool predictionMatched = actualPrediction == expectedPrediction;
      allPredictionsMatched = allPredictionsMatched && predictionMatched;
      if (!predictionMatched)
      {
        throw std::runtime_error(patientId + ": predicted class index mismatch");
      }

      std::cout << patientId << ": max_abs_error=" << error.maximumAbsoluteError
                << " max_logit_index=" << error.flatIndex
                << " predicted_class_index=" << actualPrediction << '\n';
    }

    if (!allPredictionsMatched)
    {
      throw std::runtime_error("At least one predicted class index did not match");
    }

    std::cout << "All seven golden ONNX Runtime predictions matched." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Golden parity test failed: " << error.what() << '\n';
    return 1;
  }
}
