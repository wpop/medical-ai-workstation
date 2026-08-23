#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace
{

using Json = nlohmann::json;

Json readJsonFile(const std::filesystem::path& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::runtime_error("Failed to open cardiac deployment metadata file: " + path.string());
  }

  try
  {
    Json document;
    input >> document;
    return document;
  }
  catch (const Json::parse_error& error)
  {
    throw std::runtime_error("Failed to parse cardiac deployment metadata file '" +
                             path.string() + "': " + error.what());
  }
}

const Json& requireMember(const Json& object, const char* key, std::string_view context)
{
  if (!object.is_object())
  {
    throw std::runtime_error(std::string(context) + " must be a JSON object");
  }
  const auto member = object.find(key);
  if (member == object.end())
  {
    throw std::runtime_error(std::string(context) + " is missing required field '" + key + "'");
  }
  return *member;
}

void requireExactString(const Json& object,
                        const char* key,
                        const char* expected,
                        std::string_view context)
{
  const Json& value = requireMember(object, key, context);
  if (!value.is_string() || value.get_ref<const std::string&>() != expected)
  {
    throw std::runtime_error(std::string(context) + "." + key + " must equal '" + expected +
                             "'");
  }
}

void requireExactInteger(const Json& object,
                         const char* key,
                         std::int64_t expected,
                         std::string_view context)
{
  const Json& value = requireMember(object, key, context);
  if (!value.is_number_integer() || value != expected)
  {
    throw std::runtime_error(std::string(context) + "." + key + " must equal " +
                             std::to_string(expected));
  }
}

std::string requireString(const Json& object, const char* key, std::string_view context)
{
  const Json& value = requireMember(object, key, context);
  if (!value.is_string())
  {
    throw std::runtime_error(std::string(context) + "." + key + " must be a string");
  }
  return value.get<std::string>();
}

void requireInputShape(const Json& shape)
{
  constexpr std::array<std::int64_t, 4> kTrailingDimensions{2, 14, 144, 144};
  if (!shape.is_array() || shape.size() != 5 || !shape[0].is_string() ||
      shape[0].get_ref<const std::string&>() != "N")
  {
    throw std::runtime_error("deployment.json onnx.input.shape must equal [\"N\",2,14,144,144]");
  }

  for (std::size_t index = 0; index < kTrailingDimensions.size(); ++index)
  {
    const Json& dimension = shape[index + 1];
    if (!dimension.is_number_integer() || dimension != kTrailingDimensions[index])
    {
      throw std::runtime_error(
          "deployment.json onnx.input.shape must equal [\"N\",2,14,144,144]");
    }
  }
}

void requireOutputShape(const Json& shape)
{
  if (!shape.is_array() || shape.size() != 2 || !shape[0].is_string() ||
      shape[0].get_ref<const std::string&>() != "N" || !shape[1].is_number_integer() ||
      shape[1] != static_cast<std::int64_t>(
                      maiw::cardiac::CardiacMriDeploymentMetadata::kClassCount))
  {
    throw std::runtime_error("deployment.json onnx.output.shape must equal [\"N\",5]");
  }
}

maiw::cardiac::CardiacMriDeploymentMetadata::ClassNames readClassNames(const Json& deployment)
{
  const Json& classes = requireMember(deployment, "classes", "deployment.json");
  if (!classes.is_array() ||
      classes.size() != maiw::cardiac::CardiacMriDeploymentMetadata::kClassCount)
  {
    throw std::runtime_error("deployment.json classes must contain exactly five entries");
  }

  maiw::cardiac::CardiacMriDeploymentMetadata::ClassNames names;
  std::unordered_set<std::string> uniqueNames;
  for (std::size_t index = 0; index < names.size(); ++index)
  {
    const std::string context = "deployment.json classes[" + std::to_string(index) + "]";
    const Json& entry = classes[index];
    requireExactInteger(entry, "index", static_cast<std::int64_t>(index), context);
    names[index] = requireString(entry, "name", context);
    if (names[index].empty())
    {
      throw std::runtime_error(context + ".name must not be empty");
    }
    if (!uniqueNames.insert(names[index]).second)
    {
      throw std::runtime_error("deployment.json class names must be unique");
    }
  }
  return names;
}

void validateClassMapping(
    const Json& mapping,
    const maiw::cardiac::CardiacMriDeploymentMetadata::ClassNames& classNames)
{
  if (!mapping.is_object() || mapping.size() != classNames.size())
  {
    throw std::runtime_error("class_mapping.json must contain exactly keys '0' through '4'");
  }

  for (std::size_t index = 0; index < classNames.size(); ++index)
  {
    const std::string key = std::to_string(index);
    const auto value = mapping.find(key);
    if (value == mapping.end() || !value->is_string())
    {
      throw std::runtime_error("class_mapping.json must contain string value for key '" + key +
                               "'");
    }
    if (value->get_ref<const std::string&>() != classNames[index])
    {
      throw std::runtime_error("class_mapping.json value for key '" + key +
                               "' disagrees with deployment.json");
    }
  }
}

std::filesystem::path validateModelPath(const std::filesystem::path& packageRoot,
                                        const std::string& filename)
{
  if (filename.empty())
  {
    throw std::runtime_error("deployment.json onnx.filename must not be empty");
  }

  const std::filesystem::path relativePath(filename);
  if (relativePath.is_absolute() || relativePath.has_root_name() || relativePath.has_root_directory() ||
      relativePath != relativePath.filename() || relativePath == "." || relativePath == ".." ||
      filename.find('/') != std::string::npos || filename.find('\\') != std::string::npos)
  {
    throw std::runtime_error("deployment.json onnx.filename must be a safe filename without a path");
  }

  const std::filesystem::path modelPath = packageRoot / relativePath;
  std::error_code error;
  const bool isRegularFile = std::filesystem::is_regular_file(modelPath, error);
  if (error || !isRegularFile)
  {
    throw std::runtime_error("Cardiac ONNX model is not a regular file: " + modelPath.string());
  }
  return modelPath;
}

} // namespace

namespace maiw::cardiac
{

CardiacMriDeploymentMetadata CardiacMriDeploymentMetadata::load(
    const std::filesystem::path& packageRoot)
{
  if (packageRoot.empty())
  {
    throw std::invalid_argument("Cardiac deployment package root must not be empty");
  }

  std::error_code error;
  const bool isDirectory = std::filesystem::is_directory(packageRoot, error);
  if (error || !isDirectory)
  {
    throw std::invalid_argument("Cardiac deployment package root is not a directory: " +
                                packageRoot.string());
  }

  const Json deployment = readJsonFile(packageRoot / "deployment.json");
  if (!deployment.is_object())
  {
    throw std::runtime_error("deployment.json root must be a JSON object");
  }

  requireExactInteger(deployment, "schema_version", 1, "deployment.json");
  requireExactString(deployment, "model_format", "onnx", "deployment.json");
  requireExactString(deployment, "model_id", "cardiac_mri_pathology", "deployment.json");

  const Json& onnx = requireMember(deployment, "onnx", "deployment.json");
  requireExactInteger(onnx, "opset", 18, "deployment.json onnx");
  const Json& softmaxInsideModel =
      requireMember(onnx, "softmax_inside_model", "deployment.json onnx");
  if (!softmaxInsideModel.is_boolean() || softmaxInsideModel.get<bool>())
  {
    throw std::runtime_error("deployment.json onnx.softmax_inside_model must equal false");
  }

  const Json& input = requireMember(onnx, "input", "deployment.json onnx");
  requireExactString(input, "name", "cine_mri", "deployment.json onnx.input");
  requireExactString(input, "dtype", "float32", "deployment.json onnx.input");
  requireInputShape(requireMember(input, "shape", "deployment.json onnx.input"));

  const Json& output = requireMember(onnx, "output", "deployment.json onnx");
  requireExactString(output, "name", "logits", "deployment.json onnx.output");
  requireExactString(output, "dtype", "float32", "deployment.json onnx.output");
  requireOutputShape(requireMember(output, "shape", "deployment.json onnx.output"));

  ClassNames classNames = readClassNames(deployment);
  validateClassMapping(readJsonFile(packageRoot / "class_mapping.json"), classNames);

  const std::filesystem::path modelPath =
      validateModelPath(packageRoot, requireString(onnx, "filename", "deployment.json onnx"));

  return CardiacMriDeploymentMetadata(modelPath,
                                      requireString(input, "name", "deployment.json onnx.input"),
                                      requireString(output, "name", "deployment.json onnx.output"),
                                      std::move(classNames));
}

const std::filesystem::path& CardiacMriDeploymentMetadata::modelPath() const noexcept
{
  return modelPath_;
}

const std::string& CardiacMriDeploymentMetadata::inputName() const noexcept
{
  return inputName_;
}

const std::string& CardiacMriDeploymentMetadata::outputName() const noexcept
{
  return outputName_;
}

const CardiacMriDeploymentMetadata::ClassNames&
CardiacMriDeploymentMetadata::classNames() const noexcept
{
  return classNames_;
}

CardiacMriDeploymentMetadata::CardiacMriDeploymentMetadata(std::filesystem::path modelPath,
                                                           std::string inputName,
                                                           std::string outputName,
                                                           ClassNames classNames)
  : modelPath_(std::move(modelPath)),
    inputName_(std::move(inputName)),
    outputName_(std::move(outputName)),
    classNames_(std::move(classNames))
{
}

} // namespace maiw::cardiac
