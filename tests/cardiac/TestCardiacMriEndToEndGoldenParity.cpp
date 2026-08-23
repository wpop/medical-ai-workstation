#include "NpyReader.h"
#include "Sha256.h"
#include "maiw/cardiac/CardiacMriPreprocessing.h"
#include "maiw/onnx/OnnxRuntimeSession.h"
#include "qtviewerpro/io/MedicalVolumeLoaderRegistry.h"

#include <nlohmann/json.hpp>

#include <QString>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using Json = nlohmann::json;

constexpr std::size_t kExpectedPatientCount = 7;
constexpr std::size_t kClassCount = 5;
constexpr int kExpectedPreprocessingFixtureVersion = 1;
constexpr double kExpectedAbsoluteTolerance = 1e-5;

constexpr std::string_view kExpectedInputName = "cine_mri";
constexpr std::string_view kExpectedOutputName = "logits";

constexpr std::array<std::size_t, 4> kPreprocessingTensorShape{2, 14, 144, 144};
constexpr std::array<std::size_t, 5> kInferenceInputShape{1, 2, 14, 144, 144};
constexpr std::array<std::size_t, 2> kGoldenLogitsShape{1, 5};
constexpr std::array<std::int64_t, 5> kRuntimeInputShape{1, 2, 14, 144, 144};
constexpr std::array<std::int64_t, 2> kRuntimeLogitsShape{1, 5};

constexpr std::array<std::string_view, kExpectedPatientCount> kExpectedPatientIds{
    "patient001", "patient006", "patient021", "patient042",
    "patient063", "patient085", "patient094"};

/**
 * @brief Load and parse one JSON validation contract.
 *
 * @throws std::runtime_error If the file cannot be opened or parsed.
 */
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

/**
 * @brief Return a required JSON object member.
 *
 * @throws std::runtime_error If the key is missing or the parent value is not an object.
 */
const Json& requireObjectMember(const Json& object, const char* key)
{
  if (!object.is_object() || !object.contains(key))
  {
    throw std::runtime_error(std::string("JSON object is missing key: ") + key);
  }

  return object.at(key);
}

/**
 * @brief Read a required string value from a JSON object.
 *
 * @throws std::runtime_error If the key is missing or does not contain a string.
 */
std::string requireString(const Json& object, const char* key)
{
  const Json& value = requireObjectMember(object, key);
  if (!value.is_string())
  {
    throw std::runtime_error(std::string("JSON value must be a string: ") + key);
  }

  return value.get<std::string>();
}

/**
 * @brief Read a required integer value from a JSON object.
 *
 * @throws std::runtime_error If the key is missing or does not contain an integer.
 */
int requireInt(const Json& object, const char* key)
{
  const Json& value = requireObjectMember(object, key);
  if (!value.is_number_integer())
  {
    throw std::runtime_error(std::string("JSON value must be an integer: ") + key);
  }

  return value.get<int>();
}

/**
 * @brief Read a required finite floating-point value from a JSON object.
 *
 * @throws std::runtime_error If the value is missing, non-numeric, or non-finite.
 */
double requireDouble(const Json& object, const char* key)
{
  const Json& value = requireObjectMember(object, key);
  if (!value.is_number())
  {
    throw std::runtime_error(std::string("JSON value must be numeric: ") + key);
  }

  const double parsed = value.get<double>();
  if (!std::isfinite(parsed))
  {
    throw std::runtime_error(std::string("JSON numeric value must be finite: ") + key);
  }

  return parsed;
}

/**
 * @brief Read a positive tensor shape from a JSON array.
 *
 * @throws std::runtime_error If the shape is missing or contains invalid dimensions.
 * @throws std::overflow_error If a dimension cannot be represented by std::size_t.
 */
std::vector<std::size_t> requireShape(const Json& object, const char* key)
{
  const Json& value = requireObjectMember(object, key);
  if (!value.is_array())
  {
    throw std::runtime_error(std::string("JSON shape value must be an array: ") + key);
  }

  std::vector<std::size_t> shape;
  shape.reserve(value.size());

  for (const Json& dimension : value)
  {
    if (dimension.is_number_unsigned())
    {
      const auto parsed = dimension.get<std::uint64_t>();
      if (parsed == 0 ||
          parsed > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      {
        throw std::overflow_error(
            std::string("JSON shape dimension is outside size_t range: ") + key);
      }

      shape.push_back(static_cast<std::size_t>(parsed));
      continue;
    }

    if (!dimension.is_number_integer())
    {
      throw std::runtime_error(
          std::string("JSON shape dimension must be an integer: ") + key);
    }

    const auto parsed = dimension.get<std::int64_t>();
    if (parsed <= 0)
    {
      throw std::runtime_error(
          std::string("JSON shape dimension must be positive: ") + key);
    }

    shape.push_back(static_cast<std::size_t>(parsed));
  }

  return shape;
}

/**
 * @brief Require an exact match between a parsed fixture shape and a fixed contract shape.
 */
template <std::size_t N>
void requireExactShape(const std::vector<std::size_t>& actual,
                       const std::array<std::size_t, N>& expected,
                       std::string_view context)
{
  if (actual.size() != expected.size() ||
      !std::equal(actual.begin(), actual.end(), expected.begin()))
  {
    throw std::runtime_error(std::string(context) + " shape mismatch");
  }
}

/**
 * @brief Require an exact match between an ONNX Runtime shape and a fixed runtime shape.
 */
template <std::size_t N>
void requireExactRuntimeShape(const std::vector<std::int64_t>& actual,
                              const std::array<std::int64_t, N>& expected,
                              std::string_view context)
{
  if (actual.size() != expected.size() ||
      !std::equal(actual.begin(), actual.end(), expected.begin()))
  {
    throw std::runtime_error(std::string(context) + " shape mismatch");
  }
}

/**
 * @brief Resolve a manifest value that must identify a file within an already selected root.
 *
 * Restricting fixture values to filenames prevents a manifest entry from redirecting
 * validation outside the configured package or ACDC patient directory.
 */
std::filesystem::path requireFilename(const Json& object, const char* key)
{
  const std::filesystem::path path{requireString(object, key)};

  if (path.empty() || path.is_absolute() || path.has_parent_path())
  {
    throw std::runtime_error(std::string("Manifest value must be a filename: ") + key);
  }

  return path;
}

/**
 * @brief Resolve a manifest value that must remain within a configured fixture root.
 *
 * Relative subdirectories are permitted because frozen inference artifacts may be
 * organized below the golden inference root. Absolute paths and parent traversal
 * are rejected so the manifest cannot redirect validation outside that root.
 */
std::filesystem::path requireRelativeFixturePath(const Json& object, const char* key)
{
  const std::filesystem::path path{requireString(object, key)};

  if (path.empty() || path.is_absolute())
  {
    throw std::runtime_error(
        std::string("Manifest value must be a non-empty relative path: ") + key);
  }

  for (const auto& component : path)
  {
    if (component == "..")
    {
      throw std::runtime_error(
          std::string("Manifest path must not contain parent traversal: ") + key);
    }
  }

  return path.lexically_normal();
}

/**
 * @brief Verify that a required artifact exists and matches its frozen SHA-256 digest.
 *
 * @throws std::runtime_error If the file is missing or its digest does not match.
 */
void requireFileSha256(const std::filesystem::path& path,
                       const std::string& expectedSha256,
                       const std::string& context)
{
  if (!std::filesystem::is_regular_file(path))
  {
    throw std::runtime_error(context + " file does not exist: " + path.string());
  }

  const std::string actualSha256 = maiw::test::sha256FileHex(path);
  if (actualSha256 != expectedSha256)
  {
    throw std::runtime_error(context + " SHA-256 mismatch: expected " +
                             expectedSha256 + ", actual " + actualSha256);
  }
}

/**
 * @brief Load one required medical volume through the production qtviewerpro IO path.
 *
 * @throws std::runtime_error If the production loader cannot read the volume.
 */
qvp::VolumeData loadRequiredVolume(const std::filesystem::path& path,
                                   const std::string& context)
{
  const qvp::VolumeLoadResult result =
      qvp::loadMedicalVolume(QString::fromStdString(path.string()));

  if (!result.success)
  {
    throw std::runtime_error(context + " load failed: " +
                             result.errorMessage.toStdString());
  }

  return result.volume;
}

/**
 * @brief Validate the frozen patient set and deterministic ordering in one manifest.
 */
void validateOrderedPatients(const Json& manifest, std::string_view manifestName)
{
  const Json& patients = requireObjectMember(manifest, "patients");

  if (!patients.is_array() || patients.size() != kExpectedPatientCount)
  {
    throw std::runtime_error(
        std::string(manifestName) + " must contain exactly seven patients");
  }

  for (std::size_t index = 0; index < kExpectedPatientIds.size(); ++index)
  {
    const std::string patientId = requireString(patients.at(index), "patient_id");
    if (patientId != kExpectedPatientIds[index])
    {
      throw std::runtime_error(
          std::string(manifestName) + " patient set/order mismatch");
    }
  }
}

/**
 * @brief Validate the frozen preprocessing manifest contract used by Phase 6.
 */
void validatePreprocessingManifest(const Json& manifest)
{
  if (requireString(manifest, "dataset") != "ACDC")
  {
    throw std::runtime_error("Golden preprocessing manifest dataset is not ACDC");
  }

  if (requireInt(manifest, "fixture_version") != kExpectedPreprocessingFixtureVersion)
  {
    throw std::runtime_error("Golden preprocessing fixture version mismatch");
  }

  if (requireString(manifest, "dtype") != "float32")
  {
    throw std::runtime_error("Golden preprocessing manifest dtype is not float32");
  }

  if (requireString(manifest, "source_array_order") != "XYZ")
  {
    throw std::runtime_error("Golden preprocessing source array order is not XYZ");
  }

  if (requireString(manifest, "model_array_order") != "ZYX")
  {
    throw std::runtime_error("Golden preprocessing model array order is not ZYX");
  }

  if (requireString(manifest, "preprocessing_contract") !=
      "docs/preprocessing_contract.md")
  {
    throw std::runtime_error("Golden preprocessing contract path mismatch");
  }

  requireExactShape(requireShape(manifest, "tensor_shape"),
                    kPreprocessingTensorShape,
                    "golden preprocessing manifest tensor");

  const Json& channelSemantics =
      requireObjectMember(manifest, "channel_semantics");

  if (requireString(channelSemantics, "0") != "ED" ||
      requireString(channelSemantics, "1") != "ES")
  {
    throw std::runtime_error(
        "Golden preprocessing channel semantics must be ED then ES");
  }

  validateOrderedPatients(manifest, "Golden preprocessing manifest");
}

/**
 * @brief Validate the frozen inference manifest contract used by Phase 6.
 */
void validateInferenceManifest(const Json& manifest)
{
  if (requireDouble(manifest, "absolute_tolerance") != kExpectedAbsoluteTolerance)
  {
    throw std::runtime_error(
        "Golden inference manifest absolute tolerance is not 1e-5");
  }

  if (!requireObjectMember(manifest, "relative_tolerance").is_null())
  {
    throw std::runtime_error(
        "Golden inference manifest relative tolerance must be null/disabled");
  }

  if (requireString(manifest, "input_name") != kExpectedInputName ||
      requireString(manifest, "output_name") != kExpectedOutputName)
  {
    throw std::runtime_error(
        "Golden inference manifest input/output names do not match the ONNX contract");
  }

  validateOrderedPatients(manifest, "Golden inference manifest");
}

/**
 * @brief Validate model names and element types declared by deployment.json.
 */
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

/**
 * @brief Return true when a model batch dimension accepts a single inference sample.
 */
bool acceptsSingleBatch(std::int64_t dimension) noexcept
{
  return dimension < 0 || dimension == 1;
}

/**
 * @brief Validate the production ONNX model input contract.
 */
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

  if (input.shape.size() != kRuntimeInputShape.size())
  {
    throw std::runtime_error("Model input rank must be 5");
  }

  if (!acceptsSingleBatch(input.shape[0]) ||
      input.shape[1] != kRuntimeInputShape[1] ||
      input.shape[2] != kRuntimeInputShape[2] ||
      input.shape[3] != kRuntimeInputShape[3] ||
      input.shape[4] != kRuntimeInputShape[4])
  {
    throw std::runtime_error(
        "Model input shape is not compatible with [N,2,14,144,144]");
  }
}

/**
 * @brief Validate the production ONNX model output contract.
 */
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

  if (output.shape.size() != kRuntimeLogitsShape.size())
  {
    throw std::runtime_error("Model output rank must be 2");
  }

  if (!acceptsSingleBatch(output.shape[0]) ||
      output.shape[1] != kRuntimeLogitsShape[1])
  {
    throw std::runtime_error(
        "Model output shape is not compatible with [N,5]");
  }
}

/**
 * @brief Compute the class index with the largest raw logit.
 *
 * @throws std::runtime_error If the input tensor is empty.
 */
std::size_t argmax(std::span<const float> values)
{
  if (values.empty())
  {
    throw std::runtime_error("Cannot compute argmax of an empty tensor");
  }

  return static_cast<std::size_t>(
      std::distance(values.begin(),
                    std::max_element(values.begin(), values.end())));
}

/**
 * @brief Diagnostic summary for one raw-logit parity comparison.
 */
struct LogitComparison
{
  double maximumAbsoluteError = 0.0;
  std::size_t worstIndex = 0;
  float expectedAtWorst = 0.0F;
  float actualAtWorst = 0.0F;
  bool withinTolerance = true;
};

/**
 * @brief Compare actual and golden raw logits using the frozen absolute tolerance.
 *
 * Relative tolerance is intentionally not applied because the inference manifest
 * explicitly disables it.
 */
LogitComparison compareLogits(std::span<const float> actual,
                              std::span<const float> expected)
{
  if (actual.size() != expected.size())
  {
    throw std::runtime_error("Actual and expected logits have different sizes");
  }

  if (actual.empty())
  {
    throw std::runtime_error("Logit tensors must not be empty");
  }

  LogitComparison comparison;
  comparison.expectedAtWorst = expected.front();
  comparison.actualAtWorst = actual.front();

  for (std::size_t index = 0; index < actual.size(); ++index)
  {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
    {
      throw std::runtime_error("Logit tensors contain non-finite values");
    }

    const double absoluteError =
        std::fabs(static_cast<double>(actual[index]) -
                  static_cast<double>(expected[index]));

    if (absoluteError > comparison.maximumAbsoluteError)
    {
      comparison.maximumAbsoluteError = absoluteError;
      comparison.worstIndex = index;
      comparison.expectedAtWorst = expected[index];
      comparison.actualAtWorst = actual[index];
    }

    if (absoluteError > kExpectedAbsoluteTolerance)
    {
      comparison.withinTolerance = false;
    }
  }

  return comparison;
}

/**
 * @brief Resolved artifacts required to execute one real-patient end-to-end case.
 */
struct PatientPaths
{
  std::filesystem::path ed;
  std::filesystem::path es;
  std::filesystem::path info;
  std::filesystem::path logits;
};

/**
 * @brief Permanent Phase 6 integration parity gate for the cardiac MRI deployment path.
 *
 * The test composes the production medical-image loader, CardiacMriPreprocessor,
 * and OnnxRuntimeSession without injecting a frozen preprocessing tensor between
 * preprocessing and inference. Frozen Python logits remain the external numerical
 * oracle for the final ONNX output.
 */
class EndToEndGoldenParityTest final
{
public:
  /**
   * @brief Construct the integration test with external deployment and ACDC roots.
   *
   * The test does not own or modify either external asset tree.
   */
  EndToEndGoldenParityTest(std::filesystem::path packageRoot,
                           std::filesystem::path datasetRoot)
      : packageRoot_(std::move(packageRoot)),
        datasetRoot_(std::move(datasetRoot))
  {
  }

  /**
   * @brief Execute the complete seven-patient end-to-end parity gate.
   *
   * Global contract failures abort the test immediately. Patient-specific failures
   * are reported individually so one failing patient does not hide later results.
   *
   * @throws std::runtime_error If a global contract fails or any patient fails parity.
   */
  void run() const
  {
    validateRoots();

    const Json deployment = readJson(packageRoot_ / "deployment.json");
    const Json preprocessingManifest =
        readJson(packageRoot_ / "golden/preprocessing/manifest.json");
    const Json inferenceManifest =
        readJson(packageRoot_ / "golden/inference/manifest.json");

    validateDeploymentContract(deployment);
    validatePreprocessingManifest(preprocessingManifest);
    validateInferenceManifest(inferenceManifest);

    validateGlobalArtifacts(deployment, preprocessingManifest);

    const Json& onnx = requireObjectMember(deployment, "onnx");
    const std::filesystem::path modelPath =
        packageRoot_ / requireFilename(onnx, "filename");

    Ort::Env environment(
        ORT_LOGGING_LEVEL_ERROR,
        "medical-ai-workstation-end-to-end-golden-parity");

    maiw::onnx::OnnxRuntimeSession session(environment, modelPath);
    validateModelContract(session);

    const Json& preprocessingPatients =
        requireObjectMember(preprocessingManifest, "patients");
    const Json& inferencePatients =
        requireObjectMember(inferenceManifest, "patients");

    const maiw::cardiac::CardiacMriPreprocessor preprocessor;

    bool allPatientsPassed = true;

    for (std::size_t index = 0; index < kExpectedPatientCount; ++index)
    {
      const std::string patientId{kExpectedPatientIds[index]};

      try
      {
        const bool patientPassed =
            runPatient(patientId,
                       preprocessingPatients.at(index),
                       inferencePatients.at(index),
                       preprocessor,
                       session);

        allPatientsPassed = allPatientsPassed && patientPassed;
      }
      catch (const std::exception& error)
      {
        allPatientsPassed = false;
        std::cerr << patientId
                  << ": status=FAIL error="
                  << error.what()
                  << '\n';
      }
    }

    if (!allPatientsPassed)
    {
      throw std::runtime_error(
          "End-to-end cardiac pipeline parity failed for one or more patients");
    }

    std::cout
        << "All seven real ACDC end-to-end pipeline cases passed."
        << '\n';
  }

private:
  /**
   * @brief Validate that both externally configured asset roots are available.
   */
  void validateRoots() const
  {
    if (!std::filesystem::is_directory(packageRoot_))
    {
      throw std::runtime_error(
          "Package root does not exist: " + packageRoot_.string());
    }

    if (!std::filesystem::is_directory(datasetRoot_))
    {
      throw std::runtime_error(
          "ACDC training root does not exist: " + datasetRoot_.string());
    }
  }

  /**
   * @brief Validate immutable artifacts shared by every patient case.
   */
  void validateGlobalArtifacts(const Json& deployment,
                               const Json& preprocessingManifest) const
  {
    const std::filesystem::path preprocessingContractPath =
        packageRoot_ / requireString(preprocessingManifest,
                                     "preprocessing_contract");

    requireFileSha256(
        preprocessingContractPath,
        requireString(preprocessingManifest,
                      "preprocessing_contract_sha256"),
        "preprocessing contract");

    const Json& onnx = requireObjectMember(deployment, "onnx");
    const std::filesystem::path modelPath =
        packageRoot_ / requireFilename(onnx, "filename");

    requireFileSha256(
        modelPath,
        requireString(onnx, "sha256"),
        "classifier.onnx");
  }

  /**
   * @brief Validate the runtime model metadata exposed by OnnxRuntimeSession.
   */
  static void validateModelContract(const maiw::onnx::OnnxRuntimeSession& session)
  {
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
  }

  /**
   * @brief Validate patient-specific preprocessing and inference manifest entries.
   */
  static void validatePatientContracts(
      std::string_view expectedPatientId,
      const Json& preprocessingPatient,
      const Json& inferencePatient)
  {
    if (requireString(preprocessingPatient, "patient_id") != expectedPatientId ||
        requireString(inferencePatient, "patient_id") != expectedPatientId)
    {
      throw std::runtime_error(
          std::string(expectedPatientId) +
          ": preprocessing/inference patient identity mismatch");
    }

    requireExactShape(requireShape(preprocessingPatient, "tensor_shape"),
                      kPreprocessingTensorShape,
                      "preprocessing tensor");

    if (requireString(preprocessingPatient, "tensor_dtype") != "float32")
    {
      throw std::runtime_error(
          std::string(expectedPatientId) +
          ": preprocessing tensor dtype is not float32");
    }

    requireExactShape(requireShape(inferencePatient, "input_shape"),
                      kInferenceInputShape,
                      "inference input");

    requireExactShape(requireShape(inferencePatient, "logits_shape"),
                      kGoldenLogitsShape,
                      "inference logits");

    if (requireString(inferencePatient, "input_dtype") != "float32" ||
        requireString(inferencePatient, "logits_dtype") != "float32")
    {
      throw std::runtime_error(
          std::string(expectedPatientId) +
          ": inference manifest dtype is not float32");
    }
  }

  /**
   * @brief Resolve all external artifacts consumed by one patient execution.
   *
   * Frozen inference input tensors are intentionally absent because Phase 6 must
   * feed the production preprocessing output directly into ONNX Runtime.
   */
  [[nodiscard]] PatientPaths resolvePatientPaths(
      std::string_view patientId,
      const Json& preprocessingPatient,
      const Json& inferencePatient) const
  {
    const std::filesystem::path patientRoot =
        datasetRoot_ / std::string(patientId);

    return PatientPaths{
        patientRoot / requireFilename(preprocessingPatient, "source_ed_filename"),
        patientRoot / requireFilename(preprocessingPatient, "source_es_filename"),
        patientRoot / requireFilename(preprocessingPatient, "info_filename"),
        packageRoot_ / "golden/inference" /
            requireRelativeFixturePath(inferencePatient, "logits_filename")};
  }

  /**
   * @brief Execute preprocessing, inference, and golden-logit validation for one patient.
   *
   * @return true when both raw-logit tolerance and predicted class parity pass.
   */
  [[nodiscard]] bool runPatient(
      const std::string& patientId,
      const Json& preprocessingPatient,
      const Json& inferencePatient,
      const maiw::cardiac::CardiacMriPreprocessor& preprocessor,
      maiw::onnx::OnnxRuntimeSession& session) const
  {
    validatePatientContracts(
        patientId,
        preprocessingPatient,
        inferencePatient);

    const PatientPaths paths =
        resolvePatientPaths(patientId,
                            preprocessingPatient,
                            inferencePatient);

    requireFileSha256(
        paths.ed,
        requireString(preprocessingPatient, "source_ed_sha256"),
        patientId + " ED");

    requireFileSha256(
        paths.es,
        requireString(preprocessingPatient, "source_es_sha256"),
        patientId + " ES");

    requireFileSha256(
        paths.info,
        requireString(preprocessingPatient, "info_sha256"),
        patientId + " Info.cfg");

    requireFileSha256(
        paths.logits,
        requireString(inferencePatient, "logits_sha256"),
        patientId + " golden logits");

    const qvp::VolumeData edVolume =
        loadRequiredVolume(paths.ed, patientId + " ED");

    const qvp::VolumeData esVolume =
        loadRequiredVolume(paths.es, patientId + " ES");

    const maiw::cardiac::CardiacMriInputTensor inputTensor =
        preprocessor.preprocess(edVolume, esVolume);

    if (inputTensor.shapeCdhw() != kPreprocessingTensorShape)
    {
      throw std::runtime_error(
          patientId + ": preprocessing output shape mismatch");
    }

    if (inputTensor.span().size() !=
        maiw::cardiac::CardiacMriInputTensor::elementCount())
    {
      throw std::runtime_error(
          patientId + ": preprocessing output element count mismatch");
    }

    /*
     * This call is the Phase 6 integration boundary. The contiguous production
     * preprocessing buffer is passed directly to ONNX Runtime. Adding N=1 changes
     * only the tensor shape metadata and does not copy or reorder voxel values.
     */
    const maiw::onnx::FloatTensor actualLogits =
        session.run(kExpectedInputName,
                    inputTensor.span(),
                    kRuntimeInputShape,
                    kExpectedOutputName);

    requireExactRuntimeShape(actualLogits.shape,
                             kRuntimeLogitsShape,
                             patientId + " runtime logits");

    if (actualLogits.data.size() != kClassCount)
    {
      throw std::runtime_error(
          patientId + ": runtime logits element count mismatch");
    }

    const maiw::test::NpyFloat32Array expectedLogits =
        maiw::test::readNpyFloat32(paths.logits);

    requireExactShape(expectedLogits.shape,
                      kGoldenLogitsShape,
                      patientId + " golden logits");

    if (expectedLogits.data.size() != kClassCount)
    {
      throw std::runtime_error(
          patientId + ": golden logits element count mismatch");
    }

    const int expectedClass =
        requireInt(inferencePatient, "predicted_class_index");

    if (expectedClass < 0 ||
        expectedClass >= static_cast<int>(kClassCount))
    {
      throw std::runtime_error(
          patientId + ": predicted_class_index is outside [0,4]");
    }

    const std::size_t goldenClass = argmax(expectedLogits.data);
    if (goldenClass != static_cast<std::size_t>(expectedClass))
    {
      throw std::runtime_error(
          patientId +
          ": golden logits disagree with predicted_class_index");
    }

    const LogitComparison comparison =
        compareLogits(actualLogits.data, expectedLogits.data);

    const std::size_t actualClass = argmax(actualLogits.data);

    const bool predictionMatches =
        actualClass == static_cast<std::size_t>(expectedClass);

    const bool patientPassed =
        comparison.withinTolerance && predictionMatches;

    printPatientResult(patientId,
                       comparison,
                       actualClass,
                       static_cast<std::size_t>(expectedClass),
                       patientPassed);

    return patientPassed;
  }

  /**
   * @brief Print stable numerical diagnostics for one completed patient case.
   */
  static void printPatientResult(
      const std::string& patientId,
      const LogitComparison& comparison,
      std::size_t actualClass,
      std::size_t expectedClass,
      bool passed)
  {
    std::cout
        << std::scientific
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << patientId
        << ": max_abs_error=" << comparison.maximumAbsoluteError
        << " worst_logit_index=" << comparison.worstIndex
        << " expected_at_worst="
        << static_cast<double>(comparison.expectedAtWorst)
        << " actual_at_worst="
        << static_cast<double>(comparison.actualAtWorst)
        << " predicted_class_index=" << actualClass
        << " expected_class_index=" << expectedClass
        << " status=" << (passed ? "PASS" : "FAIL")
        << '\n';
  }

  std::filesystem::path packageRoot_;
  std::filesystem::path datasetRoot_;
};

} // namespace

int main()
{
  try
  {
    const EndToEndGoldenParityTest test{
        std::filesystem::path{MAIW_CARDIAC_MRI_PACKAGE_DIR},
        std::filesystem::path{MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR}};

    test.run();
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr
        << "End-to-end cardiac pipeline parity test failed: "
        << error.what()
        << '\n';

    return 1;
  }
}
