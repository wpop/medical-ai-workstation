#include "NpyReader.h"
#include "Sha256.h"
#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "qtviewerpro/io/MedicalVolumeLoaderRegistry.h"

#include <nlohmann/json.hpp>

#include <QString>
#include <onnxruntime_cxx_api.h>

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
#include <numeric>
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
constexpr std::size_t kClassCount =
    maiw::cardiac::CardiacMriDeploymentMetadata::kClassCount;
constexpr int kExpectedPreprocessingFixtureVersion = 1;
constexpr double kExpectedAbsoluteLogitTolerance = 1e-5;
constexpr double kProbabilityNormalizationTolerance = 1e-12;

// The softmax Jacobian has an L-infinity row norm of at most 0.5. This
// preserves the frozen raw-logit tolerance while allowing only double-rounding slack.
constexpr double kExpectedProbabilityTolerance =
    (0.5 * kExpectedAbsoluteLogitTolerance) + 1e-12;

constexpr std::array<std::string_view, kExpectedPatientCount> kExpectedPatientIds{
    "patient001", "patient006", "patient021", "patient042",
    "patient063", "patient085", "patient094"};

constexpr std::array<std::size_t, 4> kPreprocessingTensorShape{
    maiw::cardiac::CardiacMriInputTensor::kChannels,
    maiw::cardiac::CardiacMriInputTensor::kDepth,
    maiw::cardiac::CardiacMriInputTensor::kHeight,
    maiw::cardiac::CardiacMriInputTensor::kWidth};
constexpr std::array<std::size_t, 5> kInferenceInputShape{
    1,
    maiw::cardiac::CardiacMriInputTensor::kChannels,
    maiw::cardiac::CardiacMriInputTensor::kDepth,
    maiw::cardiac::CardiacMriInputTensor::kHeight,
    maiw::cardiac::CardiacMriInputTensor::kWidth};
constexpr std::array<std::size_t, 2> kGoldenLogitsShape{1, kClassCount};

using ProbabilityArray = std::array<double, kClassCount>;

Json readJson(const std::filesystem::path& path)
{
  std::ifstream input(path);
  if (!input)
  {
    throw std::runtime_error("Failed to open JSON file: " + path.string());
  }

  try
  {
    Json document;
    input >> document;
    return document;
  }
  catch (const Json::parse_error& error)
  {
    throw std::runtime_error("Failed to parse JSON file '" + path.string() + "': " +
                             error.what());
  }
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

  const double parsed = value.get<double>();
  if (!std::isfinite(parsed))
  {
    throw std::runtime_error(std::string("JSON numeric value must be finite: ") + key);
  }
  return parsed;
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
  for (const Json& dimension : value)
  {
    if (!dimension.is_number_integer())
    {
      throw std::runtime_error(std::string("JSON shape dimension must be an integer: ") + key);
    }
    const std::int64_t parsed = dimension.get<std::int64_t>();
    if (parsed <= 0)
    {
      throw std::runtime_error(std::string("JSON shape dimension must be positive: ") + key);
    }
    shape.push_back(static_cast<std::size_t>(parsed));
  }
  return shape;
}

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

std::filesystem::path requireFilename(const Json& object, const char* key)
{
  const std::filesystem::path path{requireString(object, key)};
  if (path.empty() || path.is_absolute() || path.has_parent_path() || path == "." ||
      path == "..")
  {
    throw std::runtime_error(std::string("Manifest value must be a filename: ") + key);
  }
  return path;
}

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
    throw std::runtime_error(context + " SHA-256 mismatch: expected " + expectedSha256 +
                             ", actual " + actualSha256);
  }
}

qvp::VolumeData loadRequiredVolume(const std::filesystem::path& path,
                                   const std::string& context)
{
  const qvp::VolumeLoadResult result =
      qvp::loadMedicalVolume(QString::fromStdString(path.string()));
  if (!result.success)
  {
    throw std::runtime_error(context + " load failed: " + result.errorMessage.toStdString());
  }
  return result.volume;
}

void validateOrderedPatients(const Json& manifest, std::string_view manifestName)
{
  const Json& patients = requireObjectMember(manifest, "patients");
  if (!patients.is_array() || patients.size() != kExpectedPatientCount)
  {
    throw std::runtime_error(std::string(manifestName) +
                             " must contain exactly seven patients");
  }

  for (std::size_t index = 0; index < kExpectedPatientIds.size(); ++index)
  {
    if (requireString(patients.at(index), "patient_id") != kExpectedPatientIds[index])
    {
      throw std::runtime_error(std::string(manifestName) + " patient set/order mismatch");
    }
  }
}

void validatePreprocessingManifest(const Json& manifest)
{
  if (requireString(manifest, "dataset") != "ACDC" ||
      requireInt(manifest, "fixture_version") != kExpectedPreprocessingFixtureVersion ||
      requireString(manifest, "dtype") != "float32" ||
      requireString(manifest, "source_array_order") != "XYZ" ||
      requireString(manifest, "model_array_order") != "ZYX")
  {
    throw std::runtime_error("Golden preprocessing manifest contract mismatch");
  }
  if (requireString(manifest, "preprocessing_contract") !=
      "docs/preprocessing_contract.md")
  {
    throw std::runtime_error("Golden preprocessing contract path mismatch");
  }

  requireExactShape(requireShape(manifest, "tensor_shape"),
                    kPreprocessingTensorShape,
                    "golden preprocessing manifest tensor");
  const Json& channelSemantics = requireObjectMember(manifest, "channel_semantics");
  if (requireString(channelSemantics, "0") != "ED" ||
      requireString(channelSemantics, "1") != "ES")
  {
    throw std::runtime_error("Golden preprocessing channel semantics must be ED then ES");
  }
  validateOrderedPatients(manifest, "Golden preprocessing manifest");
}

void validateInferenceManifest(
    const Json& manifest,
    const maiw::cardiac::CardiacMriDeploymentMetadata& metadata)
{
  if (requireInt(manifest, "schema_version") != 1)
  {
    throw std::runtime_error("Golden inference manifest schema version mismatch");
  }
  if (requireDouble(manifest, "absolute_tolerance") != kExpectedAbsoluteLogitTolerance)
  {
    throw std::runtime_error("Golden inference manifest absolute tolerance is not 1e-5");
  }
  if (!requireObjectMember(manifest, "relative_tolerance").is_null())
  {
    throw std::runtime_error(
        "Golden inference manifest relative tolerance must be null/disabled");
  }
  if (requireString(manifest, "input_name") != metadata.inputName() ||
      requireString(manifest, "output_name") != metadata.outputName())
  {
    throw std::runtime_error(
        "Golden inference manifest tensor names disagree with deployment metadata");
  }

  const Json& classOrder = requireObjectMember(manifest, "class_order");
  if (!classOrder.is_array() || classOrder.size() != kClassCount)
  {
    throw std::runtime_error("Golden inference class order must contain exactly five entries");
  }
  for (std::size_t index = 0; index < kClassCount; ++index)
  {
    if (requireInt(classOrder.at(index), "index") != static_cast<int>(index) ||
        requireString(classOrder.at(index), "name") != metadata.classNames()[index])
    {
      throw std::runtime_error(
          "Golden inference class order disagrees with deployment metadata");
    }
  }
  validateOrderedPatients(manifest, "Golden inference manifest");
}

template <typename Values>
std::size_t firstMaximumIndex(const Values& values)
{
  if (values.empty())
  {
    throw std::runtime_error("Cannot compute argmax of an empty sequence");
  }
  return static_cast<std::size_t>(std::distance(values.begin(),
                                                std::max_element(values.begin(), values.end())));
}

ProbabilityArray stableSoftmax(std::span<const float> logits)
{
  if (logits.size() != kClassCount)
  {
    throw std::runtime_error("Golden logits must contain exactly five values");
  }
  for (const float logit : logits)
  {
    if (!std::isfinite(logit))
    {
      throw std::runtime_error("Golden logits contain a non-finite value");
    }
  }

  const float maximumLogit = *std::max_element(logits.begin(), logits.end());
  ProbabilityArray probabilities{};
  double denominator = 0.0;
  for (std::size_t index = 0; index < logits.size(); ++index)
  {
    probabilities[index] =
        std::exp(static_cast<double>(logits[index]) - static_cast<double>(maximumLogit));
    if (!std::isfinite(probabilities[index]))
    {
      throw std::runtime_error("Golden softmax produced a non-finite exponential term");
    }
    denominator += probabilities[index];
  }
  if (!std::isfinite(denominator) || denominator <= 0.0)
  {
    throw std::runtime_error("Golden softmax produced an invalid denominator");
  }

  for (double& probability : probabilities)
  {
    probability /= denominator;
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
    {
      throw std::runtime_error("Golden softmax produced an invalid probability");
    }
  }
  const double probabilitySum =
      std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
  if (!std::isfinite(probabilitySum) ||
      std::fabs(probabilitySum - 1.0) > kProbabilityNormalizationTolerance)
  {
    throw std::runtime_error("Golden softmax probabilities are not normalized");
  }
  return probabilities;
}

struct MaximumError
{
  double value = 0.0;
  std::size_t index = 0;
};

MaximumError maximumAbsoluteError(std::span<const float> actual,
                                  std::span<const float> expected)
{
  if (actual.size() != kClassCount || expected.size() != kClassCount)
  {
    throw std::runtime_error("Actual and golden logits must each contain exactly five values");
  }

  MaximumError maximum;
  for (std::size_t index = 0; index < kClassCount; ++index)
  {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
    {
      throw std::runtime_error("Actual or golden logits contain a non-finite value");
    }
    const double error = std::fabs(static_cast<double>(actual[index]) -
                                   static_cast<double>(expected[index]));
    if (error > maximum.value)
    {
      maximum.value = error;
      maximum.index = index;
    }
  }
  return maximum;
}

MaximumError maximumAbsoluteError(std::span<const double> actual,
                                  std::span<const double> expected)
{
  if (actual.size() != kClassCount || expected.size() != kClassCount)
  {
    throw std::runtime_error("Actual and expected probabilities must each contain five values");
  }

  MaximumError maximum;
  for (std::size_t index = 0; index < kClassCount; ++index)
  {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
        actual[index] < 0.0 || actual[index] > 1.0)
    {
      throw std::runtime_error("Actual or expected probabilities violate their contract");
    }
    const double error = std::fabs(actual[index] - expected[index]);
    if (error > maximum.value)
    {
      maximum.value = error;
      maximum.index = index;
    }
  }
  return maximum;
}

struct PatientPaths
{
  std::filesystem::path ed;
  std::filesystem::path es;
  std::filesystem::path info;
  std::filesystem::path logits;
};

class CardiacMriClassificationServiceGoldenParityTest final
{
public:
  CardiacMriClassificationServiceGoldenParityTest(std::filesystem::path packageRoot,
                                                   std::filesystem::path datasetRoot)
    : packageRoot_(std::move(packageRoot)),
      datasetRoot_(std::move(datasetRoot))
  {
  }

  void run() const
  {
    validateRoots();

    const Json deployment = readJson(packageRoot_ / "deployment.json");
    const Json preprocessingManifest =
        readJson(packageRoot_ / "golden/preprocessing/manifest.json");
    const Json inferenceManifest =
        readJson(packageRoot_ / "golden/inference/manifest.json");
    const auto metadata =
        maiw::cardiac::CardiacMriDeploymentMetadata::load(packageRoot_);

    validatePreprocessingManifest(preprocessingManifest);
    validateInferenceManifest(inferenceManifest, metadata);
    validateGlobalArtifacts(deployment, preprocessingManifest, inferenceManifest, metadata);

    Ort::Env environment(
        ORT_LOGGING_LEVEL_ERROR,
        "medical-ai-workstation-classification-service-golden-parity");
    maiw::cardiac::CardiacMriClassificationService service(environment, metadata);

    const Json& preprocessingPatients =
        requireObjectMember(preprocessingManifest, "patients");
    const Json& inferencePatients = requireObjectMember(inferenceManifest, "patients");

    bool allPatientsPassed = true;
    for (std::size_t index = 0; index < kExpectedPatientCount; ++index)
    {
      const std::string patientId{kExpectedPatientIds[index]};
      try
      {
        const bool passed = runPatient(patientId,
                                       preprocessingPatients.at(index),
                                       inferencePatients.at(index),
                                       metadata,
                                       service);
        allPatientsPassed = allPatientsPassed && passed;
      }
      catch (const std::exception& error)
      {
        allPatientsPassed = false;
        std::cerr << patientId
                  << ": max_abs_error=unavailable"
                  << " predicted_class_index=unavailable"
                  << " expected_class_index=unavailable"
                  << " predicted_class_name=unavailable"
                  << " status=FAIL error=" << error.what() << '\n';
      }
    }

    if (!allPatientsPassed)
    {
      throw std::runtime_error(
          "Cardiac MRI classification service golden parity failed for one or more patients");
    }
    std::cout << "All seven cardiac MRI classification service golden cases passed." << '\n';
  }

private:
  void validateRoots() const
  {
    if (!std::filesystem::is_directory(packageRoot_))
    {
      throw std::runtime_error("Package root does not exist: " + packageRoot_.string());
    }
    if (!std::filesystem::is_directory(datasetRoot_))
    {
      throw std::runtime_error("ACDC training root does not exist: " + datasetRoot_.string());
    }
  }

  void validateGlobalArtifacts(
      const Json& deployment,
      const Json& preprocessingManifest,
      const Json& inferenceManifest,
      const maiw::cardiac::CardiacMriDeploymentMetadata& metadata) const
  {
    const Json& onnx = requireObjectMember(deployment, "onnx");
    const std::string deploymentModelSha = requireString(onnx, "sha256");
    if (requireString(inferenceManifest, "model_sha256") != deploymentModelSha)
    {
      throw std::runtime_error(
          "Golden inference manifest model digest disagrees with deployment.json");
    }
    requireFileSha256(metadata.modelPath(), deploymentModelSha, "classifier.onnx");

    const std::filesystem::path preprocessingContract =
        packageRoot_ / requireRelativeFixturePath(preprocessingManifest,
                                                  "preprocessing_contract");
    requireFileSha256(preprocessingContract,
                      requireString(preprocessingManifest,
                                    "preprocessing_contract_sha256"),
                      "preprocessing contract");
  }

  static void validatePatientContracts(
      std::string_view patientId,
      const Json& preprocessingPatient,
      const Json& inferencePatient,
      const maiw::cardiac::CardiacMriDeploymentMetadata& metadata)
  {
    if (requireString(preprocessingPatient, "patient_id") != patientId ||
        requireString(inferencePatient, "patient_id") != patientId)
    {
      throw std::runtime_error(std::string(patientId) + ": manifest identity mismatch");
    }
    requireExactShape(requireShape(preprocessingPatient, "tensor_shape"),
                      kPreprocessingTensorShape,
                      "preprocessing tensor");
    requireExactShape(requireShape(inferencePatient, "input_shape"),
                      kInferenceInputShape,
                      "inference input");
    requireExactShape(requireShape(inferencePatient, "logits_shape"),
                      kGoldenLogitsShape,
                      "inference logits");
    if (requireString(preprocessingPatient, "tensor_dtype") != "float32" ||
        requireString(inferencePatient, "input_dtype") != "float32" ||
        requireString(inferencePatient, "logits_dtype") != "float32")
    {
      throw std::runtime_error(std::string(patientId) + ": manifest dtype mismatch");
    }

    const int expectedClass = requireInt(inferencePatient, "predicted_class_index");
    if (expectedClass < 0 || expectedClass >= static_cast<int>(kClassCount))
    {
      throw std::runtime_error(std::string(patientId) +
                               ": predicted_class_index is outside [0,4]");
    }
    const std::string& expectedName = metadata.classNames()[expectedClass];
    if (requireInt(preprocessingPatient, "class_index") != expectedClass ||
        requireString(preprocessingPatient, "class_name") != expectedName ||
        requireString(inferencePatient, "class_name") != expectedName ||
        requireString(inferencePatient, "predicted_class_name") != expectedName)
    {
      throw std::runtime_error(std::string(patientId) +
                               ": frozen class metadata is inconsistent");
    }
  }

  [[nodiscard]] PatientPaths resolvePatientPaths(
      std::string_view patientId,
      const Json& preprocessingPatient,
      const Json& inferencePatient) const
  {
    const std::filesystem::path patientRoot = datasetRoot_ / std::string(patientId);
    return PatientPaths{
        patientRoot / requireFilename(preprocessingPatient, "source_ed_filename"),
        patientRoot / requireFilename(preprocessingPatient, "source_es_filename"),
        patientRoot / requireFilename(preprocessingPatient, "info_filename"),
        packageRoot_ / "golden/inference" /
            requireRelativeFixturePath(inferencePatient, "logits_filename")};
  }

  [[nodiscard]] bool runPatient(
      const std::string& patientId,
      const Json& preprocessingPatient,
      const Json& inferencePatient,
      const maiw::cardiac::CardiacMriDeploymentMetadata& metadata,
      maiw::cardiac::CardiacMriClassificationService& service) const
  {
    validatePatientContracts(patientId, preprocessingPatient, inferencePatient, metadata);
    const PatientPaths paths =
        resolvePatientPaths(patientId, preprocessingPatient, inferencePatient);

    requireFileSha256(paths.ed,
                      requireString(preprocessingPatient, "source_ed_sha256"),
                      patientId + " ED");
    requireFileSha256(paths.es,
                      requireString(preprocessingPatient, "source_es_sha256"),
                      patientId + " ES");
    requireFileSha256(paths.info,
                      requireString(preprocessingPatient, "info_sha256"),
                      patientId + " Info.cfg");
    requireFileSha256(paths.logits,
                      requireString(inferencePatient, "logits_sha256"),
                      patientId + " golden logits");

    const qvp::VolumeData edVolume = loadRequiredVolume(paths.ed, patientId + " ED");
    const qvp::VolumeData esVolume = loadRequiredVolume(paths.es, patientId + " ES");
    const maiw::cardiac::CardiacMriClassificationResult result =
        service.classify(edVolume, esVolume);
    const maiw::test::NpyFloat32Array goldenLogits = maiw::test::readNpyFloat32(paths.logits);

    requireExactShape(goldenLogits.shape, kGoldenLogitsShape, patientId + " golden logits");
    if (result.rawLogits().size() != kClassCount || goldenLogits.data.size() != kClassCount)
    {
      throw std::runtime_error(patientId + ": raw or golden logit count is not five");
    }

    const std::size_t expectedClass =
        static_cast<std::size_t>(requireInt(inferencePatient, "predicted_class_index"));
    if (firstMaximumIndex(goldenLogits.data) != expectedClass)
    {
      throw std::runtime_error(patientId +
                               ": golden logits disagree with predicted_class_index");
    }

    const MaximumError logitError = maximumAbsoluteError(result.rawLogits(), goldenLogits.data);
    const ProbabilityArray expectedProbabilities = stableSoftmax(goldenLogits.data);
    const MaximumError probabilityError =
        maximumAbsoluteError(result.probabilities(), expectedProbabilities);
    const double probabilitySum =
        std::accumulate(result.probabilities().begin(), result.probabilities().end(), 0.0);

    const bool passed =
        logitError.value <= kExpectedAbsoluteLogitTolerance &&
        result.predictedClassIndex() == expectedClass &&
        result.predictedClassName() == metadata.classNames()[expectedClass] &&
        firstMaximumIndex(result.rawLogits()) == expectedClass &&
        probabilityError.value <= kExpectedProbabilityTolerance &&
        std::isfinite(probabilitySum) &&
        std::fabs(probabilitySum - 1.0) <= kProbabilityNormalizationTolerance &&
        firstMaximumIndex(expectedProbabilities) == expectedClass &&
        firstMaximumIndex(result.probabilities()) == expectedClass;

    printPatientResult(patientId,
                       logitError,
                       probabilityError,
                       result.predictedClassIndex(),
                       expectedClass,
                       result.predictedClassName(),
                       passed);
    return passed;
  }

  static void printPatientResult(const std::string& patientId,
                                 const MaximumError& logitError,
                                 const MaximumError& probabilityError,
                                 std::size_t predictedClass,
                                 std::size_t expectedClass,
                                 const std::string& predictedClassName,
                                 bool passed)
  {
    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<double>::max_digits10)
              << patientId
              << ": max_abs_error=" << logitError.value
              << " worst_logit_index=" << logitError.index
              << " max_probability_error=" << probabilityError.value
              << " worst_probability_index=" << probabilityError.index
              << " predicted_class_index=" << predictedClass
              << " expected_class_index=" << expectedClass
              << " predicted_class_name=" << predictedClassName
              << " status=" << (passed ? "PASS" : "FAIL") << '\n';
  }

  std::filesystem::path packageRoot_;
  std::filesystem::path datasetRoot_;
};

} // namespace

int main()
{
  try
  {
    const CardiacMriClassificationServiceGoldenParityTest test{
        std::filesystem::path{MAIW_CARDIAC_MRI_PACKAGE_DIR},
        std::filesystem::path{MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR}};
    test.run();
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI classification service golden parity test failed: "
              << error.what() << '\n';
    return 1;
  }
}
