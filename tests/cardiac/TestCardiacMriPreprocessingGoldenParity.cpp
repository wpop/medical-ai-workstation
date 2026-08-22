#include "NpyReader.h"
#include "Sha256.h"
#include "maiw/cardiac/CardiacMriPreprocessing.h"
#include "qtviewerpro/io/MedicalVolumeLoaderRegistry.h"

#include <nlohmann/json.hpp>

#include <QString>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Json = nlohmann::json;

constexpr std::size_t kExpectedPatientCount = 7;
constexpr int kExpectedFixtureVersion = 1;

const std::vector<std::size_t> kExpectedTensorShape{2, 14, 144, 144};

const std::array<std::string, kExpectedPatientCount> kExpectedPatientIds{
    "patient001",
    "patient006",
    "patient021",
    "patient042",
    "patient063",
    "patient085",
    "patient094",
};

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

std::size_t requireSizeT(const Json& object, const char* key)
{
  const Json& value = requireObjectMember(object, key);
  if (value.is_number_unsigned())
  {
    const auto parsed = value.get<unsigned long long>();
    if (parsed > std::numeric_limits<std::size_t>::max())
    {
      throw std::overflow_error(std::string("JSON integer value exceeds size limits: ") + key);
    }
    return static_cast<std::size_t>(parsed);
  }

  if (!value.is_number_integer())
  {
    throw std::runtime_error(std::string("JSON value must be an integer: ") + key);
  }

  const auto parsed = value.get<long long>();
  if (parsed < 0)
  {
    throw std::runtime_error(std::string("JSON integer value must be non-negative: ") + key);
  }

  return static_cast<std::size_t>(parsed);
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

  for (const auto& dimension : value)
  {
    if (!dimension.is_number_integer() && !dimension.is_number_unsigned())
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

void requireEqualShape(const std::vector<std::size_t>& actual,
                       const std::vector<std::size_t>& expected,
                       const std::string& context)
{
  if (actual != expected)
  {
    throw std::runtime_error(context + " shape mismatch");
  }
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

std::vector<std::size_t> shapeXyzFromDimensions(const maiw::cardiac::VolumeDimensions& dimensions)
{
  return {dimensions.width, dimensions.height, dimensions.depth};
}

void printShape(std::ostream& output, const std::vector<std::size_t>& shape)
{
  output << '[';
  for (std::size_t index = 0; index < shape.size(); ++index)
  {
    if (index != 0)
    {
      output << ',';
    }
    output << shape[index];
  }
  output << ']';
}

// Fixture paths are constrained to filenames so the manifest cannot redirect
// validation outside the configured package or ACDC training roots.
std::filesystem::path requireFilename(const Json& object, const char* key)
{
  const std::filesystem::path path{requireString(object, key)};
  if (path.empty() || path.is_absolute() || path.has_parent_path())
  {
    throw std::runtime_error(std::string("Manifest value must be a filename: ") + key);
  }
  return path;
}

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

// The manifest is part of the validation contract, not just test input data.
void validateManifestContract(const Json& manifest)
{
  if (requireString(manifest, "dataset") != "ACDC")
  {
    throw std::runtime_error("Golden preprocessing manifest dataset is not ACDC");
  }

  if (requireInt(manifest, "fixture_version") != kExpectedFixtureVersion)
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

  if (requireString(manifest, "preprocessing_contract") != "docs/preprocessing_contract.md")
  {
    throw std::runtime_error("Golden preprocessing contract path mismatch");
  }

  requireEqualShape(requireShape(manifest, "tensor_shape"),
                    kExpectedTensorShape,
                    "golden preprocessing manifest tensor");

  const Json& channelSemantics = requireObjectMember(manifest, "channel_semantics");
  if (requireString(channelSemantics, "0") != "ED" ||
      requireString(channelSemantics, "1") != "ES")
  {
    throw std::runtime_error("Golden preprocessing channel semantics must be ED then ES");
  }

  const Json& patients = requireObjectMember(manifest, "patients");
  if (!patients.is_array() || patients.size() != kExpectedPatientCount)
  {
    throw std::runtime_error("Golden preprocessing manifest must contain exactly seven patients");
  }

  for (std::size_t index = 0; index < kExpectedPatientIds.size(); ++index)
  {
    const std::string patientId = requireString(patients.at(index), "patient_id");
    if (patientId != kExpectedPatientIds[index])
    {
      throw std::runtime_error("Golden preprocessing patient set/order mismatch");
    }
  }
}

struct ErrorSummary
{
  std::size_t differingElements = 0;
  double maximumAbsoluteError = 0.0;
  double meanAbsoluteError = 0.0;
  std::size_t worstFlatIndex = 0;
  float expectedAtWorst = 0.0F;
  float actualAtWorst = 0.0F;
};

ErrorSummary compareTensors(const std::vector<float>& actual,
                            const std::vector<float>& expected,
                            const std::string& patientId)
{
  if (actual.size() != expected.size())
  {
    throw std::runtime_error(patientId + ": actual/expected tensor size mismatch");
  }

  if (actual.empty())
  {
    throw std::runtime_error(patientId + ": tensor must not be empty");
  }

  ErrorSummary summary;
  summary.expectedAtWorst = expected.front();
  summary.actualAtWorst = actual.front();

  double absoluteErrorSum = 0.0;

  // Exact float32 equality is the first parity gate; error statistics are
  // diagnostic only and never relax the pass/fail decision.
  for (std::size_t index = 0; index < actual.size(); ++index)
  {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
    {
      throw std::runtime_error(patientId + ": tensor contains non-finite values");
    }

    if (actual[index] != expected[index])
    {
      ++summary.differingElements;
    }

    const double absoluteError =
        std::fabs(static_cast<double>(actual[index]) -
                  static_cast<double>(expected[index]));

    absoluteErrorSum += absoluteError;

    if (absoluteError > summary.maximumAbsoluteError)
    {
      summary.maximumAbsoluteError = absoluteError;
      summary.worstFlatIndex = index;
      summary.expectedAtWorst = expected[index];
      summary.actualAtWorst = actual[index];
    }
  }

  summary.meanAbsoluteError =
      absoluteErrorSum / static_cast<double>(actual.size());

  return summary;
}

std::array<std::size_t, 4> decodeCdhwIndex(std::size_t flatIndex)
{
  const std::size_t width =
      flatIndex % maiw::cardiac::CardiacMriInputTensor::kWidth;
  flatIndex /= maiw::cardiac::CardiacMriInputTensor::kWidth;

  const std::size_t height =
      flatIndex % maiw::cardiac::CardiacMriInputTensor::kHeight;
  flatIndex /= maiw::cardiac::CardiacMriInputTensor::kHeight;

  const std::size_t depth =
      flatIndex % maiw::cardiac::CardiacMriInputTensor::kDepth;
  flatIndex /= maiw::cardiac::CardiacMriInputTensor::kDepth;

  const std::size_t channel = flatIndex;

  return {channel, depth, height, width};
}

struct StagedPreprocessingDiagnostics
{
  maiw::cardiac::CardiacMriInputTensor tensor;
  std::vector<std::size_t> resampledShapeXyz;
  std::size_t validDepth = 0;
  std::size_t zPaddingLower = 0;
  std::size_t zPaddingUpper = 0;
  maiw::cardiac::CardiacMriNormalizationMetadata normalization;
};

StagedPreprocessingDiagnostics runStagedPreprocessingDiagnostics(
    const qvp::VolumeData& edVolume,
    const qvp::VolumeData& esVolume)
{
  // This diagnostic path deliberately calls the public production stages in the
  // same order as the orchestrator so metadata can be localized without
  // duplicating preprocessing algorithms.
  const qvp::VolumeData edLps = maiw::cardiac::normalizeVolumeDataToLps(edVolume);
  const qvp::VolumeData esLps = maiw::cardiac::normalizeVolumeDataToLps(esVolume);
  maiw::cardiac::validateLpsOrientedVolumePair(edLps, esLps);

  const maiw::cardiac::CardiacMriXyzVolumePair resampled =
      maiw::cardiac::resampleOrientedPairToEdDerivedGrid(edLps, esLps);
  const maiw::cardiac::CardiacMriXyzVolumePair cropped =
      maiw::cardiac::cropResampledPairToFrozenXy(resampled.ed, resampled.es);
  const maiw::cardiac::CardiacMriNormalizationResult normalized =
      maiw::cardiac::normalizeCroppedPairIntensities(cropped.ed, cropped.es);

  const std::size_t validDepth = normalized.volumes.ed.dimensions.depth;
  const std::size_t targetDepth = maiw::cardiac::CardiacMriInputTensor::kDepth;
  if (validDepth > targetDepth)
  {
    throw std::runtime_error("Diagnostic staged preprocessing depth exceeds frozen depth");
  }
  const std::size_t totalZPadding = targetDepth - validDepth;
  const std::size_t zPaddingLower = totalZPadding / 2U;
  const std::size_t zPaddingUpper = totalZPadding - zPaddingLower;

  const maiw::cardiac::NormalizedCardiacMriXyzVolume edPadded =
      maiw::cardiac::padNormalizedVolumeZToFrozenDepth(normalized.volumes.ed);
  const maiw::cardiac::NormalizedCardiacMriXyzVolume esPadded =
      maiw::cardiac::padNormalizedVolumeZToFrozenDepth(normalized.volumes.es);
  maiw::cardiac::CardiacMriInputTensor tensor =
      maiw::cardiac::stackNormalizedPairToInputTensor(edPadded, esPadded);

  return StagedPreprocessingDiagnostics{
      std::move(tensor),
      shapeXyzFromDimensions(resampled.ed.dimensions),
      validDepth,
      zPaddingLower,
      zPaddingUpper,
      normalized.metadata};
}

void printDiagnostics(const std::string& patientId,
                      const ErrorSummary& summary)
{
  // Decode the C-contiguous CDHW offset so a mismatch can be traced directly
  // back to channel, slice, row, and column.
  const auto index = decodeCdhwIndex(summary.worstFlatIndex);

  std::cout << std::scientific
            << std::setprecision(std::numeric_limits<double>::max_digits10)
            << patientId
            << ": exact_equal="
            << (summary.differingElements == 0 ? "true" : "false")
            << " differing_elements=" << summary.differingElements
            << " max_abs_error=" << summary.maximumAbsoluteError
            << " mean_abs_error=" << summary.meanAbsoluteError
            << " worst_flat_index=" << summary.worstFlatIndex
            << " worst_cdhw=["
            << index[0] << ','
            << index[1] << ','
            << index[2] << ','
            << index[3] << ']'
            << " expected=" << static_cast<double>(summary.expectedAtWorst)
            << " actual=" << static_cast<double>(summary.actualAtWorst)
            << '\n';
}

void printShapeMetadataDiagnostic(const std::string& patientId,
                                  const char* key,
                                  const std::vector<std::size_t>& expected,
                                  const std::vector<std::size_t>& actual)
{
  const bool exact = expected == actual;
  std::cout << patientId
            << " metadata " << key
            << ": exact_equal=" << (exact ? "true" : "false");
  if (!exact)
  {
    std::cout << " structural_mismatch=true";
  }
  std::cout << " python_expected=";
  printShape(std::cout, expected);
  std::cout << " cpp_actual=";
  printShape(std::cout, actual);
  std::cout << '\n';
}

void printIntegerMetadataDiagnostic(const std::string& patientId,
                                    const char* key,
                                    std::size_t expected,
                                    std::size_t actual)
{
  const bool exact = expected == actual;
  std::cout << patientId
            << " metadata " << key
            << ": exact_equal=" << (exact ? "true" : "false");
  if (!exact)
  {
    std::cout << " structural_mismatch=true";
  }
  std::cout << " python_expected=" << expected
            << " cpp_actual=" << actual
            << '\n';
}

void printFloatingMetadataDiagnostic(const std::string& patientId,
                                     const char* key,
                                     double expected,
                                     double actual)
{
  const double absoluteError = std::fabs(actual - expected);
  std::cout << std::scientific
            << std::setprecision(std::numeric_limits<double>::max_digits10)
            << patientId
            << " metadata " << key
            << ": exact_equal=" << (actual == expected ? "true" : "false")
            << " python_expected=" << expected
            << " cpp_actual=" << actual
            << " abs_error=" << absoluteError
            << '\n';
}

void printMetadataDiagnostics(const std::string& patientId,
                              const Json& patient,
                              const StagedPreprocessingDiagnostics& diagnostics)
{
  // Manifest metadata deltas are printed for localization only; the Python
  // golden tensor exact comparison remains the active parity gate.
  printShapeMetadataDiagnostic(patientId,
                               "resampled_shape_xyz",
                               requireShape(patient, "resampled_shape_xyz"),
                               diagnostics.resampledShapeXyz);
  printIntegerMetadataDiagnostic(patientId,
                                 "valid_depth",
                                 requireSizeT(patient, "valid_depth"),
                                 diagnostics.validDepth);
  printIntegerMetadataDiagnostic(patientId,
                                 "z_padding_lower",
                                 requireSizeT(patient, "z_padding_lower"),
                                 diagnostics.zPaddingLower);
  printIntegerMetadataDiagnostic(patientId,
                                 "z_padding_upper",
                                 requireSizeT(patient, "z_padding_upper"),
                                 diagnostics.zPaddingUpper);

  const Json& normalization = requireObjectMember(patient, "normalization");
  printFloatingMetadataDiagnostic(patientId,
                                  "normalization.clip_lower",
                                  requireDouble(normalization, "clip_lower"),
                                  diagnostics.normalization.clipLower);
  printFloatingMetadataDiagnostic(patientId,
                                  "normalization.clip_upper",
                                  requireDouble(normalization, "clip_upper"),
                                  diagnostics.normalization.clipUpper);
  printFloatingMetadataDiagnostic(patientId,
                                  "normalization.mean",
                                  requireDouble(normalization, "mean"),
                                  diagnostics.normalization.mean);
  printFloatingMetadataDiagnostic(patientId,
                                  "normalization.std",
                                  requireDouble(normalization, "std"),
                                  diagnostics.normalization.std);
}

} // namespace

int main()
{
  try
  {
    const std::filesystem::path packageRoot{MAIW_CARDIAC_MRI_PACKAGE_DIR};
    const std::filesystem::path datasetRoot{MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR};

    if (!std::filesystem::is_directory(packageRoot))
    {
      throw std::runtime_error("Package root does not exist: " +
                               packageRoot.string());
    }

    if (!std::filesystem::is_directory(datasetRoot))
    {
      throw std::runtime_error("ACDC training root does not exist: " +
                               datasetRoot.string());
    }

    const std::filesystem::path goldenRoot =
        packageRoot / "golden/preprocessing";
    const std::filesystem::path manifestPath =
        goldenRoot / "manifest.json";

    const Json manifest = readJson(manifestPath);
    validateManifestContract(manifest);

    // Hash validation ties every golden run to the exact source files and
    // preprocessing contract used to produce the fixture tensors.
    const std::filesystem::path contractPath =
        packageRoot / requireString(manifest, "preprocessing_contract");

    requireFileSha256(
        contractPath,
        requireString(manifest, "preprocessing_contract_sha256"),
        "preprocessing contract");

    const Json& patients = requireObjectMember(manifest, "patients");
    const maiw::cardiac::CardiacMriPreprocessor preprocessor;

    bool allExact = true;
    bool allStagedPathsMatchOrchestrator = true;

    for (const auto& patient : patients)
    {
      const std::string patientId = requireString(patient, "patient_id");

      requireEqualShape(requireShape(patient, "tensor_shape"),
                        kExpectedTensorShape,
                        patientId + " manifest tensor");

      if (requireString(patient, "tensor_dtype") != "float32")
      {
        throw std::runtime_error(patientId +
                                 ": manifest tensor dtype is not float32");
      }

      const Json& normalization =
          requireObjectMember(patient, "normalization");
      if (requireString(normalization, "scope") != "joint_ed_es")
      {
        throw std::runtime_error(patientId +
                                 ": normalization scope is not joint_ed_es");
      }

      const std::filesystem::path patientRoot =
          datasetRoot / patientId;

      const std::filesystem::path edPath =
          patientRoot / requireFilename(patient, "source_ed_filename");
      const std::filesystem::path esPath =
          patientRoot / requireFilename(patient, "source_es_filename");
      const std::filesystem::path infoPath =
          patientRoot / requireFilename(patient, "info_filename");
      const std::filesystem::path tensorPath =
          goldenRoot / requireFilename(patient, "tensor_filename");

      requireFileSha256(
          edPath,
          requireString(patient, "source_ed_sha256"),
          patientId + " ED");

      requireFileSha256(
          esPath,
          requireString(patient, "source_es_sha256"),
          patientId + " ES");

      requireFileSha256(
          infoPath,
          requireString(patient, "info_sha256"),
          patientId + " Info.cfg");

      requireFileSha256(
          tensorPath,
          requireString(patient, "tensor_sha256"),
          patientId + " golden tensor");

      const qvp::VolumeData edVolume =
          loadRequiredVolume(edPath, patientId + " ED");
      const qvp::VolumeData esVolume =
          loadRequiredVolume(esPath, patientId + " ES");

      const StagedPreprocessingDiagnostics stagedDiagnostics =
          runStagedPreprocessingDiagnostics(edVolume, esVolume);
      printMetadataDiagnostics(patientId, patient, stagedDiagnostics);

      const maiw::cardiac::CardiacMriInputTensor actualTensor =
          preprocessor.preprocess(edVolume, esVolume);

      const ErrorSummary stagedPathSummary =
          compareTensors(stagedDiagnostics.tensor.values(),
                         actualTensor.values(),
                         patientId + " staged-vs-orchestrator");
      printDiagnostics(patientId + " staged-vs-orchestrator",
                       stagedPathSummary);
      if (stagedPathSummary.differingElements != 0)
      {
        allStagedPathsMatchOrchestrator = false;
      }

      if (actualTensor.shapeCdhw() !=
          std::array<std::size_t, 4>{2, 14, 144, 144})
      {
        throw std::runtime_error(patientId +
                                 ": C++ tensor shape mismatch");
      }

      if (actualTensor.values().size() !=
          maiw::cardiac::CardiacMriInputTensor::elementCount())
      {
        throw std::runtime_error(patientId +
                                 ": C++ tensor element count mismatch");
      }

      const maiw::test::NpyFloat32Array expectedTensor =
          maiw::test::readNpyFloat32(tensorPath);

      // The shared NPY reader enforces float32 and C-contiguous storage before
      // this test verifies the frozen CDHW tensor shape.
      requireEqualShape(
          expectedTensor.shape,
          kExpectedTensorShape,
          patientId + " golden tensor");

      if (expectedTensor.data.size() !=
          maiw::cardiac::CardiacMriInputTensor::elementCount())
      {
        throw std::runtime_error(patientId +
                                 ": golden tensor element count mismatch");
      }

      const ErrorSummary summary =
          compareTensors(actualTensor.values(),
                         expectedTensor.data,
                         patientId);

      printDiagnostics(patientId, summary);

      // Keep evaluating all patients so a single run reports the full mismatch
      // surface before the final exact-parity failure.
      if (summary.differingElements != 0)
      {
        allExact = false;
      }
    }

    if (!allStagedPathsMatchOrchestrator)
    {
      throw std::runtime_error(
          "Diagnostic staged preprocessing did not exactly reconstruct the "
          "production orchestrator tensor; inspect the staged-vs-orchestrator "
          "diagnostics above");
    }

    if (!allExact)
    {
      throw std::runtime_error(
          "Python/C++ preprocessing exact float32 parity was not achieved; "
          "inspect the per-patient diagnostics above");
    }

    std::cout
        << "All seven cardiac MRI preprocessing tensors matched exactly."
        << '\n';

    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr
        << "Cardiac MRI preprocessing golden parity test failed: "
        << error.what()
        << '\n';

    return 1;
  }
}
