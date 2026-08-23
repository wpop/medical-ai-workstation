#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"

#include "qtviewerpro/io/MedicalVolumeLoaderRegistry.h"

#include <QString>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>

namespace
{

constexpr double kProbabilitySumTolerance = 1e-12;

qvp::VolumeData loadRequiredVolume(const QString& path, const char* name)
{
  const qvp::VolumeLoadResult result = qvp::loadMedicalVolume(path);
  if (!result.success)
  {
    throw std::runtime_error(std::string(name) + " load failed: " +
                             result.errorMessage.toStdString());
  }
  return result.volume;
}

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

template <typename Values>
std::size_t firstMaximumIndex(const Values& values)
{
  return static_cast<std::size_t>(std::distance(values.begin(),
                                                std::max_element(values.begin(), values.end())));
}

void validateResult(
    const maiw::cardiac::CardiacMriClassificationResult& result,
    const maiw::cardiac::CardiacMriDeploymentMetadata::ClassNames& classNames)
{
  for (const float logit : result.rawLogits())
  {
    require(std::isfinite(logit), "service result contains a non-finite raw logit");
  }

  for (const double probability : result.probabilities())
  {
    require(std::isfinite(probability), "service result contains a non-finite probability");
    require(probability >= 0.0 && probability <= 1.0,
            "service probability is outside the closed unit interval");
  }

  const double probabilitySum =
      std::accumulate(result.probabilities().begin(), result.probabilities().end(), 0.0);
  require(std::fabs(probabilitySum - 1.0) <= kProbabilitySumTolerance,
          "service probabilities are not normalized");

  const std::size_t predictedIndex = result.predictedClassIndex();
  require(predictedIndex < maiw::cardiac::CardiacMriDeploymentMetadata::kClassCount,
          "service predicted class index is out of range");
  require(result.predictedClassName() == classNames[predictedIndex],
          "service predicted class name does not match deployment ordering");
  require(firstMaximumIndex(result.rawLogits()) == predictedIndex,
          "service raw-logit argmax does not match the prediction");
  require(firstMaximumIndex(result.probabilities()) == predictedIndex,
          "service probability argmax does not match the prediction");
}

} // namespace

int main()
{
  try
  {
    const qvp::VolumeData edVolume =
        loadRequiredVolume(QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH), "ED volume");
    const qvp::VolumeData esVolume =
        loadRequiredVolume(QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ES_PATH), "ES volume");

    const auto metadata = maiw::cardiac::CardiacMriDeploymentMetadata::load(
        std::filesystem::path{MAIW_CARDIAC_MRI_PACKAGE_DIR});
    Ort::Env environment(ORT_LOGGING_LEVEL_ERROR,
                         "medical-ai-workstation-classification-service-smoke");
    maiw::cardiac::CardiacMriClassificationService service(environment, metadata);

    const maiw::cardiac::CardiacMriClassificationResult result =
        service.classify(edVolume, esVolume);
    validateResult(result, metadata.classNames());

    std::cout << "Cardiac MRI classification service smoke passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI classification service smoke failed: " << error.what() << '\n';
    return 1;
  }
}
