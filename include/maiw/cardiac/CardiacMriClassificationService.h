#pragma once

#include "maiw/cardiac/CardiacMriClassificationResult.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/cardiac/CardiacMriPreprocessing.h"
#include "maiw/onnx/OnnxRuntimeSession.h"

namespace maiw::cardiac
{

/**
 * @brief Synchronous headless cardiac MRI classification service.
 *
 * The service owns its deployment metadata, cardiac preprocessor, and ONNX
 * Runtime session. The Ort::Env supplied at construction remains externally
 * owned and must outlive the service.
 */
class CardiacMriClassificationService final
{
public:
  /**
   * @brief Construct the service and validate the deployed ONNX model contract.
   *
   * @param environment Externally owned ONNX Runtime environment that must
   * outlive this service.
   * @param metadata Validated cardiac deployment metadata copied or moved into
   * the service.
   * @throws std::runtime_error If the runtime model violates the cardiac input
   * or output contract.
   */
  CardiacMriClassificationService(Ort::Env& environment,
                                  CardiacMriDeploymentMetadata metadata);

  /**
   * @brief Preprocess ED/ES volumes, run inference, and return a validated result.
   *
   * @throws std::invalid_argument If the volumes or inference inputs are invalid.
   * @throws std::overflow_error If preprocessing dimensions exceed supported sizes.
   * @throws std::runtime_error If inference or runtime output validation fails.
   */
  [[nodiscard]] CardiacMriClassificationResult classify(
      const qvp::VolumeData& edVolume,
      const qvp::VolumeData& esVolume);

private:
  void validateModelContract() const;

  CardiacMriDeploymentMetadata metadata_;
  CardiacMriPreprocessor preprocessor_;
  maiw::onnx::OnnxRuntimeSession session_;
};

} // namespace maiw::cardiac
