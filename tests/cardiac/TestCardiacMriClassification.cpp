#include "maiw/cardiac/CardiacMriClassificationResult.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace
{

using maiw::cardiac::CardiacMriClassificationResult;
using maiw::cardiac::CardiacMriDeploymentMetadata;

constexpr double kProbabilityTolerance = 1e-12;

const CardiacMriDeploymentMetadata::ClassNames kClassNames{
    "Class Zero", "Class One", "Class Two", "Class Three", "Class Four"};

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void requireNear(double actual,
                 double expected,
                 double tolerance,
                 const std::string& message)
{
  if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance)
  {
    throw std::runtime_error(message + ": expected " + std::to_string(expected) +
                             ", actual " + std::to_string(actual));
  }
}

template <typename Exception, typename Function>
void requireThrows(Function&& function, const std::string& message)
{
  try
  {
    std::forward<Function>(function)();
  }
  catch (const Exception&)
  {
    return;
  }
  catch (const std::exception& error)
  {
    throw std::runtime_error(message + ": wrong exception type: " + error.what());
  }
  throw std::runtime_error(message + ": expected exception was not thrown");
}

class TemporaryPackage final
{
public:
  TemporaryPackage()
  {
    static std::atomic<std::uint64_t> counter{0};
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto sequence = counter.fetch_add(1, std::memory_order_relaxed);
    root_ = std::filesystem::temp_directory_path() /
            ("maiw-cardiac-classification-" + std::to_string(timestamp) + "-" +
             std::to_string(sequence));
    if (!std::filesystem::create_directory(root_))
    {
      throw std::runtime_error("Failed to create temporary cardiac package: " + root_.string());
    }

    try
    {
      writeDeployment();
      writeClassMapping();
      writeFile(root_ / "classifier.onnx", "");
    }
    catch (...)
    {
      std::error_code error;
      std::filesystem::remove_all(root_, error);
      throw;
    }
  }

  ~TemporaryPackage()
  {
    std::error_code error;
    std::filesystem::remove_all(root_, error);
  }

  TemporaryPackage(const TemporaryPackage&) = delete;
  TemporaryPackage& operator=(const TemporaryPackage&) = delete;
  TemporaryPackage(TemporaryPackage&&) = delete;
  TemporaryPackage& operator=(TemporaryPackage&&) = delete;

  [[nodiscard]] const std::filesystem::path& root() const noexcept
  {
    return root_;
  }

  void writeDeployment(const std::string& filename = "classifier.onnx",
                       bool softmaxInsideModel = false,
                       const std::string& inputShape = "[\"N\",2,14,144,144]",
                       const std::string& outputShape = "[\"N\",5]") const
  {
    const std::string document =
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"model_format\": \"onnx\",\n"
        "  \"model_id\": \"cardiac_mri_pathology\",\n"
        "  \"onnx\": {\n"
        "    \"filename\": \"" +
        filename +
        "\",\n"
        "    \"opset\": 18,\n"
        "    \"input\": {\"name\": \"cine_mri\", \"dtype\": \"float32\", \"shape\": " +
        inputShape +
        "},\n"
        "    \"output\": {\"name\": \"logits\", \"dtype\": \"float32\", \"shape\": " +
        outputShape +
        "},\n"
        "    \"softmax_inside_model\": " +
        (softmaxInsideModel ? "true" : "false") +
        "\n"
        "  },\n"
        "  \"classes\": [\n"
        "    {\"index\": 0, \"name\": \"Class Zero\"},\n"
        "    {\"index\": 1, \"name\": \"Class One\"},\n"
        "    {\"index\": 2, \"name\": \"Class Two\"},\n"
        "    {\"index\": 3, \"name\": \"Class Three\"},\n"
        "    {\"index\": 4, \"name\": \"Class Four\"}\n"
        "  ]\n"
        "}\n";
    writeFile(root_ / "deployment.json", document);
  }

  void writeClassMapping(const std::string& thirdClassName = "Class Two") const
  {
    const std::string document =
        "{\"0\":\"Class Zero\",\"1\":\"Class One\",\"2\":\"" + thirdClassName +
        "\",\"3\":\"Class Three\",\"4\":\"Class Four\"}\n";
    writeFile(root_ / "class_mapping.json", document);
  }

  void removeModel() const
  {
    std::error_code error;
    if (!std::filesystem::remove(root_ / "classifier.onnx", error) || error)
    {
      throw std::runtime_error("Failed to remove temporary model fixture");
    }
  }

private:
  static void writeFile(const std::filesystem::path& path, const std::string& contents)
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
      throw std::runtime_error("Failed to write temporary fixture file: " + path.string());
    }
    output << contents;
    if (!output)
    {
      throw std::runtime_error("Failed to finish temporary fixture file: " + path.string());
    }
  }

  std::filesystem::path root_;
};

void requireValidProbabilities(const CardiacMriClassificationResult::Probabilities& probabilities)
{
  for (const double probability : probabilities)
  {
    require(std::isfinite(probability), "probability must be finite");
    require(probability >= 0.0 && probability <= 1.0,
            "probability must be within the closed unit interval");
  }
  const double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
  requireNear(sum, 1.0, kProbabilityTolerance, "probabilities must be normalized");
}

void testOrdinaryLogits()
{
  const CardiacMriClassificationResult::RawLogits logits{
      -1.25F, 0.5F, 3.0F, 1.0F, -0.25F};
  const CardiacMriClassificationResult result =
      CardiacMriClassificationResult::fromLogits(logits, kClassNames);

  for (std::size_t index = 0; index < logits.size(); ++index)
  {
    require(std::bit_cast<std::uint32_t>(result.rawLogits()[index]) ==
                std::bit_cast<std::uint32_t>(logits[index]),
            "raw logits must be preserved bit-exactly");
  }
  require(result.predictedClassIndex() == 2, "ordinary logits predicted index mismatch");
  require(result.predictedClassName() == "Class Two", "ordinary logits predicted name mismatch");
  requireValidProbabilities(result.probabilities());

  double denominator = 0.0;
  for (const float logit : logits)
  {
    denominator += std::exp(static_cast<double>(logit) - 3.0);
  }
  for (std::size_t index = 0; index < logits.size(); ++index)
  {
    const double expected = std::exp(static_cast<double>(logits[index]) - 3.0) / denominator;
    requireNear(result.probabilities()[index],
                expected,
                kProbabilityTolerance,
                "ordinary logits probability mismatch");
  }
}

void testExtremeFiniteLogits()
{
  const CardiacMriClassificationResult::RawLogits positive{
      997.0F, 999.0F, 1000.0F, 998.0F, 996.0F};
  const auto positiveResult = CardiacMriClassificationResult::fromLogits(positive, kClassNames);
  requireValidProbabilities(positiveResult.probabilities());
  require(positiveResult.predictedClassIndex() == 2,
          "large positive logits predicted index mismatch");

  const CardiacMriClassificationResult::RawLogits negative{
      -1003.0F, -1002.0F, -1004.0F, -1000.0F, -1001.0F};
  const auto negativeResult = CardiacMriClassificationResult::fromLogits(negative, kClassNames);
  requireValidProbabilities(negativeResult.probabilities());
  require(negativeResult.predictedClassIndex() == 3,
          "large negative logits predicted index mismatch");
}

void testEqualLogits()
{
  const CardiacMriClassificationResult::RawLogits logits{
      42.0F, 42.0F, 42.0F, 42.0F, 42.0F};
  const auto result = CardiacMriClassificationResult::fromLogits(logits, kClassNames);
  for (const double probability : result.probabilities())
  {
    requireNear(probability, 0.2, kProbabilityTolerance, "equal-logit probability mismatch");
  }
  require(result.predictedClassIndex() == 0, "equal logits must select the first class");
  require(result.predictedClassName() == "Class Zero",
          "equal logits must map to the first class name");
}

void testInvalidResultContracts()
{
  requireThrows<std::invalid_argument>(
      [] {
        const std::array<float, 4> logits{0.0F, 1.0F, 2.0F, 3.0F};
        static_cast<void>(CardiacMriClassificationResult::fromLogits(logits, kClassNames));
      },
      "wrong logit count must be rejected");

  for (const float invalidValue : {std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity()})
  {
    requireThrows<std::invalid_argument>(
        [invalidValue] {
          const CardiacMriClassificationResult::RawLogits logits{
              0.0F, 1.0F, invalidValue, 3.0F, 4.0F};
          static_cast<void>(CardiacMriClassificationResult::fromLogits(logits, kClassNames));
        },
        "non-finite logit must be rejected");
  }

  requireThrows<std::invalid_argument>(
      [] {
        auto classNames = kClassNames;
        classNames[3].clear();
        const CardiacMriClassificationResult::RawLogits logits{};
        static_cast<void>(CardiacMriClassificationResult::fromLogits(logits, classNames));
      },
      "empty class name must be rejected");
  requireThrows<std::invalid_argument>(
      [] {
        auto classNames = kClassNames;
        classNames[4] = classNames[1];
        const CardiacMriClassificationResult::RawLogits logits{};
        static_cast<void>(CardiacMriClassificationResult::fromLogits(logits, classNames));
      },
      "duplicate class name must be rejected");
}

void testValidDeploymentMetadata()
{
  TemporaryPackage package;
  const CardiacMriDeploymentMetadata metadata =
      CardiacMriDeploymentMetadata::load(package.root());

  require(metadata.modelPath() == package.root() / "classifier.onnx",
          "validated model path mismatch");
  require(metadata.inputName() == "cine_mri", "validated input name mismatch");
  require(metadata.outputName() == "logits", "validated output name mismatch");
  require(metadata.classNames() == kClassNames, "validated class names mismatch");
}

void testInvalidDeploymentMetadata()
{
  {
    TemporaryPackage package;
    package.writeClassMapping("Disagrees");
    requireThrows<std::runtime_error>(
        [&package] { static_cast<void>(CardiacMriDeploymentMetadata::load(package.root())); },
        "class mapping disagreement must be rejected");
  }
  {
    TemporaryPackage package;
    package.writeDeployment("classifier.onnx", true);
    requireThrows<std::runtime_error>(
        [&package] { static_cast<void>(CardiacMriDeploymentMetadata::load(package.root())); },
        "softmax inside model must be rejected");
  }
  {
    TemporaryPackage package;
    package.writeDeployment("classifier.onnx", false, "[\"N\",2,14,144,143]");
    requireThrows<std::runtime_error>(
        [&package] { static_cast<void>(CardiacMriDeploymentMetadata::load(package.root())); },
        "invalid input shape must be rejected");
  }
  {
    TemporaryPackage package;
    package.writeDeployment("classifier.onnx", false, "[\"N\",2,14,144,144]", "[\"N\",6]");
    requireThrows<std::runtime_error>(
        [&package] { static_cast<void>(CardiacMriDeploymentMetadata::load(package.root())); },
        "invalid output shape must be rejected");
  }
  {
    TemporaryPackage package;
    package.writeDeployment("../classifier.onnx");
    requireThrows<std::runtime_error>(
        [&package] { static_cast<void>(CardiacMriDeploymentMetadata::load(package.root())); },
        "model path traversal must be rejected");
  }
  {
    TemporaryPackage package;
    package.removeModel();
    requireThrows<std::runtime_error>(
        [&package] { static_cast<void>(CardiacMriDeploymentMetadata::load(package.root())); },
        "missing model file must be rejected");
  }
}

} // namespace

int main()
{
  try
  {
    testOrdinaryLogits();
    testExtremeFiniteLogits();
    testEqualLogits();
    testInvalidResultContracts();
    testValidDeploymentMetadata();
    testInvalidDeploymentMetadata();
    std::cout << "Cardiac MRI classification contract tests passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI classification contract tests failed: " << error.what() << '\n';
    return 1;
  }
}
