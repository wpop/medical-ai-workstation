#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>

namespace maiw::cardiac
{

/**
 * @brief Validated runtime contract for a selected cardiac MRI deployment package.
 *
 * Instances can only be created by loading both deployment metadata files and
 * validating the fixed model interface required by the cardiac classifier.
 */
class CardiacMriDeploymentMetadata final
{
public:
  static constexpr std::size_t kClassCount = 5;
  using ClassNames = std::array<std::string, kClassCount>;

  /**
   * @brief Load and validate the cardiac runtime contract from a package root.
   * @throws std::invalid_argument If packageRoot is empty or is not a directory.
   * @throws std::runtime_error If metadata, its contract, or the model file is invalid.
   */
  [[nodiscard]] static CardiacMriDeploymentMetadata load(
      const std::filesystem::path& packageRoot);

  /**
   * @brief Return the validated path to the package's ONNX model file.
   */
  [[nodiscard]] const std::filesystem::path& modelPath() const noexcept;

  /**
   * @brief Return the validated ONNX input tensor name.
   */
  [[nodiscard]] const std::string& inputName() const noexcept;

  /**
   * @brief Return the validated ONNX output tensor name.
   */
  [[nodiscard]] const std::string& outputName() const noexcept;

  /**
   * @brief Return class names in deployment output order.
   */
  [[nodiscard]] const ClassNames& classNames() const noexcept;

private:
  CardiacMriDeploymentMetadata(std::filesystem::path modelPath,
                               std::string inputName,
                               std::string outputName,
                               ClassNames classNames);

  std::filesystem::path modelPath_;
  std::string inputName_;
  std::string outputName_;
  ClassNames classNames_;
};

} // namespace maiw::cardiac
