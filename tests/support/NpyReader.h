#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace maiw::test
{

struct NpyFloat32Array
{
  std::vector<float> data;
  std::vector<std::size_t> shape;
  std::string descriptor;
};

NpyFloat32Array readNpyFloat32(const std::filesystem::path& path);

} // namespace maiw::test
