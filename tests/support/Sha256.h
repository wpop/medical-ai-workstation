#pragma once

#include <filesystem>
#include <string>

namespace maiw::test
{

std::string sha256FileHex(const std::filesystem::path& path);

} // namespace maiw::test
