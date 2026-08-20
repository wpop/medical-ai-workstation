#include "NpyReader.h"

#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace
{

constexpr std::string_view kMagicPrefix = "\x93NUMPY";

std::uint16_t readLittleEndianU16(std::istream& input)
{
  unsigned char bytes[2]{};
  input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!input)
  {
    throw std::runtime_error("Failed to read NPY v1 header length");
  }
  return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
}

std::uint32_t readLittleEndianU32(std::istream& input)
{
  unsigned char bytes[4]{};
  input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!input)
  {
    throw std::runtime_error("Failed to read NPY v2/v3 header length");
  }
  return static_cast<std::uint32_t>(bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) |
                                    (bytes[3] << 24));
}

std::string extractQuotedValue(std::string_view header, std::string_view key)
{
  const std::size_t keyPosition = header.find(key);
  if (keyPosition == std::string_view::npos)
  {
    throw std::runtime_error("NPY header is missing required key");
  }

  const std::size_t colon = header.find(':', keyPosition + key.size());
  if (colon == std::string_view::npos)
  {
    throw std::runtime_error("NPY header key is missing ':'");
  }

  const std::size_t quote = header.find_first_of("'\"", colon + 1);
  if (quote == std::string_view::npos)
  {
    throw std::runtime_error("NPY header value is not quoted");
  }

  const char quoteChar = header[quote];
  const std::size_t endQuote = header.find(quoteChar, quote + 1);
  if (endQuote == std::string_view::npos)
  {
    throw std::runtime_error("NPY header quoted value is unterminated");
  }

  return std::string(header.substr(quote + 1, endQuote - quote - 1));
}

bool extractFortranOrder(std::string_view header)
{
  constexpr std::string_view key = "fortran_order";
  const std::size_t keyPosition = header.find(key);
  if (keyPosition == std::string_view::npos)
  {
    throw std::runtime_error("NPY header is missing fortran_order");
  }

  const std::size_t colon = header.find(':', keyPosition + key.size());
  if (colon == std::string_view::npos)
  {
    throw std::runtime_error("NPY fortran_order is missing ':'");
  }

  const std::size_t truePosition = header.find("True", colon + 1);
  const std::size_t falsePosition = header.find("False", colon + 1);
  if (falsePosition != std::string_view::npos &&
      (truePosition == std::string_view::npos || falsePosition < truePosition))
  {
    return false;
  }
  if (truePosition != std::string_view::npos)
  {
    return true;
  }

  throw std::runtime_error("NPY fortran_order must be True or False");
}

std::vector<std::size_t> extractShape(std::string_view header)
{
  constexpr std::string_view key = "shape";
  const std::size_t keyPosition = header.find(key);
  if (keyPosition == std::string_view::npos)
  {
    throw std::runtime_error("NPY header is missing shape");
  }

  const std::size_t open = header.find('(', keyPosition + key.size());
  const std::size_t close = header.find(')', open + 1);
  if (open == std::string_view::npos || close == std::string_view::npos)
  {
    throw std::runtime_error("NPY shape tuple is malformed");
  }

  std::vector<std::size_t> shape;
  std::size_t position = open + 1;
  while (position < close)
  {
    while (position < close && (header[position] == ' ' || header[position] == ','))
    {
      ++position;
    }
    if (position >= close)
    {
      break;
    }

    if (header[position] < '0' || header[position] > '9')
    {
      throw std::runtime_error("NPY shape contains a non-numeric dimension");
    }

    std::size_t value = 0;
    while (position < close && header[position] >= '0' && header[position] <= '9')
    {
      const std::size_t digit = static_cast<std::size_t>(header[position] - '0');
      if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10)
      {
        throw std::overflow_error("NPY shape dimension exceeds size limits");
      }
      value = (value * 10) + digit;
      ++position;
    }
    if (value == 0)
    {
      throw std::runtime_error("NPY shape contains a zero dimension");
    }
    shape.push_back(value);
  }

  if (shape.empty())
  {
    throw std::runtime_error("NPY shape must contain at least one dimension");
  }
  return shape;
}

std::size_t checkedElementCount(const std::vector<std::size_t>& shape)
{
  std::size_t count = 1;
  for (const std::size_t dimension : shape)
  {
    if (count > std::numeric_limits<std::size_t>::max() / dimension)
    {
      throw std::overflow_error("NPY element count exceeds size limits");
    }
    count *= dimension;
  }
  return count;
}

} // namespace

namespace maiw::test
{

NpyFloat32Array readNpyFloat32(const std::filesystem::path& path)
{
  static_assert(std::endian::native == std::endian::little,
                "The focused NPY reader currently requires a little-endian host");

  std::ifstream input(path, std::ios::binary);
  if (!input)
  {
    throw std::runtime_error("Failed to open NPY file: " + path.string());
  }

  char magic[6]{};
  input.read(magic, sizeof(magic));
  if (!input || std::string_view(magic, sizeof(magic)) != kMagicPrefix)
  {
    throw std::runtime_error("File is not a NumPy .npy array: " + path.string());
  }

  unsigned char major = 0;
  [[maybe_unused]] unsigned char minor = 0;
  input.read(reinterpret_cast<char*>(&major), 1);
  input.read(reinterpret_cast<char*>(&minor), 1);
  if (!input)
  {
    throw std::runtime_error("Failed to read NPY version: " + path.string());
  }

  std::uint32_t headerLength = 0;
  if (major == 1)
  {
    headerLength = readLittleEndianU16(input);
  }
  else if (major == 2 || major == 3)
  {
    headerLength = readLittleEndianU32(input);
  }
  else
  {
    throw std::runtime_error("Unsupported NPY version in " + path.string());
  }

  std::string header(headerLength, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input)
  {
    throw std::runtime_error("Failed to read NPY header: " + path.string());
  }

  const std::string descriptor = extractQuotedValue(header, "descr");
  if (descriptor != "<f4" && descriptor != "|f4" && descriptor != "=f4")
  {
    throw std::runtime_error("NPY file is not little-endian/native float32: " + path.string());
  }

  if (extractFortranOrder(header))
  {
    throw std::runtime_error("NPY file is Fortran-contiguous, expected C-contiguous: " +
                             path.string());
  }

  const std::vector<std::size_t> shape = extractShape(header);
  const std::size_t elementCount = checkedElementCount(shape);
  if (elementCount > std::numeric_limits<std::size_t>::max() / sizeof(float))
  {
    throw std::overflow_error("NPY byte count exceeds size limits");
  }

  std::vector<float> data(elementCount);
  input.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(elementCount * sizeof(float)));
  if (!input)
  {
    throw std::runtime_error("Failed to read NPY float32 data: " + path.string());
  }

  char extra = '\0';
  if (input.read(&extra, 1))
  {
    throw std::runtime_error("NPY file contains trailing bytes: " + path.string());
  }

  return NpyFloat32Array{std::move(data), shape, descriptor};
}

} // namespace maiw::test
