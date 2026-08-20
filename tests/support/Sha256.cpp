#include "Sha256.h"

#include <openssl/evp.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

using EvpMdContextPtr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;

std::string toLowerHex(const unsigned char* bytes, unsigned int size)
{
  constexpr char kHexDigits[] = "0123456789abcdef";

  std::string hex;
  hex.reserve(static_cast<std::size_t>(size) * 2U);
  for (unsigned int index = 0; index < size; ++index)
  {
    const unsigned char value = bytes[index];
    hex.push_back(kHexDigits[value >> 4U]);
    hex.push_back(kHexDigits[value & 0x0FU]);
  }
  return hex;
}

} // namespace

namespace maiw::test
{

std::string sha256FileHex(const std::filesystem::path& path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input)
  {
    throw std::runtime_error("Failed to open file for SHA-256: " + path.string());
  }

  EvpMdContextPtr context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context)
  {
    throw std::runtime_error("Failed to allocate OpenSSL digest context");
  }
  if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
  {
    throw std::runtime_error("Failed to initialize SHA-256 digest");
  }

  std::array<char, 64 * 1024> buffer{};
  while (input)
  {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize bytesRead = input.gcount();
    if (bytesRead > 0)
    {
      if (EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(bytesRead)) != 1)
      {
        throw std::runtime_error("Failed to update SHA-256 digest for: " + path.string());
      }
    }
  }
  if (!input.eof())
  {
    throw std::runtime_error("Failed while reading file for SHA-256: " + path.string());
  }

  unsigned char digest[EVP_MAX_MD_SIZE]{};
  unsigned int digestSize = 0;
  if (EVP_DigestFinal_ex(context.get(), digest, &digestSize) != 1)
  {
    throw std::runtime_error("Failed to finalize SHA-256 digest for: " + path.string());
  }

  return toLowerHex(digest, digestSize);
}

} // namespace maiw::test
