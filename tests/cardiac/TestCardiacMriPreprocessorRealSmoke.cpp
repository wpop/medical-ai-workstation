#include "maiw/cardiac/CardiacMriPreprocessing.h"

#include "qtviewerpro/io/MedicalVolumeLoaderRegistry.h"

#include <QString>

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace
{

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

void require(bool condition, const char* message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
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

    const maiw::cardiac::CardiacMriPreprocessor preprocessor;
    const maiw::cardiac::CardiacMriInputTensor tensor =
        preprocessor.preprocess(edVolume, esVolume);

    require(tensor.shapeCdhw() == std::array<std::size_t, 4>{2, 14, 144, 144},
            "real smoke tensor shape mismatch");
    require(tensor.values().size() == maiw::cardiac::CardiacMriInputTensor::elementCount(),
            "real smoke tensor value count mismatch");
    for (const float value : tensor.values())
    {
      require(std::isfinite(value), "real smoke tensor contains a non-finite value");
    }

    std::cout << "Cardiac MRI real preprocessing smoke passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI real preprocessing smoke failed: " << error.what() << '\n';
    return 1;
  }
}
