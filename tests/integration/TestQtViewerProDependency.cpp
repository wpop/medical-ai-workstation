#include "qtviewerpro/core/VolumeData.h"
#include "qtviewerpro/core/VolumeInformation.h"
#include "qtviewerpro/io/MedicalVolumeLoaderRegistry.h"

#include <QString>

#include <cstddef>
#include <iostream>
#include <vector>

namespace
{

bool requireCondition(bool condition, const char* message)
{
  if (!condition)
  {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

} // namespace

int main()
{
  qvp::VolumeData volume(
      1,
      1,
      1,
      1.0F,
      1.0F,
      1.0F,
      std::vector<float>{42.0F});

  const qvp::VolumeInformation information = qvp::makeVolumeInformation(volume);
  if (!requireCondition(volume.isValid(), "VolumeData should be valid"))
  {
    return 1;
  }
  if (!requireCondition(information.width == std::size_t{1}, "VolumeInformation width mismatch"))
  {
    return 1;
  }
  if (!requireCondition(information.voxelCount == std::size_t{1},
                        "VolumeInformation voxel count mismatch"))
  {
    return 1;
  }
  if (!requireCondition(information.hasIntensityRange, "VolumeInformation intensity range missing"))
  {
    return 1;
  }

  const qvp::VolumeLoadResult emptyPathResult = qvp::loadMedicalVolume(QString{});
  if (!requireCondition(!emptyPathResult.success, "Empty path should not load a medical volume"))
  {
    return 1;
  }

  return 0;
}
