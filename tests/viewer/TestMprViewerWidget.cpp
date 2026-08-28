#include "maiw/viewer/MprViewerWidget.h"

#include <QApplication>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void requireOrigin(const qvp::VoxelIndex3D& position,
                   const std::string& context)
{
  require(position.x == 0 && position.y == 0 && position.z == 0,
          context + ": voxel position is not the origin");
}

void requireEmptyViewer(const maiw::viewer::SliceViewerWidget& viewer,
                        qvp::SliceOrientation expectedOrientation,
                        const std::string& context)
{
  require(viewer.orientation() == expectedOrientation,
          context + ": orientation is incorrect");
  require(!viewer.hasVolume(),
          context + ": volume is unexpectedly assigned");
  require(viewer.sliceCount() == 0,
          context + ": slice count is not zero");
  require(viewer.sliceIndex() == 0,
          context + ": slice index is not zero");
}

void requireEmptyState(const maiw::viewer::MprViewerWidget& widget,
                       const std::string& context)
{
  require(!widget.hasVolume(), context + ": volume is unexpectedly assigned");
  requireOrigin(widget.voxelPosition(), context);
  require(!widget.physicalPosition().has_value(),
          context + ": physical position is unexpectedly available");
  requireEmptyViewer(widget.axialViewer(),
                     qvp::SliceOrientation::Axial,
                     context + " axial viewer");
  requireEmptyViewer(widget.sagittalViewer(),
                     qvp::SliceOrientation::Sagittal,
                     context + " sagittal viewer");
  requireEmptyViewer(widget.coronalViewer(),
                     qvp::SliceOrientation::Coronal,
                     context + " coronal viewer");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);

    {
      maiw::viewer::MprViewerWidget widget;
      requireEmptyState(widget, "initial state");

      const auto maximum = std::numeric_limits<std::size_t>::max();
      widget.setVoxelPosition(qvp::VoxelIndex3D{maximum, maximum, maximum});
      requireEmptyState(widget, "empty navigation state");

      widget.clearVolume();
      widget.clearVolume();
      widget.setVolume(nullptr);
      requireEmptyState(widget, "repeated clear state");
    }

    std::cout << "MPR viewer widget test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "MPR viewer widget test failed: "
              << error.what() << '\n';
    return 1;
  }
}
