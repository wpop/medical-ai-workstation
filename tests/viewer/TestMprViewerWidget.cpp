#include "maiw/viewer/MprViewerWidget.h"

#include <QApplication>
#include <QLayout>
#include <QRect>

#include <cmath>
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
  require(!viewer.crosshairPosition().has_value(),
          context + ": crosshair is unexpectedly visible");
  require(viewer.imageSize().isEmpty(),
          context + ": slice image is unexpectedly available");
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

void requireGridGeometry(const maiw::viewer::MprViewerWidget& widget)
{
  const QRect axial = widget.axialViewer().geometry();
  const QRect sagittal = widget.sagittalViewer().geometry();
  const QRect coronal = widget.coronalViewer().geometry();

  require(axial.width() > 0 && axial.height() > 0 &&
              sagittal.width() > 0 && sagittal.height() > 0 &&
              coronal.width() > 0 && coronal.height() > 0,
          "MPR grid contains a collapsed viewport");
  require(std::abs(axial.width() - sagittal.width()) * 100 <=
              widget.width() * 2,
          "MPR top-row viewport widths are not balanced");
  require(std::abs(axial.y() - sagittal.y()) * 100 <=
              widget.height() * 2,
          "axial and sagittal viewports do not share the top row");
  require(coronal.y() > axial.y() && coronal.y() > sagittal.y(),
          "coronal viewport is not below the top MPR row");
  require(coronal.width() * 100 >= widget.width() * 95,
          "coronal viewport does not span the MPR grid width");
  require(std::abs(axial.height() - coronal.height()) * 100 <=
              widget.height() * 2,
          "MPR top and bottom row heights are not balanced");
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

      widget.resize(900, 600);
      require(widget.layout() != nullptr,
              "MPR viewer widget has no presentation layout");
      widget.layout()->activate();
      requireGridGeometry(widget);

      const auto maximum = std::numeric_limits<std::size_t>::max();
      widget.setVoxelPosition(qvp::VoxelIndex3D{maximum, maximum, maximum});
      requireEmptyState(widget, "empty navigation state");

      widget.setPositionFromImagePoint(qvp::SliceOrientation::Axial,
                                       QPointF(12.5, 24.5));
      widget.setPositionFromImagePoint(qvp::SliceOrientation::Sagittal,
                                       QPointF(-1.0, -1.0));
      widget.setPositionFromImagePoint(qvp::SliceOrientation::Coronal,
                                       QPointF(maximum, maximum));
      requireEmptyState(widget, "empty image-point navigation state");

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
