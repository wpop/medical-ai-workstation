#include "maiw/viewer/SliceViewerWidget.h"

#include <QApplication>

#include <iostream>
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

void requireEmptyState(const maiw::viewer::SliceViewerWidget& widget,
                       const std::string& context)
{
  require(!widget.hasVolume(), context + ": volume is unexpectedly assigned");
  require(widget.sliceCount() == 0, context + ": slice count is not zero");
  require(widget.sliceIndex() == 0, context + ": slice index is not zero");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);

    {
      maiw::viewer::SliceViewerWidget widget;
      requireEmptyState(widget, "initial state");
      require(widget.orientation() == qvp::SliceOrientation::Axial,
              "initial orientation is not axial");

      widget.setOrientation(qvp::SliceOrientation::Coronal);
      require(widget.orientation() == qvp::SliceOrientation::Coronal,
              "coronal orientation was not retained");
      requireEmptyState(widget, "coronal empty state");

      widget.setSliceIndex(42);
      requireEmptyState(widget, "empty slice selection state");

      widget.setOrientation(qvp::SliceOrientation::Sagittal);
      require(widget.orientation() == qvp::SliceOrientation::Sagittal,
              "sagittal orientation was not retained");
      requireEmptyState(widget, "sagittal empty state");

      widget.clearVolume();
      widget.clearVolume();
      widget.setVolume(nullptr);
      widget.refresh();
      requireEmptyState(widget, "repeated clear state");
    }

    std::cout << "Slice viewer widget test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Slice viewer widget test failed: "
              << error.what() << '\n';
    return 1;
  }
}
