#include "maiw/viewer/VolumeRenderingWidget.h"

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

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);

    {
      maiw::viewer::VolumeRenderingWidget widget;
      require(!widget.hasVolume(), "3D renderer is not initially empty");

      widget.resetView();
      widget.clearVolume();
      widget.clearVolume();
      widget.setVolume(maiw::viewer::SharedVolume{});
      require(!widget.hasVolume(), "empty assignment created a 3D volume state");

      widget.setRenderPreset(qvp::VolumeRenderPreset::CtBone);
      require(widget.transferFunctionState().renderPreset ==
                  qvp::VolumeRenderPreset::CtBone,
              "3D render preset was not forwarded");

      widget.setGlobalOpacity(-1.0F);
      require(widget.transferFunctionState().globalOpacity == 0.0F,
              "3D opacity lower bound was not enforced by qtvp");
      widget.setGlobalOpacity(2.0F);
      require(widget.transferFunctionState().globalOpacity == 1.0F,
              "3D opacity upper bound was not enforced by qtvp");

      widget.setManualIntensityRange(300.0F, -100.0F);
      const qvp::VolumeTransferFunctionState state =
          widget.transferFunctionState();
      require(state.renderPreset == qvp::VolumeRenderPreset::Custom,
              "manual 3D intensity range did not select the Custom preset");
      require(state.intensityMinimum == -100.0F &&
                  state.intensityMaximum == 300.0F,
              "manual 3D intensity range was not ordered by qtvp");

      widget.resetView();
      require(!widget.hasVolume(),
              "empty 3D controls unexpectedly created a volume state");
    }

    std::cout << "Volume rendering widget test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Volume rendering widget test failed: "
              << error.what() << '\n';
    return 1;
  }
}
