#include "maiw/viewer/ViewerWorkspaceWidget.h"

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

void requireEmpty(const maiw::viewer::ViewerWorkspaceWidget& workspace,
                  const std::string& context)
{
  require(!workspace.hasVolume(), context + ": workspace has a volume");
  require(!workspace.mprViewer().hasVolume(),
          context + ": MPR child has a volume");
  require(!workspace.volumeRenderingWidget().hasVolume(),
          context + ": 3D child has a volume");
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);

    {
      maiw::viewer::ViewerWorkspaceWidget workspace;
      const auto* const mprIdentity = &workspace.mprViewer();
      const auto* const volumeRenderingIdentity =
          &workspace.volumeRenderingWidget();

      requireEmpty(workspace, "initial state");
      require(&workspace.mprViewer() == mprIdentity,
              "MPR accessor did not return a stable child");
      require(&workspace.volumeRenderingWidget() == volumeRenderingIdentity,
              "3D accessor did not return a stable child");

      workspace.clearVolume();
      workspace.clearVolume();
      workspace.setVolume(maiw::viewer::SharedVolume{});
      requireEmpty(workspace, "repeated clear state");
      require(&workspace.mprViewer() == mprIdentity,
              "MPR child changed after clear operations");
      require(&workspace.volumeRenderingWidget() == volumeRenderingIdentity,
              "3D child changed after clear operations");
    }

    std::cout << "Viewer workspace widget test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Viewer workspace widget test failed: "
              << error.what() << '\n';
    return 1;
  }
}
