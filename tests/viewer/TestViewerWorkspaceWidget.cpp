#include "maiw/viewer/ViewerWorkspaceWidget.h"

#include <QApplication>
#include <QEventLoop>
#include <QString>
#include <QThread>
#include <QTimer>

#include <cstddef>
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
  require(workspace.volumeObserver().expired(),
          context + ": workspace exposes a volume observer");
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
      require(!workspace.isLoading(),
              "workspace is unexpectedly loading in its initial state");
      require(&workspace.mprViewer() == mprIdentity,
              "MPR accessor did not return a stable child");
      require(&workspace.volumeRenderingWidget() == volumeRenderingIdentity,
              "3D accessor did not return a stable child");

      workspace.clearVolume();
      workspace.clearVolume();
      workspace.setVolume(maiw::viewer::SharedVolume{});
      requireEmpty(workspace, "repeated clear state");
      require(!workspace.isLoading(),
              "clear or null assignment started volume loading");
      require(&workspace.mprViewer() == mprIdentity,
              "MPR child changed after clear operations");
      require(&workspace.volumeRenderingWidget() == volumeRenderingIdentity,
              "3D child changed after clear operations");

      std::size_t startedCount = 0;
      std::size_t successCount = 0;
      std::size_t failureCount = 0;
      bool failureDeliveredOnMainThread = false;
      QString failureMessage;

      QObject::connect(
          &workspace,
          &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingStarted,
          &application,
          [&startedCount]()
          {
            ++startedCount;
          });
      QObject::connect(
          &workspace,
          &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingSucceeded,
          &application,
          [&successCount]()
          {
            ++successCount;
          });
      QObject::connect(
          &workspace,
          &maiw::viewer::ViewerWorkspaceWidget::volumeLoadingFailed,
          &application,
          [&application,
           &failureCount,
           &failureDeliveredOnMainThread,
           &failureMessage](const QString& message)
          {
            ++failureCount;
            failureDeliveredOnMainThread =
                QThread::currentThread() == application.thread();
            failureMessage = message;
          });

      workspace.loadVolume(QStringLiteral(" \t "));
      require(startedCount == 0,
              "empty path unexpectedly started asynchronous loading");
      require(successCount == 0,
              "empty path unexpectedly reported loading success");
      require(failureCount == 1,
              "empty path did not report exactly one loading failure");
      require(failureDeliveredOnMainThread,
              "empty-path failure was not delivered on the main Qt thread");
      require(!failureMessage.isEmpty(),
              "empty-path failure did not provide a diagnostic");
      require(!workspace.isLoading(),
              "empty path left the workspace loading");
      requireEmpty(workspace, "empty-path failure state");

      QEventLoop eventLoop;
      QTimer::singleShot(0, &eventLoop, &QEventLoop::quit);
      eventLoop.exec();
      require(startedCount == 0 && successCount == 0 && failureCount == 1,
              "empty-path failure produced deferred duplicate signals");
      requireEmpty(workspace, "post-event-loop failure state");
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
