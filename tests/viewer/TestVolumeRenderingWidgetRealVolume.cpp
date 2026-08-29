#include "maiw/viewer/VolumeLoadWorkflow.h"
#include "maiw/viewer/VolumeRenderingWidget.h"

#include "qtviewerpro/core/VolumeData.h"

#include <QApplication>
#include <QEventLoop>
#include <QString>
#include <QThread>
#include <QTimer>

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

constexpr int kLoadTimeoutMilliseconds = 30000;

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

maiw::viewer::SharedVolume loadRealVolume(QApplication& application)
{
  maiw::viewer::VolumeLoadWorkflow workflow;
  QEventLoop eventLoop;
  QTimer watchdog;
  watchdog.setSingleShot(true);
  watchdog.setInterval(kLoadTimeoutMilliseconds);

  std::size_t startedCount = 0;
  std::size_t successCount = 0;
  std::size_t failureCount = 0;
  bool timedOut = false;
  bool deliveredOnMainThread = false;
  QString failureMessage;
  maiw::viewer::SharedVolume volume;

  QObject::connect(
      &workflow,
      &maiw::viewer::VolumeLoadWorkflow::loadingStarted,
      &application,
      [&startedCount]()
      {
        ++startedCount;
      });
  QObject::connect(
      &workflow,
      &maiw::viewer::VolumeLoadWorkflow::loadingSucceeded,
      &eventLoop,
      [&application,
       &eventLoop,
       &watchdog,
       &successCount,
       &deliveredOnMainThread,
       &volume](maiw::viewer::SharedVolume loadedVolume)
      {
        ++successCount;
        deliveredOnMainThread =
            QThread::currentThread() == application.thread();
        volume = std::move(loadedVolume);
        watchdog.stop();
        eventLoop.quit();
      },
      Qt::QueuedConnection);
  QObject::connect(
      &workflow,
      &maiw::viewer::VolumeLoadWorkflow::loadingFailed,
      &eventLoop,
      [&eventLoop,
       &watchdog,
       &failureCount,
       &failureMessage](const QString& message)
      {
        ++failureCount;
        failureMessage = message;
        watchdog.stop();
        eventLoop.quit();
      });
  QObject::connect(
      &watchdog,
      &QTimer::timeout,
      &eventLoop,
      [&eventLoop, &timedOut]()
      {
        timedOut = true;
        eventLoop.quit();
      });

  watchdog.start();
  workflow.startLoading(QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH));
  require(workflow.isRunning(),
          "real ACDC 3D load did not enter the running state");
  eventLoop.exec();

  require(!timedOut, "timed out loading the real ACDC 3D volume");
  require(startedCount == 1,
          "real ACDC 3D load did not start exactly once");
  require(successCount == 1,
          "real ACDC 3D load did not succeed exactly once");
  require(failureCount == 0,
          "real ACDC 3D load failed: " + failureMessage.toStdString());
  require(deliveredOnMainThread,
          "real ACDC 3D load was not delivered on the main Qt thread");
  require(!workflow.isRunning(),
          "workflow remained running after real ACDC 3D load");
  require(volume != nullptr,
          "real ACDC 3D load returned a null shared volume");
  require(volume->isValid() && !volume->isEmpty(),
          "real ACDC 3D volume is invalid or empty");

  return volume;
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QApplication application(argc, argv);
    maiw::viewer::SharedVolume externalVolume = loadRealVolume(application);
    maiw::viewer::SharedVolume retainedExternalVolume = externalVolume;
    const qvp::VolumeData* const volumeIdentity = externalVolume.get();
    std::weak_ptr<const qvp::VolumeData> volumeObserver = externalVolume;

    require(externalVolume.use_count() == 2,
            "unexpected shared ownership before 3D assignment");

    {
      maiw::viewer::VolumeRenderingWidget widget;
      widget.setVolume(externalVolume);
      require(widget.hasVolume(),
              "3D widget did not accept the real ACDC volume");
      require(externalVolume.use_count() == 4,
              "3D assignment did not share the existing volume instance");

      externalVolume.reset();
      require(widget.hasVolume(),
              "3D widget lost its volume after an external copy was reset");
      require(!volumeObserver.expired() &&
                  volumeObserver.lock().get() == volumeIdentity,
              "3D widget did not preserve the assigned volume instance");

      widget.resetView();
      widget.setRenderPreset(qvp::VolumeRenderPreset::Default);
      widget.setGlobalOpacity(0.75F);
      widget.setManualIntensityRange(volumeIdentity->intensityMinimum(),
                                     volumeIdentity->intensityMaximum());

      widget.clearVolume();
      require(!widget.hasVolume(),
              "3D widget retained its volume after clear");
      require(retainedExternalVolume.use_count() == 1,
              "3D clear retained unexpected shared volume ownership");

      widget.clearVolume();
      widget.setVolume(retainedExternalVolume);
      require(widget.hasVolume(),
              "3D widget did not accept a reassigned real volume");
      require(retainedExternalVolume.use_count() == 3,
              "3D reassignment did not share the existing volume instance");
    }

    require(retainedExternalVolume != nullptr &&
                retainedExternalVolume.get() == volumeIdentity &&
                retainedExternalVolume->isValid() &&
                !retainedExternalVolume->isEmpty(),
            "external volume became invalid after 3D widget destruction");
    require(retainedExternalVolume.use_count() == 1,
            "3D widget destruction retained shared volume ownership");

    retainedExternalVolume.reset();
    require(volumeObserver.expired(),
            "real ACDC volume remained owned after all copies were released");

    std::cout << "Real-volume rendering widget test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Real-volume rendering widget test failed: "
              << error.what() << '\n';
    return 1;
  }
}
