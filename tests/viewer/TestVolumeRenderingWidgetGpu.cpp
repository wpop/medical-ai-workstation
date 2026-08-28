#include "maiw/viewer/VolumeLoadWorkflow.h"
#include "maiw/viewer/VolumeRenderingWidget.h"

#include "qtviewerpro/core/VolumeData.h"

#include <QApplication>
#include <QString>
#include <QTimer>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace
{

constexpr int kValidationTimeoutMilliseconds = 60000;
constexpr int kVolumeLoadFailureExitCode = 1;
constexpr int kGpuUploadFailureExitCode = 2;
constexpr int kTimeoutExitCode = 3;
constexpr int kDimensionMismatchExitCode = 4;
constexpr int kUnexpectedExitCode = 5;

} // namespace

int main(int argc, char* argv[])
{
  QApplication application(argc, argv);
  application.setQuitOnLastWindowClosed(false);

  maiw::viewer::VolumeLoadWorkflow workflow;
  maiw::viewer::SharedVolume volume;
  maiw::viewer::VolumeRenderingWidget widget;
  QTimer watchdog;
  watchdog.setSingleShot(true);
  watchdog.setInterval(kValidationTimeoutMilliseconds);

  bool finished = false;
  bool volumeAssigned = false;
  std::size_t uploadSuccessCount = 0;
  std::size_t uploadFailureCount = 0;

  auto finishWithFailure =
      [&application, &watchdog, &finished](int exitCode, const QString& message)
      {
        if (finished)
        {
          return;
        }

        finished = true;
        watchdog.stop();
        std::cerr << message.toStdString() << '\n';
        application.exit(exitCode);
      };

  QObject::connect(
      &workflow,
      &maiw::viewer::VolumeLoadWorkflow::loadingSucceeded,
      &widget,
      [&volume,
       &widget,
       &volumeAssigned,
       &finishWithFailure](maiw::viewer::SharedVolume loadedVolume)
      {
        if (!loadedVolume || !loadedVolume->isValid() || loadedVolume->isEmpty())
        {
          finishWithFailure(
              kVolumeLoadFailureExitCode,
              QStringLiteral("Real ACDC volume load returned invalid data."));
          return;
        }

        volume = std::move(loadedVolume);
        volumeAssigned = true;
        widget.setVolume(volume);
      },
      Qt::QueuedConnection);

  QObject::connect(
      &workflow,
      &maiw::viewer::VolumeLoadWorkflow::loadingFailed,
      &widget,
      [&finishWithFailure](const QString& message)
      {
        finishWithFailure(
            kVolumeLoadFailureExitCode,
            QStringLiteral("Real ACDC volume load failed: %1").arg(message));
      });

  QObject::connect(
      &widget,
      &maiw::viewer::VolumeRenderingWidget::volumeUploadSucceeded,
      &application,
      [&application,
       &watchdog,
       &finished,
       &volume,
       &uploadSuccessCount,
       &uploadFailureCount,
       &finishWithFailure](int width, int height, int depth)
      {
        ++uploadSuccessCount;
        if (uploadSuccessCount != 1 || uploadFailureCount != 0)
        {
          finishWithFailure(
              kGpuUploadFailureExitCode,
              QStringLiteral("GPU upload emitted an unexpected result count."));
          return;
        }

        if (!volume || width <= 0 || height <= 0 || depth <= 0 ||
            static_cast<std::size_t>(width) != volume->width() ||
            static_cast<std::size_t>(height) != volume->height() ||
            static_cast<std::size_t>(depth) != volume->depth())
        {
          const QString expected = volume
                                       ? QStringLiteral("%1x%2x%3")
                                             .arg(volume->width())
                                             .arg(volume->height())
                                             .arg(volume->depth())
                                       : QStringLiteral("unavailable");
          finishWithFailure(
              kDimensionMismatchExitCode,
              QStringLiteral("GPU texture dimension mismatch: uploaded %1x%2x%3, expected %4.")
                  .arg(width)
                  .arg(height)
                  .arg(depth)
                  .arg(expected));
          return;
        }

        QTimer::singleShot(
            0,
            &application,
            [&application,
             &watchdog,
             &finished,
             &uploadSuccessCount,
             &uploadFailureCount,
             &finishWithFailure,
             width,
             height,
             depth]()
            {
              if (finished)
              {
                return;
              }

              if (uploadSuccessCount != 1 || uploadFailureCount != 0)
              {
                finishWithFailure(
                    kGpuUploadFailureExitCode,
                    QStringLiteral("GPU upload emitted an unexpected result count."));
                return;
              }

              finished = true;
              watchdog.stop();
              std::cout << "Real ACDC GPU volume upload passed: "
                        << width << 'x' << height << 'x' << depth << '\n';
              application.exit(EXIT_SUCCESS);
            });
      });

  QObject::connect(
      &widget,
      &maiw::viewer::VolumeRenderingWidget::volumeUploadFailed,
      &application,
      [&uploadFailureCount, &finishWithFailure](const QString& message)
      {
        ++uploadFailureCount;
        finishWithFailure(
            kGpuUploadFailureExitCode,
            QStringLiteral("GPU texture upload failed: %1").arg(message));
      });

  QObject::connect(
      &watchdog,
      &QTimer::timeout,
      &application,
      [&volumeAssigned, &finishWithFailure]()
      {
        const QString message =
            volumeAssigned
                ? QStringLiteral("Timed out waiting for GPU texture upload.")
                : QStringLiteral("Timed out loading the real ACDC volume before GPU upload.");
        finishWithFailure(kTimeoutExitCode, message);
      });

  widget.resize(900, 700);
  widget.show();
  watchdog.start();

  QTimer::singleShot(
      0,
      &workflow,
      [&workflow]()
      {
        workflow.startLoading(
            QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH));
      });

  const int exitCode = application.exec();
  if (!finished)
  {
    std::cerr << "GPU validation event loop exited before a result." << '\n';
    return kUnexpectedExitCode;
  }

  return exitCode;
}
