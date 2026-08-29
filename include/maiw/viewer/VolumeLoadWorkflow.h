#pragma once

#include "qtviewerpro/core/VolumeData.h"

#include <QFutureWatcher>
#include <QMetaType>
#include <QObject>
#include <QString>

#include <memory>

namespace maiw::viewer
{

/**
 * @brief Shared immutable ownership of a loaded medical volume.
 */
using SharedVolume = std::shared_ptr<const qvp::VolumeData>;

/**
 * @brief Loads one medical volume asynchronously for Qt viewer code.
 *
 * File-format detection and loading are delegated to qt-viewer-pro. A
 * successful load moves the volume into shared immutable ownership so the same
 * instance can later be observed by 2D widgets and shared with 3D rendering.
 * Only one load may be active at a time.
 */
class VolumeLoadWorkflow final : public QObject
{
  Q_OBJECT

public:
  /**
   * @brief Construct an idle medical-volume loading workflow.
   *
   * @param parent Optional QObject parent.
   */
  explicit VolumeLoadWorkflow(QObject* parent = nullptr);

  /**
   * @brief Wait for any active load operation before destruction.
   */
  ~VolumeLoadWorkflow() override;

  /**
   * @brief Start loading one medical volume from the supplied path.
   *
   * Empty paths and overlapping requests are rejected synchronously through
   * loadingFailed(). File loading and format detection run outside the Qt event
   * loop.
   *
   * @param path Path to the medical volume or supported series directory.
   */
  void startLoading(QString path);

  /**
   * @brief Return true while a medical-volume load is active.
   */
  [[nodiscard]] bool isRunning() const noexcept;

signals:
  /**
   * @brief Emitted when an asynchronous load operation has started.
   */
  void loadingStarted();

  /**
   * @brief Emitted after a valid, non-empty volume has been loaded.
   *
   * The signal transfers a shared ownership copy suitable for queued delivery.
   *
   * @param volume Shared immutable ownership of the loaded volume.
   */
  void loadingSucceeded(SharedVolume volume);

  /**
   * @brief Emitted when a load request cannot complete successfully.
   *
   * @param message User-presentable error description.
   */
  void loadingFailed(const QString& message);

private:
  struct AsyncResult
  {
    SharedVolume volume;
    QString errorMessage;
  };

  void handleFinished();

  QFutureWatcher<AsyncResult> watcher_;
  bool running_ = false;
};

} // namespace maiw::viewer

Q_DECLARE_METATYPE(maiw::viewer::SharedVolume)
