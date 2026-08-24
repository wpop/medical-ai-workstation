#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationWindow.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"

#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QMessageBox>
#include <QString>

#include <onnxruntime_cxx_api.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <utility>

namespace
{

/**
 * @brief Return the deployment-package command-line option.
 *
 * Phase 8 intentionally uses an explicit runtime path rather than introducing
 * package discovery, installation, or deployment-management infrastructure.
 */
QCommandLineOption cardiacPackageOption()
{
  return QCommandLineOption(
      QStringList{QStringLiteral("cardiac-package")},
      QStringLiteral("Path to the cardiac MRI deployment package."),
      QStringLiteral("path"));
}

} // namespace

int main(int argc, char* argv[])
{
  QApplication application(argc, argv);

  QCoreApplication::setApplicationName(QStringLiteral("Medical AI Workstation"));
  QCoreApplication::setApplicationVersion(QStringLiteral("0.1.0"));

  QCommandLineParser parser;
  parser.setApplicationDescription(
      QStringLiteral("Standalone cardiac MRI classification workstation."));
  parser.addHelpOption();
  parser.addVersionOption();

  const QCommandLineOption packageOption = cardiacPackageOption();
  parser.addOption(packageOption);
  parser.process(application);

  const QString packagePath = parser.value(packageOption).trimmed();
  if (packagePath.isEmpty())
  {
    QMessageBox::critical(
        nullptr,
        QStringLiteral("Configuration Error"),
        QStringLiteral(
            "The cardiac MRI deployment package must be supplied with "
            "--cardiac-package <path>."));
    return EXIT_FAILURE;
  }

  try
  {
    /*
     * Deployment metadata is validated before any inference object is created.
     * The class names are copied for presentation before ownership of the
     * metadata is transferred into the synchronous classification service.
     */
    auto metadata =
        maiw::cardiac::CardiacMriDeploymentMetadata::load(
            std::filesystem::path{packagePath.toStdString()});

    const auto classNames = metadata.classNames();

    /*
     * Declaration order defines the required lifetime hierarchy:
     *
     * Ort::Env
     *   outlives CardiacMriClassificationService
     *     outlives CardiacMriClassificationWorkflow
     *       outlives CardiacMriClassificationWindow
     *
     * Stack destruction occurs in the reverse order, preserving every
     * non-owning reference contract without shared ownership or global state.
     */
    Ort::Env environment(
        ORT_LOGGING_LEVEL_WARNING,
        "medical-ai-workstation");

    maiw::cardiac::CardiacMriClassificationService service(
        environment,
        std::move(metadata));

    maiw::qt::CardiacMriClassificationWorkflow workflow(service);

    maiw::qt::CardiacMriClassificationWindow window(
        workflow,
        classNames);

    window.show();

    return application.exec();
  }
  catch (const std::exception& exception)
  {
    QMessageBox::critical(
        nullptr,
        QStringLiteral("Application Startup Error"),
        QString::fromUtf8(exception.what()));
    return EXIT_FAILURE;
  }
  catch (...)
  {
    QMessageBox::critical(
        nullptr,
        QStringLiteral("Application Startup Error"),
        QStringLiteral("The application could not initialize the cardiac MRI workflow."));
    return EXIT_FAILURE;
  }
}
