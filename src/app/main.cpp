#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"
#include "maiw/qt/MedicalAiWorkstationWindow.h"

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
 */
QCommandLineOption cardiacPackageOption()
{
  return QCommandLineOption(
      QStringList{QStringLiteral("cardiac-package")},
      QStringLiteral("Override the installed cardiac MRI deployment package path."),
      QStringLiteral("path"));
}

/**
 * @brief Resolve the explicit or install-relative cardiac deployment package path.
 *
 * @param commandLinePath Parsed value of the --cardiac-package option.
 * @return The explicit non-empty path, or the package path relative to the
 *         installed application executable.
 */
std::filesystem::path resolveCardiacPackagePath(const QString& commandLinePath)
{
  const QString explicitPath = commandLinePath.trimmed();
  if (!explicitPath.isEmpty())
  {
    return std::filesystem::path{explicitPath.toStdString()};
  }

  const auto applicationDirectory =
      std::filesystem::path{QCoreApplication::applicationDirPath().toStdString()};
  const auto relativeDataDirectory =
      std::filesystem::path{MAIW_INSTALL_DATADIR_FROM_BINDIR}.lexically_normal();

  return (applicationDirectory
          / relativeDataDirectory
          / "medical-ai-workstation"
          / "cardiac_mri_pathology")
      .lexically_normal();
}

} // namespace

int main(int argc, char* argv[])
{
  QApplication application(argc, argv);

  QCoreApplication::setApplicationName(QStringLiteral("Medical AI Workstation"));
  QCoreApplication::setApplicationVersion(QStringLiteral("0.1.0"));

  QCommandLineParser parser;
  parser.setApplicationDescription(
      QStringLiteral("Unified medical-image viewing and cardiac MRI classification workstation."));
  parser.addHelpOption();
  parser.addVersionOption();

  const QCommandLineOption packageOption = cardiacPackageOption();
  parser.addOption(packageOption);
  parser.process(application);

  const std::filesystem::path packagePath =
      resolveCardiacPackagePath(parser.value(packageOption));

  try
  {
    /*
     * Deployment metadata is validated before any inference object is created.
     * The class names are copied for presentation before ownership of the
     * metadata is transferred into the synchronous classification service.
     */
    auto metadata =
        maiw::cardiac::CardiacMriDeploymentMetadata::load(
            packagePath);

    const auto classNames = metadata.classNames();

    /*
     * Declaration order defines the required lifetime hierarchy:
     *
     * Ort::Env
     *   outlives CardiacMriClassificationService
     *     outlives CardiacMriClassificationWorkflow
     *       outlives MedicalAiWorkstationWindow
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

    maiw::qt::MedicalAiWorkstationWindow window(
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
