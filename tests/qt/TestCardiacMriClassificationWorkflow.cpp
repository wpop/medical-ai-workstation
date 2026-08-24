#include "maiw/cardiac/CardiacMriClassificationService.h"
#include "maiw/cardiac/CardiacMriDeploymentMetadata.h"
#include "maiw/qt/CardiacMriClassificationWorkflow.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QThread>
#include <QTimer>
#include <QUuid>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

constexpr double kProbabilitySumTolerance = 1e-12;
constexpr int kTestTimeoutMilliseconds = 300000;

void require(bool condition, const std::string& message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

template <typename Values>
std::size_t firstMaximumIndex(const Values& values)
{
  return static_cast<std::size_t>(std::distance(values.begin(),
                                                std::max_element(values.begin(), values.end())));
}

void validateResult(
    const maiw::cardiac::CardiacMriClassificationResult& result,
    const maiw::cardiac::CardiacMriDeploymentMetadata::ClassNames& classNames)
{
  for (const float logit : result.rawLogits())
  {
    require(std::isfinite(logit), "workflow result contains a non-finite raw logit");
  }

  for (const double probability : result.probabilities())
  {
    require(std::isfinite(probability),
            "workflow result contains a non-finite probability");
    require(probability >= 0.0 && probability <= 1.0,
            "workflow probability is outside the closed unit interval");
  }

  const double probabilitySum =
      std::accumulate(result.probabilities().begin(), result.probabilities().end(), 0.0);
  require(std::fabs(probabilitySum - 1.0) <= kProbabilitySumTolerance,
          "workflow probabilities are not normalized");

  const std::size_t predictedIndex = result.predictedClassIndex();
  require(predictedIndex < maiw::cardiac::CardiacMriDeploymentMetadata::kClassCount,
          "workflow predicted class index is out of range");
  require(result.predictedClassName() == classNames[predictedIndex],
          "workflow predicted class name does not match deployment ordering");
  require(firstMaximumIndex(result.rawLogits()) == predictedIndex,
          "workflow raw-logit argmax does not match the prediction");
  require(firstMaximumIndex(result.probabilities()) == predictedIndex,
          "workflow probability argmax does not match the prediction");
}

enum class TestStage
{
  RealClassification,
  InvalidPath,
  Finished
};

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    QCoreApplication application(argc, argv);

    const auto metadata = maiw::cardiac::CardiacMriDeploymentMetadata::load(
        std::filesystem::path{MAIW_CARDIAC_MRI_PACKAGE_DIR});
    Ort::Env environment(ORT_LOGGING_LEVEL_ERROR,
                         "medical-ai-workstation-classification-workflow-test");
    maiw::cardiac::CardiacMriClassificationService service(environment, metadata);
    maiw::qt::CardiacMriClassificationWorkflow workflow(service);

    const QString realEdPath = QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ED_PATH);
    const QString realEsPath = QString::fromUtf8(MAIW_CARDIAC_MRI_REAL_ES_PATH);
    const QString invalidEdPath =
        QDir::temp().filePath(QStringLiteral("maiw-missing-%1.nii.gz")
                                  .arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
    require(!QFileInfo::exists(invalidEdPath),
            "generated invalid ED path unexpectedly exists");

    TestStage stage = TestStage::RealClassification;
    std::size_t responsiveTickCount = 0;
    bool successDeliveredOnMainThread = false;
    bool failureDeliveredOnMainThread = false;
    std::string failure;

    QTimer responsivenessTimer;
    responsivenessTimer.setInterval(0);
    QObject::connect(&responsivenessTimer,
                     &QTimer::timeout,
                     &application,
                     [&workflow, &stage, &responsiveTickCount]()
                     {
                       if (stage == TestStage::RealClassification && workflow.isRunning())
                       {
                         ++responsiveTickCount;
                       }
                     });

    QTimer watchdogTimer;
    watchdogTimer.setSingleShot(true);
    watchdogTimer.setInterval(kTestTimeoutMilliseconds);

    const auto finishWithFailure =
        [&application, &responsivenessTimer, &watchdogTimer, &stage, &failure](
            std::string message)
        {
          failure = std::move(message);
          stage = TestStage::Finished;
          responsivenessTimer.stop();
          watchdogTimer.stop();
          application.quit();
        };

    QObject::connect(&watchdogTimer,
                     &QTimer::timeout,
                     &application,
                     [&finishWithFailure]()
                     {
                       finishWithFailure("workflow test timed out");
                     });

    QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationSucceeded,
        &application,
        [&application,
         &workflow,
         &metadata,
         &invalidEdPath,
         &realEsPath,
         &stage,
         &responsiveTickCount,
         &successDeliveredOnMainThread,
         &finishWithFailure](
            const maiw::cardiac::CardiacMriClassificationResult& result)
        {
          if (stage != TestStage::RealClassification)
          {
            finishWithFailure("unexpected classification success");
            return;
          }

          successDeliveredOnMainThread =
              QThread::currentThread() == application.thread();

          try
          {
            require(successDeliveredOnMainThread,
                    "classification success was not delivered on the main thread");
            require(responsiveTickCount > 0,
                    "Qt event loop was not responsive during classification");
            validateResult(result, metadata.classNames());
          }
          catch (const std::exception& error)
          {
            finishWithFailure(error.what());
            return;
          }

          stage = TestStage::InvalidPath;
          workflow.startClassification(invalidEdPath, realEsPath);
        });

    QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationFailed,
        &application,
        [&application,
         &responsivenessTimer,
         &watchdogTimer,
         &stage,
         &failureDeliveredOnMainThread,
         &failure,
         &finishWithFailure](const QString& message)
        {
          if (stage == TestStage::RealClassification)
          {
            finishWithFailure("real classification failed: " + message.toStdString());
            return;
          }
          if (stage != TestStage::InvalidPath)
          {
            finishWithFailure("unexpected classification failure");
            return;
          }

          failureDeliveredOnMainThread =
              QThread::currentThread() == application.thread();
          if (!failureDeliveredOnMainThread)
          {
            finishWithFailure("classification failure was not delivered on the main thread");
            return;
          }
          if (!message.startsWith(QStringLiteral("Failed to load the ED volume: ")))
          {
            finishWithFailure("invalid ED path produced an unexpected error message: " +
                              message.toStdString());
            return;
          }

          failure.clear();
          stage = TestStage::Finished;
          responsivenessTimer.stop();
          watchdogTimer.stop();
          application.quit();
        });

    responsivenessTimer.start();
    watchdogTimer.start();
    QTimer::singleShot(0,
                       &workflow,
                       [&workflow, &realEdPath, &realEsPath]()
                       {
                         workflow.startClassification(realEdPath, realEsPath);
                       });

    application.exec();

    require(failure.empty(), failure);
    require(stage == TestStage::Finished, "workflow test did not finish");
    require(successDeliveredOnMainThread,
            "classification success main-thread delivery was not observed");
    require(failureDeliveredOnMainThread,
            "classification failure main-thread delivery was not observed");

    std::cout << "Cardiac MRI classification workflow test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI classification workflow test failed: " << error.what() << '\n';
    return 1;
  }
}
