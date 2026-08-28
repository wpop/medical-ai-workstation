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
constexpr int kTestTimeoutMilliseconds = 30000;
const QString kRequiredPathsError =
    QStringLiteral("Both ED and ES medical volume paths are required.");
const QString kConcurrentRequestError =
    QStringLiteral("A cardiac MRI classification is already in progress.");

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
  MissingEs,
  ConcurrentPrimary,
  SequentialSuccess,
  RetryFailure,
  RetrySuccess,
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
        QDir::temp().filePath(QStringLiteral("maiw-missing-ed-%1.nii.gz")
                                  .arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
    const QString invalidEsPath =
        QDir::temp().filePath(QStringLiteral("maiw-missing-es-%1.nii.gz")
                                  .arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
    require(!QFileInfo::exists(invalidEdPath),
            "generated invalid ED path unexpectedly exists");
    require(!QFileInfo::exists(invalidEsPath),
            "generated invalid ES path unexpectedly exists");

    std::size_t startedCount = 0;
    const auto startedCountConnection = QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationStarted,
        &application,
        [&startedCount]()
        {
          ++startedCount;
        });

    std::size_t synchronousFailureCount = 0;
    QString synchronousFailureMessage;
    const auto synchronousFailureConnection = QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationFailed,
        &application,
        [&synchronousFailureCount, &synchronousFailureMessage](const QString& message)
        {
          ++synchronousFailureCount;
          synchronousFailureMessage = message;
        });

    workflow.startClassification(QString(), realEsPath);
    require(synchronousFailureCount == 1,
            "empty ED path did not emit exactly one controlled failure");
    require(synchronousFailureMessage == kRequiredPathsError,
            "empty ED path produced an unexpected validation failure");
    require(startedCount == 0, "empty ED path unexpectedly started a worker");
    require(!workflow.isRunning(), "empty ED path left the workflow running");

    workflow.startClassification(realEdPath, QString());
    require(synchronousFailureCount == 2,
            "empty ES path did not emit exactly one controlled failure");
    require(synchronousFailureMessage == kRequiredPathsError,
            "empty ES path produced an unexpected validation failure");
    require(startedCount == 0, "empty ES path unexpectedly started a worker");
    require(!workflow.isRunning(), "empty ES path left the workflow running");

    QObject::disconnect(synchronousFailureConnection);

    TestStage stage = TestStage::MissingEs;
    std::size_t successCount = 0;
    std::size_t asynchronousFailureCount = 0;
    std::size_t responsiveTickCount = 0;
    bool concurrentRequestIssued = false;
    bool concurrentRequestRejected = false;
    bool allAsyncResultsDeliveredOnMainThread = true;
    std::string failure;

    QTimer responsivenessTimer;
    responsivenessTimer.setInterval(0);
    QObject::connect(&responsivenessTimer,
                     &QTimer::timeout,
                     &application,
                     [&workflow, &responsiveTickCount]()
                     {
                       if (workflow.isRunning())
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

    const auto finishSuccessfully =
        [&application, &responsivenessTimer, &watchdogTimer, &stage]()
        {
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
                       finishWithFailure("workflow state-machine test timed out");
                     });

    QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationStarted,
        &application,
        [&workflow,
         &invalidEdPath,
         &realEsPath,
         &stage,
         &concurrentRequestIssued,
         &concurrentRequestRejected,
         &finishWithFailure]()
        {
          try
          {
            require(workflow.isRunning(),
                    "classificationStarted was emitted while the workflow was not running");
            if (stage == TestStage::ConcurrentPrimary && !concurrentRequestIssued)
            {
              concurrentRequestIssued = true;
              workflow.startClassification(invalidEdPath, realEsPath);
              require(concurrentRequestRejected,
                      "concurrent request was not rejected synchronously");
              require(workflow.isRunning(),
                      "concurrent rejection stopped the primary classification");
            }
          }
          catch (const std::exception& error)
          {
            finishWithFailure(error.what());
          }
        });

    QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationSucceeded,
        &application,
        [&application,
         &workflow,
         &metadata,
         &invalidEdPath,
         &realEdPath,
         &realEsPath,
         &stage,
         &successCount,
         &concurrentRequestRejected,
         &allAsyncResultsDeliveredOnMainThread,
         &finishWithFailure,
         &finishSuccessfully](const maiw::cardiac::CardiacMriClassificationResult& result)
        {
          try
          {
            allAsyncResultsDeliveredOnMainThread =
                allAsyncResultsDeliveredOnMainThread &&
                QThread::currentThread() == application.thread();
            require(!workflow.isRunning(),
                    "workflow remained running when success was delivered");
            validateResult(result, metadata.classNames());
            ++successCount;

            switch (stage)
            {
            case TestStage::ConcurrentPrimary:
              require(concurrentRequestRejected,
                      "primary classification completed without concurrent rejection");
              stage = TestStage::SequentialSuccess;
              workflow.startClassification(realEdPath, realEsPath);
              break;
            case TestStage::SequentialSuccess:
              stage = TestStage::RetryFailure;
              workflow.startClassification(invalidEdPath, realEsPath);
              break;
            case TestStage::RetrySuccess:
              finishSuccessfully();
              break;
            default:
              finishWithFailure("unexpected workflow classification success");
              break;
            }
          }
          catch (const std::exception& error)
          {
            finishWithFailure(error.what());
          }
        });

    QObject::connect(
        &workflow,
        &maiw::qt::CardiacMriClassificationWorkflow::classificationFailed,
        &application,
        [&application,
         &workflow,
         &realEdPath,
         &realEsPath,
         &stage,
         &asynchronousFailureCount,
         &concurrentRequestRejected,
         &allAsyncResultsDeliveredOnMainThread,
         &finishWithFailure](const QString& message)
        {
          try
          {
            if (stage == TestStage::ConcurrentPrimary && message == kConcurrentRequestError)
            {
              require(workflow.isRunning(),
                      "concurrent rejection reported the primary workflow as stopped");
              concurrentRequestRejected = true;
              return;
            }

            allAsyncResultsDeliveredOnMainThread =
                allAsyncResultsDeliveredOnMainThread &&
                QThread::currentThread() == application.thread();
            require(!workflow.isRunning(),
                    "workflow remained running when asynchronous failure was delivered");
            ++asynchronousFailureCount;

            switch (stage)
            {
            case TestStage::MissingEs:
              require(message.startsWith(QStringLiteral("Failed to load the ES volume: ")),
                      "missing ES path produced an unexpected error message");
              stage = TestStage::ConcurrentPrimary;
              workflow.startClassification(realEdPath, realEsPath);
              break;
            case TestStage::RetryFailure:
              require(message.startsWith(QStringLiteral("Failed to load the ED volume: ")),
                      "retry setup produced an unexpected ED load error message");
              stage = TestStage::RetrySuccess;
              workflow.startClassification(realEdPath, realEsPath);
              break;
            default:
              finishWithFailure("unexpected workflow classification failure: " +
                                message.toStdString());
              break;
            }
          }
          catch (const std::exception& error)
          {
            finishWithFailure(error.what());
          }
        });

    responsivenessTimer.start();
    watchdogTimer.start();
    QTimer::singleShot(0,
                       &workflow,
                       [&workflow, &realEdPath, &invalidEsPath]()
                       {
                         workflow.startClassification(realEdPath, invalidEsPath);
                       });

    application.exec();

    QObject::disconnect(startedCountConnection);
    require(failure.empty(), failure);
    require(stage == TestStage::Finished, "workflow state-machine test did not finish");
    require(!workflow.isRunning(), "workflow remained running after the final success");
    require(startedCount == 5,
            "workflow started an unexpected number of asynchronous workers");
    require(successCount == 3,
            "workflow did not complete all three expected successful runs");
    require(asynchronousFailureCount == 2,
            "workflow did not deliver both expected asynchronous load failures");
    require(concurrentRequestIssued && concurrentRequestRejected,
            "concurrent request rejection was not fully observed");
    require(allAsyncResultsDeliveredOnMainThread,
            "an asynchronous result was not delivered on the main Qt thread");
    require(responsiveTickCount > 0,
            "Qt event loop was not responsive during classification");

    std::cout << "Cardiac MRI classification workflow state-machine test passed." << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Cardiac MRI classification workflow state-machine test failed: "
              << error.what() << '\n';
    return 1;
  }
}
