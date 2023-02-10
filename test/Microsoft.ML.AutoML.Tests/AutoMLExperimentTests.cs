// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TorchSharp;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class AutoMLExperimentTests : BaseTestClass
    {
        public AutoMLExperimentTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public async Task AutoMLExperiment_throw_timeout_exception_when_ct_is_canceled_and_no_trial_completed_Async()
        {
            var context = new MLContext(1);
            var experiment = context.Auto().CreateExperiment();

            experiment.SetTrainingTimeInSeconds(1)
                      .SetTrialRunner((serviceProvider) =>
                      {
                          var channel = serviceProvider.GetService<IChannel>();
                          var settings = serviceProvider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
                          return new DummyTrialRunner(settings, 5, channel);
                      })
                      .SetTuner<RandomSearchTuner>();

            var cts = new CancellationTokenSource();

            context.Log += (o, e) =>
            {
                if (e.RawMessage.Contains("Update Running Trial"))
                {
                    cts.Cancel();
                }
            };

            var runExperimentAction = async () => await experiment.RunAsync(cts.Token);

            await runExperimentAction.Should().ThrowExactlyAsync<TimeoutException>();
        }

        [Fact]
        public async Task AutoMLExperiment_cancel_trial_when_exceeds_memory_limit_Async()
        {
            var context = new MLContext(1);
            var experiment = context.Auto().CreateExperiment();
            context.Log += (o, e) =>
            {
                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    this.Output.WriteLine(e.RawMessage);
                }
            };
            // the following experiment set memory usage limit to 0.01mb
            // so all trials should be canceled and there should be no successful trials.
            // therefore when experiment finishes, it should throw timeout exception with no model trained message.
            experiment.SetMaxModelToExplore(10)
                      .SetTrialRunner((serviceProvider) =>
                      {
                          var channel = serviceProvider.GetService<IChannel>();
                          var settings = serviceProvider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
                          return new DummyTrialRunner(settings, 5, channel);
                      })
                      .SetTuner<RandomSearchTuner>()
                      .SetMaximumMemoryUsageInMegaByte(0.01);

            var runExperimentAction = async () => await experiment.RunAsync();
            await runExperimentAction.Should().ThrowExactlyAsync<TimeoutException>();
        }

        [LightGBMFact]
        public async Task AutoMLExperiment_lgbm_cancel_trial_when_exceeds_memory_limit_Async()
        {
            // this test is to verify that lightGbm can be cancelled during training booster.
            var context = new MLContext(1);
            context.Log += (o, e) =>
            {
                if (e.Message.Contains("LightGBM objective"))
                {
                    context.CancelExecution();
                }
            };
            var data = DatasetUtil.GetUciAdultDataView();
            var experiment = context.Auto().CreateExperiment();
            var pipeline = context.Auto().Featurizer(data, "_Features_", excludeColumns: new[] { DatasetUtil.UciAdultLabel })
                                .Append(context.BinaryClassification.Trainers.LightGbm(DatasetUtil.UciAdultLabel, "_Features_", numberOfIterations: 10000));

            experiment.SetDataset(context.Data.TrainTestSplit(data))
                    .SetBinaryClassificationMetric(BinaryClassificationMetric.AreaUnderRocCurve, DatasetUtil.UciAdultLabel)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(10)
                    .SetMaximumMemoryUsageInMegaByte(10);

            var runExperimentAction = async () => await experiment.RunAsync();
            await runExperimentAction.Should().ThrowExactlyAsync<TimeoutException>();
        }

        [Fact]
        public async Task AutoMLExperiment_return_current_best_trial_when_ct_is_canceled_with_trial_completed_Async()
        {
            var context = new MLContext(1);
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var experiment = context.Auto().CreateExperiment();
            experiment.SetTrainingTimeInSeconds(10)
                      .SetTrialRunner((serviceProvider) =>
                      {
                          var channel = serviceProvider.GetService<IChannel>();
                          var settings = serviceProvider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
                          return new DummyTrialRunner(settings, 1, channel);
                      })
                      .SetTuner<RandomSearchTuner>();

            var cts = new CancellationTokenSource();

            context.Log += (o, e) =>
            {
                if (e.RawMessage.Contains("Update Completed Trial"))
                {
                    cts.CancelAfter(100);
                }
            };
            var res = await experiment.RunAsync(cts.Token);

            stopWatch.Stop();
            stopWatch.ElapsedMilliseconds.Should().BeLessOrEqualTo(2 * 1000 + 500);
            cts.IsCancellationRequested.Should().BeTrue();
            res.Metric.Should().BeGreaterThan(0);
        }

        [Fact]
        public async Task AutoMLExperiment_finish_training_when_time_is_up_Async()
        {
            var context = new MLContext(1);

            var experiment = context.Auto().CreateExperiment();
            experiment.SetTrainingTimeInSeconds(5)
                      .SetTrialRunner((serviceProvider) =>
                      {
                          var channel = serviceProvider.GetService<IChannel>();
                          var settings = serviceProvider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
                          return new DummyTrialRunner(settings, 1, channel);
                      })
                      .SetTuner<RandomSearchTuner>();

            var cts = new CancellationTokenSource();
            cts.CancelAfter(10 * 1000);

            var res = await experiment.RunAsync(cts.Token);
            res.Metric.Should().BeGreaterThan(0);
            cts.IsCancellationRequested.Should().BeFalse();
        }

        [Fact]
        public async Task AutoMLExperiment_finish_training_when_reach_to_max_model_async()
        {
            var context = new MLContext(1);
            var experiment = context.Auto().CreateExperiment();
            experiment.SetMaxModelToExplore(5)
                      .SetTrialRunner((serviceProvider) =>
                      {
                          var channel = serviceProvider.GetService<IChannel>();
                          var settings = serviceProvider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
                          return new DummyTrialRunner(settings, 1, channel);
                      })
                      .SetTuner<RandomSearchTuner>();

            var runModelCounts = 0;
            context.Log += (o, e) =>
            {
                if (e.RawMessage.Contains("Update Completed Trial"))
                {
                    runModelCounts++;
                }
            };
            await experiment.RunAsync();
            runModelCounts.Should().Be(5);
        }


        [Fact]
        public async Task AutoMLExperiment_UCI_Adult_Train_Test_Split_Test()
        {
            var context = new MLContext(1);
            context.Log += (o, e) =>
            {
                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    this.Output.WriteLine(e.RawMessage);
                }
            };
            var data = DatasetUtil.GetUciAdultDataView();
            var experiment = context.Auto().CreateExperiment();
            var pipeline = context.Auto().Featurizer(data, "_Features_", excludeColumns: new[] { DatasetUtil.UciAdultLabel })
                                .Append(context.Auto().BinaryClassification(DatasetUtil.UciAdultLabel, "_Features_", useLgbm: false, useSdcaLogisticRegression: false, useLbfgsLogisticRegression: false));

            experiment.SetDataset(context.Data.TrainTestSplit(data))
                    .SetBinaryClassificationMetric(BinaryClassificationMetric.AreaUnderRocCurve, DatasetUtil.UciAdultLabel)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(1);

            var result = await experiment.RunAsync();
            result.Metric.Should().BeGreaterThan(0.8);
        }

        [Fact]
        public async Task AutoMLExperiment_UCI_Adult_CV_5_Test()
        {
            var context = new MLContext(1);
            context.Log += (o, e) =>
            {
                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    this.Output.WriteLine(e.RawMessage);
                }
            };
            var data = DatasetUtil.GetUciAdultDataView();
            var experiment = context.Auto().CreateExperiment();
            var pipeline = context.Auto().Featurizer(data, "_Features_", excludeColumns: new[] { DatasetUtil.UciAdultLabel })
                                .Append(context.Auto().BinaryClassification(DatasetUtil.UciAdultLabel, "_Features_", useLgbm: false, useSdcaLogisticRegression: false, useLbfgsLogisticRegression: false));
            var cts = new CancellationTokenSource();
            cts.CancelAfter(50000);
            experiment.SetDataset(data, 5)
                    .SetBinaryClassificationMetric(BinaryClassificationMetric.AreaUnderRocCurve, DatasetUtil.UciAdultLabel)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(100);

            var result = await experiment.RunAsync(cts.Token);
            result.Metric.Should().BeGreaterThan(0.8);
        }

        [Fact]
        public async Task AutoMLExperiment_Iris_CV_5_Test()
        {
            var context = new MLContext(1);
            context.Log += (o, e) =>
            {
                if (e.RawMessage.Contains("Trial"))
                {
                    this.Output.WriteLine(e.RawMessage);
                }
            };
            var data = DatasetUtil.GetIrisDataView();
            var experiment = context.Auto().CreateExperiment();
            var label = "Label";
            var pipeline = context.Auto().Featurizer(data, excludeColumns: new[] { label })
                                .Append(context.Transforms.Conversion.MapValueToKey(label, label))
                                .Append(context.Auto().MultiClassification(label, useLgbm: false, useSdcaMaximumEntrophy: false, useLbfgsMaximumEntrophy: false));

            experiment.SetDataset(data, 5)
                    .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MacroAccuracy, label)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(10);

            var result = await experiment.RunAsync();
            result.Metric.Should().BeGreaterThan(0.8);
        }

        [Fact]
        public async Task AutoMLExperiment_Iris_Train_Test_Split_Test()
        {
            var context = new MLContext(1);
            context.Log += (o, e) =>
            {
                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    this.Output.WriteLine(e.RawMessage);
                }
            };
            var data = DatasetUtil.GetIrisDataView();
            var experiment = context.Auto().CreateExperiment();
            var label = "Label";
            var pipeline = context.Auto().Featurizer(data, excludeColumns: new[] { label })
                                .Append(context.Transforms.Conversion.MapValueToKey(label, label))
                                .Append(context.Auto().MultiClassification(label, useLgbm: false, useSdcaMaximumEntrophy: false, useLbfgsMaximumEntrophy: false));

            experiment.SetDataset(context.Data.TrainTestSplit(data))
                    .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MacroAccuracy, label)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(10);

            var result = await experiment.RunAsync();
            result.Metric.Should().BeGreaterThan(0.8);
        }

        [Fact]
        public async Task AutoMLExperiment_Taxi_Fare_Train_Test_Split_Test()
        {
            var context = new MLContext(1);
            context.Log += (o, e) =>
            {
                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    this.Output.WriteLine(e.RawMessage);
                }
            };
            var train = DatasetUtil.GetTaxiFareTrainDataView();
            var test = DatasetUtil.GetTaxiFareTestDataView();
            var experiment = context.Auto().CreateExperiment();
            var label = DatasetUtil.TaxiFareLabel;
            var pipeline = context.Auto().Featurizer(train, excludeColumns: new[] { label })
                                .Append(context.Auto().Regression(label, useLgbm: false, useSdca: false, useLbfgsPoissonRegression: false));

            experiment.SetDataset(train, test)
                    .SetRegressionMetric(RegressionMetric.RSquared, label)
                    .SetPipeline(pipeline)
                    .SetMaxModelToExplore(1);

            var result = await experiment.RunAsync();
            result.Metric.Should().BeGreaterThan(0.5);
        }

        [Fact]
        public async Task AutoMLExperiment_Taxi_Fare_CV_5_Test()
        {
            var context = new MLContext(1);
            var train = DatasetUtil.GetTaxiFareTrainDataView();
            var experiment = context.Auto().CreateExperiment();
            var label = DatasetUtil.TaxiFareLabel;
            var pipeline = context.Auto().Featurizer(train, excludeColumns: new[] { label })
                                .Append(context.Auto().Regression(label, useLgbm: false, useSdca: false, useLbfgsPoissonRegression: false));

            experiment.SetDataset(train, 5)
                    .SetRegressionMetric(RegressionMetric.RSquared, label)
                    .SetPipeline(pipeline)
                    .SetMaxModelToExplore(1);

            var result = await experiment.RunAsync();
            result.Metric.Should().BeGreaterThan(0.5);
        }

        [Fact]
        public void AutoMLExperiment_should_use_seed_from_context_if_provided()
        {
            var context = new MLContext();
            var experiment = context.Auto().CreateExperiment();
            var settings = experiment.ServiceCollection.BuildServiceProvider().GetRequiredService<AutoMLExperiment.AutoMLExperimentSettings>();
            settings.Seed.Should().BeNull();

            context = new MLContext(1);
            experiment = context.Auto().CreateExperiment();
            settings = experiment.ServiceCollection.BuildServiceProvider().GetRequiredService<AutoMLExperiment.AutoMLExperimentSettings>();
            settings.Seed.Should().Be(1);
        }

        [Fact]
        public async Task AutoMLExperiment_should_cancel_text_classification()
        {
            var context = new MLContext(1);
            context.Log += (o, e) =>
            {
                if (e.Message.Contains("AutoMLExperiment") || e.Message.Contains("NasBertTrainer"))
                {
                    this.Output.WriteLine(e.RawMessage);
                }
            };
            var dataView = context.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new TestSingleSentenceData()
                    {   // Testing longer than 512 words.
                        Sentence1 = "ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community . ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .ultimately feels as flat as the scruffy sands of its titular community .",
                        Sentiment = "Negative"
                    },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "with a sharp script and strong performances",
                         Sentiment = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "that director m. night shyamalan can weave an eerie spell and",
                         Sentiment = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "comfortable",
                         Sentiment = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "does have its charms .",
                         Sentiment = "Positive"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "banal as the telling",
                         Sentiment = "Negative"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Sentiment = "Negative"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "leguizamo 's best movie work so far",
                         Sentiment = "Negative"
                     }
                }));

            var experiment = context.Auto().CreateExperiment();
            var chain = new EstimatorChain<ITransformer>();
            var pipeline = chain.Append(context.Transforms.Conversion.MapValueToKey("Label", "Sentiment"))
                            .Append(context.MulticlassClassification.Trainers.TextClassification(outputColumnName: "PredictedLabel", maxEpochs: 100))
                            .Append(SweepableEstimatorFactory.CreateMapKeyToValue(new MapKeyToValueOption
                            {
                                InputColumnName = "PredictedLabel",
                                OutputColumnName = "PredictedLabel",
                            }));
            experiment.SetPipeline(pipeline)
                      .SetDataset(dataView, dataView)
                      .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MacroAccuracy, "Label")
                      .SetMaxModelToExplore(1);
            var cts = new CancellationTokenSource();
            cts.CancelAfter(15_000);
            await experiment.RunAsync(cts.Token);
        }
    }
    class TestSingleSentenceData
    {
        public string Sentence1;
        public string Sentiment;
    }

    class DummyTrialRunner : ITrialRunner
    {
        private readonly int _finishAfterNSeconds;
        private readonly IChannel _logger;

        public DummyTrialRunner(AutoMLExperiment.AutoMLExperimentSettings automlSettings, int finishAfterNSeconds, IChannel logger)
        {
            _finishAfterNSeconds = finishAfterNSeconds;
            _logger = logger;
        }

        public void Dispose()
        {
        }

        public async Task<TrialResult> RunAsync(TrialSettings settings, CancellationToken ct)
        {
            _logger.Info("Update Running Trial");
            await Task.Delay(_finishAfterNSeconds * 1000, ct);
            ct.ThrowIfCancellationRequested();
            _logger.Info("Update Completed Trial");
            var metric = 1.000 + 0.01 * settings.TrialId;
            return new TrialResult
            {
                TrialSettings = settings,
                DurationInMilliseconds = _finishAfterNSeconds * 1000,
                Metric = metric,
                Loss = - -metric,
            };
        }
    }

    class DummyPeformanceMonitor : IPerformanceMonitor
    {
        private readonly int _checkIntervalInMilliseconds;
        private System.Timers.Timer _timer;

        public DummyPeformanceMonitor()
        {
            _checkIntervalInMilliseconds = 1000;
        }

        public event EventHandler<TrialPerformanceMetrics> PerformanceMetricsUpdated;

        public void Dispose()
        {
        }

        public double? GetPeakCpuUsage()
        {
            return 100;
        }

        public double? GetPeakMemoryUsageInMegaByte()
        {
            return 1000;
        }

        public void OnPerformanceMetricsUpdatedHandler(TrialSettings trialSettings, TrialPerformanceMetrics metrics, CancellationTokenSource trialCancellationTokenSource)
        {
        }

        public void Start()
        {
            if (_timer == null)
            {
                _timer = new System.Timers.Timer(_checkIntervalInMilliseconds);
                _timer.Elapsed += (o, e) =>
                {
                    PerformanceMetricsUpdated?.Invoke(this, new TrialPerformanceMetrics() { PeakCpuUsage = 100, PeakMemoryUsage = 1000 });
                };

                _timer.AutoReset = true;
            }
            _timer.Enabled = true;
        }

        public void Pause()
        {
            _timer.Enabled = false;
        }

        public void Stop()
        {
            _timer?.Stop();
            _timer?.Dispose();
            _timer = null;
        }
    }
}
