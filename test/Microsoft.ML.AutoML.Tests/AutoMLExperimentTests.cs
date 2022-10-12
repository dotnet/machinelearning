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
using Microsoft.ML.Data;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Tensorflow;
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
            experiment.SetTrainingTimeInSeconds(10)
                      .SetTrialRunner((serviceProvider) =>
                      {
                          var channel = serviceProvider.GetService<IChannel>();
                          var settings = serviceProvider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
                          return new DummyTrialRunner(settings, 5, channel);
                      })
                      .SetTuner<RandomSearchTuner>()
                      .SetMaximumMemoryUsageInMegaByte(0.01)
                      .SetPerformanceMonitor<DummyPeformanceMonitor>();

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
                                .Append(context.Auto().BinaryClassification(DatasetUtil.UciAdultLabel, "_Features_", useLgbm: false, useSdca: false, useLbfgs: false));

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
                                .Append(context.Auto().BinaryClassification(DatasetUtil.UciAdultLabel, "_Features_", useLgbm: false, useSdca: false, useLbfgs: false));

            experiment.SetDataset(data, 5)
                    .SetBinaryClassificationMetric(BinaryClassificationMetric.AreaUnderRocCurve, DatasetUtil.UciAdultLabel)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(10);

            var result = await experiment.RunAsync();
            result.Metric.Should().BeGreaterThan(0.8);
        }

        [Fact]
        public async Task AutoMLExperiment_Iris_CV_5_Test()
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
                                .Append(context.Auto().MultiClassification(label, useLgbm: false, useSdca: false, useLbfgs: false));

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
                                .Append(context.Auto().MultiClassification(label, useLgbm: false, useSdca: false, useLbfgs: false));

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
                                .Append(context.Auto().Regression(label, useLgbm: false, useSdca: false, useLbfgs: false));

            experiment.SetDataset(train, test)
                    .SetRegressionMetric(RegressionMetric.RSquared, label)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(50);

            var result = await experiment.RunAsync();
            result.Metric.Should().BeGreaterThan(0.5);
        }

        [Fact]
        public async Task AutoMLExperiment_Taxi_Fare_CV_5_Test()
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
            var experiment = context.Auto().CreateExperiment();
            var label = DatasetUtil.TaxiFareLabel;
            var pipeline = context.Auto().Featurizer(train, excludeColumns: new[] { label })
                                .Append(context.Auto().Regression(label, useLgbm: false, useSdca: false, useLbfgs: false));

            experiment.SetDataset(train, 5)
                    .SetRegressionMetric(RegressionMetric.RSquared, label)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(50);

            var result = await experiment.RunAsync();
            result.Metric.Should().BeGreaterThan(0.5);
        }

        public async Task Generate_300GB_csv()
        {
            var rnd = new Random();
            ModelInput GenerateRandomRow()
            {
                return new ModelInput
                {
                    _data0 = rnd.NextSingle() > 0.5 ? "a" : "b",
                    ignoreData1 = rnd.NextSingle(),
                    _data = Enumerable.Repeat(rnd.NextSingle(), 4204).ToArray(),
                    _ignoreData4206 = rnd.NextSingle(),
                    _ignoreData4207 = rnd.NextSingle(),
                    _ignoreData4208 = rnd.NextSingle(),
                    _label = rnd.NextSingle() > 0.5 ? "True" : "False",
                };
            }

            var filePath = @"D:/large_csv.csv";
            var fileInfo = new FileInfo(filePath);

            using (var fileStream = fileInfo.Open(FileMode.Append))
            using (var stream = new StreamWriter(fileStream))
            {
                var i = 0;
                while ((!fileInfo.Exists || fileInfo.Length < 300.0 * 1024 * 1024 * 1024) && i < 100)
                {
                    fileInfo.Refresh();
                    Output.WriteLine($"{fileInfo.Length / (1024 * 1024 * 1024 * 1.0)}");
                    var taskNum = 10;
                    var taskPool = new Task<string>[taskNum];
                    for (int _i = 0; _i != taskNum; ++_i)
                    {
                        var t = Task.Factory.StartNew(() =>
                        {
                            var sb = new StringBuilder();
                            var rows = Enumerable.Range(0, 10000).Select(i => GenerateRandomRow());
                            foreach (var row in rows)
                            {
                                var line = $"\"{row._data0}\",{row.ignoreData1.ToString("F2")},{string.Join(",", row._data.Select(d => d.ToString("F2")))}, {row._ignoreData4206.ToString("F2")}, {row._ignoreData4207.ToString("F2")}, {row._ignoreData4208.ToString("F2")},\"{row._label}\"";
                                sb.AppendLine(line);
                            }

                            return sb.ToString();
                        });
                        taskPool[_i] = t;
                    }

                    var getRandomRow = await Task.WhenAll(taskPool);
                    foreach (var row in getRandomRow)
                    {
                        await stream.WriteAsync(row);
                    }

                    await stream.FlushAsync();
                    i++;
                }
            }
        }

        public async Task Large_csv_test()
        {
            var context = new MLContext(1);
            context.Log += (o, e) =>
            {
                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    this.Output.WriteLine(e.RawMessage);
                }
            };
            var trainPath = @"D:/large_csv.csv";
            var dataset = context.Data.LoadFromTextFile<ModelInput>(trainPath, ',', hasHeader: false);
            var experiment = context.Auto().CreateExperiment();
            var label = "Entry(Text)";
            var pipeline = context.Auto().Featurizer(dataset, excludeColumns: new[] { label })
                                .Append(context.Transforms.Conversion.MapValueToKey(label, label))
                                .Append(context.Auto().MultiClassification(label));

            experiment.SetDataset(context.Data.TrainTestSplit(dataset))
                    .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MacroAccuracy, label)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(50);

            var result = await experiment.RunAsync();
            result.Metric.Should().BeGreaterThan(0.5);
        }
    }

    class DummyTrialRunner : ITrialRunner
    {
        private readonly int _finishAfterNSeconds;
        private readonly CancellationToken _ct;
        private readonly IChannel _logger;

        public DummyTrialRunner(AutoMLExperiment.AutoMLExperimentSettings automlSettings, int finishAfterNSeconds, IChannel logger)
        {
            _finishAfterNSeconds = finishAfterNSeconds;
            _ct = automlSettings.CancellationToken;
            _logger = logger;
        }

        public void Dispose()
        {
        }

        public async Task<TrialResult> RunAsync(TrialSettings settings, CancellationToken ct)
        {
            _logger.Info("Update Running Trial");
            await Task.Delay(_finishAfterNSeconds * 1000, ct);
            _ct.ThrowIfCancellationRequested();
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

        public event EventHandler<double> CpuUsage;

        public event EventHandler<double> MemoryUsageInMegaByte;

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

        public void Start()
        {
            if (_timer == null)
            {
                _timer = new System.Timers.Timer(_checkIntervalInMilliseconds);
                _timer.Elapsed += (o, e) =>
                {
                    CpuUsage?.Invoke(this, 100);
                    MemoryUsageInMegaByte?.Invoke(this, 1000);
                };

                _timer.AutoReset = true;
                _timer.Enabled = true;
            }
        }

        public void Stop()
        {
            _timer?.Stop();
            _timer?.Dispose();
            _timer = null;
        }
    }

    class ModelInput
    {
        [LoadColumn(0), NoColumn]
        public string _data0 { get; set; }

        [LoadColumn(1), NoColumn]
        public float ignoreData1 { get; set; }

        [LoadColumn(2, 4205)]
        public float[] _data { get; set; }

        [LoadColumn(4206), NoColumn]//(4206,4208)]
        public float _ignoreData4206 { get; set; }
        [LoadColumn(4207), NoColumn]//(4206,4208)]
        public float _ignoreData4207 { get; set; }
        [LoadColumn(4208), NoColumn]//(4206,4208)]
        public float _ignoreData4208 { get; set; }

        [LoadColumn(4209), ColumnName("Entry(Text)")]
        public string _label { get; set; }
    }
}
