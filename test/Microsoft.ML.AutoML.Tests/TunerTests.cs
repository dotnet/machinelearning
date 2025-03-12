// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading;
using FluentAssertions;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.SearchSpace.Tuner;
using Microsoft.ML.TestFramework;
using Tensorflow.Contexts;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class TunerTests : BaseTestClass
    {
        public TunerTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void tuner_e2e_test()
        {
            var context = new MLContext(1);
            var searchSpace = new SearchSpace<LbfgsOption>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            var smac = new SmacTuner(context, searchSpace, seed: 1);
            var tunerCandidates = new Dictionary<string, ITuner>()
            {
                {"cfo", cfo },
                {"smac", smac },
            };
            foreach (var kv in tunerCandidates)
            {
                var tuner = kv.Value;
                for (int i = 0; i != 1000; ++i)
                {
                    var trialSettings = new TrialSettings()
                    {
                        TrialId = i,
                    };

                    var param = tuner.Propose(trialSettings);
                    trialSettings.Parameter = param;
                    var option = param.AsType<LbfgsOption>();

                    option.L1Regularization.Should().BeInRange(0.03125f, 32768.0f);
                    option.L2Regularization.Should().BeInRange(0.03125f, 32768.0f);

                    tuner.Update(new TrialResult()
                    {
                        DurationInMilliseconds = i * 1000,
                        Metric = i,
                        TrialSettings = trialSettings,
                    });
                }
            }
        }

        [Fact]
        public void Smac_should_ignore_fail_trials_during_initialize()
        {
            // fix for https://github.com/dotnet/machinelearning-modelbuilder/issues/2721
            var context = new MLContext(1);
            var searchSpace = new SearchSpace<LbfgsOption>();
            var tuner = new SmacTuner(context, searchSpace, seed: 1);
            for (int i = 0; i != 1000; ++i)
            {
                var trialSettings = new TrialSettings()
                {
                    TrialId = i,
                };

                var param = tuner.Propose(trialSettings);
                trialSettings.Parameter = param;
                var option = param.AsType<LbfgsOption>();

                option.L1Regularization.Should().BeInRange(0.03125f, 32768.0f);
                option.L2Regularization.Should().BeInRange(0.03125f, 32768.0f);

                tuner.Update(new TrialResult()
                {
                    DurationInMilliseconds = i * 1000,
                    Loss = double.NaN,
                    TrialSettings = trialSettings,
                });
            }

            tuner.Candidates.Count.Should().Be(0);
            tuner.Histories.Count.Should().Be(0);
        }

        [Fact]
        public void CFO_should_be_recoverd_if_history_provided()
        {
            // this test verify that cfo can be recovered by replaying history.
            var searchSpace = new SearchSpace<LSE3DSearchSpace>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var seed = 0;
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues), seed: seed);
            var history = new List<TrialResult>();
            for (int i = 0; i != 100; ++i)
            {
                var settings = new TrialSettings()
                {
                    TrialId = i,
                };

                var param = cfo.Propose(settings);
                settings.Parameter = param;
                var lseParam = param.AsType<LSE3DSearchSpace>();
                var x = lseParam.X;
                var y = lseParam.Y;
                var z = lseParam.Z;
                var loss = -LSE3D(x, y, z);
                var result = new TrialResult()
                {
                    Loss = loss,
                    DurationInMilliseconds = 1 * 1000,
                    TrialSettings = settings,
                };
                cfo.Update(result);
                history.Add(result);
            }

            var newCfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues), history.Take(99), seed);
            var lastResult = history.Last();
            var trialSettings = lastResult.TrialSettings;
            var nextParameterFromNewCfo = newCfo.Propose(trialSettings);
            var lseParameterFromNewCfo = nextParameterFromNewCfo.AsType<LSE3DSearchSpace>();
            var lossFromNewCfo = -LSE3D(lseParameterFromNewCfo.X, lseParameterFromNewCfo.Y, lseParameterFromNewCfo.Z);
            lastResult.Loss.Should().BeApproximately(-10.1537, 1e-3);
            lossFromNewCfo.Should().BeApproximately(-11.0986, 0.01);
        }

        [Fact]
        public void CFO_should_start_from_init_point_if_provided()
        {
            var trialSettings = new TrialSettings()
            {
                TrialId = 0,
            };
            var searchSpace = new SearchSpace<LSE3DSearchSpace>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            var param = cfo.Propose(trialSettings).AsType<LSE3DSearchSpace>();
            var x = param.X;
            var y = param.Y;
            var z = param.Z;

            (x * x + y * y + z * z).Should().Be(0);
        }

        [Fact]
        public void EciCfo_should_handle_trial_result_with_nan_value()
        {
            // this test verify if tuner can find max value for LSE.
            var context = new MLContext(1);
            var pipeline = this.CreateDummySweepablePipeline(context);
            var searchSpace = new SearchSpace.SearchSpace();
            searchSpace["_pipeline_"] = pipeline.SearchSpace;
            var tuner = new EciCostFrugalTuner(pipeline, new AutoMLExperiment.AutoMLExperimentSettings
            {
                SearchSpace = searchSpace,
                Seed = 1,
            });
            var invalidLosses = Enumerable.Repeat(new[] { double.NaN, double.NegativeInfinity, double.PositiveInfinity }, 100)
                                .SelectMany(loss => loss);
            var id = 0;
            foreach (var loss in invalidLosses)
            {
                var trialSetting = new TrialSettings
                {
                    TrialId = id++,
                    Parameter = Parameter.CreateNestedParameter(),
                };
                var parameter = tuner.Propose(trialSetting);
                trialSetting.Parameter = parameter;
                var trialResult = new TrialResult
                {
                    TrialSettings = trialSetting,
                    DurationInMilliseconds = 10000,
                    Loss = loss,
                };
                tuner.Update(trialResult);
            }
        }

        [Fact]
        public void EciCfo_should_handle_trial_result_with_no_improvements_over_losses()
        {
            // this test verify if tuner can find max value for LSE.
            var context = new MLContext(1);
            var pipeline = this.CreateDummySweepablePipeline(context);
            var searchSpace = new SearchSpace.SearchSpace();
            searchSpace["_pipeline_"] = pipeline.SearchSpace;
            var tuner = new EciCostFrugalTuner(pipeline, new AutoMLExperiment.AutoMLExperimentSettings
            {
                SearchSpace = searchSpace,
                Seed = 1,
            });
            var zeroLosses = Enumerable.Repeat(0.0, 100);
            var randomLosses = Enumerable.Range(0, 100).Select(i => i * 0.1);
            var id = 0;
            foreach (var loss in zeroLosses.Concat(randomLosses))
            {
                var trialSetting = new TrialSettings
                {
                    TrialId = id++,
                    Parameter = Parameter.CreateNestedParameter(),
                };
                var parameter = tuner.Propose(trialSetting);
                trialSetting.Parameter = parameter;
                var trialResult = new TrialResult
                {
                    TrialSettings = trialSetting,
                    DurationInMilliseconds = 10000,
                    Loss = loss,
                };
                tuner.Update(trialResult);
            }
        }

        [Fact]
        public void LSE_maximize_test()
        {
            // this test verify if tuner can find max value for LSE.
            var context = new MLContext(1);
            var searchSpace = new SearchSpace<LSE3DSearchSpace>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            var smac = new SmacTuner(context, searchSpace, seed: 1, numberOfTrees: 3);
            var randomTuner = new RandomSearchTuner(searchSpace, seed: 1);
            var tunerCandidates = new Dictionary<string, ITuner>()
            {
                {"cfo", cfo },
                {"smac", smac },
                {"rnd", randomTuner },
            };

            foreach (var kv in tunerCandidates)
            {
                Output.WriteLine($"verify tuner {kv.Key}");
                var tuner = kv.Value;
                double bestMetric = double.MinValue;
                for (int i = 0; i != 100; ++i)
                {
                    var trialSettings = new TrialSettings()
                    {
                        TrialId = 0,
                    };

                    var param = tuner.Propose(trialSettings);
                    trialSettings.Parameter = param;
                    var lseParam = param.AsType<LSE3DSearchSpace>();
                    var x = lseParam.X;
                    var y = lseParam.Y;
                    var z = lseParam.Z;
                    var metric = LSE3D(x, y, z);
                    bestMetric = Math.Max(bestMetric, metric);
                    tuner.Update(new TrialResult()
                    {
                        Loss = -metric,
                        DurationInMilliseconds = 1 * 1000,
                        Metric = metric,
                        TrialSettings = trialSettings,
                    });
                }

                Output.WriteLine($"best metric: {bestMetric}");

                // 10.5 is the best metric from random tuner
                // and the other tuners should achieve better metric comparing with random tuner.
                bestMetric.Should().BeGreaterThanOrEqualTo(10.5);
            }
        }

        [Fact]
        public void LSE_minimize_test()
        {
            // this test verify if tuner can find min value for LSE.
            var context = new MLContext(1);

            var searchSpace = new SearchSpace<LSE3DSearchSpace>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            var smac = new SmacTuner(context, searchSpace, numberOfTrees: 3, seed: 1);
            var randomTuner = new RandomSearchTuner(searchSpace, seed: 1);
            var tunerCandidates = new Dictionary<string, ITuner>()
            {
                {"cfo", cfo },
                {"smac", smac },
                {"rnd", randomTuner },
            };

            foreach (var kv in tunerCandidates)
            {
                Output.WriteLine($"verify tuner {kv.Key}");
                var tuner = kv.Value;
                double bestLoss = double.MaxValue;
                for (int i = 0; i != 200; ++i)
                {
                    var trialSettings = new TrialSettings()
                    {
                        TrialId = i,
                    };

                    var param = tuner.Propose(trialSettings);
                    trialSettings.Parameter = param;
                    var lseParam = param.AsType<LSE3DSearchSpace>();
                    var x = lseParam.X;
                    var y = lseParam.Y;
                    var z = lseParam.Z;
                    var loss = LSE3D(x, y, z);
                    bestLoss = Math.Min(bestLoss, loss);
                    tuner.Update(new TrialResult()
                    {
                        Loss = loss,
                        DurationInMilliseconds = 1000,
                        TrialSettings = trialSettings,
                    });
                }

                Output.WriteLine($"best metric: {bestLoss}");
                bestLoss.Should().BeLessThan(-7);
            }
        }

        [Fact]
        public void F1_minimize_test()
        {
            var context = new MLContext(1);
            var searchSpace = new SearchSpace<LSE3DSearchSpace>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            var smac = new SmacTuner(context, searchSpace, seed: 1, localSearchParentCount: 10);
            var randomTuner = new RandomSearchTuner(searchSpace, seed: 1);
            var tunerCandidates = new Dictionary<string, ITuner>()
            {
                {"cfo", cfo },
                {"rnd", randomTuner },
                {"smac", smac },
            };
            foreach (var kv in tunerCandidates)
            {
                var tuner = kv.Value;
                double bestMetric = double.MaxValue;
                for (int i = 0; i != 100; ++i)
                {
                    var trialSettings = new TrialSettings()
                    {
                        TrialId = i,
                    };
                    var param = tuner.Propose(trialSettings);
                    trialSettings.Parameter = param;
                    var lseParam = param.AsType<LSE3DSearchSpace>();
                    var x = lseParam.X;
                    var y = lseParam.Y;
                    var z = lseParam.Z;
                    var metric = F1(x, y, z);
                    bestMetric = Math.Min(bestMetric, metric);
                    tuner.Update(new TrialResult()
                    {
                        DurationInMilliseconds = 1,
                        Metric = metric,
                        TrialSettings = trialSettings,
                        Loss = metric,
                    });
                }
                Output.WriteLine($"{kv.Key} - best metric {bestMetric}");

                // 6.1 is the best metric from random tuner.
                // and we assume that other tuners should achieve better result than random tuner
                bestMetric.Should().BeLessThan(6.1);
            }
        }

        [Fact]
        public void Hyper_parameters_from_CFO_should_be_culture_invariant_string()
        {
            var searchSpace = new SearchSpace<LSE3DSearchSpace>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            var originalCuture = Thread.CurrentThread.CurrentCulture;
            var usCulture = new CultureInfo("en-US", false);
            Thread.CurrentThread.CurrentCulture = usCulture;

            Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator.Should().Be(".");
            for (int i = 0; i != 100; ++i)
            {
                var trialSettings = new TrialSettings()
                {
                    TrialId = i,
                };
                var param = cfo.Propose(trialSettings).AsType<LSE3DSearchSpace>();
                param.X.Should().BeInRange(-10, 10);
            }

            var frCulture = new CultureInfo("fr-FR", false);
            Thread.CurrentThread.CurrentCulture = frCulture;
            Thread.CurrentThread.CurrentCulture.NumberFormat.NumberDecimalSeparator.Should().Be(",");
            for (int i = 0; i != 100; ++i)
            {
                var trialSettings = new TrialSettings()
                {
                    TrialId = i,
                };
                var param = cfo.Propose(trialSettings).AsType<LSE3DSearchSpace>();
                param.X.Should().BeInRange(-10, 10);
            }

            Thread.CurrentThread.CurrentCulture = originalCuture;
        }

        /// <summary>
        /// LSE is strictly convex, use this as test object function.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <returns></returns>
        private double LSE3D(double x, double y, double z)
        {
            return Math.Log(Math.Exp(x) + Math.Exp(y) + Math.Exp(z));
        }

        private double F1(double x, double y, double z)
        {
            return x * x + 2 * x + 1 + y * y - 2 * y + 1 + z * z;
        }

        private class LSE3DSearchSpace
        {
            [Range(-10.0, 10.0, 0.0, false)]
            public double X { get; set; }

            [Range(-10.0, 10.0, 0.0, false)]
            public double Y { get; set; }

            [Range(-10.0, 10.0, 0.0, false)]
            public double Z { get; set; }
        }

        private SweepablePipeline CreateDummySweepablePipeline(MLContext context)
        {
            var mapKeyToValue = SweepableEstimatorFactory.CreateMapKeyToValue(new MapKeyToValueOption
            {
                InputColumnName = "input",
                OutputColumnName = "output",
            });

            return mapKeyToValue.Append(context.Auto().BinaryClassification());
        }
    }
}
