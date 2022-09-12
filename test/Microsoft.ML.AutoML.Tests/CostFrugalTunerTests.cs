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
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    public class CostFrugalTunerTests : BaseTestClass
    {
        public CostFrugalTunerTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void CFO_e2e_test()
        {
            var searchSpace = new SearchSpace<LbfgsOption>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            for (int i = 0; i != 1000; ++i)
            {
                var trialSettings = new TrialSettings()
                {
                    TrialId = i,
                };

                var param = cfo.Propose(trialSettings);
                trialSettings.Parameter = param;
                var option = param.AsType<LbfgsOption>();

                option.L1Regularization.Should().BeInRange(0.03125f, 32768.0f);
                option.L2Regularization.Should().BeInRange(0.03125f, 32768.0f);

                cfo.Update(new TrialResult()
                {
                    DurationInMilliseconds = i * 1000,
                    Metric = i,
                    TrialSettings = trialSettings,
                });
            }
        }

        [Fact]
        public void CFO_should_be_recoverd_if_history_provided()
        {
            // this test verify that cfo can be recovered by replaying history.
            var searchSpace = new SearchSpace<LbfgsOption>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
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
                if (x == 10 && y == 10 && z == 10)
                {
                    break;
                }

                var result = new TrialResult()
                {
                    Loss = loss,
                    DurationInMilliseconds = 1 * 1000,
                    TrialSettings = settings,
                };
                cfo.Update(result);
                history.Add(result);
            }

            var newCfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            foreach (var result in history.Take(99))
            {
                newCfo.Update(result);
            }

            var lastResult = history.Last();
            var trialSettings = lastResult.TrialSettings;

            var nextParameterFromNewCfo = newCfo.Propose(trialSettings);
            var lseParameterFromNewCfo = nextParameterFromNewCfo.AsType<LSE3DSearchSpace>();
            var lossFromNewCfo = -LSE3D(lseParameterFromNewCfo.X, lseParameterFromNewCfo.Y, lseParameterFromNewCfo.Z);
            lossFromNewCfo.Should().BeApproximately(lastResult.Loss, 0.1);
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
        public void CFO_should_find_maximum_value_when_function_is_convex()
        {
            var searchSpace = new SearchSpace<LSE3DSearchSpace>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            double bestMetric = 0;
            for (int i = 0; i != 100; ++i)
            {
                var trialSettings = new TrialSettings()
                {
                    TrialId = 0,
                };

                var param = cfo.Propose(trialSettings);
                trialSettings.Parameter = param;
                var lseParam = param.AsType<LSE3DSearchSpace>();
                var x = lseParam.X;
                var y = lseParam.Y;
                var z = lseParam.Z;
                var metric = LSE3D(x, y, z);
                bestMetric = Math.Max(bestMetric, metric);
                Output.WriteLine($"{i} x: {x} y: {y} z: {z}");
                if (x == 10 && y == 10 && z == 10)
                {
                    break;
                }
                cfo.Update(new TrialResult()
                {
                    Loss = -metric,
                    DurationInMilliseconds = 1 * 1000,
                    Metric = metric,
                    TrialSettings = trialSettings,
                });
            }

            bestMetric.Should().BeGreaterThan(LSE3D(10, 10, 10) - 2);
        }

        [Fact]
        public void CFO_should_find_minimum_value_when_function_is_convex()
        {
            var searchSpace = new SearchSpace<LSE3DSearchSpace>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            double loss = 0;
            for (int i = 0; i != 100; ++i)
            {
                var trialSettings = new TrialSettings()
                {
                    TrialId = i,
                };

                var param = cfo.Propose(trialSettings);
                trialSettings.Parameter = param;
                var lseParam = param.AsType<LSE3DSearchSpace>();
                var x = lseParam.X;
                var y = lseParam.Y;
                var z = lseParam.Z;
                loss = LSE3D(x, y, z);
                Output.WriteLine(loss.ToString());
                Output.WriteLine($"{i} x: {x} y: {y} z: {z}");

                if (x == -10 && y == -10 && z == -10)
                {
                    break;
                }

                cfo.Update(new TrialResult()
                {
                    Loss = loss,
                    DurationInMilliseconds = 1000,
                    Metric = loss,
                    TrialSettings = trialSettings,
                });
            }

            loss.Should().BeLessThan(LSE3D(-10, -10, -10) + 2);
        }

        [Fact]
        public void CFO_should_find_minimum_value_when_function_is_F1()
        {
            var searchSpace = new SearchSpace<LSE3DSearchSpace>();
            var initValues = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var cfo = new CostFrugalTuner(searchSpace, Parameter.FromObject(initValues));
            double bestMetric = 0;
            for (int i = 0; i != 1000; ++i)
            {
                var trialSettings = new TrialSettings()
                {
                    TrialId = i,
                };
                var param = cfo.Propose(trialSettings);
                trialSettings.Parameter = param;
                var lseParam = param.AsType<LSE3DSearchSpace>();
                var x = lseParam.X;
                var y = lseParam.Y;
                var z = lseParam.Z;
                var metric = F1(x, y, z);
                bestMetric = Math.Min(bestMetric, metric);
                Output.WriteLine($"{i} x: {x} y: {y} z: {z}");

                if (x == -1 && y == 1 && z == 0)
                {
                    break;
                }

                cfo.Update(new TrialResult()
                {
                    DurationInMilliseconds = 1,
                    Metric = metric,
                    TrialSettings = trialSettings,
                });
            }

            bestMetric.Should().BeLessThan(F1(-1, 1, 0) + 2);
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
    }
}
