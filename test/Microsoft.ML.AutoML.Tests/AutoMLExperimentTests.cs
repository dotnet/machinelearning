// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Microsoft.Data.Analysis;
using Microsoft.ML.TestFramework;
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
            var pipeline = context.Transforms.Concatenate("Features", "Features")
                            .Append(context.Auto().Regression());

            var experiment = context.Auto().CreateExperiment();
            experiment.SetPipeline(pipeline)
                      .SetDataset(GetDummyData(), 10)
                      .SetEvaluateMetric(RegressionMetric.RootMeanSquaredError, "Label")
                      .SetTrainingTimeInSeconds(100);

            var cts = new CancellationTokenSource();

            context.Log += (o, e) =>
            {
                if (e.RawMessage.Contains("Update Running Trial"))
                {
                    cts.CancelAfter(1);
                }
            };

            var runExperimentAction = async () => await experiment.RunAsync(cts.Token);

            await runExperimentAction.Should().ThrowExactlyAsync<TimeoutException>();
        }

        [Fact]
        public async Task AutoMLExperiment_return_current_best_trial_when_ct_is_canceled_with_trial_completed_Async()
        {
            var context = new MLContext(1);
            var pipeline = context.Transforms.Concatenate("Features", "Features")
                            .Append(context.Auto().Regression());

            var experiment = context.Auto().CreateExperiment();
            experiment.SetPipeline(pipeline)
                      .SetDataset(GetDummyData(), 10)
                      .SetEvaluateMetric(RegressionMetric.RootMeanSquaredError, "Label")
                      .SetTrainingTimeInSeconds(100);
            var cts = new CancellationTokenSource();

            context.Log += (o, e) =>
            {
                if (e.RawMessage.Contains("Update Completed Trial"))
                {
                    cts.CancelAfter(100);
                }
            };

            var res = await experiment.RunAsync(cts.Token);
            res.Metric.Should().BeGreaterThan(0);
        }

        private IDataView GetDummyData()
        {
            var x = Enumerable.Range(-10000, 10000).Select(value => value * 1f).ToArray();
            var y = x.Select(value => value * value);

            var df = new DataFrame();
            df["Features"] = DataFrameColumn.Create("Features", x);
            df["Label"] = DataFrameColumn.Create("Label", y);

            return df;
        }
    }
}
