// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: EntryPointModule(typeof(TimeSeriesProcessingEntryPoints))]

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// Entry points for text anylytics transforms.
    /// </summary>
    internal static class TimeSeriesProcessingEntryPoints
    {
        [TlcModule.EntryPoint(Desc = ExponentialAverageTransform.Summary, UserName = ExponentialAverageTransform.UserName, ShortName = ExponentialAverageTransform.ShortName)]
        internal static CommonOutputs.TransformOutput ExponentialAverage(IHostEnvironment env, ExponentialAverageTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ExponentialAverageTransform", input);
            var xf = new ExponentialAverageTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Desc = TimeSeries.IidChangePointDetector.Summary,
            UserName = TimeSeries.IidChangePointDetector.UserName,
            ShortName = TimeSeries.IidChangePointDetector.ShortName)]
        internal static CommonOutputs.TransformOutput IidChangePointDetector(IHostEnvironment env, IidChangePointDetector.Options options)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "IidChangePointDetector", options);
            var view = new IidChangePointEstimator(h, options).Fit(options.Data).Transform(options.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, options.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = TimeSeries.IidSpikeDetector.Summary,
            UserName = TimeSeries.IidSpikeDetector.UserName,
            ShortName = TimeSeries.IidSpikeDetector.ShortName)]
        internal static CommonOutputs.TransformOutput IidSpikeDetector(IHostEnvironment env, IidSpikeDetector.Options options)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "IidSpikeDetector", options);
            var view = new IidSpikeEstimator(h, options).Fit(options.Data).Transform(options.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, options.Data),
                OutputData = view
            };
        }

        public sealed class TimeSeriesPredictionInput : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Model file path", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, SortOrder = 2)]
            public string ModelPath;
        }
# pragma warning disable MSML_GeneralName
        private class TimeSeriesData
        {
            public float t1;
            public float t2;
            public float t3;

            public TimeSeriesData(float value)
            {
                t1 = 1;
                t2 = value;
                t3 = 1;
            }
        }

        private class SsaSpikePrediction
        {
            public double[] t2_spikes { get; set; }
        }

        [TlcModule.EntryPoint(Name = "TimeSeries.OnlineLearning", Desc = "Runs predictions on new observations and updates the model file",
            UserName = "TBD",
            ShortName = "TBD")]
        public static CommonOutputs.TransformOutput TimeSeriesPredictionEngine(IHostEnvironment env, TimeSeriesPredictionInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TimeSeriesPrediction");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var ml = new MLContext();
            var model = ml.Model.Load(input.ModelPath, out DataViewSchema inputSchema);
            var model1 = new StatefulTimeseriesTransformer(env, model);
            var predictions = model1.Transform(input.Data);
            // model.Save()

            // Create a time series prediction engine from the loaded model.
            // var engine = model.CreateTimeSeriesEngine<TimeSeriesData, SsaSpikePrediction>(host);

            //var predictions = engine.Predict(new TimeSeriesData(10));
            // var predictions1 = engine.Transform(input.Data);
            // TBD this will take effect ONLY if input.Data is replayed through model
            //engine.CheckPoint(host, input.ModelPath);

            return new CommonOutputs.TransformOutput()
            {
                OutputData = predictions,
                Model = null
            };
        }

        [TlcModule.EntryPoint(Desc = TimeSeries.PercentileThresholdTransform.Summary,
            UserName = TimeSeries.PercentileThresholdTransform.UserName,
            ShortName = TimeSeries.PercentileThresholdTransform.ShortName)]
        internal static CommonOutputs.TransformOutput PercentileThresholdTransform(IHostEnvironment env, PercentileThresholdTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "PercentileThresholdTransform", input);
            var view = new PercentileThresholdTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = TimeSeries.PValueTransform.Summary,
            UserName = TimeSeries.PValueTransform.UserName,
            ShortName = TimeSeries.PValueTransform.ShortName)]
        internal static CommonOutputs.TransformOutput PValueTransform(IHostEnvironment env, PValueTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "PValueTransform", input);
            var view = new PValueTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = TimeSeries.SlidingWindowTransform.Summary,
            UserName = TimeSeries.SlidingWindowTransform.UserName,
            ShortName = TimeSeries.SlidingWindowTransform.ShortName)]
        internal static CommonOutputs.TransformOutput SlidingWindowTransform(IHostEnvironment env, SlidingWindowTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SlidingWindowTransform", input);
            var view = new SlidingWindowTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = TimeSeries.SsaChangePointDetector.Summary,
            UserName = TimeSeries.SsaChangePointDetector.UserName,
            ShortName = TimeSeries.SsaChangePointDetector.ShortName)]
        internal static CommonOutputs.TransformOutput SsaChangePointDetector(IHostEnvironment env, SsaChangePointDetector.Options options)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SsaChangePointDetector", options);
            var view = new SsaChangePointEstimator(h, options).Fit(options.Data).Transform(options.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, options.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = TimeSeries.SsaSpikeDetector.Summary,
            UserName = TimeSeries.SsaSpikeDetector.UserName,
            ShortName = TimeSeries.SsaSpikeDetector.ShortName)]
        public static CommonOutputs.TransformOutput SsaSpikeDetector(IHostEnvironment env, SsaSpikeDetector.Options options)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SsaSpikeDetector", options);
            var view = new SsaSpikeEstimator(h, options).Fit(options.Data).Transform(options.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, options.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = TimeSeries.SsaForecastingTransformer.Summary,
            UserName = TimeSeries.SsaForecastingTransformer.UserName,
            ShortName = TimeSeries.SsaForecastingTransformer.ShortName)]
        internal static CommonOutputs.TransformOutput SsaForecasting(IHostEnvironment env, SsaForecastingTransformer.Options options)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SsaForecasting", options);
            var view = new SsaForecastingEstimator(h, options).Fit(options.Data).Transform(options.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, options.Data),
                OutputData = view
            };
        }
    }
}
