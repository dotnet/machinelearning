// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
    }
}
