﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.TimeSeriesProcessing;

[assembly: EntryPointModule(typeof(TimeSeriesProcessing))]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// Entry points for text anylytics transforms.
    /// </summary>
    public static class TimeSeriesProcessing
    {
        [TlcModule.EntryPoint(Desc = ExponentialAverageTransform.Summary, UserName = ExponentialAverageTransform.UserName, ShortName = ExponentialAverageTransform.ShortName)]
        public static CommonOutputs.TransformOutput ExponentialAverage(IHostEnvironment env, ExponentialAverageTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "ExponentialAverageTransform", input);
            var xf = new ExponentialAverageTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Desc = Runtime.TimeSeriesProcessing.IidChangePointDetector.Summary, UserName = Runtime.TimeSeriesProcessing.IidChangePointDetector.UserName, ShortName = Runtime.TimeSeriesProcessing.IidChangePointDetector.ShortName)]
        public static CommonOutputs.TransformOutput IidChangePointDetector(IHostEnvironment env, IidChangePointDetector.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "IidChangePointDetector", input);
            var view = new IidChangePointEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = Runtime.TimeSeriesProcessing.IidSpikeDetector.Summary, UserName = Runtime.TimeSeriesProcessing.IidSpikeDetector.UserName, ShortName = Runtime.TimeSeriesProcessing.IidSpikeDetector.ShortName)]
        public static CommonOutputs.TransformOutput IidSpikeDetector(IHostEnvironment env, IidSpikeDetector.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "IidSpikeDetector", input);
            var view = new IidSpikeEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = Runtime.TimeSeriesProcessing.PercentileThresholdTransform.Summary, UserName = Runtime.TimeSeriesProcessing.PercentileThresholdTransform.UserName, ShortName = Runtime.TimeSeriesProcessing.PercentileThresholdTransform.ShortName)]
        public static CommonOutputs.TransformOutput PercentileThresholdTransform(IHostEnvironment env, PercentileThresholdTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "PercentileThresholdTransform", input);
            var view = new PercentileThresholdTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = Runtime.TimeSeriesProcessing.PValueTransform.Summary, UserName = Runtime.TimeSeriesProcessing.PValueTransform.UserName, ShortName = Runtime.TimeSeriesProcessing.PValueTransform.ShortName)]
        public static CommonOutputs.TransformOutput PValueTransform(IHostEnvironment env, PValueTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "PValueTransform", input);
            var view = new PValueTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = Runtime.TimeSeriesProcessing.SlidingWindowTransform.Summary, UserName = Runtime.TimeSeriesProcessing.SlidingWindowTransform.UserName, ShortName = Runtime.TimeSeriesProcessing.SlidingWindowTransform.ShortName)]
        public static CommonOutputs.TransformOutput SlidingWindowTransform(IHostEnvironment env, SlidingWindowTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SlidingWindowTransform", input);
            var view = new SlidingWindowTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = Runtime.TimeSeriesProcessing.SsaChangePointDetector.Summary, UserName = Runtime.TimeSeriesProcessing.SsaChangePointDetector.UserName, ShortName = Runtime.TimeSeriesProcessing.SsaChangePointDetector.ShortName)]
        public static CommonOutputs.TransformOutput SsaChangePointDetector(IHostEnvironment env, SsaChangePointDetector.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SsaChangePointDetector", input);
            var view = new SsaChangePointEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Desc = Runtime.TimeSeriesProcessing.SsaSpikeDetector.Summary, UserName = Runtime.TimeSeriesProcessing.SsaSpikeDetector.UserName, ShortName = Runtime.TimeSeriesProcessing.SsaSpikeDetector.ShortName)]
        public static CommonOutputs.TransformOutput SsaSpikeDetector(IHostEnvironment env, SsaSpikeDetector.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SsaSpikeDetector", input);
            var view = new SsaSpikeEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
