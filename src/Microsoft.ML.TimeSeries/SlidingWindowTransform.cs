// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;

[assembly: LoadableClass(SlidingWindowTransform.Summary, typeof(SlidingWindowTransform), typeof(SlidingWindowTransform.Arguments), typeof(SignatureDataTransform),
    SlidingWindowTransform.UserName, SlidingWindowTransform.LoaderSignature, SlidingWindowTransform.ShortName)]
[assembly: LoadableClass(SlidingWindowTransform.Summary, typeof(SlidingWindowTransform), null, typeof(SignatureLoadDataTransform),
    SlidingWindowTransform.UserName, SlidingWindowTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// Outputs a sliding window on a time series of type Single.
    /// </summary>
    public sealed class SlidingWindowTransform : SlidingWindowTransformBase<Single>
    {
        public const string Summary = "Returns the last values for a time series [y(t-d-l+1), y(t-d-l+2), ..., y(t-l-1), y(t-l)] where d is the size of the window, l the lag and y is a Float.";
        public const string LoaderSignature = "SlideWinTransform";
        public const string UserName = "Sliding Window Transform";
        public const string ShortName = "SlideWin";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SWINTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SlidingWindowTransform).Assembly.FullName);
        }

        public SlidingWindowTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(args, LoaderSignature, env, input)
        {
        }

        public SlidingWindowTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            : base(env, ctx, LoaderSignature, input)
        {
            // *** Binary format ***
            // <base>
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            base.Save(ctx);
        }
    }
}
