// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;

[assembly: LoadableClass(ExponentialAverageTransform.Summary, typeof(ExponentialAverageTransform), typeof(ExponentialAverageTransform.Arguments), typeof(SignatureDataTransform),
    ExponentialAverageTransform.UserName, ExponentialAverageTransform.LoaderSignature, ExponentialAverageTransform.ShortName)]
[assembly: LoadableClass(ExponentialAverageTransform.Summary, typeof(ExponentialAverageTransform), null, typeof(SignatureLoadDataTransform),
    ExponentialAverageTransform.UserName, ExponentialAverageTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// ExponentialAverageTransform is a weighted average of the values: ExpAvg(y_t) = a * y_t + (1-a) * ExpAvg(y_(t-1)).
    /// </summary>
    public sealed class ExponentialAverageTransform : SequentialTransformBase<Single, Single, ExponentialAverageTransform.State>
    {
        public const string Summary = "Applies a Exponential average on a time series.";
        public const string LoaderSignature = "ExpAverageTransform";
        public const string UserName = "Exponential Average Transform";
        public const string ShortName = "ExpAvg";

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column", ShortName = "name",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Coefficient d in: d m(y_t) = d * y_t + (1-d) * m(y_(t-1)), it should be in [0, 1].",
                ShortName = "d", SortOrder = 4)]
            public Single Decay = 0.9f;
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EXAVTRNS",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ExponentialAverageTransform).Assembly.FullName);
        }

        private readonly Single _decay;

        public ExponentialAverageTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(1, 1, args.Source, args.Name, LoaderSignature, env, input)
        {
            Host.CheckUserArg(0 <= args.Decay && args.Decay <= 1, nameof(args.Decay), "Should be in [0, 1].");
            _decay = args.Decay;
        }

        public ExponentialAverageTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            : base(env, ctx, LoaderSignature, input)
        {
            // *** Binary format ***
            // <base>
            // Single _decay

            _decay = ctx.Reader.ReadSingle();

            Host.CheckDecode(0 <= _decay && _decay <= 1);
            Host.CheckDecode(WindowSize == 1);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(WindowSize >= 1);
            Host.Assert(0 <= _decay && _decay <= 1);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // Single _decay

            base.Save(ctx);
            ctx.Writer.Write(_decay);
        }

        public static Single ComputeExponentialAverage(Single input, Single decay, Single previousAverage)
        {
            return decay * input + (1 - decay) * previousAverage;
        }

        public sealed class State : StateBase
        {
            private Single _previousAverage;
            private bool _firstIteration;
            private Single _decay;

            public State()
            {
                _firstIteration = true;
            }

            private protected override void SetNaOutput(ref Single output)
            {
                output = Single.NaN;
            }

            private protected override void TransformCore(ref Single input, FixedSizeQueue<Single> windowedBuffer, long iteration, ref Single output)
            {
                if (_firstIteration)
                {
                    // we only need the buffer at the first iteration
                    _previousAverage = windowedBuffer[0];
                    _firstIteration = false;
                }
                output = ComputeExponentialAverage(input, _decay, _previousAverage);
                // we keep the previous average in memory
                _previousAverage = output;
            }

            private protected override void InitializeStateCore()
            {
                _firstIteration = true;
                _decay = ((ExponentialAverageTransform)ParentTransform)._decay;
            }

            private protected override void LearnStateFromDataCore(FixedSizeQueue<Single> data)
            {
                // This method is empty because there is no need for parameter learning from the initial windowed buffer for this transform.
            }
        }
    }
}
