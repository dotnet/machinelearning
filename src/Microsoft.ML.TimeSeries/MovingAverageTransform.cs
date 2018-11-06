// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TimeSeriesProcessing;

[assembly: LoadableClass(MovingAverageTransform.Summary, typeof(MovingAverageTransform), typeof(MovingAverageTransform.Arguments), typeof(SignatureDataTransform),
    "Moving Average Transform", MovingAverageTransform.LoaderSignature, "MoAv")]
[assembly: LoadableClass(MovingAverageTransform.Summary, typeof(MovingAverageTransform), null, typeof(SignatureLoadDataTransform),
    "Moving Average Transform", MovingAverageTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// MovingAverageTransform is a weighted average of the values in
    /// the sliding window.
    /// </summary>
    public sealed class MovingAverageTransform : SequentialTransformBase<Single, Single, MovingAverageTransform.State>
    {
        public const string Summary = "Applies a moving average on a time series. Only finite values are taken into account.";
        public const string LoaderSignature = "MovingAverageTransform";

        public sealed class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column", ShortName = "name",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing the moving average", ShortName = "wnd", SortOrder = 3)]
            public int WindowSize = 2;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Lag between current observation and last observation from the sliding window", ShortName = "l", SortOrder = 4)]
            public int Lag = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "(optional) Comma separated list of weights, the first weight is applied to the oldest value. " +
                "An empty value will be replaced by uniform weights.",
                ShortName = "w", SortOrder = 5)]
            public string Weights = null;
        }

        private int _lag;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MOAVTRNS",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MovingAverageTransform).Assembly.FullName);
        }

        // _weights is null means a uniform moving average is computed.
        private readonly Single[] _weights;

        public MovingAverageTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(args.WindowSize + args.Lag - 1, args.WindowSize + args.Lag - 1, args.Source, args.Name, LoaderSignature, env, input)
        {
            Host.CheckUserArg(args.WindowSize >= 1, nameof(args.WindowSize), "Should be at least 1.");
            Host.CheckUserArg(args.Lag >= 0, nameof(args.Lag), "Should be positive.");
            Host.CheckUserArg(args.Lag != 0 || args.WindowSize > 1, nameof(args.Lag),
                "If lag=0 and wnd=1, the transform just copies the column. Use CopyColumn instead.");
            _weights = string.IsNullOrWhiteSpace(args.Weights) ? null : args.Weights.Split(',').Select(c => Convert.ToSingle(c)).ToArray();
            if (_weights != null && _weights.Length != args.WindowSize)
                throw Host.ExceptUserArg(nameof(args.Weights), string.Format("{0} weights are provided, but {1} are expected (or none)'", Utils.Size(_weights), args.WindowSize));
            _lag = args.Lag;
        }

        public MovingAverageTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            : base(env, ctx, LoaderSignature, input)
        {
            // *** Binary format ***
            // <base>
            // int: lag
            // Single[]: _weights

            _lag = ctx.Reader.ReadInt32();
            _weights = ctx.Reader.ReadFloatArray();

            Host.CheckDecode(WindowSize >= 1);
            Host.CheckDecode(_weights == null || Utils.Size(_weights) == WindowSize + 1 - _lag);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(WindowSize >= 1);
            Host.Assert(_lag >= 0);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // int: _lag
            // Single[]: _weights

            base.Save(ctx);
            ctx.Writer.Write(_lag);
            Host.Assert(_weights == null || Utils.Size(_weights) == WindowSize + 1 - _lag);
            ctx.Writer.WriteSingleArray(_weights);
        }

        private static Single ComputeMovingAverageUniformInitialisation(FixedSizeQueue<Single> others, Single input, int lag,
                                                  Single lastDropped, ref Single currentSum,
                                                  ref int nbNanValues)
        {
            Single sumValues = 0;
            nbNanValues = 0;
            int n;
            if (lag == 0)
            {
                if (Single.IsNaN(input))
                    nbNanValues = 1;
                else
                    sumValues = input;
                n = others.Count;
            }
            else
                n = others.Count - lag + 1;

            for (int i = 0; i < n; ++i)
            {
                if (Single.IsNaN(others[i]))
                    ++nbNanValues;
                else
                    sumValues += others[i];
            }
            int nb = others.Count + 1 - nbNanValues;
            currentSum = sumValues;
            return nb == 0 ? Single.NaN : sumValues / nb;
        }

        public static Single ComputeMovingAverageNonUniform(FixedSizeQueue<Single> others, Single input, Single[] weights, int lag)
        {
            Single sumWeights = 0;
            Single sumValues = 0;
            int n;
            if (lag == 0)
            {
                if (!Single.IsNaN(input))
                {
                    sumWeights = weights[weights.Length - 1];
                    sumValues = sumWeights * input;
                }
                n = others.Count;
            }
            else
                n = others.Count - lag + 1;

            for (int i = 0; i < n; ++i)
            {
                if (!Single.IsNaN(others[i]))
                {
                    sumWeights += weights[i];
                    sumValues += weights[i] * others[i];
                }
            }
            return sumWeights != 0 ? sumValues / sumWeights : Single.NaN;
        }

        /// <summary>
        /// Possible returns:
        ///
        /// Finite Value: no infinite value in the sliding window and at least a non NaN value
        /// NaN value: only NaN values in the sliding window or +/- Infinite
        /// Inifinite value: one infinite value in the sliding window (sign is no relevant)
        /// </summary>
        public static Single ComputeMovingAverageUniform(FixedSizeQueue<Single> others, Single input, int lag,
                                                         Single lastDropped, ref Single currentSum,
                                                         ref bool initUniformMovingAverage,
                                                         ref int nbNanValues)
        {
            if (initUniformMovingAverage)
            {
                initUniformMovingAverage = false;
                return ComputeMovingAverageUniformInitialisation(others, input, lag,
                                                lastDropped, ref currentSum, ref nbNanValues);
            }
            else
            {
                if (Single.IsNaN(lastDropped))
                    --nbNanValues;
                else if (!FloatUtils.IsFinite(lastDropped))
                    // One infinite value left,
                    // we need to recompute everything as we don't know how many infinite values are in the sliding window.
                    return ComputeMovingAverageUniformInitialisation(others, input, lag,
                                                lastDropped, ref currentSum, ref nbNanValues);
                else
                    currentSum -= lastDropped;

                // lastDropped is finite
                Contracts.Assert(FloatUtils.IsFinite(lastDropped) || Single.IsNaN(lastDropped));

                var newValue = lag == 0 ? input : others[others.Count - lag];
                if (!Single.IsNaN(newValue) && !FloatUtils.IsFinite(newValue))
                    // One infinite value entered,
                    // we need to recompute everything as we don't know how many infinite values are in the sliding window.
                    return ComputeMovingAverageUniformInitialisation(others, input, lag,
                                                lastDropped, ref currentSum, ref nbNanValues);

                // lastDropped is finite and input is finite or NaN
                Contracts.Assert(FloatUtils.IsFinite(newValue) || Single.IsNaN(newValue));

                if (!Single.IsNaN(currentSum) && !FloatUtils.IsFinite(currentSum))
                {
                    if (Single.IsNaN(newValue))
                    {
                        ++nbNanValues;
                        return currentSum;
                    }
                    else
                        return FloatUtils.IsFinite(newValue) ? currentSum : (currentSum + newValue);
                }

                // lastDropped is finite, input is finite or NaN, currentSum is finite or NaN
                Contracts.Assert(FloatUtils.IsFinite(currentSum) || Single.IsNaN(currentSum));

                if (Single.IsNaN(newValue))
                {
                    ++nbNanValues;
                    int nb = (lag == 0 ? others.Count + 1 : others.Count - lag + 1) - nbNanValues;
                    return nb == 0 ? Single.NaN : currentSum / nb;
                }
                else
                {
                    int nb = lag == 0 ? others.Count + 1 - nbNanValues : others.Count + 1 - nbNanValues - lag;
                    currentSum += input;
                    return nb == 0 ? Single.NaN : currentSum / nb;
                }
            }
        }

        public sealed class State : StateBase
        {
            private Single[] _weights;
            private int _lag;

            // This is only needed when we compute a uniform moving average.
            // A temptation could be to extend the buffer size but then the moving average would
            // start producing values 1 iteration later than expected.
            private Single _lastDroppedValue;
            private Single _currentSum;

            // When the moving average is uniform, the computational is incremental,
            // except for the first iteration or after encountering infinities.
            private bool _initUniformMovingAverage;

            // When the moving aveage is uniform, we need to remember how many NA values
            // take part of the computation.
            private int _nbNanValues;

            protected override void SetNaOutput(ref Single output)
            {
                output = Single.NaN;
            }

            /// <summary>
            /// input is not included
            /// </summary>
            /// <param name="input"></param>
            /// <param name="windowedBuffer"></param>
            /// <param name="iteration"></param>
            /// <param name="output"></param>
            protected override void TransformCore(ref Single input, FixedSizeQueue<Single> windowedBuffer, long iteration, ref Single output)
            {
                if (_weights == null)
                    output = ComputeMovingAverageUniform(windowedBuffer, input, _lag, _lastDroppedValue, ref _currentSum, ref _initUniformMovingAverage, ref _nbNanValues);
                else
                    output = ComputeMovingAverageNonUniform(windowedBuffer, input, _weights, _lag);
                _lastDroppedValue = windowedBuffer[0];
            }

            protected override void InitializeStateCore()
            {
                _weights = ((MovingAverageTransform)ParentTransform)._weights;
                _lag = ((MovingAverageTransform)ParentTransform)._lag;
                _initUniformMovingAverage = true;
            }

            protected override void LearnStateFromDataCore(FixedSizeQueue<Single> data)
            {
                // This method is empty because there is no need for parameter learning from the initial windowed buffer for this transform.
            }
        }
    }
}
