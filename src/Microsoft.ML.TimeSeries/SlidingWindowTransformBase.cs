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
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// SlidingWindowTransformBase outputs a sliding window as a VBuffer from a series of any type.
    /// The VBuffer contains n consecutives observations delayed or not from the current one.
    /// Let's denote y(t) a timeseries, the transform returns a vector of values for each time t
    /// which corresponds to [y(t-d-l+1), y(t-d-l+2), ..., y(t-l-1), y(t-l)] where d is the size of the window
    /// and l is the delay.
    /// </summary>

    public abstract class SlidingWindowTransformBase<TInput> : SequentialTransformBase<TInput, VBuffer<TInput>, SlidingWindowTransformBase<TInput>.StateSlide>
    {
        /// <summary>
        /// Defines what should be done about the first rows.
        /// </summary>
        public enum BeginOptions : byte
        {
            /// <summary>
            /// Fill first rows with NaN values.
            /// </summary>
            NaNValues = 0,

            /// <summary>
            /// Copy the first value of the series.
            /// </summary>
            FirstValue = 1
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing the moving average", ShortName = "wnd", SortOrder = 3)]
            public int WindowSize = 2;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Lag between current observation and last observation from the sliding window", ShortName = "l", SortOrder = 4)]
            public int Lag = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Define how to populate the first rows of the produced series", SortOrder = 5)]
            public BeginOptions Begin = BeginOptions.NaNValues;
        }

        private readonly int _lag;
        private BeginOptions _begin;
        private TInput _nanValue;

        protected SlidingWindowTransformBase(Arguments args, string loaderSignature, IHostEnvironment env, IDataView input)
            : base(args.WindowSize + args.Lag - 1, args.WindowSize + args.Lag - 1, args.Source, args.Name, loaderSignature, env, input)
        {
            Host.CheckUserArg(args.WindowSize >= 1, nameof(args.WindowSize), "Must be at least 1.");
            Host.CheckUserArg(args.Lag >= 0, nameof(args.Lag), "Must be positive.");
            if (args.Lag == 0 && args.WindowSize <= 1)
            {
                Host.Assert(args.WindowSize == 1);
                throw Host.ExceptUserArg(nameof(args.Lag),
                    $"If {args.Lag}=0 and {args.WindowSize}=1, the transform just copies the column. Use {CopyColumnsTransform.LoaderSignature} transform instead.");
            }
            Host.CheckUserArg(Enum.IsDefined(typeof(BeginOptions), args.Begin), nameof(args.Begin), "Undefined value.");
            _lag = args.Lag;
            _begin = args.Begin;
            _nanValue = GetNaValue();
        }

        protected SlidingWindowTransformBase(IHostEnvironment env, ModelLoadContext ctx, string loaderSignature,  IDataView input)
            : base(env, ctx, loaderSignature,  input)
        {
            // *** Binary format ***
            // <base>
            // Int32 lag
            // byte begin

            Host.CheckDecode(WindowSize >= 1);
            _lag = ctx.Reader.ReadInt32();
            Host.CheckDecode(_lag >= 0);
            byte r = ctx.Reader.ReadByte();
            Host.CheckDecode(Enum.IsDefined(typeof(BeginOptions), r));
            _begin = (BeginOptions)r;
            _nanValue = GetNaValue();
        }

        private TInput GetNaValue()
        {
            var sch = Schema;
            int index;
            sch.TryGetColumnIndex(InputColumnName, out index);
            ColumnType col = sch.GetColumnType(index);
            TInput nanValue = Conversions.Instance.GetNAOrDefault<TInput>(col);

            // We store the nan_value here to avoid getting it each time a state is instanciated.
            return nanValue;
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(WindowSize >= 1);
            Host.Assert(_lag >= 0);
            Host.Assert(Enum.IsDefined(typeof(BeginOptions), _begin));
            ctx.CheckAtModel();

            // *** Binary format ***
            // <base>
            // Int32 lag
            // byte begin

            base.Save(ctx);
            ctx.Writer.Write(_lag);
            ctx.Writer.Write((byte)_begin);
        }

        public sealed class StateSlide : StateBase
        {
            private SlidingWindowTransformBase<TInput> _parentSliding;

            protected override void SetNaOutput(ref VBuffer<TInput> output)
            {

                int size = _parentSliding.WindowSize - _parentSliding._lag + 1;
                var result = output.Values;
                if (Utils.Size(result) < size)
                    result = new TInput[size];

                TInput value = _parentSliding._nanValue;
                switch (_parentSliding._begin)
                {
                case BeginOptions.NaNValues:
                    value = _parentSliding._nanValue;
                    break;
                case BeginOptions.FirstValue:
                    // REVIEW: will complete the implementation
                    // if the design looks good
                    throw new NotImplementedException();
                }

                for (int i = 0; i < size; ++i)
                    result[i] = value;
                output = new VBuffer<TInput>(size, result, output.Indices);
            }

            protected override void TransformCore(ref TInput input, FixedSizeQueue<TInput> windowedBuffer, long iteration, ref VBuffer<TInput> output)
            {
                int size = _parentSliding.WindowSize - _parentSliding._lag + 1;
                var result = output.Values;
                if (Utils.Size(result) < size)
                    result = new TInput[size];

                if (_parentSliding._lag == 0)
                {
                    for (int i = 0; i < _parentSliding.WindowSize; ++i)
                        result[i] = windowedBuffer[i];
                    result[_parentSliding.WindowSize] = input;
                }
                else
                {
                    for (int i = 0; i < size; ++i)
                        result[i] = windowedBuffer[i];
                }
                output = new VBuffer<TInput>(size, result, output.Indices);
            }

            protected override void InitializeStateCore()
            {
                _parentSliding = (SlidingWindowTransformBase<TInput>)base.ParentTransform;
            }

            protected override void LearnStateFromDataCore(FixedSizeQueue<TInput> data)
            {
                // This method is empty because there is no need for parameter learning from the initial windowed buffer for this transform.
            }
        }
    }
}
