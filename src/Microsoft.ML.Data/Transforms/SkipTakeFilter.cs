// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(SkipTakeFilter.SkipTakeFilterSummary, typeof(SkipTakeFilter), typeof(SkipTakeFilter.Options), typeof(SignatureDataTransform),
    SkipTakeFilter.SkipTakeFilterUserName, "SkipTakeFilter", SkipTakeFilter.SkipTakeFilterShortName)]

[assembly: LoadableClass(SkipTakeFilter.SkipFilterSummary, typeof(SkipTakeFilter), typeof(SkipTakeFilter.SkipOptions), typeof(SignatureDataTransform),
    SkipTakeFilter.SkipFilterUserName, "SkipFilter", SkipTakeFilter.SkipFilterShortName)]

[assembly: LoadableClass(SkipTakeFilter.TakeFilterSummary, typeof(SkipTakeFilter), typeof(SkipTakeFilter.TakeOptions), typeof(SignatureDataTransform),
    SkipTakeFilter.TakeFilterUserName, "TakeFilter", SkipTakeFilter.TakeFilterShortName)]

[assembly: LoadableClass(SkipTakeFilter.SkipTakeFilterSummary, typeof(SkipTakeFilter), null, typeof(SignatureLoadDataTransform),
    "Skip and Take Filter", SkipTakeFilter.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Allows limiting input to a subset of row at an optional offset.  Can be used to implement data paging.
    /// </summary>
    [BestFriend]
    internal sealed class SkipTakeFilter : FilterBase, ITransformTemplate
    {
        public const string LoaderSignature = "SkipTakeFilter";
        private const string ModelSignature = "SKIPTKFL";
        private const string RegistrationName = "SkipTakeFilter";

        public const string SkipTakeFilterSummary = "Allows limiting input to a subset of rows at an optional offset.  Can be used to implement data paging.";
        public const string TakeFilterSummary = "Allows limiting input to a subset of rows by taking N first rows.";
        public const string SkipFilterSummary = "Allows limiting input to a subset of rows by skipping a number of rows.";
        public const string SkipTakeFilterUserName = "Skip and Take Filter";
        public const string SkipTakeFilterShortName = "SkipTake";
        public const string SkipFilterUserName = "Skip Filter";
        public const string SkipFilterShortName = "Skip";
        public const string TakeFilterUserName = "Take Filter";
        public const string TakeFilterShortName = "Take";

        public sealed class Options : TransformInputBase
        {
            internal const string SkipHelp = "Number of items to skip";
            internal const string TakeHelp = "Number of items to take";
            internal const long DefaultSkip = 0;
            internal const long DefaultTake = long.MaxValue;

            [Argument(ArgumentType.AtMostOnce, HelpText = SkipHelp, ShortName = "s", SortOrder = 1)]
            public long? Skip;

            [Argument(ArgumentType.AtMostOnce, HelpText = TakeHelp, ShortName = "t", SortOrder = 2)]
            public long? Take;
        }

        public sealed class TakeOptions : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = Options.TakeHelp, ShortName = "c,n,t", SortOrder = 1)]
            public long Count = Options.DefaultTake;
        }

        public sealed class SkipOptions : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = Options.SkipHelp, ShortName = "c,n,s", SortOrder = 1)]
            public long Count = Options.DefaultSkip;
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: ModelSignature,
                verWrittenCur: 0x00010001,          // initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SkipTakeFilter).Assembly.FullName);
        }

        private readonly long _skip;
        private readonly long _take;

        private SkipTakeFilter(long skip, long take, IHostEnvironment env, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.Assert(skip >= 0);
            Host.Assert(take >= 0);

            _skip = skip;
            _take = take;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="SkipTakeFilter"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="options">Options for the skip operation.</param>
        /// <param name="input">Input <see cref="IDataView"/>.</param>
        public SkipTakeFilter(IHostEnvironment env, SkipOptions options, IDataView input)
            : this(options.Count, Options.DefaultTake, env, input)
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="SkipTakeFilter"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="options">Options for the take operation.</param>
        /// <param name="input">Input <see cref="IDataView"/>.</param>
        public SkipTakeFilter(IHostEnvironment env, TakeOptions options, IDataView input)
            : this(Options.DefaultSkip, options.Count, env, input)
        {
        }

        IDataTransform ITransformTemplate.ApplyToData(IHostEnvironment env, IDataView newSource)
            => new SkipTakeFilter(_skip, _take, env, newSource);

        public static SkipTakeFilter Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            long skip = options.Skip ?? Options.DefaultSkip;
            long take = options.Take ?? Options.DefaultTake;
            env.CheckUserArg(skip >= 0, nameof(options.Skip), "should be non-negative");
            env.CheckUserArg(take >= 0, nameof(options.Take), "should be non-negative");
            return new SkipTakeFilter(skip, take, env, input);
        }

        public static SkipTakeFilter Create(IHostEnvironment env, SkipOptions options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckUserArg(options.Count >= 0, nameof(options.Count), "should be non-negative");
            return new SkipTakeFilter(options.Count, Options.DefaultTake, env, input);
        }

        public static SkipTakeFilter Create(IHostEnvironment env, TakeOptions options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckUserArg(options.Count >= 0, nameof(options.Count), "should be non-negative");
            return new SkipTakeFilter(Options.DefaultSkip, options.Count, env, input);
        }

        /// <summary>Creates instance of class from context.</summary>
        public static SkipTakeFilter Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // long: skip
            // long: take
            long skip = ctx.Reader.ReadInt64();
            h.CheckDecode(skip >= 0);
            long take = ctx.Reader.ReadInt64();
            h.CheckDecode(take >= 0);
            return h.Apply("Loading Model", ch => new SkipTakeFilter(skip, take, h, input));
        }

        ///<summary>Saves class data to context</summary>
        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // long: skip
            // long: take
            Host.Assert(_skip >= 0);
            ctx.Writer.Write(_skip);
            Host.Assert(_take >= 0);
            ctx.Writer.Write(_take);
        }

        /// <summary>
        /// This filter can not shuffle
        /// </summary>
        public override bool CanShuffle { get { return false; } }

        /// <summary>
        /// Returns the computed count of rows remaining after skip and take operation.
        /// Returns null if count is unknown.
        /// </summary>
        public override long? GetRowCount()
        {
            if (_take == 0)
                return 0;
            long? count = Source.GetRowCount();
            if (count == null)
                return null;

            long afterSkip = count.GetValueOrDefault() - _skip;
            return Math.Min(Math.Max(0, afterSkip), _take);
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");
            return false;
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.AssertValueOrNull(rand);

            var input = Source.GetRowCursor(columnsNeeded);
            var activeColumns = Utils.BuildArray(OutputSchema.Count, columnsNeeded);
            return new Cursor(Host, input, OutputSchema, activeColumns, _skip, _take);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);
            return new DataViewRowCursor[] { GetRowCursorCore(columnsNeeded) };
        }

        private sealed class Cursor : LinkedRowRootCursorBase
        {
            private readonly long _skip;
            private readonly long _take;
            private long _rowsTaken;
            private bool _started;

            /// <summary>
            /// SkipTakeFilter does not support cursor sets, so this can always be zero.
            /// </summary>
            public override long Batch => 0;

            public Cursor(IChannelProvider provider, DataViewRowCursor input, DataViewSchema schema, bool[] active, long skip, long take)
                : base(provider, input, schema, active)
            {
                Ch.Assert(skip >= 0);
                Ch.Assert(take >= 0);

                _skip = skip;
                _take = take;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return Input.GetIdGetter();
            }

            protected override bool MoveNextCore()
            {
                // Exit if 1 + _rowsTaken will overflow, or if we already have taken enough rows.
                if (1 > _take - _rowsTaken)
                {
                    _rowsTaken = _take;
                    return false;
                }

                ++_rowsTaken;

                if (!_started)
                {
                    _started = true;

                    // Exit if 1 + _skip will overflow.
                    if (1 > long.MaxValue - _skip)
                    {
                        _rowsTaken = _take;
                        return false;
                    }

                    // Move foward _skip + 1 rows to get to the "first" row of the input.
                    for (long i = 0; i <= _skip; ++i)
                    {
                        if (!Root.MoveNext())
                            return false;
                    }
                    return true;
                }

                return Root.MoveNext();
            }
        }
    }
}