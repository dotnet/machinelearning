// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(ProduceIdTransform.Summary, typeof(ProduceIdTransform), typeof(ProduceIdTransform.Arguments), typeof(SignatureDataTransform),
    "", "ProduceIdTransform", "ProduceId")]

[assembly: LoadableClass(ProduceIdTransform.Summary, typeof(ProduceIdTransform), null, typeof(SignatureLoadDataTransform),
    "Produce ID Transform", ProduceIdTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Produces a column with the cursor's ID as a column. This can be useful for diagnostic purposes.
    ///
    /// This class will obviously generate different data given different IDs. So, if you save data to
    /// some other file, then apply this transform to that dataview, it may of course have a different
    /// result. This is distinct from most transforms that produce results based on data alone.
    /// </summary>
    public sealed class ProduceIdTransform : RowToRowTransformBase
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the column to produce", ShortName = "col", SortOrder = 1)]
            public string Column = "Id";
        }

        private sealed class Bindings : ColumnBindingsBase
        {
            public Bindings(ISchema input, bool user, string name)
                : base(input, user, name)
            {
                Contracts.Assert(InfoCount == 1);
            }

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(iinfo == 0);
                return NumberType.UG;
            }

            public static Bindings Create(ModelLoadContext ctx, ISchema input)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(input);

                // *** Binary format ***
                // int: id of output column name
                string name = ctx.LoadNonEmptyString();
                return new Bindings(input, true, name);
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: id of output column name
                ctx.SaveNonEmptyString(GetColumnNameCore(0));
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = GetActiveInput(predicate);
                Contracts.Assert(active.Length == Input.ColumnCount);
                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        internal const string Summary = "Produces a new column with the row ID.";
        public const string LoaderSignature = "ProduceIdTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PR ID XF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private readonly Bindings _bindings;

        public override ISchema Schema { get { return _bindings; } }

        public override bool CanShuffle { get { return Source.CanShuffle; } }

        public ProduceIdTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, LoaderSignature, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckNonWhiteSpace(args.Column, nameof(args.Column));

            _bindings = new Bindings(input.Schema, true, args.Column);
        }

        private ProduceIdTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // bindings
            _bindings = Bindings.Create(ctx, Source.Schema);
        }

        public static ProduceIdTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ProduceIdTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bindings
            _bindings.Save(ctx);
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var input = Source.GetRowCursor(inputPred, rand);
            bool active = predicate(_bindings.MapIinfoToCol(0));

            return new RowCursor(Host, _bindings, input, active);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            IRowCursor[] cursors = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
            bool active = predicate(_bindings.MapIinfoToCol(0));
            for (int c = 0; c < cursors.Length; ++c)
                cursors[c] = new RowCursor(Host, _bindings, cursors[c], active);
            return cursors;
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");
            return null;
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly bool _active;

            public ISchema Schema { get { return _bindings; } }

            public RowCursor(IChannelProvider provider, Bindings bindings, IRowCursor input, bool active)
                : base(provider, input)
            {
                Ch.CheckValue(bindings, nameof(bindings));
                _bindings = bindings;
                _active = active;
            }

            public bool IsColumnActive(int col)
            {
                Ch.CheckParam(0 <= col && col < _bindings.ColumnCount, nameof(col));
                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return Input.IsColumnActive(index);
                Ch.Assert(index == 0);
                return _active;
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.CheckParam(0 <= col && col < _bindings.ColumnCount, nameof(col));
                Ch.CheckParam(IsColumnActive(col), nameof(col));
                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return Input.GetGetter<TValue>(index);
                Ch.Assert(index == 0);
                Delegate idGetter = Input.GetIdGetter();
                Ch.AssertValue(idGetter);
                var fn = idGetter as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }
        }
    }
}
