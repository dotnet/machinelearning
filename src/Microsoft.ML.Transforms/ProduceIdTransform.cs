// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ProduceIdTransform.Summary, typeof(ProduceIdTransform), typeof(ProduceIdTransform.Arguments), typeof(SignatureDataTransform),
    "", "ProduceIdTransform", "ProduceId")]

[assembly: LoadableClass(ProduceIdTransform.Summary, typeof(ProduceIdTransform), null, typeof(SignatureLoadDataTransform),
    "Produce ID Transform", ProduceIdTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Produces a column with the cursor's ID as a column. This can be useful for diagnostic purposes.
    ///
    /// This class will obviously generate different data given different IDs. So, if you save data to
    /// some other file, then apply this transform to that dataview, it may of course have a different
    /// result. This is distinct from most transforms that produce results based on data alone.
    /// </summary>
    internal sealed class ProduceIdTransform : RowToRowTransformBase
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the column to produce", ShortName = "col", SortOrder = 1)]
            public string Column = "Id";
        }

        private sealed class Bindings : ColumnBindingsBase
        {
            public Bindings(DataViewSchema input, bool user, string name)
                : base(input, user, name)
            {
                Contracts.Assert(InfoCount == 1);
            }

            protected override DataViewType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(iinfo == 0);
                return RowIdDataViewType.Instance;
            }

            public static Bindings Create(ModelLoadContext ctx, DataViewSchema input)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(input);

                // *** Binary format ***
                // int: id of output column name
                string name = ctx.LoadNonEmptyString();
                return new Bindings(input, true, name);
            }

            internal void Save(ModelSaveContext ctx)
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
                Contracts.Assert(active.Length == Input.Count);
                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        internal const string Summary = "Produces a new column with the row ID.";
        internal const string LoaderSignature = "ProduceIdTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PR ID XF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ProduceIdTransform).Assembly.FullName);
        }

        private readonly Bindings _bindings;

        public override DataViewSchema OutputSchema => _bindings.AsSchema;

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

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bindings
            _bindings.Save(ctx);
        }

        protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            Host.AssertValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var inputPred = _bindings.GetDependencies(predicate);
            var inputCols = Source.Schema.Where(x => inputPred(x.Index));
            var input = Source.GetRowCursor(inputCols, rand);
            bool active = predicate(_bindings.MapIinfoToCol(0));

            return new Cursor(Host, _bindings, input, active);
        }

        public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            Host.CheckValueOrNull(rand);
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
            var inputPred = _bindings.GetDependencies(predicate);

            var inputCols = Source.Schema.Where(x => inputPred(x.Index));
            DataViewRowCursor[] cursors = Source.GetRowCursorSet(inputCols, n, rand);
            bool active = predicate(_bindings.MapIinfoToCol(0));
            for (int c = 0; c < cursors.Length; ++c)
                cursors[c] = new Cursor(Host, _bindings, cursors[c], active);
            return cursors;
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");
            return null;
        }

        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly Bindings _bindings;
            private readonly bool _active;

            public override DataViewSchema Schema => _bindings.AsSchema;

            public Cursor(IChannelProvider provider, Bindings bindings, DataViewRowCursor input, bool active)
                : base(provider, input)
            {
                Ch.CheckValue(bindings, nameof(bindings));
                _bindings = bindings;
                _active = active;
            }

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.CheckParam(column.Index < _bindings.ColumnCount, nameof(column));
                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return Input.IsColumnActive(Input.Schema[index]);
                Ch.Assert(index == 0);
                return _active;
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Ch.CheckParam(column.Index < _bindings.ColumnCount, nameof(column));
                Ch.CheckParam(IsColumnActive(column), nameof(column.Index));
                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, column.Index);
                if (isSrc)
                    return Input.GetGetter<TValue>(Input.Schema[index]);
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
