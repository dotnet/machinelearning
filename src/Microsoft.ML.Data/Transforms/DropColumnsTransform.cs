// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using Float = System.Single;

[assembly: LoadableClass(DropColumnsTransform.DropColumnsSummary, typeof(DropColumnsTransform), typeof(DropColumnsTransform.Arguments), typeof(SignatureDataTransform),
    DropColumnsTransform.DropUserName, "DropColumns", "DropColumnsTransform", DropColumnsTransform.DropShortName, DocName = "transform/DropKeepChooseTransforms.md")]

[assembly: LoadableClass(DropColumnsTransform.KeepColumnsSummary, typeof(DropColumnsTransform), typeof(DropColumnsTransform.KeepArguments), typeof(SignatureDataTransform),
    DropColumnsTransform.KeepUserName, "KeepColumns", "KeepColumnsTransform", DropColumnsTransform.KeepShortName, DocName = "transform/DropKeepChooseTransforms.md")]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Transform to drop columns with the given names. Note that if there are names that
    /// are not in the input schema, that is not an error.
    /// </summary>
    public sealed class DropColumnsTransform : RowToRowMapperTransformBase
    {
        public abstract class ArgumentsBase : TransformInputBase
        {
            internal abstract string[] Columns { get; }
        }

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Column name to drop", ShortName = "col", SortOrder = 1)]
            public string[] Column;

            internal override string[] Columns => Column;
        }

        public sealed class KeepArguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Column name to keep", ShortName = "col", SortOrder = 1)]
            public string[] Column;

            internal override string[] Columns => Column;
        }

        private sealed class Bindings : ISchema
        {
            public readonly ISchema Input;

            // Whether to keep (vs drop) the named columns.
            public readonly bool Keep;
            // The column names to drop/keep.
            public readonly HashSet<string> Names;
            // Map from our column indices to source column indices.
            public readonly int[] ColMap;
            // Map from names to our column indices.
            public readonly Dictionary<string, int> NameToCol;

            public Schema AsSchema { get; }

            public Bindings(ArgumentsBase args, bool keep, ISchema schemaInput)
            {
                Contracts.AssertValue(args);
                Contracts.AssertNonEmpty(args.Columns);
                Contracts.AssertValue(schemaInput);

                Keep = keep;
                Input = schemaInput;

                Names = new HashSet<string>();
                for (int i = 0; i < args.Columns.Length; i++)
                {
                    var name = args.Columns[i];
                    Contracts.CheckNonWhiteSpace(name, nameof(args.Columns));

                    // REVIEW: Should this just be a warning?
                    if (!Names.Add(name))
                        throw Contracts.ExceptUserArg(nameof(args.Columns), "Column '{0}' specified multiple times", name);
                }

                BuildMap(out ColMap, out NameToCol);

                AsSchema = Schema.Create(this);
            }

            private void BuildMap(out int[] map, out Dictionary<string, int> nameToCol)
            {
                var srcs = new List<int>();
                nameToCol = new Dictionary<string, int>();
                for (int src = 0; src < Input.ColumnCount; src++)
                {
                    string name = Input.GetColumnName(src);
                    if (Names.Contains(name) == !Keep)
                        continue;

                    // If the Input schema maps name to this src column, record it in our name table.
                    int tmp;
                    if (Input.TryGetColumnIndex(name, out tmp) && tmp == src)
                        nameToCol.Add(name, srcs.Count);
                    // Record the source column index.
                    srcs.Add(src);
                }
                map = srcs.ToArray();
            }

            public Bindings(ModelLoadContext ctx, ISchema schemaInput)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(schemaInput);

                Input = schemaInput;

                // *** Binary format ***
                // bool: whether to keep (vs drop) the named columns
                // int: number of names
                // int[]: the ids of the names
                Keep = ctx.Reader.ReadBoolByte();
                int count = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(count > 0);

                Names = new HashSet<string>();
                for (int i = 0; i < count; i++)
                {
                    string name = ctx.LoadNonEmptyString();
                    Contracts.CheckDecode(Names.Add(name));
                }

                BuildMap(out ColMap, out NameToCol);
                AsSchema = Schema.Create(this);
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // bool: whether to keep (vs drop) the named columns
                // int: number of names
                // int[]: the ids of the names
                ctx.Writer.WriteBoolByte(Keep);
                ctx.Writer.Write(Names.Count);
                foreach (var name in Names)
                    ctx.SaveNonEmptyString(name);
            }

            public int ColumnCount
            {
                get { return ColMap.Length; }
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                Contracts.CheckValueOrNull(name);

                if (name == null)
                {
                    col = default(int);
                    return false;
                }
                return NameToCol.TryGetValue(name, out col);
            }

            public string GetColumnName(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return Input.GetColumnName(ColMap[col]);
            }

            public ColumnType GetColumnType(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return Input.GetColumnType(ColMap[col]);
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return Input.GetMetadataTypes(ColMap[col]);
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return Input.GetMetadataTypeOrNull(kind, ColMap[col]);
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                Input.GetMetadata(kind, ColMap[col], ref value);
            }

            internal bool[] GetActive(Func<int, bool> predicate)
            {
                return Utils.BuildArray(ColumnCount, predicate);
            }

            internal Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);
                var active = new bool[Input.ColumnCount];
                for (int i = 0; i < ColMap.Length; i++)
                {
                    if (predicate(i))
                        active[ColMap[i]] = true;
                }
                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        public const string DropColumnsSummary = "Removes a column or columns from the dataset.";
        public const string KeepColumnsSummary = "Selects which columns from the dataset to keep.";
        public const string DropUserName = "Drop Columns Transform";
        public const string KeepUserName = "Keep Columns Transform";
        public const string DropShortName = "Drop";
        public const string KeepShortName = "Keep";

        public const string LoaderSignature = "DropColumnsTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DRPCOLST",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Added KeepColumns
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DropColumnsTransform).Assembly.FullName);
        }

        private readonly Bindings _bindings;

        private const string DropRegistrationName = "DropColumns";
        private const string KeepRegistrationName = "KeepColumns";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="columnsToDrop">Name of the columns to be dropped.</param>
        public DropColumnsTransform(IHostEnvironment env, IDataView input, params string[] columnsToDrop)
         :this(env, new Arguments() { Column = columnsToDrop }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public DropColumnsTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, DropRegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckNonEmpty(args.Column, nameof(args.Column));

            _bindings = new Bindings(args, false, Source.Schema);
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public DropColumnsTransform(IHostEnvironment env, KeepArguments args, IDataView input)
            : base(env, KeepRegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckNonEmpty(args.Column, nameof(args.Column));

            _bindings = new Bindings(args, true, Source.Schema);
        }

        private DropColumnsTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            int cbFloat = ctx.Reader.ReadInt32();
            Host.CheckDecode(cbFloat == sizeof(Float));
            _bindings = new Bindings(ctx, Source.Schema);
        }

        public static DropColumnsTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(DropRegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new DropColumnsTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            ctx.Writer.Write(sizeof(Float));
            _bindings.Save(ctx);
        }

        public override Schema Schema => _bindings.AsSchema;

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);
            // Parallel doesn't matter to this transform.
            return null;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred, rand);
            return new RowCursor(Host, _bindings, input, active);
        }

        public sealed override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
            Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
            Host.AssertNonEmpty(inputs);

            // No need to split if this is given 1 input cursor.
            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new RowCursor(Host, _bindings, inputs[i], active);
            return cursors;
        }

        protected override Func<int, bool> GetDependenciesCore(Func<int, bool> predicate)
        {
            return _bindings.GetDependencies(predicate);
        }

        protected override Delegate[] CreateGetters(IRow input, Func<int, bool> active, out Action disp)
        {
            disp = null;
            return new Delegate[0];
        }

        protected override int MapColumnIndex(out bool isSrc, int col)
        {
            isSrc = true;
            return _bindings.ColMap[col];
        }

        // REVIEW: Refactor so ChooseColumns can share the same cursor class.
        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;

            public RowCursor(IChannelProvider provider, Bindings bindings, IRowCursor input, bool[] active)
                : base(provider, input)
            {
                Ch.AssertValue(bindings);
                Ch.Assert(active == null || active.Length == bindings.ColumnCount);

                _bindings = bindings;
                _active = active;
            }

            public Schema Schema => _bindings.AsSchema;

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active == null || _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));
                return Input.GetGetter<TValue>(_bindings.ColMap[col]);
            }
        }
    }

    public class KeepColumnsTransform
    {
        /// <summary>
        /// A helper method to create <see cref="KeepColumnsTransform"/> for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="columnsToKeep">Name of the columns to be kept. All other columns will be removed.</param>
        /// <returns></returns>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, params string[] columnsToKeep)
            => new DropColumnsTransform(env, new DropColumnsTransform.KeepArguments() { Column = columnsToKeep }, input);
    }
}
