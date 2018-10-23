// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(ChooseColumnsByIndexTransform), typeof(ChooseColumnsByIndexTransform.Arguments), typeof(SignatureDataTransform),
    "", "ChooseColumnsByIndexTransform", "ChooseColumnsByIndex")]

[assembly: LoadableClass(typeof(ChooseColumnsByIndexTransform), null, typeof(SignatureLoadDataTransform),
    "", ChooseColumnsByIndexTransform.LoaderSignature, ChooseColumnsByIndexTransform.LoaderSignatureOld)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class ChooseColumnsByIndexTransform : RowToRowTransformBase
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Column index to select", ShortName = "ind")]
            public int[] Index;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "If true, selected columns are dropped instead of kept, with the order of kept columns being the same as the original", ShortName = "d")]
            public bool Drop;
        }

        private sealed class Bindings : ISchema
        {
            public readonly int[] Sources;

            private readonly ISchema _input;
            private readonly Dictionary<string, int> _nameToIndex;

            // The following argument is used only to inform serialization.
            private readonly int[] _dropped;

            public Schema AsSchema { get; }

            public Bindings(Arguments args, ISchema schemaInput)
            {
                Contracts.AssertValue(args);
                Contracts.AssertValue(schemaInput);

                _input = schemaInput;

                int[] indexCopy = args.Index == null ? new int[0] : args.Index.ToArray();
                BuildNameDict(indexCopy, args.Drop, out Sources, out _dropped, out _nameToIndex, user: true);

                AsSchema = Data.Schema.Create(this);
            }

            private void BuildNameDict(int[] indexCopy, bool drop, out int[] sources, out int[] dropped, out Dictionary<string, int> nameToCol, bool user)
            {
                Contracts.AssertValue(indexCopy);
                foreach (int col in indexCopy)
                {
                    if (col < 0 || _input.ColumnCount <= col)
                    {
                        const string fmt = "Column index {0} invalid for input with {1} columns";
                        if (user)
                            throw Contracts.ExceptUserArg(nameof(Arguments.Index), fmt, col, _input.ColumnCount);
                        else
                            throw Contracts.ExceptDecode(fmt, col, _input.ColumnCount);
                    }
                }
                if (drop)
                {
                    sources = Enumerable.Range(0, _input.ColumnCount).Except(indexCopy).ToArray();
                    dropped = indexCopy;
                }
                else
                {
                    sources = indexCopy;
                    dropped = null;
                }
                if (user)
                    Contracts.CheckUserArg(sources.Length > 0, nameof(Arguments.Index), "Choose columns by index has no output columns");
                else
                    Contracts.CheckDecode(sources.Length > 0, "Choose columns by index has no output columns");
                nameToCol = new Dictionary<string, int>();
                for (int c = 0; c < sources.Length; ++c)
                    nameToCol[_input.GetColumnName(sources[c])] = c;
            }

            public Bindings(ModelLoadContext ctx, ISchema schemaInput)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(schemaInput);

                _input = schemaInput;

                // *** Binary format ***
                // bool(as byte): whether the indicated source columns are columns to keep, or drop
                // int: number of source column indices
                // int[]: source column indices

                bool isDrop = ctx.Reader.ReadBoolByte();
                BuildNameDict(ctx.Reader.ReadIntArray() ?? new int[0], isDrop, out Sources, out _dropped, out _nameToIndex, user: false);
                AsSchema = Data.Schema.Create(this);
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // bool(as byte): whether the indicated columns are columns to keep, or drop
                // int: number of source column indices
                // int[]: source column indices

                ctx.Writer.WriteBoolByte(_dropped != null);
                ctx.Writer.WriteIntArray(_dropped ?? Sources);
            }

            public int ColumnCount
            {
                get { return Sources.Length; }
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                Contracts.CheckValueOrNull(name);
                if (name == null)
                {
                    col = default(int);
                    return false;
                }
                return _nameToIndex.TryGetValue(name, out col);
            }

            public string GetColumnName(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return _input.GetColumnName(Sources[col]);
            }

            public ColumnType GetColumnType(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return _input.GetColumnType(Sources[col]);
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return _input.GetMetadataTypes(Sources[col]);
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                return _input.GetMetadataTypeOrNull(kind, Sources[col]);
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
                _input.GetMetadata(kind, Sources[col], ref value);
            }

            internal bool[] GetActive(Func<int, bool> predicate)
            {
                return Utils.BuildArray(ColumnCount, predicate);
            }

            internal Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);
                var active = new bool[_input.ColumnCount];
                for (int i = 0; i < Sources.Length; i++)
                {
                    if (predicate(i))
                        active[Sources[i]] = true;
                }
                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        public const string LoaderSignature = "ChooseColumnsIdxTrans";
        internal const string LoaderSignatureOld = "ChooseColumnsIdxFunc";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CHSCOLIF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld,
                loaderAssemblyName: typeof(ChooseColumnsByIndexTransform).Assembly.FullName);
        }

        private readonly Bindings _bindings;

        private const string RegistrationName = "ChooseColumnsByIndex";

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public ChooseColumnsByIndexTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));

            _bindings = new Bindings(args, Source.Schema);
        }

        private ChooseColumnsByIndexTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // bindings
            _bindings = new Bindings(ctx, Source.Schema);
        }

        public static ChooseColumnsByIndexTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new ChooseColumnsByIndexTransform(h, ctx, input));
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

                var src = _bindings.Sources[col];
                return Input.GetGetter<TValue>(src);
            }
        }
    }
}
