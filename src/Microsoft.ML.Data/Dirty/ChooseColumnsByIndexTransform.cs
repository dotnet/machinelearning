// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using System;
using System.Linq;

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

        private sealed class Bindings
        {
            /// <summary>
            /// A collection of source column indexes after removing those we want to drop. Specifically, j=_sources[i] means
            /// that the i-th output column in the output schema is the j-th column in the input schema.
            /// </summary>
            private readonly int[] _sources;

            /// <summary>
            /// Input schema of this transform. It's useful when determining column dependencies and other
            /// relations between input and output schemas.
            /// </summary>
            private readonly Schema _sourceSchema;

            /// <summary>
            /// Some column indexes in the input schema. <see cref="_sources"/> is computed from <see cref="_selectedColumnIndexes"/>
            /// and <see cref="_drop"/>.
            /// </summary>
            private readonly int[] _selectedColumnIndexes;

            /// <summary>
            /// True, if this transform drops selected columns indexed by <see cref="_selectedColumnIndexes"/>.
            /// </summary>
            private readonly bool _drop;

            // This transform's output schema.
            internal Schema OutputSchema { get; }

            internal Bindings(Arguments args, Schema sourceSchema)
            {
                Contracts.AssertValue(args);
                Contracts.AssertValue(sourceSchema);

                _sourceSchema = sourceSchema;

                // Store user-specified arguments as the major state of this transform. Only the major states will
                // be saved and all other attributes can be reconstructed from them.
                _drop = args.Drop;
                _selectedColumnIndexes = args.Index;

                // Compute actually used attributes in runtime from those major states.
                ComputeSources(_drop, _selectedColumnIndexes, _sourceSchema, out _sources);

                // All necessary fields in this class are set, so we can compute output schema now.
                OutputSchema = ComputeOutputSchema();
            }

            /// <summary>
            /// Common method of computing <see cref="_sources"/> from necessary parameters. This function is used in constructors.
            /// </summary>
            private static void ComputeSources(bool drop, int[] selectedColumnIndexes, Schema sourceSchema, out int[] sources)
            {
                // Compute the mapping, <see cref="_sources"/>, from output column index to input column index.
                if (drop)
                    // Drop columns indexed by args.Index
                    sources = Enumerable.Range(0, sourceSchema.Count).Except(selectedColumnIndexes).ToArray();
                else
                    // Keep columns indexed by args.Index
                    sources = selectedColumnIndexes;

                // Make sure the output of this transform is meaningful.
                Contracts.Check(sources.Length > 0, "Choose columns by index has no output column.");
            }

            /// <summary>
            /// After <see cref="_sourceSchema"/> and <see cref="_sources"/> are set, pick up selected columns from <see cref="_sourceSchema"/> to create <see cref="OutputSchema"/>
            /// Note that <see cref="_sources"/> tells us what columns in <see cref="_sourceSchema"/> are put into <see cref="OutputSchema"/>.
            /// </summary>
            private Schema ComputeOutputSchema()
            {
                var schemaBuilder = new SchemaBuilder();
                for (int i = 0; i < _sources.Length; ++i)
                {
                    // selectedIndex is an column index of input schema. Note that the input column indexed by _sources[i] in _sourceSchema is sent
                    // to the i-th column in the output schema.
                    var selectedIndex = _sources[i];

                    // The dropped/kept columns are determined by user-specified arguments, so we throw if a bad configuration is provided.
                    string fmt = string.Format("Column index {0} invalid for input with {1} columns", selectedIndex, _sourceSchema.Count);
                    Contracts.Check(selectedIndex < _sourceSchema.Count, fmt);

                    // Copy the selected column into output schema.
                    var selectedColumn = _sourceSchema[selectedIndex];
                    schemaBuilder.AddColumn(selectedColumn.Name, selectedColumn.Type, selectedColumn.Metadata);
                }
                return schemaBuilder.GetSchema();
            }

            internal Bindings(ModelLoadContext ctx, Schema sourceSchema)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(sourceSchema);

                _sourceSchema = sourceSchema;

                // *** Binary format ***
                // bool (as byte): operation mode
                // int[]: selected source column indices
                _drop = ctx.Reader.ReadBoolByte();
                _selectedColumnIndexes = ctx.Reader.ReadIntArray();

                // Compute actually used attributes in runtime from those major states.
                ComputeSources(_drop, _selectedColumnIndexes, _sourceSchema, out _sources);

                _sourceSchema = sourceSchema;
                OutputSchema = ComputeOutputSchema();
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // bool (as byte): operation mode
                // int[]: selected source column indices
                ctx.Writer.WriteBoolByte(_drop);
                ctx.Writer.WriteIntArray(_selectedColumnIndexes);
            }

            internal bool[] GetActive(Func<int, bool> predicate)
            {
                return Utils.BuildArray(OutputSchema.Count, predicate);
            }

            internal Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);
                var active = new bool[_sourceSchema.Count];
                for (int i = 0; i < _sources.Length; i++)
                {
                    if (predicate(i))
                        active[_sources[i]] = true;
                }
                return col => 0 <= col && col < active.Length && active[col];
            }

            /// <summary>
            /// Given the column index in the output schema, this function returns its source column's index in the input schema.
            /// </summary>
            internal int GetSourceColumnIndex(int outputColumnIndex) => _sources[outputColumnIndex];
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

        public override Schema OutputSchema => _bindings.OutputSchema;

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);
            // Parallel doesn't matter to this transform.
            return null;
        }

        protected override RowCursor GetRowCursorCore(Func<int, bool> predicate, Random rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred, rand);
            return new Cursor(Host, _bindings, input, active);
        }

        public sealed override RowCursor[] GetRowCursorSet(Func<int, bool> predicate, int n, Random rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var inputs = Source.GetRowCursorSet(inputPred, n, rand);
            Host.AssertNonEmpty(inputs);

            // No need to split if this is given 1 input cursor.
            var cursors = new RowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new Cursor(Host, _bindings, inputs[i], active);
            return cursors;
        }

        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly Bindings _bindings;
            private readonly bool[] _active;

            public Cursor(IChannelProvider provider, Bindings bindings, RowCursor input, bool[] active)
                : base(provider, input)
            {
                Ch.AssertValue(bindings);
                Ch.Assert(active == null || active.Length == bindings.OutputSchema.Count);

                _bindings = bindings;
                _active = active;
            }

            public override Schema Schema => _bindings.OutputSchema;

            public override bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.OutputSchema.Count);
                return _active == null || _active[col];
            }

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                var src = _bindings.GetSourceColumnIndex(col);
                return Input.GetGetter<TValue>(src);
            }
        }
    }
}
