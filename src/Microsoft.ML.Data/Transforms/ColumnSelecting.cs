﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ColumnSelectingTransformer.Summary, typeof(IDataTransform), typeof(ColumnSelectingTransformer),
                typeof(ColumnSelectingTransformer.Arguments), typeof(SignatureDataTransform),
                ColumnSelectingTransformer.UserName, "SelectColumns", "SelectColumnsTransform", ColumnSelectingTransformer.ShortName, DocName = "transform/SelectTransforms.md")]

[assembly: LoadableClass(ColumnSelectingTransformer.Summary, typeof(IDataView), typeof(ColumnSelectingTransformer), null, typeof(SignatureLoadDataTransform),
                            ColumnSelectingTransformer.UserName, ColumnSelectingTransformer.LoaderSignature)]

[assembly: LoadableClass(ColumnSelectingTransformer.Summary, typeof(ColumnSelectingTransformer), null, typeof(SignatureLoadModel),
                            ColumnSelectingTransformer.UserName, ColumnSelectingTransformer.LoaderSignature)]

// Back-compat to handle loading of the Drop and Keep Transformer
[assembly: LoadableClass("", typeof(IDataView), typeof(ColumnSelectingTransformer), null, typeof(SignatureLoadDataTransform),
    "", ColumnSelectingTransformer.DropLoaderSignature)]

// Back-compat to handle loading of the Choose Columns Transformer
[assembly: LoadableClass("", typeof(IDataView), typeof(ColumnSelectingTransformer), null, typeof(SignatureLoadDataTransform),
    "", ColumnSelectingTransformer.ChooseLoaderSignature, ColumnSelectingTransformer.ChooseLoaderSignatureOld)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// The ColumnSelectingEstimator supports selection of specified columns to keep from a given input.
    /// </summary>
    public sealed class ColumnSelectingEstimator : TrivialEstimator<ColumnSelectingTransformer>
    {
        private readonly Func<string, bool> _selectPredicate;

        /// <summary>
        /// Constructs the Select Columns Estimator.
        /// </summary>
        /// <param name="env">Instance of the host environment.</param>
        /// <param name="keepColumns">The array of column names to keep.</param>
        private ColumnSelectingEstimator(IHostEnvironment env, params string[] keepColumns)
            : this(env, keepColumns, null, ColumnSelectingTransformer.Defaults.KeepHidden, ColumnSelectingTransformer.Defaults.IgnoreMissing)
        { }

        /// <summary>
        /// Constructs the Select Columns Estimator.
        /// </summary>
        /// <param name="env">Instance of the host environment.</param>
        /// <param name="keepColumns">The array of column names to keep, cannot be set with <paramref name="dropColumns"/>.</param>
        /// <param name="dropColumns">The array of column names to drop, cannot be set with <paramref name="keepColumns"/>.</param>
        /// <param name="keepHidden">If true will keep hidden columns and false will remove hidden columns. The argument is
        /// ignored if the Estimator is in "drop mode".</param>
        /// <param name="ignoreMissing">If false will check for any columns given in <paramref name="keepColumns"/>
        ///     or <paramref name="dropColumns"/> that are missing from the input. If a missing colums exists a
        ///     SchemaMistmatch exception is thrown. If true, the check is not made.</param>
        internal ColumnSelectingEstimator(IHostEnvironment env, string[] keepColumns,
                                    string[] dropColumns, bool keepHidden = ColumnSelectingTransformer.Defaults.KeepHidden,
                                    bool ignoreMissing = ColumnSelectingTransformer.Defaults.IgnoreMissing)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ColumnSelectingEstimator)),
                  new ColumnSelectingTransformer(env, keepColumns, dropColumns, keepHidden, ignoreMissing))
        {

            _selectPredicate = (name) => (keepColumns != null) ? keepColumns.Contains(name) : !dropColumns.Contains(name);
        }

        /// <summary>
        /// KeepColumns is used to select a list of columns that the user wants to keep on a given an input. Any column not specified
        /// will be dropped from the output output schema.
        /// </summary>
        /// <param name="env">Instance of the host environment.</param>
        /// <param name="columnsToKeep">The array of column names to keep.</param>
        public static ColumnSelectingEstimator KeepColumns(IHostEnvironment env, params string[] columnsToKeep)
        {
            return new ColumnSelectingEstimator(env, columnsToKeep);
        }

        /// <summary>
        /// DropColumns is used to select a list of columns that user wants to drop from a given input. Any column not specified will
        /// be maintained in the output schema.
        /// </summary>
        /// <param name="env">Instance of the host environment.</param>
        /// <param name="columnsToDrop">The array of column names to drop.</param>
        public static ColumnSelectingEstimator DropColumns(IHostEnvironment env, params string[] columnsToDrop)
        {
            return new ColumnSelectingEstimator(env, null, columnsToDrop);

        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            if (!Transformer.IgnoreMissing && !ColumnSelectingTransformer.IsSchemaValid(inputSchema.Select(x => x.Name),
                                                                                    Transformer.SelectColumns,
                                                                                    out IEnumerable<string> invalidColumns))
            {
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", string.Join(",", invalidColumns));
            }

            var columns = inputSchema.Where(c => _selectPredicate(c.Name));
            return new SchemaShape(columns);
        }
    }

    /// <summary>
    /// The SelectColumns Transforms allows the user to specify columns to drop or keep from a given input.
    /// </summary>
    public sealed class ColumnSelectingTransformer : ITransformer, ICanSaveModel
    {
        internal const string Summary = "Selects which columns from the dataset to keep.";
        internal const string UserName = "Select Columns Transform";
        internal const string ShortName = "Select";
        internal const string LoaderSignature = "SelectColumnsTransform";

        // Back-compat signatures to support loading Drop/Keep and Choose Transforms
        internal const string DropLoaderSignature = "DropColumnsTransform";
        internal const string ChooseLoaderSignature = "ChooseColumnsTransform";
        internal const string ChooseLoaderSignatureOld = "ChooseColumnsFunction";

        private readonly IHost _host;
        private string[] _selectedColumns;

        internal static class Defaults
        {
            public const bool KeepHidden = false;
            public const bool IgnoreMissing = false;
        };

        public bool IsRowToRowMapper => true;

        public IEnumerable<string> SelectColumns => _selectedColumns.AsReadOnly();

        public bool KeepColumns { get; }

        public bool KeepHidden { get; }
        public bool IgnoreMissing { get; }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SELCOLST",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ColumnSelectingTransformer).Assembly.FullName);
        }

        private static VersionInfo GetDropVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DRPCOLST",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Added KeepColumns
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ColumnSelectingTransformer).Assembly.FullName);
        }

        private static VersionInfo GetChooseVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CHSCOLSF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: ChooseLoaderSignature,
                loaderSignatureAlt: ChooseLoaderSignatureOld,
                loaderAssemblyName: typeof(ColumnSelectingTransformer).Assembly.FullName);
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "List of columns to keep.", ShortName = "keepcol", SortOrder = 1)]
            public string[] KeepColumns;

            [Argument(ArgumentType.Multiple, HelpText = "List of columns to drop.", ShortName = "dropcol", SortOrder = 2)]
            public string[] DropColumns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Specifies whether to keep or remove hidden columns.", ShortName = "hidden", SortOrder = 3)]
            public bool KeepHidden = Defaults.KeepHidden;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Specifies whether to ignore columns that are missing from the input.", ShortName = "ignore", SortOrder = 4)]
            public bool IgnoreMissing = Defaults.IgnoreMissing;
        }

        public ColumnSelectingTransformer(IHostEnvironment env, string[] keepColumns, string[] dropColumns,
                                        bool keepHidden = Defaults.KeepHidden, bool ignoreMissing = Defaults.IgnoreMissing)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(ColumnSelectingTransformer));
            _host.CheckValueOrNull(keepColumns);
            _host.CheckValueOrNull(dropColumns);

            bool keepValid = Utils.Size(keepColumns) > 0;
            bool dropValid = Utils.Size(dropColumns) > 0;

            // Check that both are not valid
            _host.Check(!(keepValid && dropValid), "Both " + nameof(keepColumns) + " and " + nameof(dropColumns) + " are set. Exactly one can be specified.");
            // Check that both are invalid
            _host.Check(!(!keepValid && !dropValid), "Neither " + nameof(keepColumns) + " and " + nameof(dropColumns) + " is set. Exactly one must be specified.");

            _selectedColumns = (keepValid) ? keepColumns : dropColumns;
            KeepColumns = keepValid;
            KeepHidden = keepHidden;
            IgnoreMissing = ignoreMissing;
        }

        /// <summary>
        /// Helper function to determine the model version that is being loaded.
        /// </summary>
        private static bool CheckModelVersion(ModelLoadContext ctx, VersionInfo versionInfo)
        {
            try
            {
                ctx.CheckVersionInfo(versionInfo);
                return true;
            }
            catch (Exception)
            {
                //consume
                return false;
            }
        }

        /// <summary>
        /// Back-compatibilty function that handles loading the DropColumns Transform.
        /// </summary>
        private static ColumnSelectingTransformer LoadDropColumnsTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            int cbFloat = ctx.Reader.ReadInt32();
            env.CheckDecode(cbFloat == sizeof(float));

            // *** Binary format ***
            // bool: whether to keep (vs drop) the named columns
            // int: number of names
            // int[]: the ids of the names
            var keep = ctx.Reader.ReadBoolByte();
            int count = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(count > 0);

            var names = new HashSet<string>();
            for (int i = 0; i < count; i++)
            {
                string name = ctx.LoadNonEmptyString();
                Contracts.CheckDecode(names.Add(name));
            }

            string[] keepColumns = null;
            string[] dropColumns = null;
            if (keep)
                keepColumns = names.ToArray();
            else
                dropColumns = names.ToArray();

            // Note for backward compatibility, Drop/Keep Columns always preserves
            // hidden columns
            return new ColumnSelectingTransformer(env, keepColumns, dropColumns, true);
        }

        /// <summary>
        /// Back-compatibilty that is handling the HiddenColumnOption from ChooseColumns.
        /// </summary>
        private enum HiddenColumnOption : byte
        {
            Drop = 1,
            Keep = 2,
            Rename = 3
        };

        /// <summary>
        /// Backwards compatibility helper function to convert the HiddenColumnOption to a boolean.
        /// </summary>
        private static bool GetHiddenOption(IHostEnvironment env, HiddenColumnOption option)
        {
            switch (option)
            {
                case HiddenColumnOption.Keep:
                    return true;
                case HiddenColumnOption.Drop:
                    return false;
                default:
                    throw env.Except("Unsupported hide option specified");
            };
        }

        /// <summary>
        /// Backwards compatibility helper function that loads a Choose Column Transform.
        /// </summary>
        private static ColumnSelectingTransformer LoadChooseColumnsTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            int cbFloat = ctx.Reader.ReadInt32();
            string renameNotSupportedMsg = "Rename for ChooseColumns is not backwards compatible with the SelectColumnsTranform";
            string differentHideColumnNotSupportedMsg = "Setting a hide option different from default is not compatible with SelectColumnsTransform";
            // *** Binary format ***
            // byte: default HiddenColumnOption value
            // int: number of raw column infos
            // for each raw column info
            //   int: id of output column name
            //   int: id of input column name
            //   byte: HiddenColumnOption
            var hiddenOption = (HiddenColumnOption)ctx.Reader.ReadByte();
            Contracts.Assert(Enum.IsDefined(typeof(HiddenColumnOption), hiddenOption));
            env.Check(HiddenColumnOption.Rename != hiddenOption, renameNotSupportedMsg);
            var keepHidden = GetHiddenOption(env, hiddenOption);

            int count = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(count >= 0);
            var keepHiddenCols = new HiddenColumnOption[count];

            var names = new HashSet<string>();
            for (int colIdx = 0; colIdx < count; ++colIdx)
            {
                string dst = ctx.LoadNonEmptyString();
                Contracts.CheckDecode(names.Add(dst));
                string src = ctx.LoadNonEmptyString();

                var colHiddenOption = (HiddenColumnOption)ctx.Reader.ReadByte();
                Contracts.Assert(Enum.IsDefined(typeof(HiddenColumnOption), colHiddenOption));
                env.Check(colHiddenOption != HiddenColumnOption.Rename, renameNotSupportedMsg);
                var colKeepHidden = GetHiddenOption(env, colHiddenOption);
                env.Check(colKeepHidden == keepHidden, differentHideColumnNotSupportedMsg);
            }

            return new ColumnSelectingTransformer(env, names.ToArray(), null, keepHidden);
        }

        // Factory method for SignatureLoadModelTransform.
        private static ColumnSelectingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            ctx.CheckAtModel(GetVersionInfo());
            // *** Binary format ***
            // bool: keep columns flag
            // bool: keep hidden flag
            // bool: ignore missing flag
            // int: number of added columns
            // for each added column
            //   string: selected column name
            var keepColumns = ctx.Reader.ReadBoolByte();
            var keepHidden = ctx.Reader.ReadBoolByte();
            var ignoreMissing = ctx.Reader.ReadBoolByte();
            var length = ctx.Reader.ReadInt32();
            var columns = new string[length];
            for (int i = 0; i < length; i++)
            {
                columns[i] = ctx.LoadNonEmptyString();
            }

            string[] columnsToKeep = null;
            string[] columnsToDrop = null;
            if (keepColumns)
                columnsToKeep = columns;
            else
                columnsToDrop = columns;

            return new ColumnSelectingTransformer(env, columnsToKeep, columnsToDrop, keepHidden, ignoreMissing);
        }

        // Factory method for SignatureLoadDataTransform.
        public static IDataView Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ColumnSelectingTransformer transform;

            // Determine which version of the transform is being loaded.
            if (CheckModelVersion(ctx, GetDropVersionInfo()))
            {
                transform = LoadDropColumnsTransform(env, ctx, input);
            }
            else if (CheckModelVersion(ctx, GetChooseVersionInfo()))
            {
                transform = LoadChooseColumnsTransform(env, ctx, input);
            }
            else
            {
                transform = Create(env, ctx);
            }

            return transform.Transform(input);
        }

        public static IDataTransform CreateKeep(IHostEnvironment env, IDataView input, string[] keepColumns, bool keepHidden = false)
        {
            var transform = new ColumnSelectingTransformer(env, keepColumns, null, keepHidden);
            return new SelectColumnsDataTransform(env, transform, new Mapper(transform, input.Schema), input);
        }

        public static IDataTransform CreateDrop(IHostEnvironment env, IDataView input, params string[] dropColumns)
        {
            var transform = new ColumnSelectingTransformer(env, null, dropColumns);
            return new SelectColumnsDataTransform(env, transform, new Mapper(transform, input.Schema), input);
        }

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            var transform = new ColumnSelectingTransformer(env, args.KeepColumns, args.DropColumns,
                                                            args.KeepHidden, args.IgnoreMissing);
            return new SelectColumnsDataTransform(env, transform, new Mapper(transform, input.Schema), input);
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.Writer.WriteBoolByte(KeepColumns);
            ctx.Writer.WriteBoolByte(KeepHidden);
            ctx.Writer.WriteBoolByte(IgnoreMissing);
            var length = _selectedColumns.Length;
            ctx.Writer.Write(length);
            for (int i = 0; i < length; i++)
                ctx.SaveNonEmptyString(_selectedColumns[i]);
        }

        public static bool IsSchemaValid(IEnumerable<string> inputColumns,
                                         IEnumerable<string> selectColumns,
                                         out IEnumerable<string> invalidColumns)
        {
            // Confirm that all selected columns are in the inputSchema
            var missing = selectColumns.Where(x => !inputColumns.Contains(x));
            invalidColumns = missing;
            return missing.Count() == 0;
        }

        public Schema GetOutputSchema(Schema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            if (!IgnoreMissing && !IsSchemaValid(inputSchema.Select(x => x.Name),
                                                                SelectColumns, out IEnumerable<string> invalidColumns))
            {
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", string.Join(",", invalidColumns));
            }

            return new Mapper(this, inputSchema).OutputSchema;
        }

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            if (!IgnoreMissing && !IsSchemaValid(inputSchema.Select(x => x.Name),
                                                    SelectColumns, out IEnumerable<string> invalidColumns))
            {
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", string.Join(",", invalidColumns));
            }

            return new SelectColumnsDataTransform(_host, this,
                                                  new Mapper(this, inputSchema),
                                                  new EmptyDataView(_host, inputSchema));
        }

        public IDataView Transform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            if (!IgnoreMissing && !IsSchemaValid(input.Schema.Select(x => x.Name),
                                                    SelectColumns, out IEnumerable<string> invalidColumns))
            {
                throw _host.ExceptSchemaMismatch(nameof(input), "input", string.Join(",", invalidColumns));
            }

            return new SelectColumnsDataTransform(_host, this, new Mapper(this, input.Schema), input);
        }

        private sealed class Mapper
        {
            private readonly IHost _host;
            private readonly Schema _inputSchema;
            private readonly int[] _outputToInputMap;

            public Schema InputSchema => _inputSchema;

            public Schema OutputSchema { get; }

            public Mapper(ColumnSelectingTransformer transform, Schema inputSchema)
            {
                _host = transform._host.Register(nameof(Mapper));
                _inputSchema = inputSchema;

                _outputToInputMap = BuildOutputToInputMap(transform.SelectColumns,
                                                            transform.KeepColumns,
                                                            transform.KeepHidden,
                                                            _inputSchema);
                OutputSchema = GenerateOutputSchema(_outputToInputMap, _inputSchema);
            }

            public int GetInputIndex(int outputIndex)
            {
                _host.Assert(0 <= outputIndex && outputIndex < _outputToInputMap.Length);
                return _outputToInputMap[outputIndex];
            }

            private static int[] BuildOutputToInputMap(IEnumerable<string> selectedColumns,
                bool keepColumns,
                bool keepHidden,
                Schema inputSchema)
            {
                var outputToInputMapping = new List<int>();
                var columnCount = inputSchema.Count;

                if (keepColumns)
                {
                    // With KeepColumns, the order that is specified is preserved in the mapping.
                    // For example if a given input has the columns of ABC and the select columns are
                    // specified as CA, then the output will be CA.

                    // In order to account for keeping hidden columns, build a dictionary of
                    // column name-> list of column indices. This dictionary is used for
                    // building the final mapping.
                    var columnDict = new Dictionary<string, List<int>>();
                    for (int colIdx = 0; colIdx < inputSchema.Count; ++colIdx)
                    {
                        if (!keepHidden && inputSchema[colIdx].IsHidden)
                            continue;

                        var columnName = inputSchema[colIdx].Name;
                        if (columnDict.TryGetValue(columnName, out List<int> columnList))
                            columnList.Add(colIdx);
                        else
                        {
                            columnList = new List<int>();
                            columnList.Add(colIdx);
                            columnDict.Add(columnName, columnList);
                        }
                    }

                    // Since the ordering matters, iterate through the selected columns
                    // finding the associated index that should be used.
                    foreach (var columnName in selectedColumns)
                    {
                        if (columnDict.TryGetValue(columnName, out List<int> columnList))
                        {
                            foreach (var colIdx in columnList)
                            {
                                outputToInputMapping.Add(colIdx);
                            }
                        }
                    }
                }
                else
                {
                    // Handles the drop case, removing any columns specified from the input
                    // In the case of drop, the order of the output is modeled after the input
                    // given an input of ABC and dropping column B will result in AC.
                    // In drop mode, we drop all columns with the specified names and keep all the rest,
                    // ignoring the keepHidden argument.
                    for (int colIdx = 0; colIdx < inputSchema.Count; colIdx++)
                    {
                        if (selectedColumns.Contains(inputSchema[colIdx].Name))
                            continue;

                        outputToInputMapping.Add(colIdx);
                    }
                }

                return outputToInputMapping.ToArray();
            }

            private static Schema GenerateOutputSchema(IEnumerable<int> map,
                                                        Schema inputSchema)
            {
                var outputColumns = map.Select(x => new Schema.DetachedColumn(inputSchema[x]));
                return SchemaExtensions.MakeSchema(outputColumns);
            }
        }

        private sealed class RowImpl : WrappingRow
        {
            private readonly Mapper _mapper;
            public RowImpl(Row input, Mapper mapper)
                : base(input)
            {
                _mapper = mapper;
            }

            public override Schema Schema => _mapper.OutputSchema;

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                int index = _mapper.GetInputIndex(col);
                return Input.GetGetter<TValue>(index);
            }

            public override bool IsColumnActive(int col) => true;
        }

        private sealed class SelectColumnsDataTransform : IDataTransform, IRowToRowMapper, ITransformTemplate
        {
            private readonly IHost _host;
            private readonly ColumnSelectingTransformer _transform;
            private readonly Mapper _mapper;

            public SelectColumnsDataTransform(IHostEnvironment env, ColumnSelectingTransformer transform, Mapper mapper, IDataView input)
            {
                _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(SelectColumnsDataTransform));
                _transform = transform;
                _mapper = mapper;
                Source = input;
            }

            public bool CanShuffle => Source.CanShuffle;

            public IDataView Source { get; }

            public Schema InputSchema => Source.Schema;

            Schema IDataView.Schema => OutputSchema;

            public Schema OutputSchema => _mapper.OutputSchema;

            public long? GetRowCount() => Source.GetRowCount();

            public RowCursor GetRowCursor(IEnumerable<Schema.Column> columnsNeeded, Random rand = null)
            {
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
                _host.AssertValueOrNull(rand);

                // Build out the active state for the input
                var inputPred = GetDependencies(predicate);
                var inputCols = Source.Schema.Where(x => inputPred(x.Index));

                var inputRowCursor = Source.GetRowCursor(inputCols, rand);

                // Build the active state for the output
                var active = Utils.BuildArray(_mapper.OutputSchema.Count, columnsNeeded);
                return new Cursor(_host, _mapper, inputRowCursor, active);
            }

            public RowCursor[] GetRowCursorSet(IEnumerable<Schema.Column> columnsNeeded, int n, Random rand = null)
            {
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);
                _host.CheckValueOrNull(rand);

                // Build out the active state for the input
                var inputPred = GetDependencies(predicate);
                var inputCols = Source.Schema.Where(x => inputPred(x.Index));
                var inputs = Source.GetRowCursorSet(inputCols, n, rand);

                // Build out the acitve state for the output
                var active = Utils.BuildArray(_mapper.OutputSchema.Count, columnsNeeded);
                _host.AssertNonEmpty(inputs);

                // No need to split if this is given 1 input cursor.
                var cursors = new RowCursor[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                    cursors[i] = new Cursor(_host, _mapper, inputs[i], active);
                return cursors;
            }

            public void Save(ModelSaveContext ctx) => _transform.Save(ctx);

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                var active = new bool[_mapper.InputSchema.Count];
                var columnCount = _mapper.OutputSchema.Count;
                for (int colIdx = 0; colIdx < columnCount; ++colIdx)
                {
                    if (activeOutput(colIdx))
                        active[_mapper.GetInputIndex(colIdx)] = true;
                }

                return col => active[col];
            }

            public Row GetRow(Row input, Func<int, bool> active)
            {
                return new RowImpl(input, _mapper);
            }

            public IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
                => new SelectColumnsDataTransform(env, _transform, new Mapper(_transform, newSource.Schema), newSource);
        }

        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly Mapper _mapper;
            private readonly RowCursor _inputCursor;
            private readonly bool[] _active;
            public Cursor(IChannelProvider provider, Mapper mapper, RowCursor input, bool[] active)
                : base(provider, input)
            {
                _mapper = mapper;
                _inputCursor = input;
                _active = active;
            }

            public override Schema Schema => _mapper.OutputSchema;

            public override ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                int index = _mapper.GetInputIndex(col);
                return _inputCursor.GetGetter<TValue>(index);
            }

            public override bool IsColumnActive(int col) => _active[col];
        }
    }
}
