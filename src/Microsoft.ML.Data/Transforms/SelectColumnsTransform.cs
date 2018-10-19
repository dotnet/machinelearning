// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;

[assembly: LoadableClass(SelectColumnsTransform.Summary, typeof(IDataTransform), typeof(SelectColumnsTransform),
                typeof(SelectColumnsTransform.Arguments), typeof(SignatureDataTransform),
                SelectColumnsTransform.UserName, "SelectColumns", "SelectColumnsTransform", SelectColumnsTransform.ShortName, DocName = "transform/SelectTransforms.md")]

[assembly: LoadableClass(SelectColumnsTransform.Summary, typeof(IDataView), typeof(SelectColumnsTransform), null, typeof(SignatureLoadDataTransform),
                            SelectColumnsTransform.UserName, SelectColumnsTransform.LoaderSignature)]

[assembly: LoadableClass(SelectColumnsTransform.Summary, typeof(SelectColumnsTransform), null, typeof(SignatureLoadModel),
                            SelectColumnsTransform.UserName, SelectColumnsTransform.LoaderSignature)]

// Back-compat to handle loading of the Drop and Keep Transformer
[assembly: LoadableClass("", typeof(IDataView), typeof(SelectColumnsTransform), null, typeof(SignatureLoadDataTransform),
    "", SelectColumnsTransform.DropLoaderSignature)]

// Back-compat to handle loading of the Choose Columns Transformer
[assembly: LoadableClass("", typeof(IDataView), typeof(SelectColumnsTransform), null, typeof(SignatureLoadDataTransform),
    "", SelectColumnsTransform.ChooseLoaderSignature, SelectColumnsTransform.ChooseLoaderSignatureOld)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// The SelectColumnsEstimator supports selection of specified columns to keep from a given input.
    /// </summary>
    public sealed class SelectColumnsEstimator : TrivialEstimator<SelectColumnsTransform>
    {
        private readonly Func<string, bool> _selectPredicate;

        /// <summary>
        /// Constructs the Select Columns Estimator.
        /// </summary>
        /// <param name="env">Instance of the host environment.</param>
        /// <param name="keepColumns">The array of column names to keep.</param>
        public SelectColumnsEstimator(IHostEnvironment env, params string[] keepColumns)
            : this(env, keepColumns, null, true, true)
        { }

        /// <summary>
        /// Constructs the Select Columns Estimator.
        /// </summary>
        /// <param name="env">Instance of the host environment.</param>
        /// <param name="keepColumns">The array of column names to keep, cannot be set with <paramref name="dropColumns"/>.</param>
        /// <param name="dropColumns">The array of column names to drop, cannot be set with <paramref name="keepColumns"/>.</param>
        /// <param name="keepHidden">If true will keep hidden columns and false will remove hidden columns.</param>
        /// <param name="ignoreMissing">If false will check for any columns given in <paramref name="keepColumns"/>
        ///     or <paramref name="dropColumns"/> that are missing from the input. If a missing colums exists a
        ///     SchemaMistmatch exception is thrown. If true, the check is not made.</param>
        public SelectColumnsEstimator(IHostEnvironment env, string[] keepColumns,
                                    string[] dropColumns, bool keepHidden = true,
                                    bool ignoreMissing= true)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(SelectColumnsEstimator)),
                  new SelectColumnsTransform(env, keepColumns, dropColumns, keepHidden, ignoreMissing))
        {

            _selectPredicate = (name) => (keepColumns!=null) ? keepColumns.Contains(name) : !dropColumns.Contains(name);
        }

        /// <summary>
        /// KeepColumns is used to select a list of columns that the user wants to keep on a given an input. Any column not specified
        /// will be dropped from the output output schema.
        /// </summary>
        /// <param name="env">Instance of the host environment.</param>
        /// <param name="columnsToKeep">The array of column names to keep.</param>
        public static SelectColumnsEstimator KeepColumns(IHostEnvironment env, params string[] columnsToKeep)
        {
            return new SelectColumnsEstimator(env, columnsToKeep);
        }

        /// <summary>
        /// DropColumns is used to select a list of columns that user wants to drop from a given input. Any column not specified will
        /// be maintained in the output schema.
        /// </summary>
        /// <param name="env">Instance of the host environment.</param>
        /// <param name="columnsToDrop">The array of column names to drop.</param>
        public static SelectColumnsEstimator DropColumns(IHostEnvironment env, params string[] columnsToDrop)
        {
            return new SelectColumnsEstimator(env, null, columnsToDrop);

        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            if (!Transformer.IgnoreMissing && !SelectColumnsTransform.IsSchemaValid(inputSchema.Columns.Select(x=>x.Name),
                                                                                    Transformer.SelectColumns,
                                                                                    out IEnumerable<string> invalidColumns))
            {
                  throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", string.Join(",", invalidColumns));
            }

            var columns = inputSchema.Columns.Where(c=>_selectPredicate(c.Name));
            return new SchemaShape(columns);
        }
    }

    /// <summary>
    /// The SelectColumns Transforms allows the user to specify columns to drop or keep from a given input.
    /// </summary>
    public sealed class SelectColumnsTransform : ITransformer, ICanSaveModel
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
                loaderAssemblyName: typeof(SelectColumnsTransform).Assembly.FullName);
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
                loaderAssemblyName: typeof(SelectColumnsTransform).Assembly.FullName);
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
                loaderAssemblyName: typeof(SelectColumnsTransform).Assembly.FullName);
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "List of columns to keep.", ShortName = "keepcol", SortOrder = 1)]
            public string[] KeepColumns;

            [Argument(ArgumentType.Multiple, HelpText = "List of columns to drop.", ShortName = "dropcol", SortOrder = 2)]
            public string[] DropColumns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Specifies whether to keep or remove hidden columns.", ShortName = "hidden", SortOrder = 3)]
            public bool KeepHidden = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Specifies whether to ignore columns that are missing from the input.", ShortName = "ignore", SortOrder = 4)]
            public bool IgnoreMissing = true;
        }

        public SelectColumnsTransform(IHostEnvironment env, string[] keepColumns, string[] dropColumns,
                                        bool keepHidden=true, bool ignoreMissing=true)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(SelectColumnsTransform));
            _host.CheckValueOrNull(keepColumns);
            _host.CheckValueOrNull(dropColumns);

            bool keepValid = keepColumns != null && keepColumns.Count() > 0;
            bool dropValid = dropColumns != null && dropColumns.Count() > 0;

            // Check that both are not valid
            _host.Check(!(keepValid && dropValid), "Both keepColumns and dropColumns are set, only one can be specified.");
            // Check that both are invalid
            _host.Check(!(!keepValid && !dropValid), "Neither keepColumns or dropColumns is set, one must be specified.");

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
        private static SelectColumnsTransform LoadDropColumnsTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            // *** Binary format ***
            // int: sizeof(Float)
            // bindings
            int cbFloat = ctx.Reader.ReadInt32();
            //env.CheckDecode(cbFloat == sizeof(Float));

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

            return new SelectColumnsTransform(env, keepColumns, dropColumns, keep);
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
        private static SelectColumnsTransform LoadChooseColumnsTransform(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
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

            return new SelectColumnsTransform(env, names.ToArray(), null, keepHidden);
        }

        // Factory method for SignatureLoadModelTransform.
        private static SelectColumnsTransform Create(IHostEnvironment env, ModelLoadContext ctx)
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

            return new SelectColumnsTransform(env, columnsToKeep, columnsToDrop, keepHidden, ignoreMissing);
        }

        // Factory method for SignatureLoadDataTransform.
        public static IDataView Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            SelectColumnsTransform transform;

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

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            var transform = new SelectColumnsTransform(env, args.KeepColumns, args.DropColumns,
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
            if (!IgnoreMissing && !IsSchemaValid(inputSchema.GetColumns().Select(x=>x.column.Name),
                                                                SelectColumns, out IEnumerable<string> invalidColumns))
            {
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", string.Join(",", invalidColumns));
            }

            return new Mapper(this, inputSchema).Schema;
        }

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            if (!IgnoreMissing && !IsSchemaValid(inputSchema.GetColumns().Select(x=>x.column.Name),
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
            if (!IgnoreMissing && !IsSchemaValid(input.Schema.GetColumns().Select(x=>x.column.Name),
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

            public ISchema InputSchema => _inputSchema;

            public Schema Schema { get; }

            public Mapper(SelectColumnsTransform transform, ISchema inputSchema)
            {
                _host = transform._host.Register(nameof(Mapper));
                _inputSchema = Runtime.Data.Schema.Create(inputSchema);

                _outputToInputMap = BuildOutputToInputMap(transform.SelectColumns,
                                                            transform.KeepColumns,
                                                            transform.KeepHidden,
                                                            _inputSchema);
                Schema = GenerateOutputSchema(_outputToInputMap, _inputSchema);
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
                var columnCount = inputSchema.ColumnCount;
                int outputIdx = 0;

                for (int colIdx = 0; colIdx < columnCount; ++colIdx)
                {
                    if (!keepHidden && inputSchema.IsHidden(colIdx))
                        continue;

                    var columnName = inputSchema[colIdx].Name;
                    var selected = selectedColumns.Contains(columnName);
                    selected = (keepColumns) ? selected : !selected;
                    if (selected)
                    {
                        outputToInputMapping.Add(colIdx);
                        outputIdx++;
                    }
                }

                return outputToInputMapping.ToArray();
            }

            private static Schema GenerateOutputSchema(IEnumerable<int> map,
                                                        Schema inputSchema)
            {
                IEnumerable<int> inputs = Enumerable.Range(0, inputSchema.ColumnCount);
                var outputColumns = inputs.Where(idx=> map.Contains(idx))
                    .Select(idx=>inputSchema[idx]);
                return new Schema(outputColumns);
            }
        }

        private sealed class Row : IRow
        {
            private readonly Mapper _mapper;
            private readonly IRow _input;
            public Row(IRow input, Mapper mapper)
            {
                _mapper = mapper;
                _input = input;
            }

            public long Position => _input.Position;

            public long Batch => _input.Batch;

            Schema ISchematized.Schema => _mapper.Schema;

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                int index = _mapper.GetInputIndex(col);
                return _input.GetGetter<TValue>(index);
            }

            public ValueGetter<UInt128> GetIdGetter()
                => _input.GetIdGetter();

            public bool IsColumnActive(int col) => true;
        }

        private sealed class SelectColumnsDataTransform : IDataTransform, IRowToRowMapper
        {
            private readonly IHostEnvironment _env;
            private readonly SelectColumnsTransform _transform;
            private readonly Mapper _mapper;

            public SelectColumnsDataTransform(IHostEnvironment env, SelectColumnsTransform transform, Mapper mapper, IDataView input)
            {
                _env = Contracts.CheckRef(env, nameof(env)).Register(nameof(SelectColumnsDataTransform));
                _transform = transform;
                _mapper = mapper;
                Source = input;
            }

            public bool CanShuffle => Source.CanShuffle;

            public IDataView Source { get; }

            Schema IRowToRowMapper.InputSchema => Source.Schema;

            Schema ISchematized.Schema => _mapper.Schema;

            public long? GetRowCount(bool lazy = true) => Source.GetRowCount(lazy);

            public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
            {
                _env.AssertValue(needCol, nameof(needCol));
                _env.AssertValueOrNull(rand);

                // Build out the active state for the input
                var inputPred = GetDependencies(needCol);
                var inputRowCursor = Source.GetRowCursor(inputPred, rand);

                // Build the active state for the output
                var active = Utils.BuildArray(_mapper.Schema.ColumnCount, needCol);
                return new RowCursor(_env, _mapper, inputRowCursor, active);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
            {
                _env.CheckValue(needCol, nameof(needCol));
                _env.CheckValueOrNull(rand);

                // Build out the active state for the input
                var inputPred = GetDependencies(needCol);
                var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);

                // Build out the acitve state for the output
                var active = Utils.BuildArray(_mapper.Schema.ColumnCount, needCol);
                _env.AssertNonEmpty(inputs);

                // No need to split if this is given 1 input cursor.
                var cursors = new IRowCursor[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                {
                    cursors[i] = new RowCursor(_env, _mapper, inputs[i], active);
                }
                return cursors;
            }

            public void Save(ModelSaveContext ctx) => _transform.Save(ctx);

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                var active = new bool[_mapper.InputSchema.ColumnCount];
                var columnCount = _mapper.Schema.ColumnCount;
                for (int colIdx = 0; colIdx < columnCount; ++colIdx)
                {
                    if (activeOutput(colIdx))
                        active[_mapper.GetInputIndex(colIdx)] = true;
                }

                return col => active[col];
            }

            public IRow GetRow(IRow input, Func<int, bool> active, out Action disposer)
            {
                disposer = null;
                return new Row(input, _mapper);
            }
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Mapper _mapper;
            private readonly IRowCursor _inputCursor;
            private readonly bool[] _active;
            public RowCursor(IChannelProvider provider, Mapper mapper, IRowCursor input, bool[] active)
                : base(provider, input)
            {
                _mapper = mapper;
                _inputCursor = input;
                _active = active;
            }

            Schema ISchematized.Schema => _mapper.Schema;

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                int index = _mapper.GetInputIndex(col);
                return _inputCursor.GetGetter<TValue>(index);
            }

            public bool IsColumnActive(int col) => _active[col];
        }
    }
}
