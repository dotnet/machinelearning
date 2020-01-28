// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(CountTableTransformer.Summary, typeof(IDataTransform), typeof(CountTableTransformer), typeof(CountTableTransformer.Options), typeof(SignatureDataTransform),
    CountTableTransformer.UserName, "CountTableTransform", "CountTable", "Count")]

[assembly: LoadableClass(CountTableTransformer.Summary, typeof(IDataTransform), typeof(CountTableTransformer), null, typeof(SignatureLoadDataTransform),
    CountTableTransformer.UserName, CountTableTransformer.LoaderSignature)]

[assembly: LoadableClass(CountTableTransformer.Summary, typeof(CountTableTransformer), null, typeof(SignatureLoadModel),
    CountTableTransformer.UserName, CountTableTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(CountTableTransformer), null, typeof(SignatureLoadRowMapper),
   CountTableTransformer.UserName, CountTableTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(CountTable))]

namespace Microsoft.ML.Transforms
{
    internal sealed class CountTableEstimator : IEstimator<CountTableTransformer>
    {
        internal abstract class ColumnOptionsBase
        {
            public readonly string Name;
            public readonly string InputColumnName;
            public readonly float PriorCoefficient;
            public readonly float LaplaceScale;
            public readonly int Seed;

            public ColumnOptionsBase(string name, string inputColumnName, float priorCoefficient = 1,
                float laplaceScale = 0, int seed = 314489979)
            {
                Name = name;
                InputColumnName = inputColumnName;
                PriorCoefficient = priorCoefficient;
                LaplaceScale = laplaceScale;
                Seed = seed;
            }
        }

        internal sealed class ColumnOptions : ColumnOptionsBase
        {
            public readonly CountTableBuilderBase CountTableBuilder;

            public ColumnOptions(string name, string inputColumnName, CountTableBuilderBase countTableBuilder, float priorCoefficient = 1,
                float laplaceScale = 0, int seed = 314489979)
                : base(name, inputColumnName, priorCoefficient, laplaceScale, seed)
            {
                CountTableBuilder = countTableBuilder;
            }
        }

        internal sealed class SharedColumnOptions : ColumnOptionsBase
        {
            public SharedColumnOptions(string name, string inputColumnName, float priorCoefficient = 1,
                float laplaceScale = 0, int seed = 314489979)
                : base(name, inputColumnName, priorCoefficient, laplaceScale, seed)
            {
            }
        }

        private readonly IHost _host;
        private readonly ColumnOptionsBase[] _columns;
        private readonly CountTableBuilderBase[] _builders;
        private readonly CountTableBuilderBase _sharedBuilder;
        private readonly string _labelColumnName;
        private readonly CountTableTransformer _initialCounts;

        internal CountTableEstimator(IHostEnvironment env, string labelColumnName, CountTableBuilderBase countTableBuilder, params SharedColumnOptions[] columns)
            : this(env, labelColumnName, columns)
        {
            _sharedBuilder = countTableBuilder;
        }

        internal CountTableEstimator(IHostEnvironment env, string labelColumnName,
                params ColumnOptions[] columns)
            : this(env, labelColumnName, columns.Cast<ColumnOptionsBase>().ToArray())
        {
            _builders = columns.Select(c => c.CountTableBuilder).ToArray();
        }

        internal CountTableEstimator(IHostEnvironment env, string labelColumnName, CountTableTransformer initial, params InputOutputColumnPair[] columns)
            : this(env, labelColumnName, ExtractColumnOptions(initial, columns))
        {
            _initialCounts = initial;
        }

        private static ColumnOptionsBase[] ExtractColumnOptions(CountTableTransformer initial, InputOutputColumnPair[] columns)
        {
            Contracts.CheckValue(initial, nameof(initial));
            if (columns.Length != initial.Featurizer.PriorCoef.Length)
                throw Contracts.ExceptParam(nameof(columns), $"New estimator applied {columns.Length} columns, but old transformer applied to {initial.Featurizer.PriorCoef.Length} columns");
            var cols = new ColumnOptionsBase[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                cols[i] = new SharedColumnOptions(columns[i].OutputColumnName, columns[i].InputColumnName,
                    initial.Featurizer.PriorCoef[i], initial.Featurizer.LaplaceScale[i], initial.Seeds[i]);
            }
            return cols;
        }

        private CountTableEstimator(IHostEnvironment env, string labelColumnName, ColumnOptionsBase[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(labelColumnName, nameof(labelColumnName));
            env.CheckNonEmpty(columns, nameof(columns));
            _host = env.Register(nameof(CountTableEstimator));
            _host.CheckParam(columns.All(col => col.PriorCoefficient > 0), nameof(ColumnOptionsBase.PriorCoefficient), "Must be greater than zero");
            _host.CheckParam(columns.All(col => col.LaplaceScale >= 0), nameof(ColumnOptionsBase.LaplaceScale), "Must be greater than or equal to zero.");

            _columns = columns;
            _labelColumnName = labelColumnName;
        }

        public CountTableTransformer Fit(IDataView input)
        {
            var labelCol = input.Schema.GetColumnOrNull(_labelColumnName);
            if (labelCol == null)
                throw _host.ExceptUserArg(nameof(_labelColumnName), "Label column '{0}' not found", _labelColumnName);

            CheckLabelType(new RoleMappedData(input, roles: RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Label, _labelColumnName)), out var labelCardinality);

            var labelColumnType = labelCol.GetValueOrDefault().Type;
            var labelClassNames = InitLabelClassNames(_host, labelCol.GetValueOrDefault(), labelCardinality);

            var n = _columns.Length;

            var inputColumns = new DataViewSchema.Column[_columns.Length];
            for (int i = 0; i < inputColumns.Length; i++)
            {
                var col = input.Schema.GetColumnOrNull(_columns[i].InputColumnName);
                if (col == null)
                    throw _host.Except($"Could not find column {_columns[i].InputColumnName} in input schema");
                inputColumns[i] = col.GetValueOrDefault();
            }

            _host.Assert(_initialCounts != null || _sharedBuilder != null || _builders != null);
            MultiCountTableBuilderBase multiBuilder;
            if (_initialCounts != null)
                multiBuilder = _initialCounts.Featurizer.MultiCountTable.ToBuilder(_host, inputColumns, labelCardinality);
            else if (_builders != null)
                multiBuilder = new ParallelMultiCountTableBuilder(_host, inputColumns, _builders, labelCardinality);
            else
                multiBuilder = new BagMultiCountTableBuilder(_host, inputColumns, _sharedBuilder, labelCardinality);

            var cols = new List<DataViewSchema.Column>();
            foreach (var c in _columns)
            {
                var col = input.Schema.GetColumnOrNull(c.InputColumnName);
                _host.Assert(col.HasValue);
                cols.Add(col.Value);
            }

            TrainTables(input, cols, multiBuilder, labelCol.GetValueOrDefault());

            var multiCountTable = multiBuilder.CreateMultiCountTable();

            var featurizer = new CountTargetEncodingFeaturizer(_host, _columns.Select(col => col.PriorCoefficient).ToArray(), _columns.Select(col => col.LaplaceScale).ToArray(), labelCardinality, multiCountTable);

            return new CountTableTransformer(_host, featurizer, labelClassNames,
                _columns.Select(col => col.Seed).ToArray(), _columns.Select(col => (col.Name, col.InputColumnName)).ToArray());
        }

        private void TrainTables(IDataView trainingData, List<DataViewSchema.Column> cols, MultiCountTableBuilderBase builder, DataViewSchema.Column labelColumn)
        {
            var colCount = _columns.Length;

            using (var cursor = trainingData.GetRowCursor(cols.Prepend(labelColumn)))
            {
                // populate getters
                var singleGetters = new ValueGetter<uint>[colCount];
                var vectorGetters = new ValueGetter<VBuffer<uint>>[colCount];
                for (int i = 0; i < colCount; i++)
                {
                    if (cols[i].Type is VectorDataViewType)
                        vectorGetters[i] = cursor.GetGetter<VBuffer<uint>>(cols[i]);
                    else
                        singleGetters[i] = cursor.GetGetter<uint>(cols[i]);
                }

                var labelGetter = GetLabelGetter(cursor, labelColumn);
                long labelKey = 0;
                uint srcSingleValue = 0;
                var srcBuffer = default(VBuffer<uint>);
                while (cursor.MoveNext())
                {
                    labelGetter(ref labelKey);
                    if (labelKey < 0) // Invalid label, skip the data
                        continue;
                    for (int i = 0; i < colCount; i++)
                    {
                        if (cols[i].Type is VectorDataViewType)
                        {
                            vectorGetters[i](ref srcBuffer);
                            _host.Check(srcBuffer.Length == cols[i].Type.GetVectorSize(), "value count mismatch");
                            IncrementVec(builder, i, ref srcBuffer, (uint)labelKey);
                        }
                        else
                        {
                            singleGetters[i](ref srcSingleValue);
                            builder.IncrementSlot(i, 0, srcSingleValue, (uint)labelKey);
                        }
                    }
                }
            }
        }

        private ValueGetter<long> GetLabelGetter(DataViewRow row, DataViewSchema.Column col)
        {
            // The label column type is checked as part of args validation.
            var type = col.Type;
            _host.Assert(type is KeyDataViewType || type is NumberDataViewType || type is BooleanDataViewType);

            if (type is BooleanDataViewType)
            {
                bool src = default;
                var getSrc = row.GetGetter<bool>(col);
                return
                    (ref long dst) =>
                    {
                        getSrc(ref src);
                        if (src)
                            dst = 1;
                        else
                            dst = 0;
                    };
            }
            if (type is KeyDataViewType)
            {
                _host.Assert(type.GetKeyCount() > 0);

                int size = type.GetKeyCountAsInt32();
                ulong src = 0;
                var getSrc = RowCursorUtils.GetGetterAs<ulong>(NumberDataViewType.UInt64, row, col.Index);
                return
                    (ref long dst) =>
                    {
                        getSrc(ref src);
                        // The value should fall between 0 and size inclusive, where 0 is considered
                        // missing/invalid (this is the contract of the KeyType). However, we still handle the
                        // cases of too large values correctly (by treating them as invalid).
                        if (src <= (ulong)size)
                            dst = (long)src - 1;
                        else
                            dst = -1;
                    };
            }
            else
            {
                double src = 0;
                var getSrc = RowCursorUtils.GetGetterAs<double>(NumberDataViewType.Double, row, col.Index);
                return
                    (ref long dst) =>
                    {
                        getSrc(ref src);
                        // NaN maps to -1.
                        if (double.IsNaN(src))
                            dst = -1;
                        else
                            dst = (long)src;
                    };
            }
        }

        private void IncrementVec(MultiCountTableBuilderBase builder, int iCol, ref VBuffer<uint> srcBuffer, uint labelKey)
        {
            var n = srcBuffer.Length;
            var values = srcBuffer.GetValues();
            var indices = srcBuffer.GetIndices();
            if (srcBuffer.IsDense)
            {
                for (int i = 0; i < n; i++)
                    builder.IncrementSlot(iCol, i, values[i], labelKey);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
                    builder.IncrementSlot(iCol, indices[i], values[i], labelKey);
            }
        }

        private void CheckLabelType(RoleMappedData data, out int labelCardinality)
        {
            _host.Assert(data.Schema.Label.HasValue);

            if (data.Schema.Label.Value.Type == BooleanDataViewType.Instance)
                labelCardinality = 2;
            else
            {
                TrainerUtils.CheckMulticlassLabel(data, out labelCardinality);
                _host.CheckUserArg(labelCardinality > 1, nameof(_labelColumnName), "Label column type must have known cardinality more than 1");
            }
        }

        private static string[] InitLabelClassNames(IExceptionContext ectx, DataViewSchema.Column labelCol, int labelCardinality)
        {
            if (!labelCol.HasKeyValues())
                return null;

            VBuffer<ReadOnlyMemory<char>> keyNames = default;
            labelCol.GetKeyValues(ref keyNames);
            ectx.Check(keyNames.Length == labelCardinality);
            return keyNames.DenseValues().Select(name => name.ToString()).ToArray();
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);

            if (!inputSchema.TryFindColumn(_labelColumnName, out var labelCol))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "label", _labelColumnName);

            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                if (col.Kind == SchemaShape.Column.VectorKind.VariableVector)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, "known-size vector or scalar", col.GetTypeString());

                if (!col.IsKey || !col.ItemType.Equals(NumberDataViewType.UInt32))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, "vector or scalar of U4 key type", col.GetTypeString());

                // We supply slot names if the source is a single-value column, or if it has slot names.
                var newMetadataKinds = new List<SchemaShape.Column>();
                if (col.Kind == SchemaShape.Column.VectorKind.Scalar)
                    newMetadataKinds.Add(new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false));
                else if (col.Annotations.TryFindColumn(AnnotationUtils.Kinds.SlotNames, out var slotMeta))
                    newMetadataKinds.Add(slotMeta);
                var meta = new SchemaShape(newMetadataKinds);
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, meta);
            }

            return new SchemaShape(result.Values);
        }
    }

    internal sealed class CountTableTransformer : OneToOneTransformerBase, ITransformerWithDifferentMappingAtTrainingTime
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table", SignatureType = typeof(SignatureCountTableBuilder))]
            public ICountTableBuilderFactory CountTable = new CMCountTableBuilder.Options();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The coefficient with which to apply the prior smoothing to the features", ShortName = "prior")]
            public float PriorCoefficient = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Laplacian noise diversity/scale-parameter. Suggest keeping it less than 1.", ShortName = "laplace")]
            public float LaplaceScale = 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random generator for the laplacian noise.", ShortName = "seed")]
            public int Seed = 314489979;

            [Argument(ArgumentType.Required, HelpText = "Label column", ShortName = "label,lab", Purpose = SpecialPurpose.ColumnName)]
            public string LabelColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Optional model file to load counts from. If this is specified all other options are ignored.", ShortName = "inmodel, extfile")]
            public string InitialCountsModel;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Keep counts for all columns in one shared count table", ShortName = "shared")]
            public bool SharedTable = false;
        }

        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table", SignatureType = typeof(SignatureCountTableBuilder))]
            public ICountTableBuilderFactory CountTable;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The coefficient with which to apply the prior smoothing to the features", ShortName = "prior")]
            public float? PriorCoefficient;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Laplacian noise diversity/scale-parameter. Suggest keeping it less than 1.", ShortName = "laplace")]
            public float? LaplaceScale;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random generator for the laplacian noise.", ShortName = "seed")]
            public int? Seed;

            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (CountTable != null || PriorCoefficient != null || LaplaceScale != null || Seed != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        internal static class Defaults
        {
            public const float PriorCoefficient = 1;
            public const float LaplaceScale = 0;
            public const int Seed = 314489979;
            public const bool SharedTable = false;
        }

        internal readonly CountTargetEncodingFeaturizer Featurizer;
        private readonly string[] _labelClassNames;

        internal int[] Seeds { get; }

        internal const string Summary = "Transforms the categorical column into the set of features: count of each label class, "
            + "log-odds for each label class, back-off indicator. The input columns must be keys.";

        internal const string LoaderSignature = "CountTableTransform";
        internal const string UserName = "Count Table Transform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CNTTABLE",
                 verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CountTableTransformer).Assembly.FullName);
        }

        internal CountTableTransformer(IHostEnvironment env, CountTargetEncodingFeaturizer featurizer, string[] labelClassNames,
            int[] seeds, (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CountTableTransformer)), columns)
        {
            Host.AssertValue(featurizer);
            Host.AssertValueOrNull(labelClassNames);
            Host.Assert(Utils.Size(seeds) == featurizer.ColCount);

            Featurizer = featurizer;
            _labelClassNames = labelClassNames;
            Seeds = seeds;
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns));

            CountTableEstimator estimator;
            if (!string.IsNullOrEmpty(options.InitialCountsModel))
            {
                estimator = CountTargetEncodingEstimator.LoadFromFile(env, options.InitialCountsModel, options.LabelColumn,
                    options.Columns.Select(col => new InputOutputColumnPair(col.Name, col.Source)).ToArray());
                if (estimator == null)
                    throw env.Except($"The file {options.InitialCountsModel} does not contain a CountTableTransformer");
            }
            else if (!options.SharedTable)
            {
                var columnOptions = new CountTableEstimator.ColumnOptions[options.Columns.Length];
                for (int i = 0; i < options.Columns.Length; i++)
                {
                    var c = options.Columns[i];
                    var builder = c.CountTable ?? options.CountTable;
                    env.CheckValue(builder, nameof(options.CountTable));

                    columnOptions[i] = new CountTableEstimator.ColumnOptions(c.Name,
                        c.Source,
                        builder.CreateComponent(env),
                        c.PriorCoefficient ?? options.PriorCoefficient,
                        c.LaplaceScale ?? options.LaplaceScale,
                        c.Seed ?? options.Seed);
                }
                estimator = new CountTableEstimator(env, options.LabelColumn, columnOptions);
            }
            else
            {
                var columnOptions = new CountTableEstimator.SharedColumnOptions[options.Columns.Length];
                for (int i = 0; i < options.Columns.Length; i++)
                {
                    var c = options.Columns[i];
                    env.CheckUserArg(c.CountTable == null, nameof(c.CountTable), "Can't have non-default count tables if the tables are shared");

                    columnOptions[i] = new CountTableEstimator.SharedColumnOptions(c.Name,
                        c.Source,
                        c.PriorCoefficient ?? options.PriorCoefficient,
                        c.LaplaceScale ?? options.LaplaceScale,
                        c.Seed ?? options.Seed);
                }
                var builder = options.CountTable;
                env.CheckValue(builder, nameof(options.CountTable));
                estimator = new CountTableEstimator(env, options.LabelColumn, builder.CreateComponent(env), columnOptions);
            }

            return (estimator.Fit(input) as ITransformerWithDifferentMappingAtTrainingTime).TransformForTrainingPipeline(input) as IDataTransform;
        }

        // Factory method for SignatureLoadModel.
        internal static CountTableTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(LoaderSignature);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new CountTableTransformer(host, ctx);
        }

        private CountTableTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>
            // int: number of label class names
            // int[]: ids of label class names
            // for each added column:
            //   int: seed

            // Sub-models:
            // featurizer

            var lc = ctx.Reader.ReadInt32();
            Host.CheckDecode(lc >= 0);
            if (lc > 0)
            {
                _labelClassNames = new string[lc];
                for (int i = 0; i < lc; i++)
                {
                    _labelClassNames[i] = ctx.LoadNonEmptyString();
                }
            }

            Seeds = ctx.Reader.ReadIntArray(ColumnPairs.Length);
            ctx.LoadModel<CountTargetEncodingFeaturizer, SignatureLoadModel>(host, out Featurizer, "Featurizer");
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // int: number of label class names
            // int[]: ids of label class names
            // int[]: _seeds

            // Sub-models:
            // featurizer

            SaveColumns(ctx);

            ctx.Writer.Write(Utils.Size(_labelClassNames));
            if (_labelClassNames != null)
            {
                for (int i = 0; i < _labelClassNames.Length; i++)
                {
                    Host.Assert(!string.IsNullOrEmpty(_labelClassNames[i]));
                    ctx.SaveNonEmptyString(_labelClassNames[i]);
                }
            }

            ctx.Writer.WriteIntsNoCount(Seeds);
            ctx.SaveModel(Featurizer, "Featurizer");
        }

        IDataView ITransformerWithDifferentMappingAtTrainingTime.TransformForTrainingPipeline(IDataView input)
        {
            return new RowToRowMapperTransform(Host, input, MakeRowMapperWithNoise(input.Schema), MakeRowMapperWithNoise);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private IRowMapper MakeRowMapperWithNoise(DataViewSchema schema) => new Mapper(this, schema, true);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly CountTableTransformer _parent;
            private readonly bool _addNoise;

            public Mapper(CountTableTransformer parent, DataViewSchema schema, bool addNoise = false)
                : base(parent.Host.Register(nameof(Mapper)), parent, schema)
            {
                _parent = parent;
                _addNoise = addNoise;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var outputCols = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var inputCol = InputSchema[_parent.ColumnPairs[i].inputColumnName];
                    var valueCount = inputCol.Type.GetValueCount();
                    Host.Check((long)valueCount * _parent.Featurizer.NumFeatures < int.MaxValue, "Too large output size");
                    var type = new VectorDataViewType(NumberDataViewType.Single, valueCount, _parent.Featurizer.NumFeatures);

                    // We supply slot names if the source is a single-value column, or if it has slot names.
                    if (!(inputCol.Type is VectorDataViewType) || inputCol.HasSlotNames())
                    {
                        var builder = new DataViewSchema.Annotations.Builder();
                        var getSlotNames = GetSlotNamesGetter(inputCol, i);
                        builder.AddSlotNames(type.GetVectorSize(), getSlotNames);
                        outputCols[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, type, builder.ToAnnotations());
                    }
                    else
                        outputCols[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, type);
                }
                return outputCols;
            }

            private ValueGetter<VBuffer<ReadOnlyMemory<char>>> GetSlotNamesGetter(DataViewSchema.Column inputCol, int iinfo)
            {
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                VBuffer<ReadOnlyMemory<char>> inputSlotNames = default;
                if (inputCol.Type is VectorDataViewType)
                {
                    Host.Assert(inputCol.HasSlotNames());
                    inputCol.GetSlotNames(ref inputSlotNames);
                }
                else
                    inputSlotNames = new VBuffer<ReadOnlyMemory<char>>(1, new[] { inputCol.Name.AsMemory() });

                Host.Assert(inputSlotNames.Length == inputCol.Type.GetValueCount());

                VBuffer<ReadOnlyMemory<char>> featureNames = default;
                ValueGetter<VBuffer<ReadOnlyMemory<char>>> getter =
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        _parent.Featurizer.GetFeatureNames(_parent._labelClassNames, ref featureNames);
                        int nFeatures = featureNames.Length;

                        var editor = VBufferEditor.Create(ref dst, nFeatures * inputSlotNames.Length);
                        var featureNamesValues = featureNames.GetValues();
                        foreach (var pair in inputSlotNames.Items(true))
                        {
                            int i = pair.Key;
                            var slotName = pair.Value.ToString();
                            for (int j = 0; j < nFeatures; j++)
                            {
                                editor.Values[i * nFeatures + j] = $"{slotName}_{featureNamesValues[j]}".AsMemory();
                            }
                        }

                        dst = editor.Commit();
                    };
                return getter;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                if (input.Schema[_parent.ColumnPairs[iinfo].inputColumnName].Type is VectorDataViewType)
                    return ConstructVectorGetter(input, iinfo);
                return ConstructSingleGetter(input, iinfo);
            }

            private ValueGetter<VBuffer<float>> ConstructSingleGetter(DataViewRow input, int iinfo)
            {
                Host.Assert(_parent.Featurizer.SlotCount[iinfo] == 1);
                uint src = 0;
                var srcGetter = input.GetGetter<uint>(input.Schema[_parent.ColumnPairs[iinfo].inputColumnName]);
                var outputLength = _parent.Featurizer.NumFeatures;
                var rand = _addNoise ? new Random(_parent.Seeds[iinfo]) : null;
                var featurizer = _parent.Featurizer;
                return (ref VBuffer<float> dst) =>
                {
                    srcGetter(ref src);
                    var editor = VBufferEditor.Create(ref dst, outputLength);
                    featurizer.GetFeatures(iinfo, 0, rand, src, editor.Values);
                    dst = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<float>> ConstructVectorGetter(DataViewRow input, int iinfo)
            {
                var inputCol = input.Schema[_parent.ColumnPairs[iinfo].inputColumnName];
                int n = inputCol.Type.GetValueCount();
                Host.Assert(_parent.Featurizer.SlotCount[iinfo] == n);
                VBuffer<uint> src = default;

                var outputLength = _parent.Featurizer.NumFeatures;
                var srcGetter = input.GetGetter<VBuffer<uint>>(inputCol);
                var rand = _addNoise ? new Random(_parent.Seeds[iinfo]) : null;
                return (ref VBuffer<float> dst) =>
                {
                    srcGetter(ref src);
                    var editor = VBufferEditor.Create(ref dst, n * outputLength);
                    if (src.IsDense)
                    {
                        var srcValues = src.GetValues();
                        for (int i = 0; i < n; i++)
                            _parent.Featurizer.GetFeatures(iinfo, i, rand, srcValues[i], editor.Values.Slice(i * outputLength, outputLength));
                    }
                    else
                    {
                        var srcValues = src.GetValues();
                        var srcIndices = src.GetIndices();
                        editor.Values.Clear();
                        for (int i = 0; i < srcIndices.Length; i++)
                        {
                            var index = srcIndices[i];
                            _parent.Featurizer.GetFeatures(iinfo, index, rand, srcValues[i], editor.Values.Slice(index * outputLength, outputLength));
                        }
                    }

                    dst = editor.Commit();
                };
            }
        }
    }

    internal static class CountTable
    {
        [TlcModule.EntryPoint(Name = "Transforms.CountTableBuilder", Desc = CountTableTransformer.Summary, UserName = CountTableTransformer.UserName, ShortName = "Count")]
        internal static CommonOutputs.TransformOutput Create(IHostEnvironment env, CountTableTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "CountTable", input);
            var view = CountTableTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
