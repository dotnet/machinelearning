// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
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
    public sealed class CountTableEstimator : IEstimator<CountTableTransformer>
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
        private readonly string _externalCountsFile;
        private readonly string _externalCountsSchema;

        internal CountTableEstimator(IHostEnvironment env, string labelColumnName, CountTableBuilderBase countTableBuilder, params SharedColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(labelColumnName, nameof(labelColumnName));
            env.CheckNonEmpty(columns, nameof(columns));
            _host = env.Register(nameof(CountTableEstimator));

            _columns = columns;
            _labelColumnName = labelColumnName;
            _sharedBuilder = countTableBuilder;
        }

        internal CountTableEstimator(IHostEnvironment env, string labelColumnName, string externalCountsFile = null,
                string externalCountsSchema = null, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(labelColumnName, nameof(labelColumnName));
            env.CheckNonEmpty(columns, nameof(columns));
            _host = env.Register(nameof(CountTableEstimator));

            _columns = columns;
            _labelColumnName = labelColumnName;
            _externalCountsFile = externalCountsFile;
            _externalCountsSchema = externalCountsSchema;
            _builders = columns.Select(c => c.CountTableBuilder).ToArray();
        }

        public CountTableTransformer Fit(IDataView input)
        {
            var labelCol = input.Schema.GetColumnOrNull(_labelColumnName);
            if (labelCol == null)
                throw _host.ExceptUserArg(nameof(_labelColumnName), "Label column '{0}' not found", _labelColumnName);

            var labelColumnType = labelCol.GetValueOrDefault().Type;
            CheckLabelType(labelColumnType, out var labelCardinality);

            var labelClassNames = InitLabelClassNames(_host, labelCol.GetValueOrDefault(), labelCardinality);

            var n = _columns.Length;

            //var builderArgs = _columns.Select(c => c.CountTableBuilder).ToArray();
            var inputColumns = new DataViewSchema.Column[_columns.Length];
            for (int i = 0; i < inputColumns.Length; i++)
            {
                var col = input.Schema.GetColumnOrNull(_columns[i].InputColumnName);
                if (col == null)
                    throw _host.Except($"Could not find column {_columns[i].InputColumnName} in input schema");
                inputColumns[i] = col.GetValueOrDefault();
            }

            _host.Assert((_sharedBuilder == null) != (_builders == null));
            IMultiCountTableBuilder multiBuilder;
            if (_builders != null)
                multiBuilder = new ParallelMultiCountTableBuilder(_host, inputColumns, _builders, labelCardinality, _externalCountsFile, _externalCountsSchema);
            else
                multiBuilder = new BagMultiCountTableBuilder(_host, _sharedBuilder, labelCardinality);

            var cols = new List<DataViewSchema.Column>();
            foreach (var c in _columns)
            {
                var col = input.Schema.GetColumnOrNull(c.InputColumnName);
                _host.Assert(col.HasValue);
                cols.Add(col.Value);
            }

            TrainTables(input, cols, multiBuilder, labelCol.GetValueOrDefault());

            var multiCountTable = multiBuilder.CreateMultiCountTable();

            // create featurizers
            var featurizers = new DraculaFeaturizer[n][];
            for (int i = 0; i < n; i++)
            {
                int size = cols[i].Type.GetValueCount();
                _host.Assert(size > 0);
                featurizers[i] = new DraculaFeaturizer[size];
                for (int j = 0; j < size; j++)
                {
                    featurizers[i][j] = new DraculaFeaturizer(_host, new DraculaFeaturizer.Options()
                    { LaplaceScale = _columns[i].LaplaceScale, PriorCoefficient = _columns[i].PriorCoefficient },
                    labelCardinality, multiCountTable.GetCountTable(i, j));
                }
            }

            return new CountTableTransformer(_host, featurizers, labelClassNames,
                _columns.Select(col => col.Seed).ToArray(), _columns.Select(col => (col.Name, col.InputColumnName)).ToArray());
        }

        private void TrainTables(IDataView trainingData, List<DataViewSchema.Column> cols, IMultiCountTableBuilder builder, DataViewSchema.Column labelColumn)
        {
            var colCount = _columns.Length;
            //// creating a cursor over columns we need
            //bool[] activeInput = new bool[Source.Schema.ColumnCount];
            //foreach (var colInfo in Infos)
            //    activeInput[colInfo.Source] = true;
            //activeInput[labelColumnIndex] = true;

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
                            builder.IncrementOne(i, srcSingleValue, (uint)labelKey, 1);
                        }
                    }
                }
            }
        }

        private ValueGetter<long> GetLabelGetter(DataViewRow row, DataViewSchema.Column col)
        {
            // The label column type is checked as part of args validation.
            var type = col.Type;
            _host.Assert(type is KeyDataViewType || type is NumberDataViewType);

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
                        if (src > 0)
                            dst = 1;
                        else if (src <= 0)
                            dst = 0;
                        else
                            dst = -1;
                    };
            }
        }

        private void IncrementVec(IMultiCountTableBuilder builder, int iCol, ref VBuffer<uint> srcBuffer, uint labelKey)
        {
            var n = srcBuffer.Length;
            var values = srcBuffer.GetValues();
            var indices = srcBuffer.GetIndices();
            if (srcBuffer.IsDense)
            {
                for (int i = 0; i < n; i++)
                    builder.IncrementSlot(iCol, i, values[i], labelKey, 1);
            }
            else
            {
                for (int i = 0; i < indices.Length; i++)
                    builder.IncrementSlot(iCol, indices[i], values[i], labelKey, 1);
            }
        }

        private void CheckLabelType(DataViewType labelColumnType, out int labelCardinality)
        {
            if (labelColumnType is NumberDataViewType)
                labelCardinality = 2;
            else if (labelColumnType is KeyDataViewType)
            {
                labelCardinality = labelColumnType.GetKeyCountAsInt32();
                _host.CheckUserArg(labelCardinality > 1, nameof(_labelColumnName), "Label column type must have known cardinality more than 1");
            }
            else
                throw _host.ExceptUserArg(nameof(labelColumnType), "Incorrect label column type: expected numeric or key type");
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
            throw new NotImplementedException();
        }
    }

    public sealed class CountTableTransformer : OneToOneTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table", SignatureType = typeof(SignatureCountTableBuilder))]
            public IComponentFactory<CountTableBuilderBase> CountTable = new CMCountTableBuilder.Arguments();

            [Argument(ArgumentType.AtMostOnce, HelpText = "The coefficient with which to apply the prior smoothing to the features", ShortName = "prior")]
            public float PriorCoefficient = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Laplacian noise diversity/scale-parameter. Suggest keeping it less than 1.", ShortName = "laplace")]
            public float LaplaceScale = 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random generator for the laplacian noise.", ShortName = "seed")]
            public int Seed = 314489979;

            [Argument(ArgumentType.Required, HelpText = "Label column", ShortName = "label,lab", Purpose = SpecialPurpose.ColumnName)]
            public string LabelColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Optional text file to load counts from", ShortName = "extfile")]
            public string ExternalCountsFile;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma-separated list of column IDs in the external count file", ShortName = "extschema")]
            public string ExternalCountsSchema;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Keep counts for all columns in one shared count table", ShortName = "shared")]
            public bool SharedTable = false;
        }

        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table", SignatureType = typeof(SignatureCountTableBuilder))]
            public IComponentFactory<CountTableBuilderBase> CountTable;

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

        private readonly DraculaFeaturizer[][] _featurizers; // parallel to count tables
        private readonly string[] _labelClassNames;
        private readonly int[] _seeds;

        internal const string Summary = "Transforms the categorical column into the set of features: count of each label class, "
            + "log-odds for each label class, back-off indicator. The input columns must be keys. This is a part of the Dracula transform.";

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

        internal CountTableTransformer(IHostEnvironment env, DraculaFeaturizer[][] featurizers, string[] labelClassNames,
            int[] seeds, (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CountTableTransformer)), columns)
        {
            Host.AssertNonEmpty(featurizers);
            Host.AssertValueOrNull(labelClassNames);
            Host.Assert(Utils.Size(seeds) == featurizers.Length);

            _featurizers = featurizers;
            _labelClassNames = labelClassNames;
            _seeds = seeds;
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
            if (!options.SharedTable)
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
                estimator = new CountTableEstimator(env, options.LabelColumn, options.ExternalCountsFile, options.ExternalCountsSchema, columnOptions);
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

            return estimator.Fit(input).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static CountTableTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
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
            //   int: # of slots

            // Sub-models:
            // featurizers (each in a separate folder)

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

            _featurizers = new DraculaFeaturizer[ColumnPairs.Length][];
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                var size = ctx.Reader.ReadInt32();
                Host.CheckDecode(size > 0);
                _featurizers[i] = new DraculaFeaturizer[size];
                for (int j = 0; j < size; j++)
                {
                    var featurizerName = string.Format("Feat_{0:000}_{1:000}", i, j);
                    ctx.LoadModel<DraculaFeaturizer, SignatureLoadModel>(Host, out _featurizers[i][j], featurizerName);
                }
            }
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
            // for each added column:
            //   int: # of slots

            // Sub-models:
            // featurizers (each in a separate folder)

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

            for (int i = 0; i < _featurizers.Length; i++)
            {
                var size = _featurizers[i].Length;
                Host.Assert(size > 0);
                ctx.Writer.Write(size);
                for (int j = 0; j < size; j++)
                {
                    var featurizerName = string.Format("Feat_{0:000}_{1:000}", i, j);
                    ctx.SaveSubModel(featurizerName, context => _featurizers[i][j].Save(context));
                    //ctx.SaveModel(_featurizers[i][j], featurizerName);
                }
            }
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly CountTableTransformer _parent;
            public Mapper(CountTableTransformer parent, DataViewSchema schema)
                : base(parent.Host.Register(nameof(Mapper)), parent, schema)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var outputCols = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var featurizer = _parent._featurizers[i][0];
                    var inputCol = InputSchema[_parent.ColumnPairs[i].inputColumnName];
                    var valueCount = inputCol.Type.GetValueCount();
                    Host.Check((long)valueCount * featurizer.NumFeatures < int.MaxValue, "Too large output size");
                    var type = new VectorDataViewType(NumberDataViewType.Single, valueCount, featurizer.NumFeatures);

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
                        _parent._featurizers[iinfo][0].GetFeatureNames(_parent._labelClassNames, ref featureNames);
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
                Host.Assert(Utils.Size(_parent._featurizers[iinfo]) == 1);
                uint src = 0;
                var srcGetter = input.GetGetter<uint>(input.Schema[_parent.ColumnPairs[iinfo].inputColumnName]);
                var outputLength = _parent._featurizers[iinfo][0].NumFeatures;
                var rand = new Random(_parent._seeds[iinfo]);
                return (ref VBuffer<float> dst) =>
                {
                    srcGetter(ref src);
                    var editor = VBufferEditor.Create(ref dst, outputLength);
                    _parent._featurizers[iinfo][0].GetFeatures(rand, src, editor.Values);
                    dst = editor.Commit();
                };
            }

            private ValueGetter<VBuffer<float>> ConstructVectorGetter(DataViewRow input, int iinfo)
            {
                var inputCol = input.Schema[_parent.ColumnPairs[iinfo].inputColumnName];
                int n = inputCol.Type.GetValueCount();
                Host.Assert(Utils.Size(_parent._featurizers[iinfo]) == n);
                VBuffer<uint> src = default;

                var outputLength = _parent._featurizers[iinfo][0].NumFeatures;
                var srcGetter = input.GetGetter<VBuffer<uint>>(inputCol);
                var rand = new Random(_parent._seeds[iinfo]);
                return (ref VBuffer<float> dst) =>
                {
                    srcGetter(ref src);
                    var editor = VBufferEditor.Create(ref dst, n * outputLength);
                    if (src.IsDense)
                    {
                        var srcValues = src.GetValues();
                        for (int i = 0; i < n; i++)
                            _parent._featurizers[iinfo][i].GetFeatures(rand, srcValues[i], editor.Values.Slice(i * outputLength, outputLength));
                    }
                    else
                    {
                        var srcValues = src.GetValues();
                        var srcIndices = src.GetIndices();
                        editor.Values.Clear();
                        for (int i = 0; i < srcIndices.Length; i++)
                        {
                            var index = srcIndices[i];
                            _parent._featurizers[iinfo][index].GetFeatures(rand, srcValues[i], editor.Values.Slice(index * outputLength, outputLength));
                        }
                    }

                    dst = editor.Commit();
                };
            }
        }
    }

    //public class CountTableTransform : OneToOneTransformBase, ITransformTemplate
    //{
    //    public sealed class Column : OneToOneColumn
    //    {
    //        [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table")]
    //        public ICountTableBuilderFactory CountTable;

    //        [Argument(ArgumentType.AtMostOnce, HelpText = "The coefficient with which to apply the prior smoothing to the features", ShortName = "prior")]
    //        public float? PriorCoefficient;

    //        [Argument(ArgumentType.AtMostOnce, HelpText = "Laplacian noise diversity/scale-parameter. Suggest keeping it less than 1.", ShortName = "laplace")]
    //        public float? LaplaceScale;

    //        [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random generator for the laplacian noise.", ShortName = "seed")]
    //        public int? Seed;

    //        public static Column Parse(string str)
    //        {
    //            var res = new Column();
    //            if (res.TryParse(str))
    //                return res;
    //            return null;
    //        }

    //        public bool TryUnparse(StringBuilder sb)
    //        {
    //            Contracts.AssertValue(sb);
    //            if (CountTable != null || PriorCoefficient != null || LaplaceScale != null || Seed != null)
    //                return false;
    //            return TryUnparseCore(sb);
    //        }
    //    }

    //    public sealed class Arguments : TransformInputBase
    //    {
    //        [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
    //        public Column[] Column;

    //        [Argument(ArgumentType.Multiple, HelpText = "Count table settings", ShortName = "table")]
    //        public ICountTableBuilderFactory CountTable = new CMCountTableBuilder.Arguments();

    //        [Argument(ArgumentType.AtMostOnce, HelpText = "The coefficient with which to apply the prior smoothing to the features", ShortName = "prior")]
    //        public float PriorCoefficient = 1;

    //        [Argument(ArgumentType.AtMostOnce, HelpText = "Laplacian noise diversity/scale-parameter. Suggest keeping it less than 1.", ShortName = "laplace")]
    //        public float LaplaceScale = 0;

    //        [Argument(ArgumentType.AtMostOnce, HelpText = "Seed for the random generator for the laplacian noise.", ShortName = "seed")]
    //        public int Seed = 314489979;

    //        [Argument(ArgumentType.Required, HelpText = "Label column", ShortName = "label,lab", Purpose = SpecialPurpose.ColumnName)]
    //        public string LabelColumn;

    //        [Argument(ArgumentType.AtMostOnce, HelpText = "Optional text file to load counts from", ShortName = "extfile")]
    //        public string ExternalCountsFile;

    //        [Argument(ArgumentType.AtMostOnce, HelpText = "Comma-separated list of column IDs in the external count file", ShortName = "extschema")]
    //        public string ExternalCountsSchema;

    //        [Argument(ArgumentType.AtMostOnce, HelpText = "Keep counts for all columns in one shared count table", ShortName = "shared")]
    //        public bool SharedTable = false;
    //    }

    //    internal const string Summary = "Transforms the categorical column into the set of features: count of each label class, "
    //        + "log-odds for each label class, back-off indicator. The input columns must be keys. This is a part of the Dracula transform.";

    //    internal const string UserName = "Count Table Transform";
    //    internal const string ShortName = "count";

    //    private readonly ColumnType[] _columnTypes;
    //    private readonly string[] _labelClassNames;
    //    private readonly string[][] _savedColumnFeatureNames; // the cached feature names for columns, provided by featurizers

    //    private readonly ICountFeaturizer[][] _featurizers; // parallel to count tables
    //    private readonly int _labelCardinality; // number of different values label column can have
    //    private readonly IMultiCountTable _multiCountTable;

    //    private const string RegistrationName = "CountTable";

    //    public CountTableTransform(IHostEnvironment env, Arguments args, IDataView input)
    //        : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column,
    //            input, TestColumnType)
    //    {
    //        Host.AssertNonEmpty(Infos);
    //        Host.Assert(Utils.Size(Infos) == Utils.Size(args.Column));
    //        Host.CheckUserArg(!string.IsNullOrWhiteSpace(args.LabelColumn), nameof(args.LabelColumn), "Must specify the label column name");

    //        int labelColumnIndex;
    //        if (!input.Schema.TryGetColumnIndex(args.LabelColumn, out labelColumnIndex))
    //            throw Host.ExceptUserArg(nameof(args.LabelColumn), "Label column '{0}' not found", args.LabelColumn);

    //        var labelColumnType = input.Schema.GetColumnType(labelColumnIndex);
    //        CheckLabelType(labelColumnType, out _labelCardinality);

    //        InitLabelClassNames(Host, Source.Schema, args.LabelColumn, _labelCardinality, out _labelClassNames);

    //        var n = Infos.Length;

    //        var builderArgs = args.Column.Select(c => c.CountTable).ToArray();
    //        IMultiCountTableBuilder multiBuilder;
    //        if (!args.SharedTable)
    //        {
    //            multiBuilder = new ParallelMultiCountTableBuilder(Host, Infos, builderArgs, args.CountTable,
    //                _labelCardinality);
    //            if (!string.IsNullOrEmpty(args.ExternalCountsFile))
    //                ((ParallelMultiCountTableBuilder)multiBuilder).LoadExternalCounts(args.ExternalCountsFile,
    //                    args.ExternalCountsSchema, _labelCardinality);
    //        }
    //        else
    //        {
    //            Host.CheckUserArg(args.Column.All(c => c.CountTable == null), nameof(args.Column), "Can't have non-default count tables if the tables are shared");
    //            multiBuilder = new BagMultiCountTableBuilder(Host, args.CountTable, _labelCardinality);
    //        }

    //        using (var ch = Host.Start("Training count tables"))
    //        {
    //            TrainTables(ch, input, multiBuilder, labelColumnIndex);
    //            ch.Done();
    //        }

    //        _multiCountTable = multiBuilder.CreateMultiCountTable();

    //        // create featurizers
    //        _featurizers = new ICountFeaturizer[n][];
    //        for (int i = 0; i < n; i++)
    //        {
    //            int size = Infos[i].TypeSrc.ValueCount;
    //            Host.Assert(size > 0);
    //            _featurizers[i] = new ICountFeaturizer[size];
    //            for (int j = 0; j < size; j++)
    //                _featurizers[i][j] = CreateFeaturizer(Host, args.Column[i].Featurizer ?? args.Featurizer, _multiCountTable.GetCountTable(i, j));
    //        }

    //        _columnTypes = GenerateColumnTypesAndMetadata();
    //        _savedColumnFeatureNames = new string[Infos.Length][];
    //    }

    //    #region Serialization
    //    public const string LoaderSignature = "CountTableTransform";
    //    internal const string LoaderSignatureOld = "CountTableFunction";

    //    private static VersionInfo GetVersionInfo()
    //    {
    //        return new VersionInfo(
    //            modelSignature: "CNTTBL F",
    //            // verWrittenCur: 0x00010001, // Initial
    //            // verWrittenCur: 0x00010002, // Added slot names and multinomial support
    //            // verWrittenCur: 0x00010003, // label cardinality
    //            // verWrittenCur: 0x00010004, // Single-only tables
    //            verWrittenCur: 0x00010005, // Multi-tables
    //            verReadableCur: 0x00010005,
    //            verWeCanReadBack: 0x00010005,
    //            loaderSignature: LoaderSignature,
    //            loaderSignatureAlt: LoaderSignatureOld);
    //    }

    //    public static CountTableTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
    //    {
    //        Contracts.CheckValue(env, nameof(env));
    //        var h = env.Register(RegistrationName);
    //        h.CheckValue(ctx, nameof(ctx));
    //        h.CheckValue(input, nameof(input));
    //        ctx.CheckAtModel(GetVersionInfo());
    //        return h.Apply("Loading Model",
    //            ch =>
    //            {
    //            // *** Binary format ***
    //            // int: label cardinality
    //            // <remainder handled in ctors>
    //            int labelCardinality = ctx.Reader.ReadInt32();
    //                ch.CheckDecode(labelCardinality > 1);
    //                return new CountTableTransform(h, ctx, input, labelCardinality);
    //            });
    //    }

    //    private CountTableTransform(IHostEnvironment env, CountTableTransform transform, IDataView newSource)
    //        : base(env, RegistrationName, transform, newSource, TestColumnType)
    //    {
    //        Host.AssertValue(transform, "transform");

    //        // REVIEW petelu(nihejazi): compute label class names for the new source (here and when deserializing).
    //        _labelClassNames = transform._labelClassNames;
    //        _labelCardinality = transform._labelCardinality;
    //        _multiCountTable = transform._multiCountTable;
    //        _featurizers = transform._featurizers;

    //        _columnTypes = GenerateColumnTypesAndMetadata();
    //        _savedColumnFeatureNames = new string[Infos.Length][];
    //    }

    //    private CountTableTransform(IHost host, ModelLoadContext ctx, IDataView input, int labelCardinality)
    //        : base(host, ctx, input, TestColumnType)
    //    {
    //        Host.AssertValue(ctx);
    //        Host.AssertNonEmpty(Infos);
    //        Host.Assert(labelCardinality > 1);

    //        // *** Binary format ***
    //        // <prefix handled in static Create method>
    //        // <base>
    //        // int: number of label class names (0 or the same as label cardinality)
    //        // int[]: ids of label class names
    //        // for each added column:
    //        //   int: # of slots

    //        // Sub-models:
    //        // multi-count-table (it saves one or more sub-models for count tables)
    //        // featurizers (each in a separate folder)
    //        _labelCardinality = labelCardinality;

    //        var lc = ctx.Reader.ReadInt32();
    //        Host.CheckDecode(lc == 0 || lc == _labelCardinality);
    //        if (lc > 0)
    //        {
    //            _labelClassNames = new string[lc];
    //            for (int i = 0; i < lc; i++)
    //            {
    //                _labelClassNames[i] = ctx.LoadString();
    //                Host.Assert(_labelClassNames[i] != null);
    //            }
    //        }
    //        ctx.LoadModel<IMultiCountTable, SignatureLoadModel>(Host, out _multiCountTable, "CountTable");

    //        _featurizers = new ICountFeaturizer[Infos.Length][];
    //        for (int i = 0; i < Infos.Length; i++)
    //        {
    //            var size = ctx.Reader.ReadInt32();
    //            Host.CheckDecode(size > 0);
    //            Host.CheckDecode(size == Infos[i].TypeSrc.ValueCount);
    //            _featurizers[i] = new ICountFeaturizer[size];
    //            for (int j = 0; j < size; j++)
    //            {
    //                var featurizerName = string.Format("Feat_{0:000}_{1:000}", i, j);
    //                ctx.LoadModel<ICountFeaturizer, SignatureLoadCountFeaturizer>(Host, out _featurizers[i][j], featurizerName, _multiCountTable.GetCountTable(i, j));
    //            }
    //        }

    //        _columnTypes = GenerateColumnTypesAndMetadata();
    //        _savedColumnFeatureNames = new string[Infos.Length][];
    //    }

    //    public override void Save(ModelSaveContext ctx)
    //    {
    //        Host.CheckValue(ctx, nameof(ctx));
    //        ctx.CheckAtModel();
    //        ctx.SetVersionInfo(GetVersionInfo());

    //        // *** Binary format ***
    //        // int: label cardinality
    //        // <base>
    //        // int: number of label class names (0 or the same as label cardinality)
    //        // int[]: ids of label class names
    //        // for each added column:
    //        //   int: # of slots

    //        // Sub-models:
    //        // multi-count-table (it saves one or more sub-models for count tables)
    //        // featurizers (each in a separate folder)

    //        Host.Assert(_labelCardinality > 1);
    //        ctx.Writer.Write(_labelCardinality);
    //        SaveBase(ctx);

    //        ctx.Writer.Write(Utils.Size(_labelClassNames));
    //        if (_labelClassNames != null)
    //        {
    //            Host.Assert(_labelClassNames.Length == _labelCardinality);
    //            for (int i = 0; i < _labelClassNames.Length; i++)
    //            {
    //                Host.Assert(_labelClassNames[i] != null);
    //                ctx.SaveString(_labelClassNames[i]);
    //            }
    //        }

    //        ctx.SaveModel(_multiCountTable, "CountTable");

    //        for (int i = 0; i < _featurizers.Length; i++)
    //        {
    //            var size = _featurizers[i].Length;
    //            Host.Assert(size > 0);
    //            Host.Assert(size == Infos[i].TypeSrc.ValueCount);
    //            ctx.Writer.Write(size);
    //            for (int j = 0; j < size; j++)
    //            {
    //                var featurizerName = string.Format("Feat_{0:000}_{1:000}", i, j);
    //                ctx.SaveModel(_featurizers[i][j], featurizerName);
    //            }
    //        }
    //    }

    //    public IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
    //    {
    //        Host.CheckValue(env, nameof(env));
    //        Host.CheckValue(newSource, nameof(newSource));

    //        return new CountTableTransform(env, this, newSource);
    //    }

    //    #endregion Serialization

    //    private void CheckLabelType(ColumnType labelColumnType, out int labelCardinality)
    //    {
    //        if (labelColumnType.IsNumber)
    //        {
    //            labelCardinality = 2;
    //        }
    //        else if (labelColumnType.IsKey)
    //        {
    //            labelCardinality = labelColumnType.KeyCount;
    //            Host.CheckUserArg(labelCardinality > 1, nameof(Arguments.LabelColumn), "Label column type must have known cardinality more than 1");
    //        }
    //        else
    //        {
    //            throw Host.ExceptUserArg(nameof(Arguments.LabelColumn), "Incorrect label column type: expected numeric or key type");
    //        }
    //    }

    //    private ICountFeaturizer CreateFeaturizer(IHostEnvironment env, ICountFeaturizerFactory featurizerArgs, ICountTable countTable)
    //    {
    //        return featurizerArgs.CreateComponent(env,
    //            _labelCardinality, // extra arg1: # of label bins
    //            countTable);  // extra arg2: count table
    //    }

    //    private void TrainTables(IChannel ch, IDataView trainingData, IMultiCountTableBuilder builder, int labelColumnIndex)
    //    {
    //        var colCount = Infos.Length;

    //        // creating a cursor over columns we need
    //        bool[] activeInput = new bool[Source.Schema.ColumnCount];
    //        foreach (var colInfo in Infos)
    //            activeInput[colInfo.Source] = true;
    //        activeInput[labelColumnIndex] = true;

    //        double rowCount = trainingData.GetRowCount(true) ?? double.NaN;
    //        long rowCur = 0;
    //        using (var pch = Host.StartProgressChannel("Training tables"))
    //        using (var cursor = trainingData.GetRowCursor(col => activeInput[col]))
    //        {
    //            var header = new ProgressHeader(new[] { "rows" });
    //            pch.SetHeader(header, e => { e.SetProgress(0, rowCur, rowCount); });
    //            // populate getters
    //            var singleGetters = new ValueGetter<uint>[colCount];
    //            var vectorGetters = new ValueGetter<VBuffer<uint>>[colCount];
    //            for (int i = 0; i < colCount; i++)
    //            {
    //                if (Infos[i].TypeSrc.IsVector)
    //                    vectorGetters[i] = cursor.GetGetter<VBuffer<uint>>(Infos[i].Source);
    //                else
    //                    singleGetters[i] = cursor.GetGetter<uint>(Infos[i].Source);
    //            }

    //            var labelGetter = GetLabelGetter(cursor, labelColumnIndex);
    //            long labelKey = 0;
    //            uint srcSingleValue = 0;
    //            var srcBuffer = default(VBuffer<uint>);
    //            while (cursor.MoveNext())
    //            {
    //                labelGetter(ref labelKey);
    //                if (labelKey < 0) // Invalid label, skip the data
    //                    continue;
    //                for (int i = 0; i < colCount; i++)
    //                {
    //                    var colInfo = Infos[i];
    //                    if (colInfo.TypeSrc.IsVector)
    //                    {
    //                        vectorGetters[i](ref srcBuffer);
    //                        Host.Check(srcBuffer.Length == colInfo.TypeSrc.VectorSize, "value count mismatch");
    //                        IncrementVec(builder, i, ref srcBuffer, (uint)labelKey);
    //                    }
    //                    else
    //                    {
    //                        singleGetters[i](ref srcSingleValue);
    //                        builder.IncrementOne(i, srcSingleValue, (uint)labelKey, 1);
    //                    }
    //                }
    //                rowCur++;
    //            }
    //            pch.Checkpoint(rowCur);
    //        }
    //    }

    //    private static void InitLabelClassNames(IExceptionContext ectx, ISchema schema, string labelColumn, int labelCardinality, out string[] labelClassNames)
    //    {
    //        int labelColId;
    //        const string defaultClassNameTemplate = "Class{0:000}";
    //        Contracts.Check(schema.TryGetColumnIndex(labelColumn, out labelColId));
    //        var mType = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, labelColId);

    //        if (mType != null && mType.IsVector && mType.VectorSize == labelCardinality)
    //        {
    //            VBuffer<DvText> keyNames = default(VBuffer<DvText>);
    //            schema.GetMetadata(MetadataUtils.Kinds.KeyValues, labelColId, ref keyNames);
    //            ectx.Check(keyNames.Length == labelCardinality);
    //            labelClassNames = keyNames.Items(true)
    //                .Select(
    //                    pair => !pair.Value.HasChars
    //                        ? string.Format(defaultClassNameTemplate, pair.Key)
    //                        : pair.Value.ToString())
    //                .ToArray();
    //            ectx.Assert(labelClassNames.Length == labelCardinality);
    //        }
    //        else
    //            labelClassNames = null;
    //    }

    //    /// <summary>
    //    /// This method is called once all the featurizers are created, and therefore we know the # and names of individual columns
    //    /// </summary>
    //    public ColumnType[] GenerateColumnTypesAndMetadata()
    //    {
    //        Host.AssertNonEmpty(Infos);

    //        var md = Metadata;
    //        var types = new ColumnType[Infos.Length];
    //        for (int i = 0; i < Infos.Length; i++)
    //        {
    //            var featurizer = GetFeaturizerForColumn(i);
    //            var valueCount = Infos[i].TypeSrc.ValueCount;
    //            Host.Check((long)valueCount * featurizer.NumFeatures < int.MaxValue, "Too large output size");
    //            types[i] = new VectorType(NumberType.R4, valueCount, featurizer.NumFeatures);

    //            // We supply slot names if the source is a single-value column, or if it has slot names.
    //            if (!Infos[i].TypeSrc.IsVector || IsValidSlotNameType(i))
    //            {
    //                using (var bldr = md.BuildMetadata(i))
    //                {
    //                    bldr.AddGetter<VBuffer<DvText>>(MetadataUtils.Kinds.SlotNames,
    //                        MetadataUtils.GetNamesType(types[i].VectorSize), GetSlotNames);
    //                }
    //            }
    //        }
    //        md.Seal();
    //        return types;
    //    }

    //    private static string TestColumnType(ColumnType type)
    //    {
    //        // we accept V<Key<U4>, KnownSize> and Key<U4>
    //        if (type.ValueCount > 0 && type.ItemType.IsKey && type.ItemType.RawKind == DataKind.U4)
    //            return null;
    //        return "Expected U4 Key type or vector of U4 Key type";
    //    }

    //    protected override ColumnType GetColumnTypeCore(int iinfo)
    //    {
    //        Host.Assert(iinfo >= 0 && iinfo < Infos.Length);
    //        Host.Assert(Utils.Size(_columnTypes) == Infos.Length);
    //        return _columnTypes[iinfo];
    //    }

    //    #region Metadata
    //    private void GetSlotNames(int iinfo, ref VBuffer<DvText> dst)
    //    {
    //        Host.Assert(0 <= iinfo && iinfo < Infos.Length);

    //        VBuffer<DvText> inputSlotNames = default(VBuffer<DvText>);
    //        if (Infos[iinfo].TypeSrc.IsVector)
    //            Source.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, Infos[iinfo].Source, ref inputSlotNames);
    //        else
    //            inputSlotNames = new VBuffer<DvText>(1, new[] { new DvText(Source.Schema.GetColumnName(Infos[iinfo].Source)) });

    //        Host.Check(inputSlotNames.Length == Infos[iinfo].TypeSrc.ValueCount, "unexpected number of slot names");

    //        string[] featureNames;
    //        GetColumnFeatureNames(iinfo, out featureNames);
    //        int nFeatures = featureNames.Length;

    //        DvText[] outputSlotNames;
    //        if (dst.Count >= nFeatures * inputSlotNames.Length)
    //            outputSlotNames = dst.Values;
    //        else
    //            outputSlotNames = new DvText[nFeatures * inputSlotNames.Length];

    //        foreach (var pair in inputSlotNames.Items(true))
    //        {
    //            int i = pair.Key;
    //            var slotName = pair.Value.ToString();
    //            for (int j = 0; j < nFeatures; j++)
    //            {
    //                outputSlotNames[i * nFeatures + j] = new DvText(string.Format("{0}_{1}", slotName, featureNames[j]));
    //            }
    //        }

    //        dst = new VBuffer<DvText>(nFeatures * inputSlotNames.Length, outputSlotNames, dst.Indices);
    //    }

    //    private ICountFeaturizer GetFeaturizerForColumn(int iinfo)
    //    {
    //        Host.Assert(0 <= iinfo && iinfo < Infos.Length);
    //        Host.Assert(iinfo < _featurizers.Length);
    //        Host.Assert(_featurizers[iinfo].Length >= 1);
    //        Host.AssertValue(_featurizers[iinfo][0]);
    //        return _featurizers[iinfo][0];
    //    }

    //    private void GetColumnFeatureNames(int iinfo, out string[] featureNames)
    //    {
    //        Host.Assert(0 <= iinfo && iinfo < _savedColumnFeatureNames.Length);
    //        if (_savedColumnFeatureNames[iinfo] == null)
    //        {
    //            var featurizer = GetFeaturizerForColumn(iinfo);
    //            var nFeatures = featurizer.NumFeatures;
    //            Interlocked.CompareExchange(
    //                ref _savedColumnFeatureNames[iinfo],
    //                featurizer.GetFeatureNames(_labelClassNames).ToArray(),
    //                null);
    //            Host.Check(Utils.Size(_savedColumnFeatureNames[iinfo]) == nFeatures,
    //                "unexpected # of feature names");
    //        }

    //        featureNames = _savedColumnFeatureNames[iinfo];
    //    }

    //    /// <summary>
    //    /// Determines if the given column pair has acceptable slot names metadata:
    //    /// it must exist and have the expected type and cardinality
    //    /// </summary>
    //    /// <param name="iinfo">Index of the column pair</param>
    //    /// <returns>True if this type is acceptable as slot names</returns>
    //    private bool IsValidSlotNameType(int iinfo)
    //    {
    //        var metadataType = Source.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, Infos[iinfo].Source);
    //        return metadataType != null
    //            && metadataType.IsKnownSizeVector
    //            && metadataType.VectorSize == Infos[iinfo].TypeSrc.VectorSize
    //            && metadataType.ItemType.IsText;
    //    }
    //    #endregion Metadata

    //    private ValueGetter<long> GetLabelGetter(IRow row, int col)
    //    {
    //        // The label column type is checked as part of args validation.
    //        var type = row.Schema.GetColumnType(col);
    //        Host.Assert(type.IsKey || type.IsNumber);

    //        if (type.IsKey)
    //        {
    //            Host.Assert(type.KeyCount > 0);

    //            int size = type.KeyCount;
    //            ulong src = 0;
    //            var getSrc = RowCursorUtils.GetGetterAs<ulong>(NumberType.U8, row, col);
    //            return
    //                (ref long dst) =>
    //                {
    //                    getSrc(ref src);
    //                // The value should fall between 0 and size inclusive, where 0 is considered
    //                // missing/invalid (this is the contract of the KeyType). However, we still handle the
    //                // cases of too large values correctly (by treating them as invalid).
    //                if (src <= (ulong)size)
    //                        dst = (long)src - 1;
    //                    else
    //                        dst = -1;
    //                };
    //        }
    //        else
    //        {
    //            Double src = 0;
    //            var getSrc = RowCursorUtils.GetGetterAs<Double>(NumberType.R8, row, col);
    //            return
    //                (ref long dst) =>
    //                {
    //                    getSrc(ref src);
    //                // NaN maps to -1.
    //                if (src > 0)
    //                        dst = 1;
    //                    else if (src <= 0)
    //                        dst = 0;
    //                    else
    //                        dst = -1;
    //                };
    //        }
    //    }

    //    /// <summary>
    //    /// Iterate over all count tables for the given column and increment the corresponding counts
    //    /// </summary>
    //    /// <param name="builder">Count tables to iterate over</param>
    //    /// <param name="iCol">Column index</param>
    //    /// <param name="srcBuffer">Source values to use as keys</param>
    //    /// <param name="labelKey">Label key</param>
    //    private void IncrementVec(IMultiCountTableBuilder builder, int iCol, ref VBuffer<uint> srcBuffer, uint labelKey)
    //    {
    //        var n = srcBuffer.Length;
    //        if (srcBuffer.IsDense)
    //        {
    //            for (int i = 0; i < n; i++)
    //                builder.IncrementSlot(iCol, i, srcBuffer.Values[i], labelKey, 1);
    //        }
    //        else
    //        {
    //            for (int i = 0; i < srcBuffer.Count; i++)
    //                builder.IncrementSlot(iCol, srcBuffer.Indices[i], srcBuffer.Values[i], labelKey, 1);
    //        }
    //    }

    //    public ICountTable GetCountTable(int columnIndex, int slotIndex)
    //    {
    //        Host.AssertValue(_multiCountTable);
    //        return _multiCountTable.GetCountTable(columnIndex, slotIndex);
    //    }

    //    public int GetLabelCardinality()
    //    {
    //        return _labelCardinality;
    //    }

    //    protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
    //    {
    //        Host.AssertValueOrNull(ch);
    //        Host.AssertValue(input);
    //        Host.Assert(0 <= iinfo && iinfo < Infos.Length);
    //        disposer = null;

    //        if (Infos[iinfo].TypeSrc.IsVector)
    //            return ConstructVectorGetter(input, iinfo);
    //        return ConstructSingleGetter(input, iinfo);
    //    }

    //    private ValueGetter<VBuffer<Single>> ConstructSingleGetter(IRow input, int bindingIndex)
    //    {
    //        Host.Assert(Utils.Size(_featurizers[bindingIndex]) == 1);
    //        uint src = 0;
    //        var srcGetter = GetSrcGetter<uint>(input, bindingIndex);
    //        var numFeatureColumns = _featurizers[bindingIndex][0].NumFeatures;
    //        return (ref VBuffer<Single> dst) =>
    //        {
    //            srcGetter(ref src);
    //            var values = dst.Values;
    //            if (Utils.Size(values) < numFeatureColumns)
    //                values = new Single[numFeatureColumns];
    //            _featurizers[bindingIndex][0].GetFeatures(src, values, 0);
    //            dst = new VBuffer<Single>(numFeatureColumns, values, dst.Indices);
    //        };
    //    }

    //    private ValueGetter<VBuffer<Single>> ConstructVectorGetter(IRow input, int bindingIndex)
    //    {
    //        int n = Infos[bindingIndex].TypeSrc.ValueCount;
    //        Host.Assert(Utils.Size(_featurizers[bindingIndex]) == n);
    //        VBuffer<uint> src = default(VBuffer<uint>);

    //        var numFeatureColumns = _featurizers[bindingIndex][0].NumFeatures;
    //        var srcGetter = GetSrcGetter<VBuffer<uint>>(input, bindingIndex);
    //        return (ref VBuffer<Single> dst) =>
    //        {
    //            srcGetter(ref src);
    //            var values = dst.Values;
    //            if (Utils.Size(values) < n * numFeatureColumns)
    //                values = new Single[n * numFeatureColumns];
    //            if (src.IsDense)
    //            {
    //                for (int i = 0; i < n; i++)
    //                    _featurizers[bindingIndex][i].GetFeatures(src.Values[i], values, i * numFeatureColumns);
    //            }
    //            else
    //            {
    //                for (int i = 0; i < numFeatureColumns * n; i++)
    //                    values[i] = 0;

    //                for (int i = 0; i < src.Count; i++)
    //                {
    //                    var index = src.Indices[i];
    //                    _featurizers[bindingIndex][index].GetFeatures(src.Values[i], values, index * numFeatureColumns);
    //                }
    //            }

    //            dst = new VBuffer<Single>(n * numFeatureColumns, values, dst.Indices);
    //        };
    //    }

    public static class CountTable
    {
        [TlcModule.EntryPoint(Desc = CountTableTransformer.Summary, UserName = CountTableTransformer.UserName, ShortName = "Count")]
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
