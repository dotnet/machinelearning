// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(CountTargetEncodingTransformer.Summary, typeof(IDataTransform), typeof(CountTargetEncodingEstimator), typeof(CountTargetEncodingEstimator.Options), typeof(SignatureDataTransform),
    CountTargetEncodingTransformer.UserName, CountTargetEncodingTransformer.LoaderSignature, "Dracula")]

[assembly: LoadableClass(CountTargetEncodingTransformer.Summary, typeof(CountTargetEncodingTransformer), typeof(CountTargetEncodingEstimator), null, typeof(SignatureLoadModel),
    CountTargetEncodingTransformer.UserName, CountTargetEncodingTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(CountTargetEncoder))]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Transforms a categorical column into a set of features that includes the count of each label class,
    /// the log-odds for each label class and the back-off indicator.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | Any |
    /// | Output column data type | Vector of <xref:System.Single>. |
    ///
    /// The resulting <xref:Microsoft.ML.Transforms.CountTargetEncodingTransformer> creates a new column, named as specified in the output column name parameters,
    /// containing three parts: the count of each label class, the log-odds for each label class and the back-off indicator.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="CountTargetEncodingCatalog.CountTargetEncode(TransformsCatalog, InputOutputColumnPair[], CountTargetEncodingTransformer, string)" />
    /// <seealso cref="CountTargetEncodingCatalog.CountTargetEncode(TransformsCatalog, InputOutputColumnPair[], string, CountTableBuilderBase, float, float, bool, int, bool, uint)" />
    /// <seealso cref="CountTargetEncodingCatalog.CountTargetEncode(TransformsCatalog, string, CountTargetEncodingTransformer, string, string)"/>
    /// <seealso cref="CountTargetEncodingCatalog.CountTargetEncode(TransformsCatalog, string, string, string, CountTableBuilderBase, float, float, int, bool, uint)"/>
    public class CountTargetEncodingEstimator : IEstimator<CountTargetEncodingTransformer>
    {
        /// <summary>
        /// This is a merger of arguments for <see cref="CountTableTransformer"/> and <see cref="HashJoiningTransform"/>
        /// </summary>
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                ShortName = "col", SortOrder = 1)]
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

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the values need to be combined for a single hash", SortOrder = 3)]
            public bool Combine = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of bits to hash into. Must be between 1 and 31, inclusive.",
                ShortName = "bits")]
            public int NumberOfBits = HashJoiningTransform.NumBitsLim - 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint HashingSeed = 314489979;
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

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the values need to be combined for a single hash")]
            public bool? Combine;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Which slots should be combined together. Example: 0,3,5;0,1;3;2,1,0. Overrides 'join'.")]
            public string CustomSlotMap;

            public static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }
        }

        private readonly IHost _host;
        private readonly CountTableEstimator _estimator;
        private readonly HashJoiningTransform.Arguments _hashJoinArgs;
        private readonly HashJoiningTransform _hashJoin;

        internal CountTargetEncodingEstimator(IHostEnvironment env, string labelColumnName, CountTableEstimator.ColumnOptions[] columnOptions,
            int numberOfBits = HashJoiningTransform.Defaults.NumberOfBits,
            bool combine = HashJoiningTransform.Defaults.Combine, uint hashingSeed = HashJoiningTransform.Defaults.Seed)
            : this(env, new CountTableEstimator(env, labelColumnName,
                columnOptions.Select(col => new CountTableEstimator.ColumnOptions(col.Name, col.Name, col.CountTableBuilder, col.PriorCoefficient, col.LaplaceScale, col.Seed)).ToArray()),
                  columnOptions, numberOfBits, combine, hashingSeed)
        {
        }

        internal CountTargetEncodingEstimator(IHostEnvironment env, string labelColumnName, CountTableEstimator.SharedColumnOptions[] columnOptions,
            CountTableBuilderBase countTableBuilder, int numberOfBits = HashJoiningTransform.Defaults.NumberOfBits,
            bool combine = HashJoiningTransform.Defaults.Combine, uint hashingSeed = HashJoiningTransform.Defaults.Seed)
            : this(env, new CountTableEstimator(env, labelColumnName, countTableBuilder,
                columnOptions.Select(col => new CountTableEstimator.SharedColumnOptions(col.Name, col.Name, col.PriorCoefficient, col.LaplaceScale, col.Seed)).ToArray()),
                  columnOptions, numberOfBits, combine, hashingSeed)
        {
        }

        internal CountTargetEncodingEstimator(IHostEnvironment env, string labelColumnName, CountTargetEncodingTransformer initialCounts, params InputOutputColumnPair[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(CountTargetEncodingEstimator));
            _host.CheckValue(initialCounts, nameof(initialCounts));
            _host.CheckNonEmpty(columns, nameof(columns));
            _host.Check(columns.All(c => initialCounts.HashJoin.InputSchema.GetColumnOrNull(c.InputColumnName) != null), nameof(columns));
            _host.Check(columns.All(c => initialCounts.HashJoin.OutputSchema.GetColumnOrNull(c.OutputColumnName) != null), nameof(columns));

            _estimator = new CountTableEstimator(_host, labelColumnName, initialCounts.CountTable, columns.Select(c => new InputOutputColumnPair(c.OutputColumnName, c.OutputColumnName)).ToArray());
            _hashJoin = initialCounts.HashJoin;
        }

        private CountTargetEncodingEstimator(IHostEnvironment env, CountTableEstimator estimator, CountTableEstimator.ColumnOptionsBase[] columns,
            int numberOfBits, bool combine, uint hashingSeed)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(CountTargetEncodingEstimator));

            _estimator = estimator;
            _hashJoinArgs = InitializeHashJoinArguments(columns, numberOfBits, combine, hashingSeed);
        }

        internal CountTargetEncodingEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(CountTargetEncodingEstimator));
            _host.CheckValue(options, nameof(options));
            _host.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns), "Columns must be specified");
            _host.CheckUserArg(!string.IsNullOrWhiteSpace(options.LabelColumn), nameof(options.LabelColumn), "Must specify the label column name");

            if (!string.IsNullOrEmpty(options.InitialCountsModel))
            {
                _estimator = LoadFromFile(env, options.InitialCountsModel, options.LabelColumn,
                    options.Columns.Select(col => new InputOutputColumnPair(col.Name)).ToArray());
                if (_estimator == null)
                    throw env.Except($"The file {options.InitialCountsModel} does not contain a CountTableTransformer");
            }
            else if (options.SharedTable)
            {
                var columns = new CountTableEstimator.SharedColumnOptions[options.Columns.Length];
                for (int i = 0; i < options.Columns.Length; i++)
                {
                    var column = options.Columns[i];
                    columns[i] = new CountTableEstimator.SharedColumnOptions(
                        column.Name,
                        column.Name,
                        column.PriorCoefficient ?? options.PriorCoefficient,
                        column.LaplaceScale ?? options.LaplaceScale,
                        column.Seed ?? options.Seed);
                }
                var builder = options.CountTable;
                _host.CheckValue(builder, nameof(options.CountTable));
                _estimator = new CountTableEstimator(_host, options.LabelColumn, builder.CreateComponent(_host), columns);
            }
            else
            {
                var columns = new CountTableEstimator.ColumnOptions[options.Columns.Length];
                for (int i = 0; i < options.Columns.Length; i++)
                {
                    var column = options.Columns[i];
                    var builder = column.CountTable ?? options.CountTable;
                    _host.CheckValue(builder, nameof(options.CountTable));
                    columns[i] = new CountTableEstimator.ColumnOptions(
                        column.Name,
                        column.Name,
                        builder.CreateComponent(_host),
                        column.PriorCoefficient ?? options.PriorCoefficient,
                        column.LaplaceScale ?? options.LaplaceScale,
                        column.Seed ?? options.Seed);
                }
                _estimator = new CountTableEstimator(_host, options.LabelColumn, columns);
            }

            _hashJoinArgs = InitializeHashJoinArguments(options);
        }

        internal static CountTableEstimator LoadFromFile(IHostEnvironment env, string initialCountsModel, string labelColumn, InputOutputColumnPair[] columns)
        {
            var catalog = new ModelOperationsCatalog(env);
            var transformer = catalog.Load(initialCountsModel, out _);
            CountTableTransformer countTableTransformer;
            if ((countTableTransformer = transformer as CountTableTransformer) == null)
            {
                if (transformer is CountTargetEncodingTransformer cte)
                    countTableTransformer = cte.CountTable;
                else if (transformer is TransformerChain<ITransformer> chain)
                {
                    var transformerChain = ((ITransformerChainAccessor)chain).Transformers;
                    for (int i = transformerChain.Length - 1; i >= 0; i++)
                    {
                        countTableTransformer = transformerChain[i] as CountTableTransformer;
                        if (countTableTransformer == null)
                        {
                            if ((cte = transformerChain[i] as CountTargetEncodingTransformer) != null)
                                countTableTransformer = cte.CountTable;
                        }
                        if (countTableTransformer != null)
                            break;
                    }
                }
            }
            if (countTableTransformer != null)
                return new CountTableEstimator(env, labelColumn, countTableTransformer, columns);
            return null;
        }

        private HashJoiningTransform.Arguments InitializeHashJoinArguments(CountTableEstimator.ColumnOptionsBase[] columns, int numberOfBits, bool combine, uint hashingSeed)
        {
            var cols = new HashJoiningTransform.Column[columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var column = columns[i];
                cols[i] =
                    new HashJoiningTransform.Column
                    {
                        Name = column.Name,
                        Source = column.InputColumnName,
                    };
            }

            return new HashJoiningTransform.Arguments
            {
                Columns = cols,
                NumberOfBits = numberOfBits,
                Combine = combine,
                Seed = hashingSeed,
                Ordered = false,
            };
        }

        private HashJoiningTransform.Arguments InitializeHashJoinArguments(Options options)
        {
            var columns = options.Columns;
            var cols = new HashJoiningTransform.Column[columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var column = columns[i];
                cols[i] =
                    new HashJoiningTransform.Column
                    {
                        Combine = column.Combine ?? options.Combine,
                        CustomSlotMap = column.CustomSlotMap,
                        Seed = options.HashingSeed,
                        NumberOfBits = options.NumberOfBits,
                        Name = column.Name,
                        Source = column.Source,
                    };
            }

            return new HashJoiningTransform.Arguments
            {
                Columns = cols,
                NumberOfBits = options.NumberOfBits,
                Combine = options.Combine,
                Seed = options.HashingSeed,
                Ordered = false,
            };
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns));

            var estimator = new CountTargetEncodingEstimator(env, options);
            return (estimator.Fit(input) as ITransformerWithDifferentMappingAtTrainingTime).TransformForTrainingPipeline(input) as IDataTransform;
        }

        private static CountTargetEncodingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
            => CountTargetEncodingTransformer.Create(env, ctx);

        public CountTargetEncodingTransformer Fit(IDataView input)
        {
            var hashJoinTransform = CreateHashJoiningTransform(input);
            return new CountTargetEncodingTransformer(_host, hashJoinTransform, _estimator.Fit(hashJoinTransform));
        }

        private HashJoiningTransform CreateHashJoiningTransform(IDataView input)
        {
            _host.Assert(_hashJoinArgs != null || _hashJoin != null);
            var hashJoinTransform = _hashJoinArgs != null ?
                new HashJoiningTransform(_host, _hashJoinArgs, input) :
                new HashJoiningTransform(_host, _hashJoin, input);
            return hashJoinTransform;
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var hashJoinOutputSchema = CreateHashJoinOutputSchema(inputSchema);
            return _estimator.GetOutputSchema(hashJoinOutputSchema);
        }

        private SchemaShape CreateHashJoinOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            var inputSchemaBuilder = new DataViewSchema.Builder();
            foreach (var col in inputSchema)
            {
                switch (col.Kind)
                {
                    case SchemaShape.Column.VectorKind.Scalar:
                        inputSchemaBuilder.AddColumn(col.Name, col.ItemType);
                        break;
                    case SchemaShape.Column.VectorKind.Vector:
                        var annotationsBuilder = new DataViewSchema.Annotations.Builder();
                        if (col.HasSlotNames())
                            annotationsBuilder.AddSlotNames(1, (ref VBuffer<ReadOnlyMemory<char>> dst) => { });
                        inputSchemaBuilder.AddColumn(col.Name, new VectorDataViewType(col.ItemType as PrimitiveDataViewType, 1), annotationsBuilder.ToAnnotations());
                        break;
                    case SchemaShape.Column.VectorKind.VariableVector:
                        inputSchemaBuilder.AddColumn(col.Name, new VectorDataViewType(col.ItemType as PrimitiveDataViewType, 0));
                        break;
                }
            }

            var inputData = new EmptyDataView(_host, inputSchemaBuilder.ToSchema());
            var hashJoinSchema = CreateHashJoiningTransform(inputData).OutputSchema;
            return SchemaShape.Create(hashJoinSchema);
        }
    }

    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="LpNormNormalizingEstimator"/> or <see cref="CountTargetEncodingEstimator"/>.
    /// </summary>
    public sealed class CountTargetEncodingTransformer : ITransformerWithDifferentMappingAtTrainingTime
    {
        private readonly IHost _host;
        internal readonly HashJoiningTransform HashJoin;
        internal readonly CountTableTransformer CountTable;

        internal const string Summary = "Transforms the categorical column into the set of features: count of each label class, "
            + "log-odds for each label class, back-off indicator. The columns can be of arbitrary type.";
        internal const string LoaderSignature = "CountTargetEncode";
        internal const string UserName = "Count Target Encoding Transform";

        bool ITransformer.IsRowToRowMapper => true;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DRACULA ",
                 verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CountTargetEncodingTransformer).Assembly.FullName);
        }

        internal CountTargetEncodingTransformer(IHostEnvironment env, HashJoiningTransform hashJoin, CountTableTransformer countTable)
        {
            Contracts.AssertValue(env);
            env.AssertValue(hashJoin);
            env.AssertValue(countTable);
            _host = env.Register(nameof(CountTargetEncodingTransformer));
            HashJoin = hashJoin;
            CountTable = countTable;
        }

        private CountTargetEncodingTransformer(IHost host, ModelLoadContext ctx)
        {
            _host = host;

            // *** Binary format ***
            // input schema
            // _hashJoin
            // _countTable

            var ms = new MemoryStream();
            ctx.TryLoadBinaryStream("InputSchema", reader =>
            {
                reader.BaseStream.CopyTo(ms);
            });

            ms.Position = 0;
            var loader = new BinaryLoader(_host, new BinaryLoader.Arguments(), ms);
            var view = new EmptyDataView(_host, loader.Schema);

            ctx.LoadModel<HashJoiningTransform, SignatureLoadDataTransform>(_host, out HashJoin, "HashJoin", view);
            ctx.LoadModel<CountTableTransformer, SignatureLoadModel>(_host, out CountTable, "CountTable");
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // input schema
            // _hashJoin
            // _countTable

            ctx.SaveBinaryStream("InputSchema", writer =>
            {
                using (var ch = _host.Start("Saving input schema"))
                {
                    var saver = new BinarySaver(_host, new BinarySaver.Arguments { Silent = true });
                    DataSaverUtils.SaveDataView(ch, saver, new EmptyDataView(_host, HashJoin.Source.Schema), writer.BaseStream);
                }
            });

            ctx.SaveModel(HashJoin, "HashJoin");
            ctx.SaveModel(CountTable, "CountTable");
        }

        internal static CountTargetEncodingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(LoaderSignature);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new CountTargetEncodingTransformer(host, ctx);
        }

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var chain = new TransformerChain<ITransformer>(new TransformWrapper(_host, HashJoin), CountTable);
            return chain.GetOutputSchema(inputSchema);
        }

        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            ITransformer chain = new TransformerChain<ITransformer>(new TransformWrapper(_host, HashJoin), CountTable);
            return chain.GetRowToRowMapper(inputSchema);
        }

        public IDataView Transform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            var chain = new TransformerChain<ITransformer>(new TransformWrapper(_host, HashJoin), CountTable);
            return chain.Transform(input);
        }

        IDataView ITransformerWithDifferentMappingAtTrainingTime.TransformForTrainingPipeline(IDataView input)
        {
            var hashJoin = ApplyTransformUtils.ApplyTransformToData(_host, HashJoin, input);
            return (CountTable as ITransformerWithDifferentMappingAtTrainingTime).TransformForTrainingPipeline(hashJoin);
        }
    }

    public static class CountTargetEncodingCatalog
    {
        /// <summary>
        /// Transforms a categorical column into a set of features that includes the count of each label class,
        /// the log-odds for each label class and the back-off indicator.
        /// </summary>
        /// <param name="catalog">The transforms catalog.</param>
        /// <param name="columns">The input and output columns.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="builder">The builder that creates the count tables from the training data.</param>
        /// <param name="priorCoefficient">The coefficient with which to apply the prior smoothing to the features.</param>
        /// <param name="laplaceScale">The Laplacian noise diversity/scale-parameter. Recommended values are between 0 and 1. Note that the noise
        /// will only be applied if the estimator is part of an <see cref="EstimatorChain{TLastTransformer}"/>, when fitting the next estimator in the chain.</param>
        /// <param name="sharedTable">Indicates whether to keep counts for all columns and slots in one shared count table. If true, the keys in the count table
        /// will include a hash of the column and slot indices.</param>
        /// <param name="numberOfBits">The number of bits to hash the input into. Must be between 1 and 31, inclusive.</param>
        /// <param name="combine">In case the input is a vector column, indicates whether the values should be combined into a single hash to create a single
        /// count table, or be left as a vector of hashes with multiple count tables.</param>
        /// <param name="hashingSeed">The seed used for hashing the input columns.</param>
        /// <returns></returns>
        public static CountTargetEncodingEstimator CountTargetEncode(this TransformsCatalog catalog,
            InputOutputColumnPair[] columns, string labelColumn = DefaultColumnNames.Label,
            CountTableBuilderBase builder = null,
            float priorCoefficient = CountTableTransformer.Defaults.PriorCoefficient,
            float laplaceScale = CountTableTransformer.Defaults.LaplaceScale,
            bool sharedTable = CountTableTransformer.Defaults.SharedTable,
            int numberOfBits = HashJoiningTransform.Defaults.NumberOfBits,
            bool combine = HashJoiningTransform.Defaults.Combine,
            uint hashingSeed = HashJoiningTransform.Defaults.Seed)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckValue(columns, nameof(columns));

            builder = builder ?? new CMCountTableBuilder();

            CountTargetEncodingEstimator estimator;
            if (sharedTable)
            {
                var columnOptions = new CountTableEstimator.SharedColumnOptions[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    columnOptions[i] = new CountTableEstimator.SharedColumnOptions(
                        columns[i].OutputColumnName, columns[i].InputColumnName, priorCoefficient, laplaceScale);
                }
                estimator = new CountTargetEncodingEstimator(env, labelColumn, columnOptions, builder, numberOfBits, combine, hashingSeed);
            }
            else
            {
                var columnOptions = new CountTableEstimator.ColumnOptions[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    columnOptions[i] = new CountTableEstimator.ColumnOptions(
                        columns[i].OutputColumnName, columns[i].InputColumnName, builder, priorCoefficient, laplaceScale);
                }
                estimator = new CountTargetEncodingEstimator(env, labelColumn, columnOptions, numberOfBits: numberOfBits, combine: combine, hashingSeed: hashingSeed);
            }
            return estimator;
        }

        /// <summary>
        /// Transforms a categorical column into a set of features that includes the count of each label class,
        /// the log-odds for each label class and the back-off indicator.
        /// </summary>
        /// <param name="catalog">The transforms catalog.</param>
        /// <param name="columns">The input and output columns.</param>
        /// <param name="initialCounts">A previously trained count table containing initial counts.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <returns></returns>
        public static CountTargetEncodingEstimator CountTargetEncode(this TransformsCatalog catalog,
            InputOutputColumnPair[] columns, CountTargetEncodingTransformer initialCounts, string labelColumn = "Label")
        {
            return new CountTargetEncodingEstimator(CatalogUtils.GetEnvironment(catalog), labelColumn, initialCounts, columns);
        }

        /// <summary>
        /// Transforms a categorical column into a set of features that includes the count of each label class,
        /// the log-odds for each label class and the back-off indicator.
        /// </summary>
        /// <param name="catalog">The transforms catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="builder">The builder that creates the count tables from the training data.</param>
        /// <param name="priorCoefficient">The coefficient with which to apply the prior smoothing to the features.</param>
        /// <param name="laplaceScale">The Laplacian noise diversity/scale-parameter. Recommended values are between 0 and 1. Note that the noise
        /// will only be applied if the estimator is part of an <see cref="EstimatorChain{TLastTransformer}"/>, when fitting the next estimator in the chain.</param>
        /// <param name="numberOfBits">The number of bits to hash the input into. Must be between 1 and 31, inclusive.</param>
        /// <param name="combine">In case the input is a vector column, indicates whether the values should be combined into a single hash to create a single
        /// count table, or be left as a vector of hashes with multiple count tables.</param>
        /// <param name="hashingSeed">The seed used for hashing the input columns.</param>
        public static CountTargetEncodingEstimator CountTargetEncode(this TransformsCatalog catalog, string outputColumnName, string inputColumnName = null,
            string labelColumn = DefaultColumnNames.Label,
            CountTableBuilderBase builder = null,
            float priorCoefficient = CountTableTransformer.Defaults.PriorCoefficient,
            float laplaceScale = CountTableTransformer.Defaults.LaplaceScale,
            int numberOfBits = HashJoiningTransform.Defaults.NumberOfBits,
            bool combine = HashJoiningTransform.Defaults.Combine,
            uint hashingSeed = HashJoiningTransform.Defaults.Seed)
        {
            var env = CatalogUtils.GetEnvironment(catalog);
            env.CheckNonEmpty(outputColumnName, nameof(outputColumnName));

            inputColumnName = string.IsNullOrEmpty(inputColumnName) ? outputColumnName : inputColumnName;
            builder = builder ?? new CMCountTableBuilder();

            return new CountTargetEncodingEstimator(env, labelColumn,
                new[] { new CountTableEstimator.ColumnOptions(outputColumnName, inputColumnName, builder, priorCoefficient, laplaceScale) },
                numberOfBits, combine, hashingSeed);
        }

        /// <summary>
        /// Transforms a categorical column into a set of features that includes the count of each label class,
        /// the log-odds for each label class and the back-off indicator.
        /// </summary>
        /// <param name="catalog">The transforms catalog.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="initialCounts">A previously trained count table containing initial counts.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <returns></returns>
        public static CountTargetEncodingEstimator CountTargetEncode(this TransformsCatalog catalog, string outputColumnName,
            CountTargetEncodingTransformer initialCounts,
            string inputColumnName = null, string labelColumn = "Label")
        {
            return new CountTargetEncodingEstimator(CatalogUtils.GetEnvironment(catalog), labelColumn, initialCounts, new[] { new InputOutputColumnPair(outputColumnName, inputColumnName) });
        }
    }

    internal static class CountTargetEncoder
    {
        [TlcModule.EntryPoint(Name = "Transforms.CountTargetEncoder", Desc = CountTargetEncodingTransformer.Summary, UserName = CountTableTransformer.UserName, ShortName = "Count")]
        internal static CommonOutputs.TransformOutput Create(IHostEnvironment env, CountTargetEncodingEstimator.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, nameof(CountTargetEncoder), input);
            var view = CountTargetEncodingEstimator.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
