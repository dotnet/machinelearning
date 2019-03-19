// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// A catalog of operations over data that are not transformers or estimators.
    /// This includes data loaders, saving, caching, filtering etc.
    /// </summary>
    public sealed class DataOperationsCatalog : IInternalCatalog
    {
        IHostEnvironment IInternalCatalog.Environment => _env;
        private readonly IHostEnvironment _env;

        /// <summary>
        /// A pair of datasets, for the train and test set.
        /// </summary>
        public struct TrainTestData
        {
            /// <summary>
            /// Training set.
            /// </summary>
            public readonly IDataView TrainSet;
            /// <summary>
            /// Testing set.
            /// </summary>
            public readonly IDataView TestSet;
            /// <summary>
            /// Create pair of datasets.
            /// </summary>
            /// <param name="trainSet">Training set.</param>
            /// <param name="testSet">Testing set.</param>
            internal TrainTestData(IDataView trainSet, IDataView testSet)
            {
                TrainSet = trainSet;
                TestSet = testSet;
            }
        }

        internal DataOperationsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            _env = env;
        }

        /// <summary>
        /// Create a new <see cref="IDataView"/> over an enumerable of the items of user-defined type.
        /// The user maintains ownership of the <paramref name="data"/> and the resulting data view will
        /// never alter the contents of the <paramref name="data"/>.
        /// Since <see cref="IDataView"/> is assumed to be immutable, the user is expected to support
        /// multiple enumeration of the <paramref name="data"/> that would return the same results, unless
        /// the user knows that the data will only be cursored once.
        ///
        /// One typical usage for streaming data view could be: create the data view that lazily loads data
        /// as needed, then apply pre-trained transformations to it and cursor through it for transformation
        /// results.
        /// </summary>
        /// <typeparam name="TRow">The user-defined item type.</typeparam>
        /// <param name="data">The data to wrap around.</param>
        /// <param name="schemaDefinition">The optional schema definition of the data view to create. If <c>null</c>,
        /// the schema definition is inferred from <typeparamref name="TRow"/>.</param>
        /// <returns>The constructed <see cref="IDataView"/>.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[BootstrapSample](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/DataViewEnumerable.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView LoadFromEnumerable<TRow>(IEnumerable<TRow> data, SchemaDefinition schemaDefinition = null)
            where TRow : class
        {
            _env.CheckValue(data, nameof(data));
            _env.CheckValueOrNull(schemaDefinition);
            return DataViewConstructionUtils.CreateFromEnumerable(_env, data, schemaDefinition);
        }

        public IDataView LoadFromEnumerable<TRow>(IEnumerable<TRow> data, DataViewSchema schema)
            where TRow : class
        {
            _env.CheckValue(data, nameof(data));
            _env.CheckValue(schema, nameof(schema));
            return DataViewConstructionUtils.CreateFromEnumerable(_env, data, schema);
        }

        /// <summary>
        /// Convert an <see cref="IDataView"/> into a strongly-typed <see cref="IEnumerable{TRow}"/>.
        /// </summary>
        /// <typeparam name="TRow">The user-defined row type.</typeparam>
        /// <param name="data">The underlying data view.</param>
        /// <param name="reuseRowObject">Whether to return the same object on every row, or allocate a new one per row.</param>
        /// <param name="ignoreMissingColumns">Whether to ignore the case when a requested column is not present in the data view.</param>
        /// <param name="schemaDefinition">Optional user-provided schema definition. If it is not present, the schema is inferred from the definition of T.</param>
        /// <returns>The <see cref="IEnumerable{TRow}"/> that holds the data in <paramref name="data"/>. It can be enumerated multiple times.</returns>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[BootstrapSample](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/DataViewEnumerable.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IEnumerable<TRow> CreateEnumerable<TRow>(IDataView data, bool reuseRowObject,
            bool ignoreMissingColumns = false, SchemaDefinition schemaDefinition = null)
            where TRow : class, new()
        {
            _env.CheckValue(data, nameof(data));
            _env.CheckValueOrNull(schemaDefinition);

            var engine = new PipeEngine<TRow>(_env, data, ignoreMissingColumns, schemaDefinition);
            return engine.RunPipe(reuseRowObject);
        }

        /// <summary>
        /// Take an approximate bootstrap sample of <paramref name="input"/>.
        /// </summary>
        /// <remarks>
        /// This sampler is a streaming version of <a href="https://en.wikipedia.org/wiki/Bootstrapping_(statistics)">bootstrap resampling</a>.
        /// Instead of taking the whole dataset into memory and resampling, <see cref="BootstrapSample"/> streams through the dataset and
        /// uses a <a href="https://en.wikipedia.org/wiki/Poisson_distribution">Poisson</a>(1) distribution to select the number of times a
        /// given row will be added to the sample. The <paramref name="complement"/> parameter allows for the creation of a bootstap sample
        /// and complementary out-of-bag sample by using the same <paramref name="seed"/>.
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="seed">The random seed. If unspecified, the random state will be instead derived from the <see cref="MLContext"/>.</param>
        /// <param name="complement">Whether this is the out-of-bag sample, that is, all those rows that are not selected by the transform.
        /// Can be used to create a complementary pair of samples by using the same seed.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[BootstrapSample](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/BootstrapSample.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView BootstrapSample(IDataView input,
            int? seed = null,
            bool complement = BootstrapSamplingTransformer.Defaults.Complement)
        {
            _env.CheckValue(input, nameof(input));
            return new BootstrapSamplingTransformer(
                _env,
                input,
                complement: complement,
                seed: (uint?) seed,
                shuffleInput: false,
                poolSize: 0);
        }

        /// <summary>
        /// Creates a lazy in-memory cache of <paramref name="input"/>.
        /// </summary>
        /// <remarks>
        /// Caching happens per-column. A column is only cached when it is first accessed.
        /// In addition, <paramref name="columnsToPrefetch"/> are considered 'always needed', so these columns
        /// will be cached the first time any data is requested.
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="columnsToPrefetch">The columns that must be cached whenever anything is cached. An empty array or null
        /// value means that columns are cached upon their first access.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Cache](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/Cache.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView Cache(IDataView input, params string[] columnsToPrefetch)
        {
            _env.CheckValue(input, nameof(input));
            _env.CheckValueOrNull(columnsToPrefetch);

            int[] prefetch = new int[Utils.Size(columnsToPrefetch)];
            for (int i = 0; i < prefetch.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(columnsToPrefetch[i], out prefetch[i]))
                    throw _env.ExceptSchemaMismatch(nameof(columnsToPrefetch), "prefetch", columnsToPrefetch[i]);
            }
            return new CacheDataView(_env, input, prefetch);
        }

        /// <summary>
        /// Filter the dataset by the values of a numeric column.
        /// </summary>
        /// <remarks>
        /// Keep only those rows that satisfy the range condition: the value of column <paramref name="columnName"/>
        /// must be between <paramref name="lowerBound"/> (inclusive) and <paramref name="upperBound"/> (exclusive).
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="columnName">The name of a column to use for filtering.</param>
        /// <param name="lowerBound">The inclusive lower bound.</param>
        /// <param name="upperBound">The exclusive upper bound.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FilterRowsByColumn](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/FilterRowsByColumn.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView FilterRowsByColumn(IDataView input, string columnName, double lowerBound = double.NegativeInfinity, double upperBound = double.PositiveInfinity)
        {
            _env.CheckValue(input, nameof(input));
            _env.CheckNonEmpty(columnName, nameof(columnName));
            _env.CheckParam(lowerBound < upperBound, nameof(upperBound), "Must be less than lowerBound");

            var type = input.Schema[columnName].Type;
            if (!(type is NumberDataViewType))
                throw _env.ExceptSchemaMismatch(nameof(columnName), "filter", columnName, "number", type.ToString());
            return new RangeFilter(_env, input, columnName, lowerBound, upperBound, false);
        }

        /// <summary>
        /// Filter the dataset by the values of a <see cref="KeyType"/> column.
        /// </summary>
        /// <remarks>
        /// Keep only those rows that satisfy the range condition: the value of a key column <paramref name="columnName"/>
        /// (treated as a fraction of the entire key range) must be between <paramref name="lowerBound"/> (inclusive) and <paramref name="upperBound"/> (exclusive).
        /// This filtering is useful if the <paramref name="columnName"/> is a key column obtained by some 'stable randomization',
        /// for example, hashing.
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="columnName">The name of a column to use for filtering.</param>
        /// <param name="lowerBound">The inclusive lower bound.</param>
        /// <param name="upperBound">The exclusive upper bound.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FilterRowsByKeyColumnFraction](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/FilterRowsByKeyColumnFraction.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView FilterRowsByKeyColumnFraction(IDataView input, string columnName, double lowerBound = 0, double upperBound = 1)
        {
            _env.CheckValue(input, nameof(input));
            _env.CheckNonEmpty(columnName, nameof(columnName));
            _env.CheckParam(0 <= lowerBound && lowerBound <= 1, nameof(lowerBound), "Must be in [0, 1]");
            _env.CheckParam(0 <= upperBound && upperBound <= 2, nameof(upperBound), "Must be in [0, 2]");
            _env.CheckParam(lowerBound <= upperBound, nameof(upperBound), "Must be no less than lowerBound");

            var type = input.Schema[columnName].Type;
            if (type.GetKeyCount() == 0)
                throw _env.ExceptSchemaMismatch(nameof(columnName), "filter", columnName, "KeyType", type.ToString());
            return new RangeFilter(_env, input, columnName, lowerBound, upperBound, false);
        }

        /// <summary>
        /// Drop rows where any column in <paramref name="columns"/> contains a missing value.
        /// </summary>
        /// <param name="input">The input data.</param>
        /// <param name="columns">Name of the columns to filter on. If a row is has a missing value in any of
        /// these columns, it will be dropped from the dataset.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[FilterRowsByMissingValues](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/FilterRowsByMissingValues.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView FilterRowsByMissingValues(IDataView input, params string[] columns)
        {
            _env.CheckValue(input, nameof(input));
            _env.CheckUserArg(Utils.Size(columns) > 0, nameof(columns));

            return new NAFilter(_env, input, complement: false, columns);
        }

        /// <summary>
        /// Shuffle the rows of <paramref name="input"/>.
        /// </summary>
        /// <remarks>
        /// <see cref="ShuffleRows"/> will shuffle the rows of any input <see cref="IDataView"/> using a streaming approach.
        /// In order to not load the entire dataset in memory, a pool of <paramref name="shufflePoolSize"/> rows will be used
        /// to randomly select rows to output. The pool is constructed from the first <paramref name="shufflePoolSize"/> rows
        /// in <paramref name="input"/>. Rows will then be randomly yielded from the pool and replaced with the next row from <paramref name="input"/>
        /// until all the rows have been yielded, resulting in a new <see cref="IDataView"/> of the same size as <paramref name="input"/>
        /// but with the rows in a randomized order.
        /// If the <see cref="IDataView.CanShuffle"/> property of <paramref name="input"/> is true, then it will also be read into the
        /// pool in a random order, offering two sources of randomness.
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="seed">The random seed. If unspecified, the random state will be instead derived from the <see cref="MLContext"/>.</param>
        /// <param name="shufflePoolSize">The number of rows to hold in the pool. Setting this to 1 will turn off pool shuffling and
        /// <see cref="ShuffleRows"/> will only perform a shuffle by reading <paramref name="input"/> in a random order.</param>
        /// <param name="shuffleSource">If <see langword="false"/>, the transform will not attempt to read <paramref name="input"/> in a random order and only use
        /// pooling to shuffle. This parameter has no effect if the <see cref="IDataView.CanShuffle"/> property of <paramref name="input"/> is <see langword="false"/>.
        /// </param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[ShuffleRows](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/ShuffleRows.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView ShuffleRows(IDataView input,
            int? seed = null,
            int shufflePoolSize = RowShufflingTransformer.Defaults.PoolRows,
            bool shuffleSource = !RowShufflingTransformer.Defaults.PoolOnly)
        {
            _env.CheckValue(input, nameof(input));
            _env.CheckUserArg(shufflePoolSize > 0, nameof(shufflePoolSize), "Must be positive");

            var options = new RowShufflingTransformer.Options
            {
                PoolRows = shufflePoolSize,
                PoolOnly = !shuffleSource,
                ForceShuffle = true,
                ForceShuffleSeed = seed
            };

            return new RowShufflingTransformer(_env, options, input);
        }

        /// <summary>
        /// Skip <paramref name="count"/> rows in <paramref name="input"/>.
        /// </summary>
        /// <remarks>
        /// Skips the first <paramref name="count"/> rows from <paramref name="input"/> and returns an <see cref="IDataView"/> with all other rows.
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="count">Number of rows to skip.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[SkipRows](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/SkipRows.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView SkipRows(IDataView input, long count)
        {
            _env.CheckValue(input, nameof(input));
            _env.CheckUserArg(count > 0, nameof(count), "Must be greater than zero.");

            var options = new SkipTakeFilter.SkipOptions()
            {
                Count = count
            };

            return new SkipTakeFilter(_env, options, input);
        }

        /// <summary>
        /// Take <paramref name="count"/> rows from <paramref name="input"/>.
        /// </summary>
        /// <remarks>
        /// Returns returns an <see cref="IDataView"/> with the first <paramref name="count"/> rows from <paramref name="input"/>.
        /// </remarks>
        /// <param name="input">The input data.</param>
        /// <param name="count">Number of rows to take.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[TakeRows](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/DataOperations/TakeRows.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public IDataView TakeRows(IDataView input, long count)
        {
            _env.CheckValue(input, nameof(input));
            _env.CheckUserArg(count > 0, nameof(count), "Must be greater than zero.");

            var options = new SkipTakeFilter.TakeOptions()
            {
                Count = count
            };

            return new SkipTakeFilter(_env, options, input);
        }

        /// <summary>
        /// Split the dataset into the train set and test set according to the given fraction.
        /// Respects the <paramref name="samplingKeyColumnName"/> if provided.
        /// </summary>
        /// <param name="data">The dataset to split.</param>
        /// <param name="testFraction">The fraction of data to go into the test set.</param>
        /// <param name="samplingKeyColumnName">Name of a column to use for grouping rows. If two examples share the same value of the <paramref name="samplingKeyColumnName"/>,
        /// they are guaranteed to appear in the same subset (train or test). This can be used to ensure no label leakage from the train to the test set.
        /// If <see langword="null"/> no row grouping will be performed.</param>
        /// <param name="seed">Seed for the random number generator used to select rows for the train-test split.</param>
        public TrainTestData TrainTestSplit(IDataView data, double testFraction = 0.1, string samplingKeyColumnName = null, int? seed = null)
        {
            _env.CheckValue(data, nameof(data));
            _env.CheckParam(0 < testFraction && testFraction < 1, nameof(testFraction), "Must be between 0 and 1 exclusive");
            _env.CheckValueOrNull(samplingKeyColumnName);

            EnsureGroupPreservationColumn(_env, ref data, ref samplingKeyColumnName, seed);

            var trainFilter = new RangeFilter(_env, new RangeFilter.Options()
            {
                Column = samplingKeyColumnName,
                Min = 0,
                Max = testFraction,
                Complement = true
            }, data);
            var testFilter = new RangeFilter(_env, new RangeFilter.Options()
            {
                Column = samplingKeyColumnName,
                Min = 0,
                Max = testFraction,
                Complement = false
            }, data);

            return new TrainTestData(trainFilter, testFilter);
        }

        /// <summary>
        /// Ensures the provided <paramref name="samplingKeyColumn"/> is valid for <see cref="RangeFilter"/>, hashing it if necessary, or creates a new column <paramref name="samplingKeyColumn"/> is null.
        /// </summary>
        internal static void EnsureGroupPreservationColumn(IHostEnvironment env, ref IDataView data, ref string samplingKeyColumn, int? seed = null)
        {
            // We need to handle two cases: if samplingKeyColumn is provided, we use hashJoin to
            // build a single hash of it. If it is not, we generate a random number.

            if (samplingKeyColumn == null)
            {
                samplingKeyColumn = data.Schema.GetTempColumnName("SamplingKeyColumn");
                data = new GenerateNumberTransform(env, data, samplingKeyColumn, (uint?)seed);
            }
            else
            {
                if (!data.Schema.TryGetColumnIndex(samplingKeyColumn, out int stratCol))
                    throw env.ExceptSchemaMismatch(nameof(samplingKeyColumn), "SamplingKeyColumn", samplingKeyColumn);

                var type = data.Schema[stratCol].Type;
                if (!RangeFilter.IsValidRangeFilterColumnType(env, type))
                {
                    // Hash the samplingKeyColumn.
                    // REVIEW: this could currently crash, since Hash only accepts a limited set
                    // of column types. It used to be HashJoin, but we should probably extend Hash
                    // instead of having two hash transformations.
                    var origStratCol = samplingKeyColumn;
                    int tmp;
                    int inc = 0;

                    // Generate a new column with the hashed samplingKeyColumn.
                    while (data.Schema.TryGetColumnIndex(samplingKeyColumn, out tmp))
                        samplingKeyColumn = string.Format("{0}_{1:000}", origStratCol, ++inc);
                    HashingEstimator.ColumnOptions columnOptions;
                    if (seed.HasValue)
                        columnOptions = new HashingEstimator.ColumnOptions(samplingKeyColumn, origStratCol, 30, (uint)seed.Value);
                    else
                        columnOptions = new HashingEstimator.ColumnOptions(samplingKeyColumn, origStratCol, 30);
                    data = new HashingEstimator(env, columnOptions).Fit(data).Transform(data);
                }
            }
        }
    }
}
