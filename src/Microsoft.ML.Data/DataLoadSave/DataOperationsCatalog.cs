// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    /// <summary>
    /// A catalog of operations over data that are not transformers or estimators.
    /// This includes data readers, saving, caching, filtering etc.
    /// </summary>
    public sealed class DataOperationsCatalog
    {
        [BestFriend]
        internal IHostEnvironment Environment { get; }

        internal DataOperationsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            Environment = env;
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
            Environment.CheckValue(input, nameof(input));
            return new BootstrapSamplingTransformer(
                Environment,
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
            Environment.CheckValue(input, nameof(input));
            Environment.CheckValueOrNull(columnsToPrefetch);

            int[] prefetch = new int[Utils.Size(columnsToPrefetch)];
            for (int i = 0; i < prefetch.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(columnsToPrefetch[i], out prefetch[i]))
                    throw Environment.ExceptSchemaMismatch(nameof(columnsToPrefetch), "prefetch", columnsToPrefetch[i]);
            }
            return new CacheDataView(Environment, input, prefetch);
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
            Environment.CheckValue(input, nameof(input));
            Environment.CheckNonEmpty(columnName, nameof(columnName));
            Environment.CheckParam(lowerBound < upperBound, nameof(upperBound), "Must be less than lowerBound");

            var type = input.Schema[columnName].Type;
            if (!(type is NumberDataViewType))
                throw Environment.ExceptSchemaMismatch(nameof(columnName), "filter", columnName, "number", type.ToString());
            return new RangeFilter(Environment, input, columnName, lowerBound, upperBound, false);
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
            Environment.CheckValue(input, nameof(input));
            Environment.CheckNonEmpty(columnName, nameof(columnName));
            Environment.CheckParam(0 <= lowerBound && lowerBound <= 1, nameof(lowerBound), "Must be in [0, 1]");
            Environment.CheckParam(0 <= upperBound && upperBound <= 2, nameof(upperBound), "Must be in [0, 2]");
            Environment.CheckParam(lowerBound <= upperBound, nameof(upperBound), "Must be no less than lowerBound");

            var type = input.Schema[columnName].Type;
            if (type.GetKeyCount() == 0)
                throw Environment.ExceptSchemaMismatch(nameof(columnName), "filter", columnName, "KeyType", type.ToString());
            return new RangeFilter(Environment, input, columnName, lowerBound, upperBound, false);
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
            Environment.CheckValue(input, nameof(input));
            Environment.CheckUserArg(Utils.Size(columns) > 0, nameof(columns));

            return new NAFilter(Environment, input, complement: false, columns);
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
            Environment.CheckValue(input, nameof(input));
            Environment.CheckUserArg(shufflePoolSize > 0, nameof(shufflePoolSize), "Must be positive");

            var options = new RowShufflingTransformer.Options
            {
                PoolRows = shufflePoolSize,
                PoolOnly = !shuffleSource,
                ForceShuffle = true,
                ForceShuffleSeed = seed
            };

            return new RowShufflingTransformer(Environment, options, input);
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
            Environment.CheckValue(input, nameof(input));
            Environment.CheckUserArg(count > 0, nameof(count), "Must be greater than zero.");

            var options = new SkipTakeFilter.SkipOptions()
            {
                Count = count
            };

            return new SkipTakeFilter(Environment, options, input);
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
            Environment.CheckValue(input, nameof(input));
            Environment.CheckUserArg(count > 0, nameof(count), "Must be greater than zero.");

            var options = new SkipTakeFilter.TakeOptions()
            {
                Count = count
            };

            return new SkipTakeFilter(Environment, options, input);
        }
    }
}
