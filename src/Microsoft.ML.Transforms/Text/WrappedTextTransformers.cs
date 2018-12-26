// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Transforms.Text
{

    /// <summary>
    /// Produces a bag of counts of ngrams (sequences of consecutive words) in a given text.
    /// It does so by building a dictionary of ngrams and using the id in the dictionary as the index in the bag.
    /// </summary>
    public sealed class WordBagEstimator : TrainedWrapperEstimatorBase
    {
        private readonly (string[] inputs, string output)[] _columns;
        private readonly int _ngramLength;
        private readonly int _skipLength;
        private readonly bool _allLengths;
        private readonly int _maxNumTerms;
        private readonly NgramExtractingEstimator.WeightingCriteria _weighting;

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumn"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing bag of word vector. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public WordBagEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            : this(env, new[] { (new[] { inputColumn }, outputColumn ?? inputColumn) }, ngramLength, skipLength, allLengths, maxNumTerms, weighting)
        {
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumns"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumns">The columns containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing output tokens.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public WordBagEstimator(IHostEnvironment env,
            string[] inputColumns,
            string outputColumn,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            : this(env, new[] { (inputColumns, outputColumn) }, ngramLength, skipLength, allLengths, maxNumTerms, weighting)
        {
        }

        /// <summary>
        /// Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
        /// <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
        public WordBagEstimator(IHostEnvironment env,
            (string[] inputs, string output)[] columns,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            int maxNumTerms = 10000000,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordBagEstimator)))
        {
            foreach (var (input, output) in columns)
            {
                Host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
                Host.CheckValue(output, nameof(input));
            }

            _columns = columns;
            _ngramLength = ngramLength;
            _skipLength = skipLength;
            _allLengths = allLengths;
            _maxNumTerms = maxNumTerms;
            _weighting = weighting;
        }

        public override TransformWrapper Fit(IDataView input)
        {
            // Create arguments.
            var args = new WordBagBuildingTransformer.Arguments
            {
                Column = _columns.Select(x => new WordBagBuildingTransformer.Column { Source = x.inputs, Name = x.output }).ToArray(),
                NgramLength = _ngramLength,
                SkipLength = _skipLength,
                AllLengths = _allLengths,
                MaxNumTerms = new[] { _maxNumTerms },
                Weighting = _weighting
            };

            return new TransformWrapper(Host, WordBagBuildingTransformer.Create(Host, args, input));
        }
    }

    /// <summary>
    /// Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text.
    /// It does so by hashing each ngram and using the hash value as the index in the bag.
    /// </summary>
    public sealed class WordHashBagEstimator : TrainedWrapperEstimatorBase
    {
        private readonly (string[] inputs, string output)[] _columns;
        private readonly int _hashBits;
        private readonly int _ngramLength;
        private readonly int _skipLength;
        private readonly bool _allLengths;
        private readonly uint _seed;
        private readonly bool _ordered;
        private readonly int _invertHash;

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumn"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing bag of word vector. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public WordHashBagEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : this(env, new[] { (new[] { inputColumn }, outputColumn ?? inputColumn) }, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash)
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="inputColumns"/>
        /// and outputs bag of word vector as <paramref name="outputColumn"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumns">The columns containing text to compute bag of word vector.</param>
        /// <param name="outputColumn">The column containing output tokens.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public WordHashBagEstimator(IHostEnvironment env,
            string[] inputColumns,
            string outputColumn,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : this(env, new[] { (inputColumns, outputColumn) }, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash)
        {
        }

        /// <summary>
        /// Produces a bag of counts of hashed ngrams in <paramref name="columns.inputs"/>
        /// and outputs bag of word vector for each output in <paramref name="columns.output"/>
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute bag of word vector.</param>
        /// <param name="hashBits">Number of bits to hash into. Must be between 1 and 30, inclusive.</param>
        /// <param name="ngramLength">Ngram length.</param>
        /// <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
        /// <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
        /// <param name="seed">Hashing seed.</param>
        /// <param name="ordered">Whether the position of each source column should be included in the hash (when there are multiple source columns).</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public WordHashBagEstimator(IHostEnvironment env,
            (string[] inputs, string output)[] columns,
            int hashBits = 16,
            int ngramLength = 1,
            int skipLength = 0,
            bool allLengths = true,
            uint seed = 314489979,
            bool ordered = true,
            int invertHash = 0)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordBagEstimator)))
        {
            foreach (var (input, output) in columns)
            {
                Host.CheckUserArg(Utils.Size(input) > 0, nameof(input));
                Host.CheckValue(output, nameof(input));
            }

            _columns = columns;
            _hashBits = hashBits;
            _ngramLength = ngramLength;
            _skipLength = skipLength;
            _allLengths = allLengths;
            _seed = seed;
            _ordered = ordered;
            _invertHash = invertHash;
        }

        public override TransformWrapper Fit(IDataView input)
        {
            // Create arguments.
            var args = new WordHashBagProducingTransformer.Arguments
            {
                Column = _columns.Select(x => new WordHashBagProducingTransformer.Column { Source = x.inputs, Name = x.output }).ToArray(),
                HashBits = _hashBits,
                NgramLength = _ngramLength,
                SkipLength = _skipLength,
                AllLengths = _allLengths,
                Seed = _seed,
                Ordered = _ordered,
                InvertHash = _invertHash
            };

            return new TransformWrapper(Host, WordHashBagProducingTransformer.Create(Host, args, input));
        }
    }
}