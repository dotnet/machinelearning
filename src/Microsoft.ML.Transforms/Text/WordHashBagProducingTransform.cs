// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(WordHashBagProducingTransformer.Summary, typeof(IDataTransform), typeof(WordHashBagProducingTransformer), typeof(WordHashBagProducingTransformer.Options), typeof(SignatureDataTransform),
    "Word Hash Bag Transform", "WordHashBagTransform", "WordHashBag")]

[assembly: LoadableClass(NgramHashExtractingTransformer.Summary, typeof(INgramExtractorFactory), typeof(NgramHashExtractingTransformer), typeof(NgramHashExtractingTransformer.NgramHashExtractorArguments),
    typeof(SignatureNgramExtractorFactory), "Ngram Hash Extractor Transform", "NgramHashExtractorTransform", "NgramHash", NgramHashExtractingTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(NgramHashExtractingTransformer.NgramHashExtractorArguments))]

namespace Microsoft.ML.Transforms.Text
{
    internal static class WordHashBagProducingTransformer
    {
        internal sealed class Column : NgramHashExtractingTransformer.ColumnBase
        {
            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:B:S where N is the new column name, B is the number of bits,
                // and S is source column names.
                string extra;
                if (!base.TryParse(str, out extra))
                    return false;
                if (extra == null)
                    return true;

                int bits;
                if (!int.TryParse(extra, out bits))
                    return false;
                NumberOfBits = bits;
                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NgramLength != null || SkipLength != null || Seed != null ||
                    Ordered != null || MaximumNumberOfInverts != null)
                {
                    return false;
                }
                if (NumberOfBits == null)
                    return TryUnparseCore(sb);

                string extra = NumberOfBits.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        internal sealed class Options : NgramHashExtractingTransformer.ArgumentsBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:numberOfBits:srcs)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }
        private const string RegistrationName = "WordHashBagTransform";

        internal const string Summary = "Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text. "
            + "It does so by hashing each ngram and using the hash value as the index in the bag.";

        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(options, nameof(options));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns), "Columns must be specified");

            // To each input column to the WordHashBagTransform, a tokenize transform is applied,
            // followed by applying WordHashVectorizeTransform.
            // Since WordHashBagTransform is a many-to-one column transform, for each
            // WordHashBagTransform.Column we may need to define multiple tokenize transform columns.
            // NgramHashExtractorTransform may need to define an identical number of HashTransform.Columns.
            // The intermediate columns are dropped at the end of using a DropColumnsTransform.
            IDataView view = input;

            var uniqueSourceNames = NgramExtractionUtils.GenerateUniqueSourceNames(h, options.Columns, view.Schema);
            Contracts.Assert(uniqueSourceNames.Length == options.Columns.Length);

            var tokenizeColumns = new List<WordTokenizingEstimator.ColumnOptions>();
            var extractorCols = new NgramHashExtractingTransformer.Column[options.Columns.Length];
            var colCount = options.Columns.Length;
            List<string> tmpColNames = new List<string>();
            for (int iinfo = 0; iinfo < colCount; iinfo++)
            {
                var column = options.Columns[iinfo];
                int srcCount = column.Source.Length;
                var curTmpNames = new string[srcCount];
                Contracts.Assert(uniqueSourceNames[iinfo].Length == options.Columns[iinfo].Source.Length);
                for (int isrc = 0; isrc < srcCount; isrc++)
                    tokenizeColumns.Add(new WordTokenizingEstimator.ColumnOptions(curTmpNames[isrc] = uniqueSourceNames[iinfo][isrc], options.Columns[iinfo].Source[isrc]));

                tmpColNames.AddRange(curTmpNames);
                extractorCols[iinfo] =
                    new NgramHashExtractingTransformer.Column
                    {
                        Name = column.Name,
                        Source = curTmpNames,
                        NumberOfBits = column.NumberOfBits,
                        NgramLength = column.NgramLength,
                        Seed = column.Seed,
                        SkipLength = column.SkipLength,
                        Ordered = column.Ordered,
                        MaximumNumberOfInverts = column.MaximumNumberOfInverts,
                        FriendlyNames = options.Columns[iinfo].Source,
                        UseAllLengths = column.UseAllLengths
                    };
            }

            view = new WordTokenizingEstimator(env, tokenizeColumns.ToArray()).Fit(view).Transform(view);

            var featurizeArgs =
                new NgramHashExtractingTransformer.Options
                {
                    UseAllLengths = options.UseAllLengths,
                    NumberOfBits = options.NumberOfBits,
                    NgramLength = options.NgramLength,
                    SkipLength = options.SkipLength,
                    Ordered = options.Ordered,
                    Seed = options.Seed,
                    Columns = extractorCols.ToArray(),
                    MaximumNumberOfInverts = options.MaximumNumberOfInverts
                };

            view = NgramHashExtractingTransformer.Create(h, featurizeArgs, view);

            // Since we added columns with new names, we need to explicitly drop them before we return the IDataTransform.
            return ColumnSelectingTransformer.CreateDrop(h, view, tmpColNames.ToArray()) as IDataTransform;
        }
    }

    /// <summary>
    /// A transform that turns a collection of tokenized text (vector of ReadOnlyMemory) into numerical feature vectors
    /// using the hashing trick.
    /// </summary>
    internal static class NgramHashExtractingTransformer
    {
        internal abstract class ColumnBase : ManyToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Ngram length (stores all lengths up to the specified Ngram length)", ShortName = "ngram")]
            public int? NgramLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips")]
            public int? SkipLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits")]
            public int? NumberOfBits;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint? Seed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the position of each source column should be included in the hash (when there are multiple source columns).", ShortName = "ord")]
            public bool? Ordered;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int? MaximumNumberOfInverts;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether to include all ngram lengths up to " + nameof(NgramLength) + " or only " + nameof(NgramLength),
                Name = "AllLengths", ShortName = "all", SortOrder = 4)]
            public bool? UseAllLengths;
        }

        internal sealed class Column : ColumnBase
        {
            // For all source columns, use these friendly names for the source
            // column names instead of the real column names.
            public string[] FriendlyNames;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:B:S where N is the new column name, B is the number of bits,
                // and S is source column names.
                string extra;
                if (!base.TryParse(str, out extra))
                    return false;
                if (extra == null)
                    return true;

                int bits;
                if (!int.TryParse(extra, out bits))
                    return false;
                NumberOfBits = bits;
                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NgramLength != null || SkipLength != null || Seed != null ||
                    Ordered != null || MaximumNumberOfInverts != null)
                {
                    return false;
                }
                if (NumberOfBits == null)
                    return TryUnparseCore(sb);

                string extra = NumberOfBits.Value.ToString();
                return TryUnparseCore(sb, extra);
            }
        }

        /// <summary>
        /// This class is a merger of <see cref="HashingTransformer.Options"/> and
        /// <see cref="NgramHashingTransformer.Options"/>, with the ordered option,
        /// the rehashUnigrams option and the allLength option removed.
        /// </summary>
        internal abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Ngram length", ShortName = "ngram", SortOrder = 3)]
            public int NgramLength = 1;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips", SortOrder = 4)]
            public int SkipLength = 0;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of bits to hash into. Must be between 1 and 30, inclusive.",
                ShortName = "bits", SortOrder = 2)]
            public int NumberOfBits = 16;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Hashing seed")]
            public uint Seed = 314489979;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether the position of each source column should be included in the hash (when there are multiple source columns).",
                ShortName = "ord")]
            public bool Ordered = true;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Limit the number of keys used to generate the slot name to this many. 0 means no invert hashing, -1 means no limit.",
                ShortName = "ih")]
            public int MaximumNumberOfInverts;

            [Argument(ArgumentType.AtMostOnce,
               HelpText = "Whether to include all ngram lengths up to ngramLength or only ngramLength",
               Name = "AllLengths", ShortName = "all", SortOrder = 4)]
            public bool UseAllLengths = true;
        }

        internal static class DefaultArguments
        {
            public const int NgramLength = 1;
            public const int SkipLength = 0;
            public const int NumberOfBits = 16;
            public const uint Seed = 314489979;
            public const bool Ordered = true;
            public const int MaximumNumberOfInverts = 0;
            public const bool UseAllLengths = true;
        }

        [TlcModule.Component(Name = "NGramHash", FriendlyName = "NGram Hash Extractor Transform", Alias = "NGramHashExtractorTransform,NGramHashExtractor",
                            Desc = "Extracts NGrams from text and convert them to vector using hashing trick.")]
        internal sealed class NgramHashExtractorArguments : ArgumentsBase, INgramExtractorFactoryFactory
        {
            public INgramExtractorFactory CreateComponent(IHostEnvironment env, TermLoaderArguments loaderArgs)
            {
                return Create(env, this, loaderArgs);
            }
        }

        internal sealed class Options : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:srcs)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        internal const string Summary = "A transform that turns a collection of tokenized text (vector of ReadOnlyMemory) into numerical feature vectors using the hashing trick.";

        internal const string LoaderSignature = "NgramHashExtractor";

        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input,
            TermLoaderArguments termLoaderArgs = null)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(options, nameof(options));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns), "Columns must be specified");

            // To each input column to the NgramHashExtractorArguments, a HashTransform using 31
            // bits (to minimize collisions) is applied first, followed by an NgramHashTransform.
            IDataView view = input;

            List<ValueToKeyMappingTransformer.Column> termCols = null;
            if (termLoaderArgs != null)
                termCols = new List<ValueToKeyMappingTransformer.Column>();

            var hashColumns = new List<HashingEstimator.ColumnOptions>();
            var ngramHashColumns = new NgramHashingEstimator.ColumnOptions[options.Columns.Length];

            var colCount = options.Columns.Length;
            // The NGramHashExtractor has a ManyToOne column type. To avoid stepping over the source
            // column name when a 'name' destination column name was specified, we use temporary column names.
            string[][] tmpColNames = new string[colCount][];
            for (int iinfo = 0; iinfo < colCount; iinfo++)
            {
                var column = options.Columns[iinfo];
                h.CheckUserArg(!string.IsNullOrWhiteSpace(column.Name), nameof(column.Name));
                h.CheckUserArg(Utils.Size(column.Source) > 0 &&
                    column.Source.All(src => !string.IsNullOrWhiteSpace(src)), nameof(column.Source));

                int srcCount = column.Source.Length;
                tmpColNames[iinfo] = new string[srcCount];
                for (int isrc = 0; isrc < srcCount; isrc++)
                {
                    var tmpName = input.Schema.GetTempColumnName(column.Source[isrc]);
                    tmpColNames[iinfo][isrc] = tmpName;
                    if (termLoaderArgs != null)
                    {
                        termCols.Add(
                            new ValueToKeyMappingTransformer.Column
                            {
                                Name = tmpName,
                                Source = column.Source[isrc]
                            });
                    }

                    hashColumns.Add(new HashingEstimator.ColumnOptions(tmpName, termLoaderArgs == null ? column.Source[isrc] : tmpName,
                        30, column.Seed ?? options.Seed, false, column.MaximumNumberOfInverts ?? options.MaximumNumberOfInverts));
                }

                ngramHashColumns[iinfo] =
                    new NgramHashingEstimator.ColumnOptions(column.Name, tmpColNames[iinfo],
                    column.NgramLength ?? options.NgramLength,
                    column.SkipLength ?? options.SkipLength,
                    column.UseAllLengths ?? options.UseAllLengths,
                    column.NumberOfBits ?? options.NumberOfBits,
                    column.Seed ?? options.Seed,
                    column.Ordered ?? options.Ordered,
                    column.MaximumNumberOfInverts ?? options.MaximumNumberOfInverts);
                ngramHashColumns[iinfo].FriendlyNames = column.FriendlyNames;
            }

            if (termLoaderArgs != null)
            {
                h.Assert(Utils.Size(termCols) == hashColumns.Count);
                var termArgs =
                    new ValueToKeyMappingTransformer.Options()
                    {
                        MaxNumTerms = int.MaxValue,
                        Term = termLoaderArgs.Term,
                        Terms = termLoaderArgs.Terms,
                        DataFile = termLoaderArgs.DataFile,
                        Loader = termLoaderArgs.Loader,
                        TermsColumn = termLoaderArgs.TermsColumn,
                        Sort = termLoaderArgs.Sort,
                        Columns = termCols.ToArray()
                    };
                view = ValueToKeyMappingTransformer.Create(h, termArgs, view);

                if (termLoaderArgs.DropUnknowns)
                {
                    var missingDropColumns = new (string outputColumnName, string inputColumnName)[termCols.Count];
                    for (int iinfo = 0; iinfo < termCols.Count; iinfo++)
                        missingDropColumns[iinfo] = (termCols[iinfo].Name, termCols[iinfo].Name);
                    view = new MissingValueDroppingTransformer(h, missingDropColumns).Transform(view);
                }
            }
            view = new HashingEstimator(h, hashColumns.ToArray()).Fit(view).Transform(view);
            view = new NgramHashingEstimator(h, ngramHashColumns).Fit(view).Transform(view);
            return ColumnSelectingTransformer.CreateDrop(h, view, tmpColNames.SelectMany(cols => cols).ToArray()) as IDataTransform;
        }

        internal static IDataTransform Create(NgramHashExtractorArguments extractorArgs, IHostEnvironment env, IDataView input,
            ExtractorColumn[] cols, TermLoaderArguments termLoaderArgs = null)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(extractorArgs, nameof(extractorArgs));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(extractorArgs.SkipLength < extractorArgs.NgramLength, nameof(extractorArgs.SkipLength), "Should be less than " + nameof(extractorArgs.NgramLength));
            h.CheckUserArg(Utils.Size(cols) > 0, nameof(Options.Columns), "Must be specified");
            h.AssertValueOrNull(termLoaderArgs);

            var extractorCols = new Column[cols.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                extractorCols[i] =
                    new Column
                    {
                        Name = cols[i].Name,
                        Source = cols[i].Source,
                        FriendlyNames = cols[i].FriendlyNames
                    };
            }

            var options = new Options
            {
                Columns = extractorCols,
                NgramLength = extractorArgs.NgramLength,
                SkipLength = extractorArgs.SkipLength,
                NumberOfBits = extractorArgs.NumberOfBits,
                MaximumNumberOfInverts = extractorArgs.MaximumNumberOfInverts,
                Ordered = extractorArgs.Ordered,
                Seed = extractorArgs.Seed,
                UseAllLengths = extractorArgs.UseAllLengths
            };

            return Create(h, options, input, termLoaderArgs);
        }

        internal static INgramExtractorFactory Create(IHostEnvironment env, NgramHashExtractorArguments extractorArgs,
            TermLoaderArguments termLoaderArgs)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(extractorArgs, nameof(extractorArgs));
            h.CheckValueOrNull(termLoaderArgs);

            return new NgramHashExtractorFactory(extractorArgs, termLoaderArgs);
        }
    }
}
