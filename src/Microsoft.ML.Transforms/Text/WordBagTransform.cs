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

[assembly: LoadableClass(WordBagBuildingTransformer.Summary, typeof(IDataTransform), typeof(WordBagBuildingTransformer), typeof(WordBagBuildingTransformer.Options), typeof(SignatureDataTransform),
    "Word Bag Transform", "WordBagTransform", "WordBag")]

[assembly: LoadableClass(NgramExtractorTransform.Summary, typeof(INgramExtractorFactory), typeof(NgramExtractorTransform), typeof(NgramExtractorTransform.NgramExtractorArguments),
    typeof(SignatureNgramExtractorFactory), "Ngram Extractor Transform", "NgramExtractorTransform", "Ngram", NgramExtractorTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(NgramExtractorTransform.NgramExtractorArguments))]

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// Signature for creating an INgramExtractorFactory.
    /// </summary>
    internal delegate void SignatureNgramExtractorFactory(TermLoaderArguments termLoaderArgs);

    /// <summary>
    /// A many-to-one column common to both <see cref="NgramExtractorTransform"/>
    /// and <see cref="NgramHashExtractingTransformer"/>.
    /// </summary>
    internal sealed class ExtractorColumn : ManyToOneColumn
    {
        // For all source columns, use these friendly names for the source
        // column names instead of the real column names.
        public string[] FriendlyNames;
    }

    internal static class WordBagBuildingTransformer
    {
        internal sealed class Column : ManyToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Ngram length", ShortName = "ngram")]
            public int? NgramLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips")]
            public int? SkipLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether to include all ngram lengths up to " + nameof(NgramLength) + " or only " + nameof(NgramLength),
                Name = "AllLengths", ShortName = "all")]
            public bool? UseAllLengths;

            [Argument(ArgumentType.Multiple, HelpText = "Maximum number of ngrams to store in the dictionary", ShortName = "max")]
            public int[] MaxNumTerms = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Statistical measure used to evaluate how important a word is to a document in a corpus")]
            public NgramExtractingEstimator.WeightingCriteria? Weighting;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NgramLength != null || SkipLength != null || UseAllLengths != null || Utils.Size(MaxNumTerms) > 0 ||
                    Weighting != null)
                {
                    return false;
                }
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options : NgramExtractorTransform.ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:srcs)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        private const string RegistrationName = "WordBagTransform";

        internal const string Summary = "Produces a bag of counts of ngrams (sequences of consecutive words of length 1-n) in a given text. It does so by building "
            + "a dictionary of ngrams and using the id in the dictionary as the index in the bag.";

        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(options, nameof(options));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns), "Columns must be specified");

            // Compose the WordBagTransform from a tokenize transform,
            // followed by a NgramExtractionTransform.
            // Since WordBagTransform is a many-to-one column transform, for each
            // WordBagTransform.Column with multiple sources, we first apply a ConcatTransform.

            // REVIEW: In order to not get ngrams that cross between vector slots, we need to
            // enable tokenize transforms to insert a special token between slots.

            // REVIEW: In order to make it possible to output separate bags for different columns
            // using the same dictionary, we need to find a way to make ConcatTransform remember the boundaries.

            var tokenizeColumns = new WordTokenizingEstimator.ColumnOptions[options.Columns.Length];

            var extractorArgs =
                new NgramExtractorTransform.Options()
                {
                    MaxNumTerms = options.MaxNumTerms,
                    NgramLength = options.NgramLength,
                    SkipLength = options.SkipLength,
                    UseAllLengths = options.UseAllLengths,
                    Weighting = options.Weighting,
                    Columns = new NgramExtractorTransform.Column[options.Columns.Length]
                };

            for (int iinfo = 0; iinfo < options.Columns.Length; iinfo++)
            {
                var column = options.Columns[iinfo];
                h.CheckUserArg(!string.IsNullOrWhiteSpace(column.Name), nameof(column.Name));
                h.CheckUserArg(Utils.Size(column.Source) > 0, nameof(column.Source));
                h.CheckUserArg(column.Source.All(src => !string.IsNullOrWhiteSpace(src)), nameof(column.Source));

                tokenizeColumns[iinfo] = new WordTokenizingEstimator.ColumnOptions(column.Name, column.Source.Length > 1 ? column.Name : column.Source[0]);

                extractorArgs.Columns[iinfo] =
                    new NgramExtractorTransform.Column()
                    {
                        Name = column.Name,
                        Source = column.Name,
                        MaxNumTerms = column.MaxNumTerms,
                        NgramLength = column.NgramLength,
                        SkipLength = column.SkipLength,
                        Weighting = column.Weighting,
                        UseAllLengths = column.UseAllLengths
                    };
            }

            IDataView view = input;
            view = NgramExtractionUtils.ApplyConcatOnSources(h, options.Columns, view);
            view = new WordTokenizingEstimator(env, tokenizeColumns).Fit(view).Transform(view);
            return NgramExtractorTransform.Create(h, extractorArgs, view);
        }
    }

    /// <summary>
    /// A transform that turns a collection of tokenized text (vector of ReadOnlyMemory), or vectors of keys into numerical
    /// feature vectors. The feature vectors are counts of ngrams (sequences of consecutive *tokens* -words or keys-
    /// of length 1-n).
    /// </summary>
    internal static class NgramExtractorTransform
    {
        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Ngram length (stores all lengths up to the specified Ngram length)", ShortName = "ngram")]
            public int? NgramLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips")]
            public int? SkipLength;

            [Argument(ArgumentType.AtMostOnce, HelpText =
                "Whether to include all ngram lengths up to " + nameof(NgramLength) + " or only " + nameof(NgramLength),
                Name = "AllLengths", ShortName = "all")]
            public bool? UseAllLengths;

            // REVIEW: This argument is actually confusing. If you set only one value we will use this value for all ngrams respectfully for example,
            // if we specify 3 ngrams we will have maxNumTerms * 3. And it also pick first value from this array to run term transform, so if you specify
            // something like 1,1,10000, term transform would be run with limitation of only one term.
            [Argument(ArgumentType.Multiple, HelpText = "Maximum number of ngrams to store in the dictionary", ShortName = "max")]
            public int[] MaxNumTerms = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The weighting criteria")]
            public NgramExtractingEstimator.WeightingCriteria? Weighting;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NgramLength != null || SkipLength != null || UseAllLengths != null || Utils.Size(MaxNumTerms) > 0 ||
                    Weighting != null)
                {
                    return false;
                }
                return TryUnparseCore(sb);
            }
        }

        /// <summary>
        /// This class is a merger of <see cref="ValueToKeyMappingTransformer.Options"/> and
        /// <see cref="NgramExtractingTransformer.Options"/>, with the allLength option removed.
        /// </summary>
        internal abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Ngram length", ShortName = "ngram")]
            public int NgramLength = 1;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Maximum number of tokens to skip when constructing an ngram",
                ShortName = "skips")]
            public int SkipLength = NgramExtractingEstimator.Defaults.SkipLength;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Whether to include all ngram lengths up to " + nameof(NgramLength) + " or only " + nameof(NgramLength),
                Name = "AllLengths", ShortName = "all")]
            public bool UseAllLengths = NgramExtractingEstimator.Defaults.UseAllLengths;

            [Argument(ArgumentType.Multiple, HelpText = "Maximum number of ngrams to store in the dictionary", ShortName = "max")]
            public int[] MaxNumTerms = new int[] { NgramExtractingEstimator.Defaults.MaximumNgramsCount };

            [Argument(ArgumentType.AtMostOnce, HelpText = "The weighting criteria")]
            public NgramExtractingEstimator.WeightingCriteria Weighting = NgramExtractingEstimator.Defaults.Weighting;
        }

        [TlcModule.Component(Name = "NGram", FriendlyName = "NGram Extractor Transform", Alias = "NGramExtractorTransform,NGramExtractor",
            Desc = "Extracts NGrams from text and convert them to vector using dictionary.")]
        public sealed class NgramExtractorArguments : ArgumentsBase, INgramExtractorFactoryFactory
        {
            public INgramExtractorFactory CreateComponent(IHostEnvironment env, TermLoaderArguments loaderArgs)
            {
                return Create(env, this, loaderArgs);
            }
        }

        internal sealed class Options : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        internal const string Summary = "A transform that turns a collection of tokenized text ReadOnlyMemory, or vectors of keys into numerical " +
            "feature vectors. The feature vectors are counts of ngrams (sequences of consecutive *tokens* -words or keys- of length 1-n).";

        internal const string LoaderSignature = "NgramExtractor";

        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input,
            TermLoaderArguments termLoaderArgs = null)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(options, nameof(options));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(Utils.Size(options.Columns) > 0, nameof(options.Columns), "Columns must be specified");

            IDataView view = input;
            var termCols = new List<Column>();
            var isTermCol = new bool[options.Columns.Length];

            for (int i = 0; i < options.Columns.Length; i++)
            {
                var col = options.Columns[i];

                h.CheckNonWhiteSpace(col.Name, nameof(col.Name));
                h.CheckNonWhiteSpace(col.Source, nameof(col.Source));
                int colId;
                if (input.Schema.TryGetColumnIndex(col.Source, out colId) &&
                    input.Schema[colId].Type.GetItemType() is TextDataViewType)
                {
                    termCols.Add(col);
                    isTermCol[i] = true;
                }
            }

            // If the column types of args.column are text, apply term transform to convert them to keys.
            // Otherwise, skip term transform and apply ngram transform directly.
            // This logic allows NgramExtractorTransform to handle both text and key input columns.
            // Note: ngram transform handles the validation of the types natively (in case the types
            // of args.column are not text nor keys).
            if (termCols.Count > 0)
            {
                ValueToKeyMappingTransformer.Options termArgs = null;
                string[] missingDropColumns = null;
                if (termLoaderArgs != null)
                {
                    termArgs =
                        new ValueToKeyMappingTransformer.Options()
                        {
                            MaxNumTerms = int.MaxValue,
                            Term = termLoaderArgs.Term,
                            Terms = termLoaderArgs.Terms,
                            DataFile = termLoaderArgs.DataFile,
                            Loader = termLoaderArgs.Loader,
                            TermsColumn = termLoaderArgs.TermsColumn,
                            Sort = termLoaderArgs.Sort,
                            Columns = new ValueToKeyMappingTransformer.Column[termCols.Count]
                        };
                    if (termLoaderArgs.DropUnknowns)
                        missingDropColumns = new string[termCols.Count];
                }
                else
                {
                    termArgs =
                        new ValueToKeyMappingTransformer.Options()
                        {
                            MaxNumTerms = Utils.Size(options.MaxNumTerms) > 0 ? options.MaxNumTerms[0] : NgramExtractingEstimator.Defaults.MaximumNgramsCount,
                            Columns = new ValueToKeyMappingTransformer.Column[termCols.Count]
                        };
                }

                for (int iinfo = 0; iinfo < termCols.Count; iinfo++)
                {
                    var column = termCols[iinfo];
                    termArgs.Columns[iinfo] =
                        new ValueToKeyMappingTransformer.Column()
                        {
                            Name = column.Name,
                            Source = column.Source,
                            MaxNumTerms = Utils.Size(column.MaxNumTerms) > 0 ? column.MaxNumTerms[0] : default(int?)
                        };

                    if (missingDropColumns != null)
                        missingDropColumns[iinfo] = column.Name;
                }

                view = ValueToKeyMappingTransformer.Create(h, termArgs, view);
                if (missingDropColumns != null)
                    view = new MissingValueDroppingTransformer(h, missingDropColumns.Select(x => (x, x)).ToArray()).Transform(view);
            }

            var ngramColumns = new NgramExtractingEstimator.ColumnOptions[options.Columns.Length];
            for (int iinfo = 0; iinfo < options.Columns.Length; iinfo++)
            {
                var column = options.Columns[iinfo];
                ngramColumns[iinfo] = new NgramExtractingEstimator.ColumnOptions(column.Name,
                    column.NgramLength ?? options.NgramLength,
                    column.SkipLength ?? options.SkipLength,
                    column.UseAllLengths ?? options.UseAllLengths,
                    column.Weighting ?? options.Weighting,
                    column.MaxNumTerms ?? options.MaxNumTerms,
                    isTermCol[iinfo] ? column.Name : column.Source
                    );
            }

            return new NgramExtractingEstimator(env, ngramColumns).Fit(view).Transform(view) as IDataTransform;
        }

        internal static IDataTransform Create(IHostEnvironment env, NgramExtractorArguments extractorArgs, IDataView input,
            ExtractorColumn[] cols, TermLoaderArguments termLoaderArgs = null)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(extractorArgs, nameof(extractorArgs));
            h.CheckValue(input, nameof(input));
            h.CheckUserArg(extractorArgs.SkipLength < extractorArgs.NgramLength, nameof(extractorArgs.SkipLength), "Should be less than " + nameof(extractorArgs.NgramLength));
            h.CheckUserArg(Utils.Size(cols) > 0, nameof(Options.Columns), "Must be specified");
            h.CheckValueOrNull(termLoaderArgs);

            var extractorCols = new Column[cols.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                Contracts.Check(Utils.Size(cols[i].Source) == 1, "too many source columns");
                extractorCols[i] = new Column { Name = cols[i].Name, Source = cols[i].Source[0] };
            }

            var options = new Options
            {
                Columns = extractorCols,
                NgramLength = extractorArgs.NgramLength,
                SkipLength = extractorArgs.SkipLength,
                UseAllLengths = extractorArgs.UseAllLengths,
                MaxNumTerms = extractorArgs.MaxNumTerms,
                Weighting = extractorArgs.Weighting
            };

            return Create(h, options, input, termLoaderArgs);
        }

        internal static INgramExtractorFactory Create(IHostEnvironment env, NgramExtractorArguments extractorArgs,
            TermLoaderArguments termLoaderArgs)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(extractorArgs, nameof(extractorArgs));
            h.CheckValueOrNull(termLoaderArgs);

            return new NgramExtractorFactory(extractorArgs, termLoaderArgs);
        }
    }

    /// <summary>
    /// Arguments for defining custom list of terms or data file containing the terms.
    /// The class includes a subset of <see cref="ValueToKeyMappingTransformer"/>'s arguments.
    /// </summary>
    internal sealed class TermLoaderArguments
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Comma separated list of terms", Name = "Terms", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
        public string Term;

        [Argument(ArgumentType.AtMostOnce, HelpText = "List of terms", Name = "Term", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
        public string[] Terms;

        [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Data file containing the terms", ShortName = "data", SortOrder = 2, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
        public string DataFile;

        [Argument(ArgumentType.Multiple, HelpText = "Data loader", NullName = "<Auto>", SortOrder = 3, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureDataLoader))]
        internal IComponentFactory<IMultiStreamSource, ILegacyDataLoader> Loader;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the text column containing the terms", ShortName = "termCol", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
        public string TermsColumn;

        [Argument(ArgumentType.AtMostOnce, HelpText = "How items should be ordered when vectorized. By default, they will be in the order encountered. " +
            "If by value, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').", SortOrder = 5)]
        public ValueToKeyMappingEstimator.KeyOrdinality Sort = ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Drop unknown terms instead of mapping them to NA term.", ShortName = "dropna", SortOrder = 6)]
        public bool DropUnknowns = false;
    }

    /// <summary>
    /// An ngram extractor factory interface to create an ngram extractor transform.
    /// </summary>
    internal interface INgramExtractorFactory
    {
        /// <summary>
        /// Whether the extractor transform created by this factory uses the hashing trick
        /// (by using <see cref="HashingTransformer"/> or <see cref="NgramHashingTransformer"/>, for example).
        /// </summary>
        bool UseHashingTrick { get; }

        IDataTransform Create(IHostEnvironment env, IDataView input, ExtractorColumn[] cols);
    }

    [TlcModule.ComponentKind("NgramExtractor")]
    internal interface INgramExtractorFactoryFactory : IComponentFactory<TermLoaderArguments, INgramExtractorFactory> { }

    /// <summary>
    /// An implementation of <see cref="INgramExtractorFactory"/> to create <see cref="NgramExtractorTransform"/>.
    /// </summary>
    internal class NgramExtractorFactory : INgramExtractorFactory
    {
        private readonly NgramExtractorTransform.NgramExtractorArguments _extractorArgs;
        private readonly TermLoaderArguments _termLoaderArgs;

        public bool UseHashingTrick { get { return false; } }

        public NgramExtractorFactory(NgramExtractorTransform.NgramExtractorArguments extractorArgs,
            TermLoaderArguments termLoaderArgs)
        {
            Contracts.CheckValue(extractorArgs, nameof(extractorArgs));
            Contracts.CheckValueOrNull(termLoaderArgs);
            _extractorArgs = extractorArgs;
            _termLoaderArgs = termLoaderArgs;
        }

        public IDataTransform Create(IHostEnvironment env, IDataView input, ExtractorColumn[] cols)
        {
            return NgramExtractorTransform.Create(env, _extractorArgs, input, cols, _termLoaderArgs);
        }
    }

    /// <summary>
    /// An implementation of <see cref="INgramExtractorFactory"/> to create <see cref="NgramHashExtractingTransformer"/>.
    /// </summary>
    internal class NgramHashExtractorFactory : INgramExtractorFactory
    {
        private readonly NgramHashExtractingTransformer.NgramHashExtractorArguments _extractorArgs;
        private readonly TermLoaderArguments _termLoaderArgs;

        public bool UseHashingTrick { get { return true; } }

        public NgramHashExtractorFactory(NgramHashExtractingTransformer.NgramHashExtractorArguments extractorArgs,
            TermLoaderArguments customTermsArgs = null)
        {
            Contracts.CheckValue(extractorArgs, nameof(extractorArgs));
            Contracts.CheckValueOrNull(customTermsArgs);
            _extractorArgs = extractorArgs;
            _termLoaderArgs = customTermsArgs;
        }

        public IDataTransform Create(IHostEnvironment env, IDataView input, ExtractorColumn[] cols)
        {
            return NgramHashExtractingTransformer.Create(_extractorArgs, env, input, cols, _termLoaderArgs);
        }
    }

    internal static class NgramExtractionUtils
    {
        public static IDataView ApplyConcatOnSources(IHostEnvironment env, ManyToOneColumn[] columns, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(columns, nameof(columns));
            env.CheckValue(input, nameof(input));

            IDataView view = input;
            var concatColumns = new List<ColumnConcatenatingTransformer.ColumnOptions>();
            foreach (var col in columns)
            {
                env.CheckUserArg(col != null, nameof(WordBagBuildingTransformer.Options.Columns));
                env.CheckUserArg(!string.IsNullOrWhiteSpace(col.Name), nameof(col.Name));
                env.CheckUserArg(Utils.Size(col.Source) > 0, nameof(col.Source));
                env.CheckUserArg(col.Source.All(src => !string.IsNullOrWhiteSpace(src)), nameof(col.Source));
                if (col.Source.Length > 1)
                    concatColumns.Add(new ColumnConcatenatingTransformer.ColumnOptions(col.Name, col.Source));
            }
            if (concatColumns.Count > 0)
                return new ColumnConcatenatingTransformer(env, concatColumns.ToArray()).Transform(view);

            return view;
        }

        /// <summary>
        /// Generates and returns unique names for columns source. Each element of the returned array is
        /// an array of unique source names per specific column.
        /// </summary>
        public static string[][] GenerateUniqueSourceNames(IHostEnvironment env, ManyToOneColumn[] columns, DataViewSchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(columns, nameof(columns));
            env.CheckValue(schema, nameof(schema));

            string[][] uniqueNames = new string[columns.Length][];
            int tmp = 0;
            for (int iinfo = 0; iinfo < columns.Length; iinfo++)
            {
                var col = columns[iinfo];
                env.CheckUserArg(col != null, nameof(WordHashBagProducingTransformer.Options.Columns));
                env.CheckUserArg(!string.IsNullOrWhiteSpace(col.Name), nameof(col.Name));
                env.CheckUserArg(Utils.Size(col.Source) > 0 &&
                              col.Source.All(src => !string.IsNullOrWhiteSpace(src)), nameof(col.Source));

                int srcCount = col.Source.Length;
                uniqueNames[iinfo] = new string[srcCount];
                for (int isrc = 0; isrc < srcCount; isrc++)
                {
                    string tmpColName;
                    for (; ; )
                    {
                        tmpColName = string.Format("_tmp{0:000}", tmp++);
                        int index;
                        if (!schema.TryGetColumnIndex(tmpColName, out index))
                            break;
                    }

                    uniqueNames[iinfo][isrc] = tmpColName;
                }
            }

            return uniqueNames;
        }
    }
}
