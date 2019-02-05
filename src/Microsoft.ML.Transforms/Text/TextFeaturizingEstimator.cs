﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Transforms.Projections;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(TextFeaturizingEstimator.Summary, typeof(IDataTransform), typeof(TextFeaturizingEstimator), typeof(TextFeaturizingEstimator.Arguments), typeof(SignatureDataTransform),
    TextFeaturizingEstimator.UserName, "TextTransform", TextFeaturizingEstimator.LoaderSignature)]

[assembly: LoadableClass(TextFeaturizingEstimator.Summary, typeof(ITransformer), typeof(TextFeaturizingEstimator), null, typeof(SignatureLoadModel),
    TextFeaturizingEstimator.UserName, "TextTransform", TextFeaturizingEstimator.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    using CaseNormalizationMode = TextNormalizingEstimator.CaseNormalizationMode;
    // A transform that turns a collection of text documents into numerical feature vectors. The feature vectors are counts
    // of (word or character) ngrams in a given text. It offers ngram hashing (finding the ngram token string name to feature
    // integer index mapping through hashing) as an option.
    /// <include file='doc.xml' path='doc/members/member[@name="TextFeaturizingEstimator "]/*' />
    public sealed class TextFeaturizingEstimator : IEstimator<ITransformer>
    {
        /// <summary>
        /// Text language. This enumeration is serialized.
        /// </summary>
        public enum Language
        {
            English = 1,
            French = 2,
            German = 3,
            Dutch = 4,
            Italian = 5,
            Spanish = 6,
            Japanese = 7
        }

        /// <summary>
        /// Text vector normalizer kind.
        /// </summary>
        public enum TextNormKind
        {
            None = 0,
            L1 = 1,
            L2 = 2,
            LInf = 3
        }

        internal sealed class Column : ManyToOneColumn
        {
            internal static Column Parse(string str)
            {
                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        /// <summary>
        /// This class exposes <see cref="NgramExtractorTransform"/>/<see cref="NgramHashExtractingTransformer"/> arguments.
        /// </summary>
        internal sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "New column definition (optional form: name:srcs).", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dataset language or 'AutoDetect' to detect language per row.", ShortName = "lang", SortOrder = 3)]
            public Language Language = DefaultLanguage;

            [Argument(ArgumentType.Multiple, HelpText = "Use stop remover or not.", ShortName = "remover", SortOrder = 4)]
            public bool UsePredefinedStopWordRemover = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Casing text using the rules of the invariant culture.", ShortName = "case", SortOrder = 5)]
            public CaseNormalizationMode TextCase = TextNormalizingEstimator.Defaults.TextCase;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep diacritical marks or remove them.", ShortName = "diac", SortOrder = 6)]
            public bool KeepDiacritics = TextNormalizingEstimator.Defaults.KeepDiacritics;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep punctuation marks or remove them.", ShortName = "punc", SortOrder = 7)]
            public bool KeepPunctuations = TextNormalizingEstimator.Defaults.KeepPunctuations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to keep numbers or remove them.", ShortName = "num", SortOrder = 8)]
            public bool KeepNumbers = TextNormalizingEstimator.Defaults.KeepNumbers;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to output the transformed text tokens as an additional column.", ShortName = "tokens,showtext,showTransformedText", SortOrder = 9)]
            public bool OutputTokens;

            [Argument(ArgumentType.Multiple, HelpText = "A dictionary of whitelisted terms.", ShortName = "dict", NullName = "<None>", SortOrder = 10, Hide = true)]
            public TermLoaderArguments Dictionary;

            [TGUI(Label = "Word Gram Extractor")]
            [Argument(ArgumentType.Multiple, HelpText = "Ngram feature extractor to use for words (WordBag/WordHashBag).", ShortName = "wordExtractor", NullName = "<None>", SortOrder = 11)]
            public INgramExtractorFactoryFactory WordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments();

            [TGUI(Label = "Char Gram Extractor")]
            [Argument(ArgumentType.Multiple, HelpText = "Ngram feature extractor to use for characters (WordBag/WordHashBag).", ShortName = "charExtractor", NullName = "<None>", SortOrder = 12)]
            public INgramExtractorFactoryFactory CharFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3, AllLengths = false };

            [Argument(ArgumentType.AtMostOnce, HelpText = "Normalize vectors (rows) individually by rescaling them to unit norm.", ShortName = "norm", SortOrder = 13)]
            public TextNormKind VectorNormalizer = TextNormKind.L2;
        }

        public sealed class Settings
        {
#pragma warning disable MSML_NoInstanceInitializers // No initializers on instance fields or properties
            public Language TextLanguage { get; set; } = DefaultLanguage;
            public CaseNormalizationMode TextCase { get; set; } = CaseNormalizationMode.Lower;
            public bool KeepDiacritics { get; set; } = false;
            public bool KeepPunctuations { get; set; } = true;
            public bool KeepNumbers { get; set; } = true;
            public bool OutputTokens { get; set; } = false;
            public TextNormKind VectorNormalizer { get; set; } = TextNormKind.L2;
            public bool UseStopRemover { get; set; } = false;
            public bool UseCharExtractor { get; set; } = true;
            public bool UseWordExtractor { get; set; } = true;
#pragma warning restore MSML_NoInstanceInitializers // No initializers on instance fields or properties
        }

        public readonly string OutputColumn;
        private readonly string[] _inputColumns;
        public IReadOnlyCollection<string> InputColumns => _inputColumns.AsReadOnly();
        public Settings AdvancedSettings { get; }

        // These parameters are hardcoded for now.
        // REVIEW: expose them once sub-transforms are estimators.
        private TermLoaderArguments _dictionary;
        private INgramExtractorFactoryFactory _wordFeatureExtractor;
        private INgramExtractorFactoryFactory _charFeatureExtractor;

        private readonly IHost _host;

        /// <summary>
        /// A distilled version of the TextFeaturizingEstimator  Arguments, with all fields marked readonly and
        /// only the exact set of information needed to construct the transforms preserved.
        /// </summary>
        private sealed class TransformApplierParams
        {
            public readonly INgramExtractorFactory WordExtractorFactory;
            public readonly INgramExtractorFactory CharExtractorFactory;

            public readonly TextNormKind VectorNormalizer;
            public readonly Language Language;
            public readonly bool UsePredefinedStopWordRemover;
            public readonly CaseNormalizationMode TextCase;
            public readonly bool KeepDiacritics;
            public readonly bool KeepPunctuations;
            public readonly bool KeepNumbers;
            public readonly bool OutputTextTokens;
            public readonly TermLoaderArguments Dictionary;

            public StopWordsRemovingEstimator.Language StopwordsLanguage
                => (StopWordsRemovingEstimator.Language)Enum.Parse(typeof(StopWordsRemovingEstimator.Language), Language.ToString());

            public LpNormalizingEstimatorBase.NormalizerKind LpNormalizerKind
            {
                get
                {
                    switch (VectorNormalizer)
                    {
                        case TextNormKind.L1:
                            return LpNormalizingEstimatorBase.NormalizerKind.L1Norm;
                        case TextNormKind.L2:
                            return LpNormalizingEstimatorBase.NormalizerKind.L2Norm;
                        case TextNormKind.LInf:
                            return LpNormalizingEstimatorBase.NormalizerKind.LInf;
                        default:
                            Contracts.Assert(false, "Unexpected normalizer type");
                            return LpNormalizingEstimatorBase.NormalizerKind.L2Norm;
                    }
                }
            }

            // These properties encode the logic needed to determine which transforms to apply.
            #region NeededTransforms
            public bool NeedsWordTokenizationTransform { get { return WordExtractorFactory != null || UsePredefinedStopWordRemover || OutputTextTokens; } }

            public bool NeedsNormalizeTransform
            {
                get
                {
                    return
                        TextCase != CaseNormalizationMode.None ||
                        !KeepDiacritics ||
                        !KeepPunctuations ||
                        !KeepNumbers;
                }
            }

            private bool UsesHashExtractors
            {
                get
                {
                    return
                        (WordExtractorFactory == null ? true : WordExtractorFactory.UseHashingTrick) &&
                        (CharExtractorFactory == null ? true : CharExtractorFactory.UseHashingTrick);
                }
            }

            // If we're performing language auto detection, or either of our extractors aren't hashing then
            // we need all the input text concatenated into a single ReadOnlyMemory, for the LanguageDetectionTransform
            // to operate on the entire text vector, and for the Dictionary feature extractor to build its bound dictionary
            // correctly.
            public bool NeedInitialSourceColumnConcatTransform
            {
                get
                {
                    return !UsesHashExtractors;
                }
            }
            #endregion

            public TransformApplierParams(TextFeaturizingEstimator parent)
            {
                var host = parent._host;
                host.Check(Enum.IsDefined(typeof(Language), parent.AdvancedSettings.TextLanguage));
                host.Check(Enum.IsDefined(typeof(CaseNormalizationMode), parent.AdvancedSettings.TextCase));
                WordExtractorFactory = parent._wordFeatureExtractor?.CreateComponent(host, parent._dictionary);
                CharExtractorFactory = parent._charFeatureExtractor?.CreateComponent(host, parent._dictionary);
                VectorNormalizer = parent.AdvancedSettings.VectorNormalizer;
                Language = parent.AdvancedSettings.TextLanguage;
                UsePredefinedStopWordRemover = parent.AdvancedSettings.UseStopRemover;
                TextCase = parent.AdvancedSettings.TextCase;
                KeepDiacritics = parent.AdvancedSettings.KeepDiacritics;
                KeepPunctuations = parent.AdvancedSettings.KeepPunctuations;
                KeepNumbers = parent.AdvancedSettings.KeepNumbers;
                OutputTextTokens = parent.AdvancedSettings.OutputTokens;
                Dictionary = parent._dictionary;
            }
        }

        internal const string Summary = "A transform that turns a collection of text documents into numerical feature vectors. " +
            "The feature vectors are normalized counts of (word and/or character) ngrams in a given tokenized text.";

        internal const string UserName = "Text Transform";
        internal const string LoaderSignature = "Text";

        public const Language DefaultLanguage = Language.English;

        private const string TransformedTextColFormat = "{0}_TransformedText";

        public TextFeaturizingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
            Action<Settings> advancedSettings = null)
            : this(env, outputColumnName, new[] { inputColumnName ?? outputColumnName }, advancedSettings)
        {
        }

        public TextFeaturizingEstimator(IHostEnvironment env, string name, IEnumerable<string> source,
            Action<Settings> advancedSettings = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TextFeaturizingEstimator));
            _host.CheckValue(source, nameof(source));
            _host.CheckParam(source.Any(), nameof(source));
            _host.CheckParam(!source.Any(string.IsNullOrWhiteSpace), nameof(source));
            _host.CheckNonEmpty(name, nameof(name));
            _host.CheckValueOrNull(advancedSettings);

            _inputColumns = source.ToArray();
            OutputColumn = name;

            AdvancedSettings = new Settings();
            advancedSettings?.Invoke(AdvancedSettings);

            _dictionary = null;
            if (AdvancedSettings.UseWordExtractor)
                _wordFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments();
            if (AdvancedSettings.UseCharExtractor)
                _charFeatureExtractor = new NgramExtractorTransform.NgramExtractorArguments() { NgramLength = 3, AllLengths = false };
        }

        public ITransformer Fit(IDataView input)
        {
            var h = _host;
            h.CheckValue(input, nameof(input));

            var tparams = new TransformApplierParams(this);
            string[] textCols = _inputColumns;
            string[] wordTokCols = null;
            string[] charTokCols = null;
            string wordFeatureCol = null;
            string charFeatureCol = null;
            List<string> tempCols = new List<string>();
            IDataView view = input;

            if (tparams.NeedInitialSourceColumnConcatTransform && textCols.Length > 1)
            {
                var srcCols = textCols;
                textCols = new[] { GenerateColumnName(input.Schema, OutputColumn, "InitialConcat") };
                tempCols.Add(textCols[0]);
                view = new ColumnConcatenatingTransformer(h, textCols[0], srcCols).Transform(view);
            }

            if (tparams.NeedsNormalizeTransform)
            {
                var xfCols = new (string outputColumnName, string inputColumnName)[textCols.Length];
                string[] dstCols = new string[textCols.Length];
                for (int i = 0; i < textCols.Length; i++)
                {
                    dstCols[i] = GenerateColumnName(view.Schema, textCols[i], "TextNormalizer");
                    tempCols.Add(dstCols[i]);
                    xfCols[i] = (dstCols[i], textCols[i]);
                }

                view = new TextNormalizingEstimator(h, tparams.TextCase, tparams.KeepDiacritics, tparams.KeepPunctuations, tparams.KeepNumbers, xfCols).Fit(view).Transform(view);

                textCols = dstCols;
            }

            if (tparams.NeedsWordTokenizationTransform)
            {
                var xfCols = new WordTokenizingTransformer.ColumnInfo[textCols.Length];
                wordTokCols = new string[textCols.Length];
                for (int i = 0; i < textCols.Length; i++)
                {
                    var col = new WordTokenizingTransformer.ColumnInfo(GenerateColumnName(view.Schema, textCols[i], "WordTokenizer"), textCols[i]);
                    xfCols[i] = col;
                    wordTokCols[i] = col.Name;
                    tempCols.Add(col.Name);
                }

                view = new WordTokenizingEstimator(h, xfCols).Fit(view).Transform(view);
            }

            if (tparams.UsePredefinedStopWordRemover)
            {
                Contracts.Assert(wordTokCols != null, "StopWords transform requires that word tokenization has been applied to the input text.");
                var xfCols = new StopWordsRemovingTransformer.ColumnInfo[wordTokCols.Length];
                var dstCols = new string[wordTokCols.Length];
                for (int i = 0; i < wordTokCols.Length; i++)
                {
                    var tempName = GenerateColumnName(view.Schema, wordTokCols[i], "StopWordsRemoverTransform");
                    var col = new StopWordsRemovingTransformer.ColumnInfo(tempName, wordTokCols[i], tparams.StopwordsLanguage);
                    dstCols[i] = tempName;
                    tempCols.Add(tempName);

                    xfCols[i] = col;
                }
                view = new StopWordsRemovingEstimator(h, xfCols).Fit(view).Transform(view);
                wordTokCols = dstCols;
            }

            if (tparams.WordExtractorFactory != null)
            {
                var dstCol = GenerateColumnName(view.Schema, OutputColumn, "WordExtractor");
                tempCols.Add(dstCol);
                view = tparams.WordExtractorFactory.Create(h, view, new[] {
                    new ExtractorColumn()
                    {
                        Name = dstCol,
                        Source = wordTokCols,
                        FriendlyNames = _inputColumns
                    }});
                wordFeatureCol = dstCol;
            }

            if (tparams.OutputTextTokens)
            {
                string[] srcCols = wordTokCols ?? textCols;
                view = new ColumnConcatenatingTransformer(h, string.Format(TransformedTextColFormat, OutputColumn), srcCols).Transform(view);
            }

            if (tparams.CharExtractorFactory != null)
            {
                {
                    var srcCols = tparams.UsePredefinedStopWordRemover ? wordTokCols : textCols;
                    charTokCols = new string[srcCols.Length];
                    var xfCols = new (string outputColumnName, string inputColumnName)[srcCols.Length];
                    for (int i = 0; i < srcCols.Length; i++)
                    {
                        xfCols[i] = (GenerateColumnName(view.Schema, srcCols[i], "CharTokenizer"), srcCols[i]);
                        tempCols.Add(xfCols[i].outputColumnName);
                        charTokCols[i] = xfCols[i].outputColumnName;
                    }
                    view = new TokenizingByCharactersTransformer(h, columns: xfCols).Transform(view);
                }

                {
                    charFeatureCol = GenerateColumnName(view.Schema, OutputColumn, "CharExtractor");
                    tempCols.Add(charFeatureCol);
                    view = tparams.CharExtractorFactory.Create(h, view, new[] {
                        new ExtractorColumn()
                        {
                            Source = charTokCols,
                            FriendlyNames = _inputColumns,
                            Name = charFeatureCol
                        }});
                }
            }

            if (tparams.VectorNormalizer != TextNormKind.None)
            {
                var xfCols = new List<LpNormalizingEstimator.LpNormColumnInfo>(2);

                if (charFeatureCol != null)
                {
                    var dstCol = GenerateColumnName(view.Schema, charFeatureCol, "LpCharNorm");
                    tempCols.Add(dstCol);
                    xfCols.Add(new LpNormalizingEstimator.LpNormColumnInfo(dstCol, charFeatureCol, normalizerKind: tparams.LpNormalizerKind));
                    charFeatureCol = dstCol;
                }

                if (wordFeatureCol != null)
                {
                    var dstCol = GenerateColumnName(view.Schema, wordFeatureCol, "LpWordNorm");
                    tempCols.Add(dstCol);
                    xfCols.Add(new LpNormalizingEstimator.LpNormColumnInfo(dstCol, wordFeatureCol, normalizerKind: tparams.LpNormalizerKind));
                    wordFeatureCol = dstCol;
                }

                if (xfCols.Count > 0)
                    view = new LpNormalizingTransformer(h, xfCols.ToArray()).Transform(view);
            }

            {
                var srcTaggedCols = new List<KeyValuePair<string, string>>(2);
                if (charFeatureCol != null && wordFeatureCol != null)
                {
                    // If we're producing both char and word grams, then we need to disambiguate
                    // between them (for example, the word 'a' vs. the char gram 'a').
                    srcTaggedCols.Add(new KeyValuePair<string, string>("Char", charFeatureCol));
                    srcTaggedCols.Add(new KeyValuePair<string, string>("Word", wordFeatureCol));
                }
                else
                {
                    // Otherwise, simply use the slot names, omitting the original source column names
                    // entirely. For the Concat transform setting the Key == Value of the TaggedColumn
                    // KVP signals this intent.
                    Contracts.Assert(charFeatureCol != null || wordFeatureCol != null || tparams.OutputTextTokens);
                    if (charFeatureCol != null)
                        srcTaggedCols.Add(new KeyValuePair<string, string>(charFeatureCol, charFeatureCol));
                    else if (wordFeatureCol != null)
                        srcTaggedCols.Add(new KeyValuePair<string, string>(wordFeatureCol, wordFeatureCol));
                }
                if (srcTaggedCols.Count > 0)
                {
                    view = new ColumnConcatenatingTransformer(h, new ColumnConcatenatingTransformer.ColumnInfo(OutputColumn,
                        srcTaggedCols.Select(kvp => (kvp.Value, kvp.Key))))
                        .Transform(view);
                }
            }

            view = ColumnSelectingTransformer.CreateDrop(h, view, tempCols.ToArray());
            return new Transformer(_host, input, view);
        }

        public static ITransformer Create(IHostEnvironment env, ModelLoadContext ctx)
            => new Transformer(env, ctx);

        private static string GenerateColumnName(Schema schema, string srcName, string xfTag)
        {
            return schema.GetTempColumnName(string.Format("{0}_{1}", srcName, xfTag));
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var srcName in _inputColumns)
            {
                if (!inputSchema.TryFindColumn(srcName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", srcName);
                if (!(col.ItemType is TextType))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", srcName, "scalar or vector of text", col.GetTypeString());
            }

            var metadata = new List<SchemaShape.Column>(2);
            metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false));
            if (AdvancedSettings.VectorNormalizer != TextNormKind.None)
                metadata.Add(new SchemaShape.Column(MetadataUtils.Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));

            result[OutputColumn] = new SchemaShape.Column(OutputColumn, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false,
                new SchemaShape(metadata));
            if (AdvancedSettings.OutputTokens)
            {
                string name = string.Format(TransformedTextColFormat, OutputColumn);
                result[name] = new SchemaShape.Column(name, SchemaShape.Column.VectorKind.VariableVector, TextType.Instance, false);
            }

            return new SchemaShape(result.Values);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView data)
        {
            Action<Settings> settings = s =>
            {
                s.TextLanguage = args.Language;
                s.TextCase = args.TextCase;
                s.KeepDiacritics = args.KeepDiacritics;
                s.KeepPunctuations = args.KeepPunctuations;
                s.KeepNumbers = args.KeepNumbers;
                s.OutputTokens = args.OutputTokens;
                s.VectorNormalizer = args.VectorNormalizer;
                s.UseStopRemover = args.UsePredefinedStopWordRemover;
                s.UseWordExtractor = args.WordFeatureExtractor != null;
                s.UseCharExtractor = args.CharFeatureExtractor != null;
            };

            var estimator = new TextFeaturizingEstimator(env, args.Columns.Name, args.Columns.Source ?? new[] { args.Columns.Name }, settings);
            estimator._dictionary = args.Dictionary;
            estimator._wordFeatureExtractor = args.WordFeatureExtractor;
            estimator._charFeatureExtractor = args.CharFeatureExtractor;
            return estimator.Fit(data).Transform(data) as IDataTransform;
        }

        private sealed class Transformer : ITransformer, ICanSaveModel
        {
            private const string TransformDirTemplate = "Step_{0:000}";

            private readonly IHost _host;
            private readonly IDataView _xf;

            public Transformer(IHostEnvironment env, IDataView input, IDataView view)
            {
                _host = env.Register(nameof(Transformer));
                _xf = ApplyTransformUtils.ApplyAllTransformsToData(_host, view, new EmptyDataView(_host, input.Schema), input);
            }

            public Schema GetOutputSchema(Schema inputSchema)
            {
                _host.CheckValue(inputSchema, nameof(inputSchema));
                return Transform(new EmptyDataView(_host, inputSchema)).Schema;
            }

            public IDataView Transform(IDataView input)
            {
                _host.CheckValue(input, nameof(input));
                return ApplyTransformUtils.ApplyAllTransformsToData(_host, _xf, input);
            }

            public bool IsRowToRowMapper => true;

            public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
            {
                _host.CheckValue(inputSchema, nameof(inputSchema));
                var input = new EmptyDataView(_host, inputSchema);
                var revMaps = new List<IRowToRowMapper>();
                IDataView chain;
                for (chain = ApplyTransformUtils.ApplyAllTransformsToData(_host, _xf, input); chain is IDataTransform xf; chain = xf.Source)
                {
                    // Everything in the chain ought to be a row mapper.
                    _host.Assert(xf is IRowToRowMapper);
                    revMaps.Add((IRowToRowMapper)xf);
                }
                // The walkback should have ended at the input.
                Contracts.Assert(chain == input);
                revMaps.Reverse();
                return new CompositeRowToRowMapper(inputSchema, revMaps.ToArray());
            }

            public void Save(ModelSaveContext ctx)
            {
                _host.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel();
                ctx.SetVersionInfo(GetVersionInfo());

                var dataPipe = _xf;
                var transforms = new List<IDataTransform>();
                while (dataPipe is IDataTransform xf)
                {
                    transforms.Add(xf);
                    dataPipe = xf.Source;
                    Contracts.AssertValue(dataPipe);
                }
                transforms.Reverse();

                ctx.SaveSubModel("Loader", c => BinaryLoader.SaveInstance(_host, c, dataPipe.Schema));

                ctx.Writer.Write(transforms.Count);
                for (int i = 0; i < transforms.Count; i++)
                {
                    var dirName = string.Format(TransformDirTemplate, i);
                    ctx.SaveModel(transforms[i], dirName);
                }
            }

            public Transformer(IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(Transformer));
                _host.CheckValue(ctx, nameof(ctx));

                ctx.CheckAtModel(GetVersionInfo());
                int n = ctx.Reader.ReadInt32();

                ctx.LoadModel<IDataLoader, SignatureLoadDataLoader>(env, out var loader, "Loader", new MultiFileSource(null));

                IDataView data = loader;
                for (int i = 0; i < n; i++)
                {
                    var dirName = string.Format(TransformDirTemplate, i);
                    ctx.LoadModel<IDataTransform, SignatureLoadDataTransform>(env, out var xf, dirName, data);
                    data = xf;
                }

                _xf = data;
            }

            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "TEXT XFR",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(Transformer).Assembly.FullName);
            }
        }
    }
}
