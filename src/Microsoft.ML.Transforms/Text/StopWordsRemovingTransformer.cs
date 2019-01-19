﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(StopWordsRemovingTransformer.Summary, typeof(IDataTransform), typeof(StopWordsRemovingTransformer), typeof(StopWordsRemovingTransformer.Arguments), typeof(SignatureDataTransform),
    "Stopwords Remover Transform", "StopWordsRemoverTransform", "StopWordsRemover", "StopWords")]

[assembly: LoadableClass(StopWordsRemovingTransformer.Summary, typeof(IDataTransform), typeof(StopWordsRemovingTransformer), null, typeof(SignatureLoadDataTransform),
    "Stopwords Remover Transform", StopWordsRemovingTransformer.LoaderSignature)]

[assembly: LoadableClass(StopWordsRemovingTransformer.Summary, typeof(StopWordsRemovingTransformer), null, typeof(SignatureLoadModel),
    "Stopwords Remover Transform", StopWordsRemovingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(StopWordsRemovingTransformer), null, typeof(SignatureLoadRowMapper),
    "Stopwords Remover Transform", StopWordsRemovingTransformer.LoaderSignature)]

[assembly: LoadableClass(CustomStopWordsRemovingTransform.Summary, typeof(IDataTransform), typeof(CustomStopWordsRemovingTransform), typeof(CustomStopWordsRemovingTransform.Arguments), typeof(SignatureDataTransform),
    "Custom Stopwords Remover Transform", "CustomStopWordsRemoverTransform", "CustomStopWords")]

[assembly: LoadableClass(CustomStopWordsRemovingTransform.Summary, typeof(IDataTransform), typeof(CustomStopWordsRemovingTransform), null, typeof(SignatureLoadDataTransform),
    "Custom Stopwords Remover Transform", CustomStopWordsRemovingTransform.LoaderSignature)]

[assembly: LoadableClass(CustomStopWordsRemovingTransform.Summary, typeof(CustomStopWordsRemovingTransform), null, typeof(SignatureLoadModel),
     "Custom Stopwords Remover Transform", CustomStopWordsRemovingTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(CustomStopWordsRemovingTransform), null, typeof(SignatureLoadRowMapper),
     "Custom Stopwords Remover Transform", CustomStopWordsRemovingTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(PredefinedStopWordsRemoverFactory))]
[assembly: EntryPointModule(typeof(CustomStopWordsRemovingTransform.LoaderArguments))]

namespace Microsoft.ML.Transforms.Text
{
    [TlcModule.ComponentKind("StopWordsRemover")]
    public interface IStopWordsRemoverFactory : IComponentFactory<IDataView, OneToOneColumn[], IDataTransform> { }

    [TlcModule.Component(Name = "Predefined", FriendlyName = "Predefined Stopwords List Remover", Alias = "PredefinedStopWordsRemover,PredefinedStopWords",
        Desc = "Remover with predefined list of stop words.")]
    public sealed class PredefinedStopWordsRemoverFactory : IStopWordsRemoverFactory
    {
        public IDataTransform CreateComponent(IHostEnvironment env, IDataView input, OneToOneColumn[] columns)
        {
            return new StopWordsRemovingEstimator(env, columns.Select(x => new StopWordsRemovingTransformer.ColumnInfo(x.Source, x.Name)).ToArray()).Fit(input).Transform(input) as IDataTransform;
        }
    }

    /// <summary>
    /// A Stopword remover transform based on language-specific lists of stop words (most common words)
    /// from Office Named Entity Recognition project.
    /// The transform is usually applied after tokenizing text, so it compares individual tokens
    /// (case-insensitive comparison) to the stopwords.
    /// </summary>
    public sealed class StopWordsRemovingTransformer : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Optional column to use for languages. This overrides sentence separator language value.",
                ShortName = "langscol")]
            public string LanguagesColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Stopword Language (optional).", ShortName = "lang")]
            public StopWordsRemovingEstimator.Language? Language;

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (!string.IsNullOrWhiteSpace(LanguagesColumn) || Language.HasValue)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Optional column to use for languages. This overrides language value.",
                ShortName = "langscol", SortOrder = 1,
                Purpose = SpecialPurpose.ColumnName)]
            public string LanguagesColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Language-specific stop words list.", ShortName = "lang", SortOrder = 1)]
            public StopWordsRemovingEstimator.Language Language = StopWordsRemovingEstimator.Defaults.DefaultLanguage;
        }

        internal const string Summary = "A Stopword remover transform based on language-specific lists of stop words (most common words) " +
       "from Office Named Entity Recognition project. The transform is usually applied after tokenizing text, so it compares individual tokens " +
       "(case-insensitive comparison) to the stopwords.";

        internal const string LoaderSignature = "StopWordsTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "STOPWRDR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(StopWordsRemovingTransformer).Assembly.FullName);
        }

        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();

        private readonly ColumnInfo[] _columns;
        private static volatile NormStr.Pool[] _stopWords;
        private static volatile Dictionary<ReadOnlyMemory<char>, StopWordsRemovingEstimator.Language> _langsDictionary;

        private const string RegistrationName = "StopWordsRemover";
        private const string StopWordsDirectoryName = "StopWords";

        private static NormStr.Pool[] StopWords
        {
            get
            {
                if (_stopWords == null)
                {
                    var values = Enum.GetValues(typeof(StopWordsRemovingEstimator.Language)).Cast<int>();
                    var langValues = values as int[] ?? values.ToArray();
                    int maxValue = langValues.Max();
                    Contracts.Assert(langValues.Min() >= 0);
                    Contracts.Assert(maxValue < Utils.ArrayMaxSize);

                    var stopWords = new NormStr.Pool[maxValue + 1];
                    Interlocked.CompareExchange(ref _stopWords, stopWords, null);
                }

                return _stopWords;
            }
        }

        private static Dictionary<ReadOnlyMemory<char>, StopWordsRemovingEstimator.Language> LangsDictionary
        {
            get
            {
                if (_langsDictionary == null)
                {
                    var langsDictionary = Enum.GetValues(typeof(StopWordsRemovingEstimator.Language)).Cast<StopWordsRemovingEstimator.Language>()
                        .ToDictionary(lang => lang.ToString().AsMemory(), new ReadOnlyMemoryUtils.ReadonlyMemoryCharComparer());
                    Interlocked.CompareExchange(ref _langsDictionary, langsDictionary, null);
                }

                return _langsDictionary;
            }
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly StopWordsRemovingEstimator.Language Language;
            public readonly string LanguageColumn;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="language">Language-specific stop words list.</param>
            /// <param name="languageColumn">Optional column to use for languages. This overrides language value.</param>
            public ColumnInfo(string input, string output, StopWordsRemovingEstimator.Language language = StopWordsRemovingEstimator.Defaults.DefaultLanguage, string languageColumn = null)
            {
                Input = input;
                Output = output;
                Language = language;
                LanguageColumn = languageColumn;
            }
        }

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        protected override void CheckInputColumn(Schema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            if (!StopWordsRemovingEstimator.IsColumnTypeValid(type))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, StopWordsRemovingEstimator.ExpectedColumnType, type.ToString());
        }

        /// <summary>
        /// Stopword remover removes language-specific list of stop words (most common words).
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to remove stop words from.</param>
        public StopWordsRemovingTransformer(IHostEnvironment env, params ColumnInfo[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns;
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // foreach column:
            //   int: the stopwords list language
            //   string: the id of languages column name
            SaveColumns(ctx);
            foreach (var column in _columns)
            {
                ctx.Writer.Write((int)column.Language);
                ctx.SaveStringOrNull(column.LanguageColumn);
            }
        }

        private StopWordsRemovingTransformer(IHost host, ModelLoadContext ctx) :
            base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            // *** Binary format ***
            // <base>
            // foreach column:
            //   int: the stopwords list language
            //   string: the id of languages column name
            _columns = new ColumnInfo[columnsLength];
            for (int i = 0; i < columnsLength; i++)
            {
                var lang = (StopWordsRemovingEstimator.Language)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Enum.IsDefined(typeof(StopWordsRemovingEstimator.Language), lang));
                var langColName = ctx.LoadStringOrNull();
                _columns[i] = new ColumnInfo(ColumnPairs[i].input, ColumnPairs[i].output, lang, langColName);
            }
        }

        // Factory method for SignatureLoadModel.
        private static StopWordsRemovingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new StopWordsRemovingTransformer(host, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Column[i];
                cols[i] = new ColumnInfo(item.Source ?? item.Name,
                   item.Name,
                   item.Language ?? args.Language,
                   item.LanguagesColumn ?? args.LanguagesColumn);
            }
            return new StopWordsRemovingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        private void CheckResources()
        {
            // Find required resources
            var requiredResources = new bool[StopWords.Length];
            for (int iinfo = 0; iinfo < _columns.Length; iinfo++)
                requiredResources[(int)_columns[iinfo].Language] = true;
            using (var ch = Host.Start("Check resources"))
            {
                // Check the existence of resource files
                var missings = new StringBuilder();
                foreach (StopWordsRemovingEstimator.Language lang in Enum.GetValues(typeof(StopWordsRemovingEstimator.Language)))
                {
                    if (GetResourceFileStreamOrNull(lang) == null)
                    {
                        if (requiredResources[(int)lang])
                        {
                            throw ch.Except(
                                "Missing '{0}.txt' resource.");
                        }

                        if (missings.Length > 0)
                            missings.Append(", ");
                        missings.Append(lang);
                    }
                }

                if (missings.Length > 0)
                {
                    const string wrnMsg = "Missing resources for languages: '{0}'. You can check the following help page for more info: '{1}'. "
                        + "Default stop words list (specified by 'lang' option) will be used if needed.";
                    ch.Warning(wrnMsg, missings.ToString());
                }
            }
        }

        private static void AddResourceIfNotPresent(StopWordsRemovingEstimator.Language lang)
        {
            Contracts.Assert(0 <= (int)lang & (int)lang < Utils.Size(StopWords));

            if (StopWords[(int)lang] == null)
            {
                Stream stopWordsStream = GetResourceFileStreamOrNull(lang);
                Contracts.Assert(stopWordsStream != null);
                var stopWordsList = new NormStr.Pool();
                using (StreamReader reader = new StreamReader(stopWordsStream))
                {
                    string stopWord;
                    while ((stopWord = reader.ReadLine()) != null)
                    {
                        if (!string.IsNullOrWhiteSpace(stopWord))
                            stopWordsList.Add(stopWord);
                    }
                }
                Interlocked.CompareExchange(ref StopWords[(int)lang], stopWordsList, null);
            }
        }

        private static Stream GetResourceFileStreamOrNull(StopWordsRemovingEstimator.Language lang)
        {
            Assembly assembly = Assembly.GetExecutingAssembly();
            return assembly.GetManifestResourceStream($"{assembly.GetName().Name}.Text.StopWords.{lang.ToString()}.txt");
        }

        private sealed class Mapper : MapperBase
        {
            private readonly ColumnType[] _types;
            private readonly StopWordsRemovingTransformer _parent;
            private readonly int[] _languageColumns;
            private readonly bool?[] _resourcesExist;
            private readonly Dictionary<int, int> _colMapNewToOld;

            public Mapper(StopWordsRemovingTransformer parent, Schema inputSchema)
             : base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent.ColumnPairs.Length];
                _languageColumns = new int[_parent.ColumnPairs.Length];
                _resourcesExist = new bool?[StopWords.Length];
                _colMapNewToOld = new Dictionary<int, int>();

                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int srcCol))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].input);
                    _parent.CheckInputColumn(InputSchema, i, srcCol);
                    _colMapNewToOld.Add(i, srcCol);

                    var srcType = InputSchema[srcCol].Type;
                    if (!StopWordsRemovingEstimator.IsColumnTypeValid(srcType))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", parent._columns[i].Input, StopWordsRemovingEstimator.ExpectedColumnType, srcType.ToString());

                    _types[i] = new VectorType(TextType.Instance);
                    if (!string.IsNullOrEmpty(_parent._columns[i].LanguageColumn))
                    {
                        if (!inputSchema.TryGetColumnIndex(_parent._columns[i].LanguageColumn, out int langCol))
                            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "language column", _parent._columns[i].LanguageColumn);
                        _languageColumns[i] = langCol;
                    }
                    else
                        _languageColumns[i] = -1;
                }
                _parent.CheckResources();
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new Schema.DetachedColumn(_parent.ColumnPairs[i].output, _types[i]);
                }
                return result;
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                StopWordsRemovingEstimator.Language stopWordslang = _parent._columns[iinfo].Language;
                var lang = default(ReadOnlyMemory<char>);
                var getLang = _languageColumns[iinfo] >= 0 ? input.GetGetter<ReadOnlyMemory<char>>(_languageColumns[iinfo]) : null;
                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(_colMapNewToOld[iinfo]);
                var src = default(VBuffer<ReadOnlyMemory<char>>);
                var buffer = new StringBuilder();
                var list = new List<ReadOnlyMemory<char>>();

                ValueGetter<VBuffer<ReadOnlyMemory<char>>> del =
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        var langToUse = stopWordslang;
                        UpdateLanguage(ref langToUse, getLang, ref lang);

                        getSrc(ref src);
                        list.Clear();

                        var srcValues = src.GetValues();
                        for (int i = 0; i < srcValues.Length; i++)
                        {
                            if (srcValues[i].IsEmpty)
                                continue;
                            buffer.Clear();
                            ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(srcValues[i].Span, buffer);

                            // REVIEW: Consider using a trie for string matching (Aho-Corasick, etc.)
                            if (StopWords[(int)langToUse].Get(buffer) == null)
                                list.Add(srcValues[i]);
                        }

                        VBufferUtils.Copy(list, ref dst, list.Count);
                    };

                return del;
            }

            private void UpdateLanguage(ref StopWordsRemovingEstimator.Language langToUse, ValueGetter<ReadOnlyMemory<char>> getLang, ref ReadOnlyMemory<char> langTxt)
            {
                if (getLang != null)
                {
                    getLang(ref langTxt);
                    StopWordsRemovingEstimator.Language lang;
                    if (LangsDictionary.TryGetValue(langTxt, out lang))
                        langToUse = lang;
                }

                if (!ResourceExists(langToUse))
                    langToUse = StopWordsRemovingEstimator.Defaults.DefaultLanguage;
                AddResourceIfNotPresent(langToUse);
            }

            private bool ResourceExists(StopWordsRemovingEstimator.Language lang)
            {
                int langVal = (int)lang;
                Contracts.Assert(0 <= langVal & langVal < Utils.Size(StopWords));
                // Note: Updating values in _resourcesExist does not have to be an atomic operation
                return StopWords[langVal] != null ||
                    (_resourcesExist[langVal] ?? (_resourcesExist[langVal] = GetResourceFileStreamOrNull(lang) != null).Value);
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                foreach (var pair in _colMapNewToOld)
                    if (activeOutput(pair.Key))
                    {
                        active[pair.Value] = true;
                        if (_languageColumns[pair.Key] != -1)
                            active[_languageColumns[pair.Key]] = true;
                    }
                return col => active[col];
            }

            public override void Save(ModelSaveContext ctx) => _parent.Save(ctx);
        }
    }

    /// <summary>
    /// Stopword remover removes language-specific list of stop words (most common words)
    /// This is usually applied after tokenizing text, so it compares individual tokens
    /// (case-insensitive comparison) to the stopwords.
    /// </summary>
    public sealed class StopWordsRemovingEstimator : TrivialEstimator<StopWordsRemovingTransformer>
    {
        /// <summary>
        /// Stopwords language. This enumeration is serialized.
        /// </summary>
        public enum Language
        {
            English = 0,
            French = 1,
            German = 2,
            Dutch = 3,
            Danish = 4,
            Swedish = 5,
            Italian = 6,
            Spanish = 7,
            Portuguese = 8,
#pragma warning disable MSML_GeneralName // These names correspond to file names, so this is fine in this case.
            Portuguese_Brazilian = 9,
            Norwegian_Bokmal = 10,
#pragma warning restore MSML_GeneralName
            Russian = 11,
            Polish = 12,
            Czech = 13,
            Arabic = 14,
            Japanese = 15
        }

        internal static class Defaults
        {
            public const Language DefaultLanguage = Language.English;
        }

        public static bool IsColumnTypeValid(ColumnType type) =>
            type is VectorType vectorType && vectorType.ItemType is TextType;

        internal const string ExpectedColumnType = "vector of Text type";

        /// <summary>
        /// Removes stop words from incoming token streams in <paramref name="inputColumn"/>
        /// and outputs the token streams without stopwords as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to remove stop words on.</param>
        /// <param name="outputColumn">The column containing output text. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="language">Langauge of the input text column <paramref name="inputColumn"/>.</param>
        public StopWordsRemovingEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null, Language language = Language.English)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, language)
        {
        }

        /// <summary>
        /// Removes stop words from incoming token streams in input columns
        /// and outputs the token streams without stop words as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to remove stop words on.</param>
        /// <param name="language">Langauge of the input text columns <paramref name="columns"/>.</param>
        public StopWordsRemovingEstimator(IHostEnvironment env, (string input, string output)[] columns, Language language = Language.English)
            : this(env, columns.Select(x => new StopWordsRemovingTransformer.ColumnInfo(x.input, x.output, language)).ToArray())
        {
        }

        public StopWordsRemovingEstimator(IHostEnvironment env, params StopWordsRemovingTransformer.ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(StopWordsRemovingEstimator)), new StopWordsRemovingTransformer(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (col.Kind == SchemaShape.Column.VectorKind.Scalar || !(col.ItemType is TextType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, ExpectedColumnType, col.ItemType.ToString());
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.VariableVector, TextType.Instance, false);
            }
            return new SchemaShape(result.Values);
        }
    }

    /// <summary>
    /// Custom stopword remover removes specified list of stop words.
    /// This is usually applied after tokenizing text, so it compares individual tokens
    /// (case-insensitive comparison) to the stopwords.
    /// </summary>
    public sealed class CustomStopWordsRemovingTransform : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
        {
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
                return TryUnparseCore(sb);
            }
        }

        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma separated list of stopwords", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string Stopwords;

            [Argument(ArgumentType.AtMostOnce, HelpText = "List of stopwords", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string[] Stopword;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Data file containing the stopwords", ShortName = "data", SortOrder = 2, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string DataFile;

            [Argument(ArgumentType.Multiple, HelpText = "Data loader", NullName = "<Auto>", SortOrder = 3, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<IMultiStreamSource, IDataLoader> Loader;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the text column containing the stopwords", ShortName = "stopwordsCol", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string StopwordsColumn;
        }

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        [TlcModule.Component(Name = "Custom", FriendlyName = "Custom Stopwords Remover", Alias = "CustomStopWordsRemover,CustomStopWords",
            Desc = "Remover with list of stopwords specified by the user.")]
        public sealed class LoaderArguments : ArgumentsBase, IStopWordsRemoverFactory
        {
            public IDataTransform CreateComponent(IHostEnvironment env, IDataView input, OneToOneColumn[] column)
            {
                if (Utils.Size(Stopword) > 0)
                    return new CustomStopWordsRemovingTransform(env, Stopword, column.Select(x => (x.Source, x.Name)).ToArray()).Transform(input) as IDataTransform;
                else
                    return new CustomStopWordsRemovingTransform(env, Stopwords, DataFile, StopwordsColumn, Loader, column.Select(x => (x.Source, x.Name)).ToArray()).Transform(input) as IDataTransform;
            }
        }

        internal const string Summary = "A Stopword remover transform based on a user-defined list of stopwords. " +
            "The transform is usually applied after tokenizing text, so it compares individual tokens " +
            "(case-insensitive comparison) to the stopwords.";

        internal const string LoaderSignature = "CustomStopWords";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CSTOPWRD",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CustomStopWordsRemovingTransform).Assembly.FullName);
        }

        private const string StopwordsManagerLoaderSignature = "CustomStopWordsManager";

        private static VersionInfo GetStopwordsManagerVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "STOPWRDM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: StopwordsManagerLoaderSignature,
                loaderAssemblyName: typeof(CustomStopWordsRemovingTransform).Assembly.FullName);
        }

        private static readonly ColumnType _outputType = new VectorType(TextType.Instance);

        private readonly NormStr.Pool _stopWordsMap;
        private const string RegistrationName = "CustomStopWordsRemover";

        private IDataLoader GetLoaderForStopwords(IChannel ch, string dataFile,
            IComponentFactory<IMultiStreamSource, IDataLoader> loader, ref string stopwordsCol)
        {
            Host.CheckValue(ch, nameof(ch));

            MultiFileSource fileSource = new MultiFileSource(dataFile);
            IDataLoader dataLoader;

            // First column using the file.
            if (loader == null)
            {
                // Determine the default loader from the extension.
                var ext = Path.GetExtension(dataFile);
                bool isBinary = string.Equals(ext, ".idv", StringComparison.OrdinalIgnoreCase);
                bool isTranspose = string.Equals(ext, ".tdv", StringComparison.OrdinalIgnoreCase);
                if (isBinary || isTranspose)
                {
                    ch.Assert(isBinary != isTranspose);
                    ch.CheckUserArg(!string.IsNullOrWhiteSpace(stopwordsCol), nameof(Arguments.StopwordsColumn),
                        "stopwordsColumn should be specified");
                    if (isBinary)
                        dataLoader = new BinaryLoader(Host, new BinaryLoader.Arguments(), fileSource);
                    else
                    {
                        ch.Assert(isTranspose);
                        dataLoader = new TransposeLoader(Host, new TransposeLoader.Arguments(), fileSource);
                    }
                }
                else
                {
                    if (stopwordsCol == null)
                        stopwordsCol = "Stopwords";
                    dataLoader = new TextLoader(
                        Host,
                        columns: new[]
                        {
                            new TextLoader.Column(stopwordsCol, DataKind.TX, 0)
                        },
                        dataSample: fileSource).Read(fileSource) as IDataLoader;
                }
                ch.AssertNonEmpty(stopwordsCol);
            }
            else
                dataLoader = loader.CreateComponent(Host, fileSource);
            return dataLoader;
        }

        private void LoadStopWords(IChannel ch, ReadOnlyMemory<char> stopwords, string dataFile, string stopwordsColumn,
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory, out NormStr.Pool stopWordsMap)
        {
            Host.AssertValue(ch);

            var src = default(ReadOnlyMemory<char>);
            stopWordsMap = new NormStr.Pool();
            var buffer = new StringBuilder();

            stopwords = ReadOnlyMemoryUtils.TrimSpaces(stopwords);
            if (!stopwords.IsEmpty)
            {
                bool warnEmpty = true;
                for (bool more = true; more;)
                {
                    ReadOnlyMemory<char> stopword;
                    more = ReadOnlyMemoryUtils.SplitOne(stopwords, ',', out stopword, out stopwords);
                    stopword = ReadOnlyMemoryUtils.TrimSpaces(stopword);
                    if (!stopword.IsEmpty)
                    {
                        buffer.Clear();
                        ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(stopword.Span, buffer);
                        stopWordsMap.Add(buffer);
                    }
                    else if (warnEmpty)
                    {
                        ch.Warning("Empty strings ignored in 'stopwords' specification");
                        warnEmpty = false;
                    }
                }
                ch.CheckUserArg(stopWordsMap.Count > 0, nameof(Arguments.Stopwords), "stopwords is empty");
            }
            else
            {
                string srcCol = stopwordsColumn;
                var loader = GetLoaderForStopwords(ch, dataFile, loaderFactory, ref srcCol);
                int colSrc;
                if (!loader.Schema.TryGetColumnIndex(srcCol, out colSrc))
                    throw ch.ExceptUserArg(nameof(Arguments.StopwordsColumn), "Unknown column '{0}'", srcCol);
                var typeSrc = loader.Schema[colSrc].Type;
                ch.CheckUserArg(typeSrc is TextType, nameof(Arguments.StopwordsColumn), "Must be a scalar text column");

                // Accumulate the stopwords.
                using (var cursor = loader.GetRowCursor(col => col == colSrc))
                {
                    bool warnEmpty = true;
                    var getter = cursor.GetGetter<ReadOnlyMemory<char>>(colSrc);
                    while (cursor.MoveNext())
                    {
                        getter(ref src);
                        if (!src.IsEmpty)
                        {
                            buffer.Clear();
                            ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(src.Span, buffer);
                            stopWordsMap.Add(buffer);
                        }
                        else if (warnEmpty)
                        {
                            ch.Warning("Empty rows ignored in data file");
                            warnEmpty = false;
                        }
                    }
                }
                ch.CheckUserArg(stopWordsMap.Count > 0, nameof(Arguments.DataFile), "dataFile is empty");
            }
        }

        public IReadOnlyCollection<(string input, string output)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Custom stopword remover removes specified list of stop words.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        /// <param name="columns">Pairs of columns to remove stop words from.</param>
        public CustomStopWordsRemovingTransform(IHostEnvironment env, string[] stopwords, params (string input, string output)[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
            _stopWordsMap = new NormStr.Pool();
            var buffer = new StringBuilder();
            foreach (string word in stopwords)
            {
                var stopword = word.AsSpan();
                stopword = stopword.Trim(' ');
                if (!stopword.IsEmpty)
                {
                    buffer.Clear();
                    ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(stopword, buffer);
                    _stopWordsMap.Add(buffer);
                }
            }
        }

        internal CustomStopWordsRemovingTransform(IHostEnvironment env, string stopwords,
            string dataFile, string stopwordsColumn, IComponentFactory<IMultiStreamSource, IDataLoader> loader, params (string input, string output)[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
            var ch = Host.Start("LoadStopWords");
            _stopWordsMap = new NormStr.Pool();
            LoadStopWords(ch, stopwords.AsMemory(), dataFile, stopwordsColumn, loader, out _stopWordsMap);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            SaveColumns(ctx);

            const string dir = "Stopwords";
            ctx.SaveSubModel(dir,
                c =>
                {
                    Host.CheckValue(c, nameof(ctx));
                    c.CheckAtModel();
                    c.SetVersionInfo(GetStopwordsManagerVersionInfo());

                    // *** Binary format ***
                    // int: number of stopwords
                    // int[]: stopwords string ids
                    Host.Assert(_stopWordsMap.Count > 0);
                    ctx.Writer.Write(_stopWordsMap.Count);
                    int id = 0;
                    foreach (var nstr in _stopWordsMap)
                    {
                        Host.Assert(nstr.Id == id);
                        ctx.SaveString(nstr.Value);
                        id++;
                    }

                    ctx.SaveTextStream("Stopwords.txt", writer =>
                    {
                        foreach (var nstr in _stopWordsMap)
                            writer.WriteLine("{0}\t{1}", nstr.Id, nstr.Value);
                    });
                });
        }

        private CustomStopWordsRemovingTransform(IHost host, ModelLoadContext ctx) :
            base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            // *** Binary format ***
            // <base>
            Host.AssertValue(ctx);

            using (var ch = Host.Start("Deserialization"))
            {

                const string dir = "Stopwords";
                NormStr.Pool stopwrods = null;
                bool res = ctx.TryProcessSubModel(dir,
                    c =>
                    {
                        Host.CheckValue(c, nameof(ctx));
                        c.CheckAtModel(GetStopwordsManagerVersionInfo());

                        // *** Binary format ***
                        // int: number of stopwords
                        // int[]: stopwords string ids
                        int cstr = ctx.Reader.ReadInt32();
                        Host.CheckDecode(cstr > 0);

                        stopwrods = new NormStr.Pool();
                        for (int istr = 0; istr < cstr; istr++)
                        {
                            var nstr = stopwrods.Add(ctx.LoadString());
                            Host.CheckDecode(nstr.Id == istr);
                        }

                        // All stopwords are distinct.
                        Host.CheckDecode(stopwrods.Count == cstr);
                        // The deserialized pool should not have the empty string.
                        Host.CheckDecode(stopwrods.Get("") == null);
                    });
                if (!res)
                    throw Host.ExceptDecode();

                _stopWordsMap = stopwrods;
            }
        }

        // Factory method for SignatureLoadModel.
        private static CustomStopWordsRemovingTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new CustomStopWordsRemovingTransform(host, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new (string input, string output)[args.Column.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Column[i];
                cols[i] = (item.Source ?? item.Name, item.Name);
            }
            CustomStopWordsRemovingTransform transfrom = null;
            if (Utils.Size(args.Stopword) > 0)
                transfrom = new CustomStopWordsRemovingTransform(env, args.Stopword, cols);
            else
                transfrom = new CustomStopWordsRemovingTransform(env, args.Stopwords, args.DataFile, args.StopwordsColumn, args.Loader, cols);
            return transfrom.MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
           => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ColumnType[] _types;
            private readonly CustomStopWordsRemovingTransform _parent;

            public Mapper(CustomStopWordsRemovingTransform parent, Schema inputSchema)
             : base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int srcCol);
                    var srcType = inputSchema[srcCol].Type;
                    if (!StopWordsRemovingEstimator.IsColumnTypeValid(srcType))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", parent.ColumnPairs[i].input, StopWordsRemovingEstimator.ExpectedColumnType, srcType.ToString());

                    _types[i] = new VectorType(TextType.Instance);
                }
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new Schema.DetachedColumn(_parent.ColumnPairs[i].output, _types[i]);
                }
                return result;
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(ColMapNewToOld[iinfo]);
                var src = default(VBuffer<ReadOnlyMemory<char>>);
                var buffer = new StringBuilder();
                var list = new List<ReadOnlyMemory<char>>();

                ValueGetter<VBuffer<ReadOnlyMemory<char>>> del =
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        getSrc(ref src);
                        list.Clear();

                        var srcValues = src.GetValues();
                        for (int i = 0; i < srcValues.Length; i++)
                        {
                            if (srcValues[i].IsEmpty)
                                continue;
                            buffer.Clear();
                            ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(srcValues[i].Span, buffer);

                            // REVIEW: Consider using a trie for string matching (Aho-Corasick, etc.)
                            if (_parent._stopWordsMap.Get(buffer) == null)
                                list.Add(srcValues[i]);
                        }

                        VBufferUtils.Copy(list, ref dst, list.Count);
                    };

                return del;
            }
        }
    }

    /// <summary>
    /// Custom stopword remover removes specified list of stop words.
    /// This is usually applied after tokenizing text, so it compares individual tokens
    /// (case-insensitive comparison) to the stopwords.
    /// </summary>
    public sealed class CustomStopWordsRemovingEstimator : TrivialEstimator<CustomStopWordsRemovingTransform>
    {
        internal const string ExpectedColumnType = "vector of Text type";

        /// <summary>
        /// Removes stop words from incoming token streams in <paramref name="inputColumn"/>
        /// and outputs the token streams without stopwords as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to remove stop words on.</param>
        /// <param name="outputColumn">The column containing output text. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        public CustomStopWordsRemovingEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null, params string[] stopwords)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, stopwords)
        {
        }

        /// <summary>
        /// Removes stop words from incoming token streams in input columns
        /// and outputs the token streams without stop words as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to remove stop words on.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        public CustomStopWordsRemovingEstimator(IHostEnvironment env, (string input, string output)[] columns, string[] stopwords) :
           base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CustomStopWordsRemovingEstimator)), new CustomStopWordsRemovingTransform(env, stopwords, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input);
                if (col.Kind == SchemaShape.Column.VectorKind.Scalar || !(col.ItemType is TextType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input, ExpectedColumnType, col.ItemType.ToString());
                result[colInfo.output] = new SchemaShape.Column(colInfo.output, SchemaShape.Column.VectorKind.VariableVector, TextType.Instance, false);
            }
            return new SchemaShape(result.Values);
        }
    }
}