// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(StopWordsRemovingTransformer.Summary, typeof(IDataTransform), typeof(StopWordsRemovingTransformer), typeof(StopWordsRemovingTransformer.Options), typeof(SignatureDataTransform),
    "Stopwords Remover Transform", "StopWordsRemoverTransform", "StopWordsRemover", "StopWords")]

[assembly: LoadableClass(StopWordsRemovingTransformer.Summary, typeof(IDataTransform), typeof(StopWordsRemovingTransformer), null, typeof(SignatureLoadDataTransform),
    "Stopwords Remover Transform", StopWordsRemovingTransformer.LoaderSignature)]

[assembly: LoadableClass(StopWordsRemovingTransformer.Summary, typeof(StopWordsRemovingTransformer), null, typeof(SignatureLoadModel),
    "Stopwords Remover Transform", StopWordsRemovingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(StopWordsRemovingTransformer), null, typeof(SignatureLoadRowMapper),
    "Stopwords Remover Transform", StopWordsRemovingTransformer.LoaderSignature)]

[assembly: LoadableClass(CustomStopWordsRemovingTransformer.Summary, typeof(IDataTransform), typeof(CustomStopWordsRemovingTransformer), typeof(CustomStopWordsRemovingTransformer.Options), typeof(SignatureDataTransform),
    "Custom Stopwords Remover Transform", "CustomStopWordsRemoverTransform", "CustomStopWords")]

[assembly: LoadableClass(CustomStopWordsRemovingTransformer.Summary, typeof(IDataTransform), typeof(CustomStopWordsRemovingTransformer), null, typeof(SignatureLoadDataTransform),
    "Custom Stopwords Remover Transform", CustomStopWordsRemovingTransformer.LoaderSignature)]

[assembly: LoadableClass(CustomStopWordsRemovingTransformer.Summary, typeof(CustomStopWordsRemovingTransformer), null, typeof(SignatureLoadModel),
     "Custom Stopwords Remover Transform", CustomStopWordsRemovingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(CustomStopWordsRemovingTransformer), null, typeof(SignatureLoadRowMapper),
     "Custom Stopwords Remover Transform", CustomStopWordsRemovingTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(PredefinedStopWordsRemoverFactory))]
[assembly: EntryPointModule(typeof(CustomStopWordsRemovingTransformer.LoaderArguments))]

namespace Microsoft.ML.Transforms.Text
{
    [TlcModule.ComponentKind("StopWordsRemover")]
    internal interface IStopWordsRemoverFactory : IComponentFactory<IDataView, OneToOneColumn[], IDataTransform> { }

    [TlcModule.Component(Name = "Predefined", FriendlyName = "Predefined Stopwords List Remover", Alias = "PredefinedStopWordsRemover,PredefinedStopWords",
        Desc = "Remover with predefined list of stop words.")]
    internal sealed class PredefinedStopWordsRemoverFactory : IStopWordsRemoverFactory
    {
        public IDataTransform CreateComponent(IHostEnvironment env, IDataView input, OneToOneColumn[] columns)
        {
            return new StopWordsRemovingEstimator(env, columns.Select(x => new StopWordsRemovingEstimator.ColumnOptions(x.Name, x.Source)).ToArray()).Fit(input).Transform(input) as IDataTransform;
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
        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Optional column to use for languages. This overrides sentence separator language value.",
                ShortName = "langscol")]
            public string LanguagesColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Stopword Language (optional).", ShortName = "lang")]
            public StopWordsRemovingEstimator.Language? Language;

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
                if (!string.IsNullOrWhiteSpace(LanguagesColumn) || Language.HasValue)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

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

        /// <summary>
        /// Defines the behavior of the transformer.
        /// </summary>
        internal IReadOnlyCollection<StopWordsRemovingEstimator.ColumnOptions> Columns => _columns.AsReadOnly();

        private readonly StopWordsRemovingEstimator.ColumnOptions[] _columns;
        private static volatile NormStr.Pool[] _stopWords;
        private static volatile Dictionary<ReadOnlyMemory<char>, StopWordsRemovingEstimator.Language> _langsDictionary;

        private const string RegistrationName = "StopWordsRemover";
        private const string StopWordsDirectoryName = "StopWords";

        private static NormStr.Pool[] StopWords
        {
            get
            {
                NormStr.Pool[] result = _stopWords;
                if (result == null)
                {
                    var values = Enum.GetValues(typeof(StopWordsRemovingEstimator.Language)).Cast<int>();
                    var langValues = values as int[] ?? values.ToArray();
                    int maxValue = langValues.Max();
                    Contracts.Assert(langValues.Min() >= 0);
                    Contracts.Assert(maxValue < Utils.ArrayMaxSize);

                    var stopWords = new NormStr.Pool[maxValue + 1];
                    Interlocked.CompareExchange(ref _stopWords, stopWords, null);
                    result = _stopWords;
                }

                return result;
            }
        }

        private static Dictionary<ReadOnlyMemory<char>, StopWordsRemovingEstimator.Language> LangsDictionary
        {
            get
            {
                Dictionary<ReadOnlyMemory<char>, StopWordsRemovingEstimator.Language> result = _langsDictionary;
                if (result == null)
                {
                    var langsDictionary = Enum.GetValues(typeof(StopWordsRemovingEstimator.Language)).Cast<StopWordsRemovingEstimator.Language>()
                        .ToDictionary(lang => lang.ToString().AsMemory(), new ReadOnlyMemoryUtils.ReadonlyMemoryCharComparer());
                    Interlocked.CompareExchange(ref _langsDictionary, langsDictionary, null);
                    result = _langsDictionary;
                }

                return result;
            }
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(StopWordsRemovingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            if (!StopWordsRemovingEstimator.IsColumnTypeValid(type))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, StopWordsRemovingEstimator.ExpectedColumnType, type.ToString());
        }

        /// <summary>
        /// Stopword remover removes language-specific list of stop words (most common words).
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to remove stop words from.</param>
        internal StopWordsRemovingTransformer(IHostEnvironment env, params StopWordsRemovingEstimator.ColumnOptions[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns;
        }

        private protected override void SaveModel(ModelSaveContext ctx)
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
            _columns = new StopWordsRemovingEstimator.ColumnOptions[columnsLength];
            for (int i = 0; i < columnsLength; i++)
            {
                var lang = (StopWordsRemovingEstimator.Language)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Enum.IsDefined(typeof(StopWordsRemovingEstimator.Language), lang));
                var langColName = ctx.LoadStringOrNull();
                _columns[i] = new StopWordsRemovingEstimator.ColumnOptions(ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, lang, langColName);
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
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new StopWordsRemovingEstimator.ColumnOptions[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];
                cols[i] = new StopWordsRemovingEstimator.ColumnOptions(
                   item.Name,
                   item.Source ?? item.Name,
                   item.Language ?? options.Language,
                   item.LanguagesColumn ?? options.LanguagesColumn);
            }
            return new StopWordsRemovingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

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
            private readonly DataViewType[] _types;
            private readonly StopWordsRemovingTransformer _parent;
            private readonly int[] _languageColumns;
            private readonly bool?[] _resourcesExist;
            private readonly Dictionary<int, int> _colMapNewToOld;

            public Mapper(StopWordsRemovingTransformer parent, DataViewSchema inputSchema)
             : base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _types = new DataViewType[_parent.ColumnPairs.Length];
                _languageColumns = new int[_parent.ColumnPairs.Length];
                _resourcesExist = new bool?[StopWords.Length];
                _colMapNewToOld = new Dictionary<int, int>();

                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int srcCol))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);
                    _parent.CheckInputColumn(InputSchema, i, srcCol);
                    _colMapNewToOld.Add(i, srcCol);

                    var srcType = InputSchema[srcCol].Type;
                    if (!StopWordsRemovingEstimator.IsColumnTypeValid(srcType))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", parent._columns[i].InputColumnName, StopWordsRemovingEstimator.ExpectedColumnType, srcType.ToString());

                    _types[i] = new VectorType(TextDataViewType.Instance);
                    if (!string.IsNullOrEmpty(_parent._columns[i].LanguageColumn))
                    {
                        if (!inputSchema.TryGetColumnIndex(_parent._columns[i].LanguageColumn, out int langCol))
                            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "language", _parent._columns[i].LanguageColumn);
                        _languageColumns[i] = langCol;
                    }
                    else
                        _languageColumns[i] = -1;
                }
                _parent.CheckResources();
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i]);
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                StopWordsRemovingEstimator.Language stopWordslang = _parent._columns[iinfo].Language;
                var lang = default(ReadOnlyMemory<char>);
                var getLang = _languageColumns[iinfo] >= 0 ? input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_languageColumns[iinfo]]) : null;
                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(input.Schema[_colMapNewToOld[iinfo]]);
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

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
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
        /// Use stop words remover that can remove language-specific list of stop words (most common words) already defined in the system.
        /// </summary>
        public sealed class Options : IStopWordsRemoverOptions
        {
            /// <summary>
            /// Language of the text dataset. 'English' is default.
            /// </summary>
            public TextFeaturizingEstimator.Language Language;

            public Options()
            {
                Language = TextFeaturizingEstimator.DefaultLanguage;
            }
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>Name of the column resulting from the transformation of <see cref="InputColumnName"/>.</summary>
            public readonly string Name;
            /// <summary>Name of the column to transform.</summary>
            public readonly string InputColumnName;
            /// <summary>Language-specific stop words list.</summary>
            public readonly Language Language;
            /// <summary>Optional column to use for languages. This overrides language value.</summary>
            public readonly string LanguageColumn;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="language">Language-specific stop words list.</param>
            /// <param name="languageColumn">Optional column to use for languages. This overrides language value.</param>
            public ColumnOptions(string name,
                string inputColumnName = null,
                Language language = Defaults.DefaultLanguage,
                string languageColumn = null)
            {
                Contracts.CheckNonWhiteSpace(name, nameof(name));

                Name = name;
                InputColumnName = inputColumnName ?? name;
                Language = language;
                LanguageColumn = languageColumn;
            }
        }

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

        internal static bool IsColumnTypeValid(DataViewType type) =>
            type is VectorType vectorType && vectorType.ItemType is TextDataViewType;

        internal const string ExpectedColumnType = "vector of Text type";

        /// <summary>
        /// Removes stop words from incoming token streams in <paramref name="inputColumnName"/>
        /// and outputs the token streams without stopwords as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="language">Langauge of the input text column <paramref name="inputColumnName"/>.</param>
        internal StopWordsRemovingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, Language language = Language.English)
            : this(env, new[] { (outputColumnName, inputColumnName ?? outputColumnName) }, language)
        {
        }

        /// <summary>
        /// Removes stop words from incoming token streams in input columns
        /// and outputs the token streams without stop words as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to remove stop words on.</param>
        /// <param name="language">Langauge of the input text columns <paramref name="columns"/>.</param>
        internal StopWordsRemovingEstimator(IHostEnvironment env, (string outputColumnName, string inputColumnName)[] columns, Language language = Language.English)
            : this(env, columns.Select(x => new ColumnOptions(x.outputColumnName, x.inputColumnName, language)).ToArray())
        {
        }

        internal StopWordsRemovingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(StopWordsRemovingEstimator)), new StopWordsRemovingTransformer(env, columns))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                if (col.Kind == SchemaShape.Column.VectorKind.Scalar || !(col.ItemType is TextDataViewType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, ExpectedColumnType, col.ItemType.ToString());
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.VariableVector, TextDataViewType.Instance, false);
            }
            return new SchemaShape(result.Values);
        }
    }

    /// <summary>
    /// Custom stopword remover removes specified list of stop words.
    /// This is usually applied after tokenizing text, so it compares individual tokens
    /// (case-insensitive comparison) to the stopwords.
    /// </summary>
    public sealed class CustomStopWordsRemovingTransformer : OneToOneTransformerBase
    {
        internal sealed class Column : OneToOneColumn
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

        internal abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma separated list of stopwords", Name = "Stopwords", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string Stopword;

            [Argument(ArgumentType.AtMostOnce, HelpText = "List of stopwords", Name = "Stopword", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string[] Stopwords;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Data file containing the stopwords", ShortName = "data", SortOrder = 2, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string DataFile;

            [Argument(ArgumentType.Multiple, HelpText = "Data loader", NullName = "<Auto>", SortOrder = 3, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureDataLoader))]
            internal IComponentFactory<IMultiStreamSource, ILegacyDataLoader> Loader;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the text column containing the stopwords", ShortName = "stopwordsCol", SortOrder = 4, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string StopwordsColumn;
        }

        internal sealed class Options : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        [TlcModule.Component(Name = "Custom", FriendlyName = "Custom Stopwords Remover", Alias = "CustomStopWordsRemover,CustomStopWords",
            Desc = "Remover with list of stopwords specified by the user.")]
        internal sealed class LoaderArguments : ArgumentsBase, IStopWordsRemoverFactory
        {
            public IDataTransform CreateComponent(IHostEnvironment env, IDataView input, OneToOneColumn[] column)
            {
                if (Utils.Size(Stopword) > 0)
                    return new CustomStopWordsRemovingTransformer(env, Stopwords, column.Select(x => (x.Name, x.Source)).ToArray()).Transform(input) as IDataTransform;
                else
                    return new CustomStopWordsRemovingTransformer(env, Stopword, DataFile, StopwordsColumn, Loader, column.Select(x => (x.Name, x.Source)).ToArray()).Transform(input) as IDataTransform;
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
                loaderAssemblyName: typeof(CustomStopWordsRemovingTransformer).Assembly.FullName);
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
                loaderAssemblyName: typeof(CustomStopWordsRemovingTransformer).Assembly.FullName);
        }

        private static readonly DataViewType _outputType = new VectorType(TextDataViewType.Instance);

        private readonly NormStr.Pool _stopWordsMap;
        private const string RegistrationName = "CustomStopWordsRemover";

        private ILegacyDataLoader GetLoaderForStopwords(IChannel ch, string dataFile,
            IComponentFactory<IMultiStreamSource, ILegacyDataLoader> loader, ref string stopwordsCol)
        {
            Host.CheckValue(ch, nameof(ch));

            MultiFileSource fileSource = new MultiFileSource(dataFile);
            ILegacyDataLoader dataLoader;

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
                    ch.CheckUserArg(!string.IsNullOrWhiteSpace(stopwordsCol), nameof(Options.StopwordsColumn),
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

                    // Create text loader.
                    var options = new TextLoader.Options()
                    {
                        Columns = new[]
                        {
                            new TextLoader.Column(stopwordsCol, DataKind.String, 0)
                        },
                        Separators = new[] { ',' },
                    };
                    var textLoader = new TextLoader(Host, options: options, dataSample: fileSource);

                    dataLoader = textLoader.Load(fileSource) as ILegacyDataLoader;
                }
                ch.AssertNonEmpty(stopwordsCol);
            }
            else
                dataLoader = loader.CreateComponent(Host, fileSource);
            return dataLoader;
        }

        private void LoadStopWords(IChannel ch, ReadOnlyMemory<char> stopwords, string dataFile, string stopwordsColumn,
            IComponentFactory<IMultiStreamSource, ILegacyDataLoader> loaderFactory, out NormStr.Pool stopWordsMap)
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
                ch.CheckUserArg(stopWordsMap.Count > 0, nameof(Options.Stopword), "stopwords is empty");
            }
            else
            {
                string srcCol = stopwordsColumn;
                var loader = GetLoaderForStopwords(ch, dataFile, loaderFactory, ref srcCol);

                if (!loader.Schema.TryGetColumnIndex(srcCol, out int colSrcIndex))
                    throw ch.ExceptUserArg(nameof(Options.StopwordsColumn), "Unknown column '{0}'", srcCol);
                var typeSrc = loader.Schema[colSrcIndex].Type;
                ch.CheckUserArg(typeSrc is TextDataViewType, nameof(Options.StopwordsColumn), "Must be a scalar text column");

                // Accumulate the stopwords.
                using (var cursor = loader.GetRowCursor(loader.Schema[srcCol]))
                {
                    bool warnEmpty = true;
                    var getter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor.Schema[colSrcIndex]);
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
                ch.CheckUserArg(stopWordsMap.Count > 0, nameof(Options.DataFile), "dataFile is empty");
            }
        }

        /// <summary>
        /// The names of the input output column pairs on which this transformation is applied.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Custom stopword remover removes specified list of stop words.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        /// <param name="columns">Pairs of columns to remove stop words from.</param>
        internal CustomStopWordsRemovingTransformer(IHostEnvironment env, string[] stopwords, params (string outputColumnName, string inputColumnName)[] columns) :
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

        internal CustomStopWordsRemovingTransformer(IHostEnvironment env, string stopwords,
            string dataFile, string stopwordsColumn, IComponentFactory<IMultiStreamSource, ILegacyDataLoader> loader, params (string outputColumnName, string inputColumnName)[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {
            var ch = Host.Start("LoadStopWords");
            _stopWordsMap = new NormStr.Pool();
            LoadStopWords(ch, stopwords.AsMemory(), dataFile, stopwordsColumn, loader, out _stopWordsMap);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
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

        private CustomStopWordsRemovingTransformer(IHost host, ModelLoadContext ctx) :
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
        private static CustomStopWordsRemovingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new CustomStopWordsRemovingTransformer(host, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new (string outputColumnName, string inputColumnName)[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];
                cols[i] = (item.Name, item.Source ?? item.Name);
            }
            CustomStopWordsRemovingTransformer transfrom = null;
            if (Utils.Size(options.Stopwords) > 0)
                transfrom = new CustomStopWordsRemovingTransformer(env, options.Stopwords, cols);
            else
                transfrom = new CustomStopWordsRemovingTransformer(env, options.Stopword, options.DataFile, options.StopwordsColumn, options.Loader, cols);
            return transfrom.MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
           => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly DataViewType[] _types;
            private readonly CustomStopWordsRemovingTransformer _parent;

            public Mapper(CustomStopWordsRemovingTransformer parent, DataViewSchema inputSchema)
             : base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new DataViewType[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int srcCol);
                    var srcType = inputSchema[srcCol].Type;
                    if (!StopWordsRemovingEstimator.IsColumnTypeValid(srcType))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", parent.ColumnPairs[i].inputColumnName, StopWordsRemovingEstimator.ExpectedColumnType, srcType.ToString());

                    _types[i] = new VectorType(TextDataViewType.Instance);
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _types[i]);
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(input.Schema[ColMapNewToOld[iinfo]]);
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
    public sealed class CustomStopWordsRemovingEstimator : TrivialEstimator<CustomStopWordsRemovingTransformer>
    {
        /// <summary>
        /// Use stop words remover that can removes language-specific list of stop words (most common words) already defined in the system.
        /// </summary>
        public sealed class Options : IStopWordsRemoverOptions
        {
            /// <summary>
            /// List of stop words to remove.
            /// </summary>
            public string[] StopWords;
        }

        internal const string ExpectedColumnType = "vector of Text type";

        /// <summary>
        /// Removes stop words from incoming token streams in <paramref name="inputColumnName"/>
        /// and outputs the token streams without stopwords as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        internal CustomStopWordsRemovingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, params string[] stopwords)
            : this(env, new[] { (outputColumnName, inputColumnName ?? outputColumnName) }, stopwords)
        {
        }

        /// <summary>
        /// Removes stop words from incoming token streams in input columns
        /// and outputs the token streams without stop words as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to remove stop words on.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        internal CustomStopWordsRemovingEstimator(IHostEnvironment env, (string outputColumnName, string inputColumnName)[] columns, string[] stopwords) :
           base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CustomStopWordsRemovingEstimator)), new CustomStopWordsRemovingTransformer(env, stopwords, columns))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.inputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName);
                if (col.Kind == SchemaShape.Column.VectorKind.Scalar || !(col.ItemType is TextDataViewType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.inputColumnName, ExpectedColumnType, col.ItemType.ToString());
                result[colInfo.outputColumnName] = new SchemaShape.Column(colInfo.outputColumnName, SchemaShape.Column.VectorKind.VariableVector, TextDataViewType.Instance, false);
            }
            return new SchemaShape(result.Values);
        }
    }
}
