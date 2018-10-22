// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;

[assembly: LoadableClass(StopWordsRemoverTransform.Summary, typeof(IDataTransform), typeof(StopWordsRemoverTransform), typeof(StopWordsRemoverTransform.Arguments), typeof(SignatureDataTransform),
    "Stopwords Remover Transform", "StopWordsRemoverTransform", "StopWordsRemover", "StopWords")]

/*[assembly: LoadableClass(StopWordsRemoverTransform.Summary, typeof(StopWordsRemoverTransform), null, typeof(SignatureStopWordsRemoverTransform),
    "Predefined Stopwords List Remover", "PredefinedStopWordsRemoverTransform", "PredefinedStopWordsRemover", "PredefinedStopWords", "Predefined")]*/

[assembly: LoadableClass(StopWordsRemoverTransform.Summary, typeof(IDataTransform), typeof(StopWordsRemoverTransform), null, typeof(SignatureLoadDataTransform),
    "Stopwords Remover Transform", StopWordsRemoverTransform.LoaderSignature)]

[assembly: LoadableClass(StopWordsRemoverTransform.Summary, typeof(StopWordsRemoverTransform), null, typeof(SignatureLoadModel),
    "Stopwords Remover Transform", StopWordsRemoverTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(StopWordsRemoverTransform), null, typeof(SignatureLoadRowMapper),
    "Stopwords Remover Transform", StopWordsRemoverTransform.LoaderSignature)]

[assembly: LoadableClass(CustomStopWordsRemoverTransform.Summary, typeof(IDataTransform), typeof(CustomStopWordsRemoverTransform), typeof(CustomStopWordsRemoverTransform.Arguments), typeof(SignatureDataTransform),
    "Custom Stopwords Remover Transform", "CustomStopWordsRemoverTransform", "CustomStopWords")]

/*[assembly: LoadableClass(CustomStopWordsRemoverTransform.Summary, typeof(CustomStopWordsRemoverTransform), typeof(CustomStopWordsRemoverTransform.LoaderArguments),
    typeof(SignatureStopWordsRemoverTransform), "Custom Stopwords Remover Transform", "CustomStopWordsRemoverTransform", "CustomStopWords", "Custom")]*/

[assembly: LoadableClass(CustomStopWordsRemoverTransform.Summary, typeof(IDataTransform), typeof(CustomStopWordsRemoverTransform), null, typeof(SignatureLoadDataTransform),
    "Custom Stopwords Remover Transform", CustomStopWordsRemoverTransform.LoaderSignature)]

[assembly: LoadableClass(CustomStopWordsRemoverTransform.Summary, typeof(CustomStopWordsRemoverTransform), null, typeof(SignatureLoadModel),
     "Custom Stopwords Remover Transform", StopWordsRemoverTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(CustomStopWordsRemoverTransform), null, typeof(SignatureLoadRowMapper),
     "Custom Stopwords Remover Transform", CustomStopWordsRemoverTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(PredefinedStopWordsRemoverFactory))]
[assembly: EntryPointModule(typeof(CustomStopWordsRemoverTransform.LoaderArguments))]

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// Signature for creating an IStopWordsRemoverTransform.
    /// </summary>
    public delegate void SignatureStopWordsRemoverTransform(IDataView input, OneToOneColumn[] column);

    public interface IStopWordsRemoverTransform : IDataTransform { }

    [TlcModule.ComponentKind("StopWordsRemover")]
    public interface IStopWordsRemoverFactory : IComponentFactory<IDataView, OneToOneColumn[], IStopWordsRemoverTransform> { }

    [TlcModule.Component(Name = "Predefined", FriendlyName = "Predefined Stopwords List Remover", Alias = "PredefinedStopWordsRemover,PredefinedStopWords",
        Desc = "Remover with predefined list of stop words.")]
    public sealed class PredefinedStopWordsRemoverFactory : IStopWordsRemoverFactory
    {
        public IStopWordsRemoverTransform CreateComponent(IHostEnvironment env, IDataView input, OneToOneColumn[] columns)
        {
            return new StopWordsRemoverEstimator(env, columns.Select(x => new StopWordsRemoverTransform.ColumnInfo(x.Source, x.Name)).ToArray()).Fit(input).Transform(input) as IStopWordsRemoverTransform;
        }
    }

    /// <summary>
    /// A Stopword remover transform based on language-specific lists of stop words (most common words)
    /// from Office Named Entity Recognition project.
    /// The transform is usually applied after tokenizing text, so it compares individual tokens
    /// (case-insensitive comparison) to the stopwords.
    /// </summary>
    public sealed class StopWordsRemoverTransform : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Optional column to use for languages. This overrides sentence separator language value.",
                ShortName = "langscol")]
            public string LanguagesColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Stopword Language (optional).", ShortName = "lang")]
            public StopWordsRemoverEstimator.Language? Language;

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
            public StopWordsRemoverEstimator.Language Language = StopWordsRemoverEstimator.Defaults.DefaultLanguage;
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
                loaderAssemblyName: typeof(StopWordsRemoverTransform).Assembly.FullName);
        }
        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();

        private readonly ColumnInfo[] _columns;
        private static volatile NormStr.Pool[] _stopWords;
        private static volatile Dictionary<ReadOnlyMemory<char>, StopWordsRemoverEstimator.Language> _langsDictionary;

        private const string RegistrationName = "StopWordsRemover";
        private const string StopWordsDirectoryName = "StopWords";

        private static NormStr.Pool[] StopWords
        {
            get
            {
                if (_stopWords == null)
                {
                    var values = Enum.GetValues(typeof(StopWordsRemoverEstimator.Language)).Cast<int>();
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

        private static Dictionary<ReadOnlyMemory<char>, StopWordsRemoverEstimator.Language> LangsDictionary
        {
            get
            {
                if (_langsDictionary == null)
                {
                    var langsDictionary = Enum.GetValues(typeof(StopWordsRemoverEstimator.Language)).Cast<StopWordsRemoverEstimator.Language>()
                        .ToDictionary(lang => lang.ToString().AsMemory());
                    Interlocked.CompareExchange(ref _langsDictionary, langsDictionary, null);
                }

                return _langsDictionary;
            }
        }

        public class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly StopWordsRemoverEstimator.Language Language;
            public readonly string LanguageColumn;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="language">Language-specific stop words list.</param>
            /// <param name="languageColumn">Optional column to use for languages. This overrides language value.</param>
            public ColumnInfo(string input, string output, StopWordsRemoverEstimator.Language language = StopWordsRemoverEstimator.Defaults.DefaultLanguage, string languageColumn = null)
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

        /// <summary>
        /// Stopword remover removes language-specific lists of stop words (most common words).
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to remove stop words from.</param>
        public StopWordsRemoverTransform(IHostEnvironment env, params ColumnInfo[] columns) :
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

        private StopWordsRemoverTransform(IHost host, ModelLoadContext ctx) :
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
                var lang = (StopWordsRemoverEstimator.Language)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Enum.IsDefined(typeof(StopWordsRemoverEstimator.Language), lang));
                var langColName = ctx.LoadStringOrNull();
                _columns[i] = new ColumnInfo(ColumnPairs[i].input, ColumnPairs[i].output, lang, langColName);
            }
        }

        // Factory method for SignatureLoadModel.
        private static StopWordsRemoverTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new StopWordsRemoverTransform(host, ctx);
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
            return new StopWordsRemoverTransform(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, Schema.Create(schema));

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
                foreach (StopWordsRemoverEstimator.Language lang in Enum.GetValues(typeof(StopWordsRemoverEstimator.Language)))
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

        private static void AddResourceIfNotPresent(StopWordsRemoverEstimator.Language lang)
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

        private static Stream GetResourceFileStreamOrNull(StopWordsRemoverEstimator.Language lang)
        {
            Assembly assembly = Assembly.GetExecutingAssembly();
            return assembly.GetManifestResourceStream($"{assembly.GetName().Name}.Text.StopWords.{lang.ToString()}.txt");
        }

        private sealed class Mapper : MapperBase
        {
            private readonly ColumnType[] _types;
            private readonly StopWordsRemoverTransform _parent;
            private readonly int[] _languageColumns;
            private readonly bool?[] _resourcesExist;

            public Mapper(StopWordsRemoverTransform parent, Schema inputSchema)
             : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent.ColumnPairs.Length];
                _languageColumns = new int[_parent.ColumnPairs.Length];
                _resourcesExist = new bool?[StopWords.Length];

                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    //IVAN: Host.Assert(Infos[iinfo].TypeSrc.IsVector & Infos[iinfo].TypeSrc.ItemType.IsText);
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

            //IVAN: check language column and override dependencies

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _types[i], null);
                }
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                StopWordsRemoverEstimator.Language stopWordslang = _parent._columns[iinfo].Language;
                var lang = default(ReadOnlyMemory<char>);
                var getLang = _languageColumns[iinfo] >= 0 ? input.GetGetter<ReadOnlyMemory<char>>(_languageColumns[iinfo]) : null;
                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(ColMapNewToOld[iinfo]);
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

                        for (int i = 0; i < src.Count; i++)
                        {
                            if (src.Values[i].IsEmpty)
                                continue;
                            buffer.Clear();
                            ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(src.Values[i].Span, buffer);

                            // REVIEW: Consider using a trie for string matching (Aho-Corasick, etc.)
                            if (StopWords[(int)langToUse].Get(buffer) == null)
                                list.Add(src.Values[i]);
                        }

                        VBufferUtils.Copy(list, ref dst, list.Count);
                    };

                return del;
            }

            private void UpdateLanguage(ref StopWordsRemoverEstimator.Language langToUse, ValueGetter<ReadOnlyMemory<char>> getLang, ref ReadOnlyMemory<char> langTxt)
            {
                if (getLang != null)
                {
                    getLang(ref langTxt);
                    StopWordsRemoverEstimator.Language lang;
                    if (LangsDictionary.TryGetValue(langTxt, out lang))
                        langToUse = lang;
                }

                if (!ResourceExists(langToUse))
                    langToUse = StopWordsRemoverEstimator.Defaults.DefaultLanguage;
                AddResourceIfNotPresent(langToUse);
            }

            private bool ResourceExists(StopWordsRemoverEstimator.Language lang)
            {
                int langVal = (int)lang;
                Contracts.Assert(0 <= langVal & langVal < Utils.Size(StopWords));
                // Note: Updating values in _resourcesExist does not have to be an atomic operation
                return StopWords[langVal] != null ||
                    (_resourcesExist[langVal] ?? (_resourcesExist[langVal] = GetResourceFileStreamOrNull(lang) != null).Value);
            }
        }
    }

    /// <summary>
    /// Stopword remover removes language-specific lists of stop words (most common words)
    /// This is usually applied after tokenizing text, so it compares individual tokens
    /// (case-insensitive comparison) to the stopwords.
    /// </summary>
    public sealed class StopWordsRemoverEstimator : TrivialEstimator<StopWordsRemoverTransform>
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

        public static bool IsColumnTypeValid(ColumnType type) => type.ItemType.IsText && type.IsVector;

        internal const string ExpectedColumnType = "vector of Text type";

        /// <summary>
        /// Removes stop words from incoming token streams in <paramref name="inputColumn"/>
        /// and outputs the token streams without stopwords as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to remove stop words on.</param>
        /// <param name="outputColumn">The column containing output text. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="language">Langauge of the input text column <paramref name="inputColumn"/>.</param>
        public StopWordsRemoverEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null, Language language = Language.English)
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
        public StopWordsRemoverEstimator(IHostEnvironment env, (string input, string output)[] columns, Language language = Language.English)
            : this(env, columns.Select(x => new StopWordsRemoverTransform.ColumnInfo(x.input, x.output, language)).ToArray())
        {
        }

        public StopWordsRemoverEstimator(IHostEnvironment env, params StopWordsRemoverTransform.ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(StopWordsRemoverEstimator)), new StopWordsRemoverTransform(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (col.Kind == SchemaShape.Column.VectorKind.Scalar || !col.ItemType.IsText)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, ExpectedColumnType, col.ItemType.ToString());
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.VariableVector, TextType.Instance, false);
            }
            return new SchemaShape(result.Values);
        }
    }

    public sealed class CustomStopWordsRemoverTransform : OneToOneTransformerBase
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
            public IStopWordsRemoverTransform CreateComponent(IHostEnvironment env, IDataView input, OneToOneColumn[] column)
            {
                if (Utils.Size(Stopword) > 0)
                    return new CustomStopWordsRemoverTransform(env, Stopword, column.Select(x => (x.Source, x.Name)).ToArray()).Transform(input) as IStopWordsRemoverTransform;
                else
                    return new CustomStopWordsRemoverTransform(env, Stopwords, DataFile, StopwordsColumn, Loader, column.Select(x => (x.Source, x.Name)).ToArray()).Transform(input) as IStopWordsRemoverTransform;
            }
        }

        internal const string Summary = "A Stopword remover transform based on a user-defined list of stopwords. " +
            "The transform is usually applied after tokenizing text, so it compares individual tokens " +
            "(case-insensitive comparison) to the stopwords.";

        public const string LoaderSignature = "CustomStopWords";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CSTOPWRD",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CustomStopWordsRemoverTransform).Assembly.FullName);
        }

        public const string StopwordsManagerLoaderSignature = "CustomStopWordsManager";
        private static VersionInfo GetStopwrodsManagerVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "STOPWRDM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: StopwordsManagerLoaderSignature,
                loaderAssemblyName: typeof(CustomStopWordsRemoverTransform).Assembly.FullName);
        }

        private static readonly ColumnType _outputType = new VectorType(TextType.Instance);

        private readonly NormStr.Pool _stopWordsMap;
        private readonly (string input, string output)[] _columns;
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
                    if (!string.IsNullOrWhiteSpace(stopwordsCol))
                    {
                        ch.Warning("{0} should not be specified when default loader is TextLoader. Ignoring stopwordsColumn={0}",
                            stopwordsCol);
                    }
                    dataLoader = TextLoader.Create(
                        Host,
                        new TextLoader.Arguments()
                        {
                            Separator = "tab",
                            Column = new[]
                            {
                                new TextLoader.Column("Stopwords", DataKind.TX, 0)
                            }
                        },
                        fileSource);
                    stopwordsCol = "Stopwords";
                }
                ch.AssertNonEmpty(stopwordsCol);
            }
            else
            {
                dataLoader = loader.CreateComponent(Host, fileSource);
            }

            return dataLoader;
        }

        private void LoadStopWords(IChannel ch, ReadOnlyMemory<char> stopwords, string dataFile, string stopwordsColumn, IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory, out NormStr.Pool stopWordsMap)
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
                        ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(stopwords.Span, buffer);
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
                ch.CheckUserArg(typeSrc.IsText, nameof(Arguments.StopwordsColumn), "Must be a scalar text column");

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
        public IReadOnlyCollection<(string input, string output)> Columns => _columns.AsReadOnly();

        /// <summary>
        /// Custom stopword remover removes specified list of stop words.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        /// <param name="columns">Pairs of columns to remove stop words from.</param>
        public CustomStopWordsRemoverTransform(IHostEnvironment env, string[] stopwords, params (string input, string output)[] columns) :
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
                _columns = columns;
            }
        }

        internal CustomStopWordsRemoverTransform(IHostEnvironment env, string stopwords,
            string dataFile, string stopwordsColumn, IComponentFactory<IMultiStreamSource, IDataLoader> loader, params (string input, string output)[] columns) :
         base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), columns)
        {

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
                    c.SetVersionInfo(GetStopwrodsManagerVersionInfo());

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

        private CustomStopWordsRemoverTransform(IHost host, ModelLoadContext ctx) :
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
                        c.CheckAtModel(GetStopwrodsManagerVersionInfo());

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
        private static CustomStopWordsRemoverTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new CustomStopWordsRemoverTransform(host, ctx);
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
            //IVAN propagate other options of argument class.
            return new CustomStopWordsRemoverTransform(env, args.Stopword, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, Schema.Create(schema));

        private sealed class Mapper : MapperBase
        {
            private readonly ColumnType[] _types;
            private readonly CustomStopWordsRemoverTransform _parent;

            public Mapper(CustomStopWordsRemoverTransform parent, Schema inputSchema)
             : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = new ColumnType[_parent.ColumnPairs.Length];
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _types[i], null);
                }
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
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

                        for (int i = 0; i < src.Count; i++)
                        {
                            if (src.Values[i].IsEmpty)
                                continue;
                            buffer.Clear();
                            ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(src.Values[i].Span, buffer);

                            // REVIEW: Consider using a trie for string matching (Aho-Corasick, etc.)
                            if (_parent._stopWordsMap.Get(buffer) == null)
                                list.Add(src.Values[i]);
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
    public sealed class CustomStopWordsRemoverEstimator : TrivialEstimator<CustomStopWordsRemoverTransform>
    {

        public static bool IsColumnTypeValid(ColumnType type) => type.ItemType.IsText && type.IsVector;

        internal const string ExpectedColumnType = "vector of Text type";

        /// <summary>
        /// Removes stop words from incoming token streams in <paramref name="inputColumn"/>
        /// and outputs the token streams without stopwords as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to remove stop words on.</param>
        /// <param name="outputColumn">The column containing output text. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="stopwords">Array of words to remove.</param>
        public CustomStopWordsRemoverEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null, params string[] stopwords)
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
        public CustomStopWordsRemoverEstimator(IHostEnvironment env, (string input, string output)[] columns, string[] stopwords) :
           base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CustomStopWordsRemoverEstimator)), new CustomStopWordsRemoverTransform(env, stopwords, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input);
                if (col.Kind == SchemaShape.Column.VectorKind.Scalar || !col.ItemType.IsText)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input, ExpectedColumnType, col.ItemType.ToString());
                result[colInfo.output] = new SchemaShape.Column(colInfo.output, SchemaShape.Column.VectorKind.VariableVector, TextType.Instance, false);
            }
            return new SchemaShape(result.Values);
        }
    }
}