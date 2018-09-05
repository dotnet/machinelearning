// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TextAnalytics;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;

[assembly: LoadableClass(StopWordsRemoverTransform.Summary, typeof(StopWordsRemoverTransform), typeof(StopWordsRemoverTransform.Arguments), typeof(SignatureDataTransform),
    "Stopwords Remover Transform", "StopWordsRemoverTransform", "StopWordsRemover", "StopWords")]

[assembly: LoadableClass(StopWordsRemoverTransform.Summary, typeof(StopWordsRemoverTransform), null, typeof(SignatureStopWordsRemoverTransform),
    "Predefined Stopwords List Remover", "PredefinedStopWordsRemoverTransform", "PredefinedStopWordsRemover", "PredefinedStopWords", "Predefined")]

[assembly: LoadableClass(StopWordsRemoverTransform.Summary, typeof(StopWordsRemoverTransform), null, typeof(SignatureLoadDataTransform),
    "Stopwords Remover Transform", StopWordsRemoverTransform.LoaderSignature)]

[assembly: LoadableClass(CustomStopWordsRemoverTransform.Summary, typeof(CustomStopWordsRemoverTransform), typeof(CustomStopWordsRemoverTransform.Arguments), typeof(SignatureDataTransform),
    "Custom Stopwords Remover Transform", "CustomStopWordsRemoverTransform", "CustomStopWords")]

[assembly: LoadableClass(CustomStopWordsRemoverTransform.Summary, typeof(CustomStopWordsRemoverTransform), typeof(CustomStopWordsRemoverTransform.LoaderArguments),
    typeof(SignatureStopWordsRemoverTransform), "Custom Stopwords Remover Transform", "CustomStopWordsRemoverTransform", "CustomStopWords", "Custom")]

[assembly: LoadableClass(CustomStopWordsRemoverTransform.Summary, typeof(CustomStopWordsRemoverTransform), null, typeof(SignatureLoadDataTransform),
    "Custom Stopwords Remover Transform", CustomStopWordsRemoverTransform.LoaderSignature)]

[assembly: EntryPointModule(typeof(PredefinedStopWordsRemoverFactory))]
[assembly: EntryPointModule(typeof(CustomStopWordsRemoverTransform.LoaderArguments))]

namespace Microsoft.ML.Runtime.TextAnalytics
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
        public IStopWordsRemoverTransform CreateComponent(IHostEnvironment env, IDataView input, OneToOneColumn[] column)
        {
            return new StopWordsRemoverTransform(env, input, column);
        }
    }

    /// <summary>
    /// A Stopword remover transform based on language-specific lists of stop words (most common words)
    /// from Office Named Entity Recognition project.
    /// The transform is usually applied after tokenizing text, so it compares individual tokens
    /// (case-insensitive comparison) to the stopwords.
    /// </summary>
    public sealed class StopWordsRemoverTransform : OneToOneTransformBase, IStopWordsRemoverTransform
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

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Optional column to use for languages. This overrides sentence separator language value.",
                ShortName = "langscol")]
            public string LanguagesColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Stopword Language (optional).", ShortName = "lang")]
            public Language? Language;

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
            public Language Language = DefaultLanguage;
        }

        private sealed class ColInfoEx
        {
            public readonly Language Lang;
            public readonly int LangsColIndex;

            private readonly string _langsColName;

            public ColInfoEx(ISchema input, Language language, string languagesColumn)
            {
                Lang = language;
                Contracts.CheckUserArg(Enum.IsDefined(typeof(Language), Lang), nameof(Column.Language),
                    "value does not exist in the enumeration");

                _langsColName = languagesColumn;
                if (!string.IsNullOrWhiteSpace(_langsColName))
                {
                    Bind(input, _langsColName, out LangsColIndex, true);
                    Contracts.Assert(LangsColIndex >= 0);
                }
                else
                {
                    _langsColName = null;
                    LangsColIndex = -1;
                }
            }

            /// <summary>
            /// Binds a text column with the given name using input schema and returns the column index.
            /// Fails if there is no column with the given name or if the column type is not text.
            /// </summary>
            private static void Bind(ISchema input, string name, out int index, bool exceptUser)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(name);

                if (!input.TryGetColumnIndex(name, out index))
                {
                    throw exceptUser
                        ? Contracts.ExceptUserArg(nameof(Arguments.Column), "Source column '{0}' not found", name)
                        : Contracts.ExceptDecode("Source column '{0}' not found", name);
                }

                var type = input.GetColumnType(index);
                if (type != TextType.Instance)
                {
                    throw exceptUser
                        ? Contracts.ExceptUserArg(nameof(Arguments.Column), "Source column '{0}' has type '{1}' but must be text", name, type)
                        : Contracts.ExceptDecode("Source column '{0}' has type '{1}' but must be text", name, type);
                }
            }

            public ColInfoEx(ModelLoadContext ctx, ISchema input)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(input);

                // *** Binary format ***
                // int: the stopwords list language
                // int: the id of languages column name
                Lang = (Language)ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Enum.IsDefined(typeof(Language), Lang));
                _langsColName = ctx.LoadStringOrNull();
                if (_langsColName != null)
                {
                    Bind(input, _langsColName, out LangsColIndex, false);
                    Contracts.Assert(LangsColIndex >= 0);
                }
                else
                    LangsColIndex = -1;
            }

            public void Save(ModelSaveContext ctx)
            {
                // *** Binary format ***
                // int: the stopwords list language
                // int: the id of languages column name
                ctx.Writer.Write((int)Lang);
                Contracts.Assert((LangsColIndex >= 0 && _langsColName != null)
                    || (LangsColIndex == -1 && _langsColName == null));
                ctx.SaveStringOrNull(_langsColName);
            }
        }

        internal const string Summary = "A Stopword remover transform based on language-specific lists of stop words (most common words) " +
            "from Office Named Entity Recognition project. The transform is usually applied after tokenizing text, so it compares individual tokens " +
            "(case-insensitive comparison) to the stopwords.";

        public const string LoaderSignature = "StopWordsTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "STOPWRDR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private readonly bool?[] _resourcesExist;
        private readonly ColInfoEx[] _exes;

        private static readonly ColumnType _outputType = new VectorType(TextType.Instance);

        private static volatile NormStr.Pool[] _stopWords;
        private static volatile Dictionary<ReadOnlyMemory<char>, Language> _langsDictionary;

        private const Language DefaultLanguage = Language.English;
        private const string RegistrationName = "StopWordsRemover";
        private const string StopWordsDirectoryName = "StopWords";

        private static NormStr.Pool[] StopWords
        {
            get
            {
                if (_stopWords == null)
                {
                    var values = Enum.GetValues(typeof(Language)).Cast<int>();
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

        private static Dictionary<ReadOnlyMemory<char>, Language> LangsDictionary
        {
            get
            {
                if (_langsDictionary == null)
                {
                    var langsDictionary = Enum.GetValues(typeof(Language)).Cast<Language>()
                        .ToDictionary(lang => lang.ToString().AsMemory());
                    Interlocked.CompareExchange(ref _langsDictionary, langsDictionary, null);
                }

                return _langsDictionary;
            }
        }

        public StopWordsRemoverTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column, input, TestIsTextVector)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            using (var ch = Host.Start("construction"))
            {
                _exes = new ColInfoEx[Infos.Length];
                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    var col = args.Column[iinfo];
                    string languagesCol = !string.IsNullOrWhiteSpace(col.LanguagesColumn)
                        ? col.LanguagesColumn
                        : args.LanguagesColumn;
                    _exes[iinfo] = new ColInfoEx(input.Schema, col.Language ?? args.Language, languagesCol);
                }

                _resourcesExist = new bool?[StopWords.Length];

                CheckResources(ch);
                ch.Done();
            }
            Metadata.Seal();
        }

        public StopWordsRemoverTransform(IHostEnvironment env, IDataView input, OneToOneColumn[] column)
            : base(env, RegistrationName, column, input, TestIsTextVector)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(column));
            Host.Assert(column is Column[]);

            using (var ch = Host.Start("construction"))
            {
                _exes = new ColInfoEx[Infos.Length];
                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    var col = (Column)column[iinfo];
                    _exes[iinfo] = new ColInfoEx(input.Schema, col.Language ?? DefaultLanguage, col.LanguagesColumn);
                }

                _resourcesExist = new bool?[StopWords.Length];

                CheckResources(ch);
                ch.Done();
            }
            Metadata.Seal();
        }

        private StopWordsRemoverTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsTextVector)
        {
            Host.AssertValue(ctx);

            using (var ch = Host.Start("Deserialization"))
            {
                // *** Binary format ***
                // <base>
                // for each added column
                //   ColInfoEx
                ch.AssertNonEmpty(Infos);
                _exes = new ColInfoEx[Infos.Length];
                for (int iinfo = 0; iinfo < _exes.Length; iinfo++)
                    _exes[iinfo] = new ColInfoEx(ctx, input.Schema);

                _resourcesExist = new bool?[StopWords.Length];
                CheckResources(ch);
                ch.Done();
            }
            Metadata.Seal();
        }

        private void CheckResources(IChannel ch)
        {
            Host.AssertValue(ch);

            // Find required resources
            var requiredResources = new bool[StopWords.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                requiredResources[(int)_exes[iinfo].Lang] = true;

            // Check the existence of resource files
            var missings = new StringBuilder();
            foreach (Language lang in Enum.GetValues(typeof(Language)))
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

        public static StopWordsRemoverTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = env.Register(RegistrationName);
            return h.Apply("Loading Model", ch => new StopWordsRemoverTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // for each added column
            //   ColInfoEx
            SaveBase(ctx);
            Host.Assert(_exes.Length == Infos.Length);
            foreach (var ex in _exes)
                ex.Save(ctx);
        }

        protected override void ActivateSourceColumns(int iinfo, bool[] active)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            base.ActivateSourceColumns(iinfo, active);

            if (_exes[iinfo].LangsColIndex >= 0)
            {
                Host.Assert(_exes[iinfo].LangsColIndex < active.Length);
                active[_exes[iinfo].LangsColIndex] = true;
            }
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return _outputType;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector & Infos[iinfo].TypeSrc.ItemType.IsText);
            disposer = null;

            var ex = _exes[iinfo];
            Language stopWordslang = ex.Lang;
            var lang = default(ReadOnlyMemory<char>);
            var getLang = ex.LangsColIndex >= 0 ? input.GetGetter<ReadOnlyMemory<char>>(ex.LangsColIndex) : null;

            var getSrc = GetSrcGetter<VBuffer<ReadOnlyMemory<char>>>(input, iinfo);
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
                        ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(buffer, src.Values[i]);

                        // REVIEW nihejazi: Consider using a trie for string matching (Aho-Corasick, etc.)
                        if (StopWords[(int)langToUse].Get(buffer) == null)
                            list.Add(src.Values[i]);
                    }

                    VBufferUtils.Copy(list, ref dst, list.Count);
                };

            return del;
        }

        private void UpdateLanguage(ref Language langToUse, ValueGetter<ReadOnlyMemory<char>> getLang, ref ReadOnlyMemory<char> langTxt)
        {
            if (getLang != null)
            {
                getLang(ref langTxt);
                Language lang;
                if (LangsDictionary.TryGetValue(langTxt, out lang))
                    langToUse = lang;
            }

            if (!ResourceExists(langToUse))
                langToUse = DefaultLanguage;
            AddResourceIfNotPresent(langToUse);
        }

        private bool ResourceExists(Language lang)
        {
            int langVal = (int)lang;
            Contracts.Assert(0 <= langVal & langVal < Utils.Size(StopWords));
            // Note: Updating values in _resourcesExist does not have to be an atomic operation
            return StopWords[langVal] != null ||
                (_resourcesExist[langVal] ?? (_resourcesExist[langVal] = GetResourceFileStreamOrNull(lang) != null).Value);
        }

        private static void AddResourceIfNotPresent(Language lang)
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

        private static Stream GetResourceFileStreamOrNull(Language lang)
        {
            Assembly assembly = Assembly.GetExecutingAssembly();
            return assembly.GetManifestResourceStream($"{assembly.GetName().Name}.Text.StopWords.{lang.ToString()}.txt");
        }
    }

    public sealed class CustomStopWordsRemoverTransform : OneToOneTransformBase, IStopWordsRemoverTransform
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
                return new CustomStopWordsRemoverTransform(env, this, input, column);
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
                loaderSignature: LoaderSignature);
        }

        public const string StopwrodsManagerLoaderSignature = "CustomStopWordsManager";
        private static VersionInfo GetStopwrodsManagerVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "STOPWRDM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: StopwrodsManagerLoaderSignature);
        }

        private static readonly ColumnType _outputType = new VectorType(TextType.Instance);

        private readonly NormStr.Pool _stopWordsMap;

        private const string RegistrationName = "CustomStopWordsRemover";

        private static IDataLoader LoadStopwords(IHostEnvironment env, IChannel ch, string dataFile,
            IComponentFactory<IMultiStreamSource, IDataLoader> loader, ref string stopwordsCol)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));

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
                        dataLoader = new BinaryLoader(env, new BinaryLoader.Arguments(), fileSource);
                    else
                    {
                        ch.Assert(isTranspose);
                        dataLoader = new TransposeLoader(env, new TransposeLoader.Arguments(), fileSource);
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
                        env,
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
                dataLoader = loader.CreateComponent(env, fileSource);
            }

            return dataLoader;
        }

        private void LoadStopWords(IHostEnvironment env, IChannel ch, ArgumentsBase loaderArgs, out NormStr.Pool stopWordsMap)
        {
            Contracts.AssertValue(env);
            env.AssertValue(ch);
            ch.AssertValue(loaderArgs);

            if ((!string.IsNullOrEmpty(loaderArgs.Stopwords) || Utils.Size(loaderArgs.Stopword) > 0) &&
                (!string.IsNullOrWhiteSpace(loaderArgs.DataFile) || loaderArgs.Loader != null ||
                    !string.IsNullOrWhiteSpace(loaderArgs.StopwordsColumn)))
            {
                ch.Warning("Explicit stopwords list specified. Data file arguments will be ignored");
            }

            var src = default(ReadOnlyMemory<char>);
            stopWordsMap = new NormStr.Pool();
            var buffer = new StringBuilder();

            var stopwords = loaderArgs.Stopwords.AsMemory();
            stopwords = ReadOnlyMemoryUtils.Trim(stopwords);
            if (!stopwords.IsEmpty)
            {
                bool warnEmpty = true;
                for (bool more = true; more;)
                {
                    ReadOnlyMemory<char> stopword;
                    more = ReadOnlyMemoryUtils.SplitOne(',', out stopword, out stopwords, stopwords);
                    stopword = ReadOnlyMemoryUtils.Trim(stopword);
                    if (!stopword.IsEmpty)
                    {
                        buffer.Clear();
                        ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(buffer, stopwords);
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
            else if (Utils.Size(loaderArgs.Stopword) > 0)
            {
                bool warnEmpty = true;
                foreach (string word in loaderArgs.Stopword)
                {
                    var stopword = word.AsMemory();
                    stopword = ReadOnlyMemoryUtils.Trim(stopword);
                    if (!stopword.IsEmpty)
                    {
                        buffer.Clear();
                        ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(buffer, stopword);
                        stopWordsMap.Add(buffer);
                    }
                    else if (warnEmpty)
                    {
                        ch.Warning("Empty strings ignored in 'stopword' specification");
                        warnEmpty = false;
                    }
                }
            }
            else
            {
                string srcCol = loaderArgs.StopwordsColumn;
                var loader = LoadStopwords(env, ch, loaderArgs.DataFile, loaderArgs.Loader, ref srcCol);
                int colSrc;
                if (!loader.Schema.TryGetColumnIndex(srcCol, out colSrc))
                    throw ch.ExceptUserArg(nameof(Arguments.StopwordsColumn), "Unknown column '{0}'", srcCol);
                var typeSrc = loader.Schema.GetColumnType(colSrc);
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
                            ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(buffer, src);
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

        public CustomStopWordsRemoverTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column, input, TestIsTextVector)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(!string.IsNullOrWhiteSpace(args.Stopwords) || args.Stopword != null || !string.IsNullOrWhiteSpace(args.DataFile),
                nameof(args.DataFile), "stopwords or datafile must be defined");

            using (var ch = Host.Start(RegistrationName))
            {
                LoadStopWords(env, ch, args, out _stopWordsMap);
                ch.Done();
            }
            Metadata.Seal();
        }

        /// <summary>
        /// Public constructor corresponding to SignatureStopWordsRemoverTransform. It accepts arguments of type LoaderArguments,
        /// and a separate array of columns (constructed by the caller -TextTransform- arguments).
        /// </summary>
        public CustomStopWordsRemoverTransform(IHostEnvironment env, LoaderArguments loaderArgs, IDataView input, OneToOneColumn[] column)
            : base(env, RegistrationName, column, input, TestIsTextItem)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(column));

            Host.CheckValue(loaderArgs, nameof(loaderArgs));
            Host.CheckUserArg(!string.IsNullOrWhiteSpace(loaderArgs.Stopwords) || loaderArgs.Stopword != null || !string.IsNullOrWhiteSpace(loaderArgs.DataFile),
                nameof(loaderArgs.DataFile), "stopwords or datafile must be defined");

            using (var ch = Host.Start(RegistrationName))
            {
                LoadStopWords(env, ch, loaderArgs, out _stopWordsMap);
                ch.Done();
            }
            Metadata.Seal();
        }

        private CustomStopWordsRemoverTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsTextVector)
        {
            Host.AssertValue(ctx);

            using (var ch = Host.Start("Deserialization"))
            {
                // *** Binary format ***
                // <base>
                ch.AssertNonEmpty(Infos);

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
                ch.Done();
            }
            Metadata.Seal();
        }

        public static CustomStopWordsRemoverTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = env.Register(RegistrationName);
            return h.Apply("Loading Model", ch => new CustomStopWordsRemoverTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            SaveBase(ctx);

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
                        ctx.SaveString(nstr);
                        id++;
                    }

                    ctx.SaveTextStream("Stopwords.txt", writer =>
                    {
                        foreach (var nstr in _stopWordsMap)
                            writer.WriteLine("{0}\t{1}", nstr.Id, nstr.Value);
                    });
                });
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return _outputType;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            Host.Assert(Infos[iinfo].TypeSrc.IsVector & Infos[iinfo].TypeSrc.ItemType.IsText);
            disposer = null;

            var getSrc = GetSrcGetter<VBuffer<ReadOnlyMemory<char>>>(input, iinfo);
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
                        ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(buffer, src.Values[i]);

                        // REVIEW nihejazi: Consider using a trie for string matching (Aho-Corasick, etc.)
                        if (_stopWordsMap.Get(buffer) == null)
                            list.Add(src.Values[i]);
                    }

                    VBufferUtils.Copy(list, ref dst, list.Count);
                };

            return del;
        }
    }
}