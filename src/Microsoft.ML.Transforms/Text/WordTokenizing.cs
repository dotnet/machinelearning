// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(WordTokenizingTransformer.Summary, typeof(IDataTransform), typeof(WordTokenizingTransformer), typeof(WordTokenizingTransformer.Options), typeof(SignatureDataTransform),
    "Word Tokenizer Transform", "WordTokenizeTransform", "DelimitedTokenizeTransform", "WordToken", "DelimitedTokenize", "Token")]

[assembly: LoadableClass(WordTokenizingTransformer.Summary, typeof(IDataTransform), typeof(WordTokenizingTransformer), null, typeof(SignatureLoadDataTransform),
    "Word Tokenizer Transform", WordTokenizingTransformer.LoaderSignature)]

[assembly: LoadableClass(WordTokenizingTransformer.Summary, typeof(WordTokenizingTransformer), null, typeof(SignatureLoadModel),
    "Word Tokenizer Transform", WordTokenizingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(WordTokenizingTransformer), null, typeof(SignatureLoadRowMapper),
   "Word Tokenizer Transform", WordTokenizingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{

    // The input for this transform is a ReadOnlyMemory or a vector of ReadOnlyMemory, and its output is a vector of ReadOnlyMemory<char>,
    // corresponding to the tokens in the input text, split using a set of user specified separator characters.
    // Empty strings and strings containing only spaces are dropped.
    /// <include file='doc.xml' path='doc/members/member[@name="WordTokenizer"]/*' />
    public sealed class WordTokenizingTransformer : OneToOneTransformerBase
    {
        internal class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Comma separated set of term separator(s). Commonly: 'space', 'comma', 'semicolon' or other single character.",
                ShortName = "sep")]
            public string TermSeparators;

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
                return TryUnparseCore(sb);
            }
        }

        internal abstract class ArgumentsBase : TransformInputBase
        {
            // REVIEW: Think about adding a user specified separator string, that is added as an extra token between
            // the tokens of each column
            [Argument(ArgumentType.AtMostOnce,
                Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly,
                HelpText = "Comma separated set of term separator(s). Commonly: 'space', 'comma', 'semicolon' or other single character.",
                ShortName = "sep")]
            public string TermSeparators = "space";

            [Argument(ArgumentType.AtMostOnce,
                Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly,
                HelpText = "Array of single character term separator(s). By default uses space character separator.",
                ShortName = "sep")]
            public char[] CharArrayTermSeparators;
        }

        internal sealed class Options : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        internal const string Summary = "The input to this transform is text, and the output is a vector of text containing the words (tokens) in the original text. "
            + "The separator is space, but can be specified as any other character (or multiple characters) if needed.";

        internal const string LoaderSignature = "TokenizeTextTransform";

        internal const string UserName = "Tokenize Text Transform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "WRDTOKNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(WordTokenizingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "DelimitedTokenize";

        internal IReadOnlyCollection<WordTokenizingEstimator.ColumnOptions> Columns => _columns.AsReadOnly();
        private readonly WordTokenizingEstimator.ColumnOptions[] _columns;

        private static (string name, string inputColumnName)[] GetColumnPairs(WordTokenizingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckNonEmpty(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        internal WordTokenizingTransformer(IHostEnvironment env, params WordTokenizingEstimator.ColumnOptions[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema[srcCol].Type;
            if (!WordTokenizingEstimator.IsColumnTypeValid(type))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, WordTokenizingEstimator.ExpectedColumnType, type.ToString());
        }

        private WordTokenizingTransformer(IHost host, ModelLoadContext ctx) :
            base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            _columns = new WordTokenizingEstimator.ColumnOptions[columnsLength];
            // *** Binary format ***
            // <base>
            // for each added column
            //   charArray: Separators
            for (int i = 0; i < columnsLength; i++)
            {
                var separators = ctx.Reader.ReadCharArray();
                Contracts.CheckDecode(Utils.Size(separators) > 0);
                _columns[i] = new WordTokenizingEstimator.ColumnOptions(ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, separators);
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // for each added column
            //   charArray: Separators
            SaveColumns(ctx);
            foreach (var column in _columns)
                ctx.Writer.WriteCharArray(column.SeparatorsArray);
        }

        // Factory method for SignatureLoadModel.
        private static WordTokenizingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new WordTokenizingTransformer(host, ctx);
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));
            var cols = new WordTokenizingEstimator.ColumnOptions[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];
                var separators = options.CharArrayTermSeparators ?? PredictionUtil.SeparatorFromString(item.TermSeparators ?? options.TermSeparators);
                cols[i] = new WordTokenizingEstimator.ColumnOptions(item.Name, item.Source ?? item.Name, separators);

            }
            return new WordTokenizingTransformer(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsPfa
        {
            private readonly DataViewType _type;
            private readonly WordTokenizingTransformer _parent;
            private readonly bool[] _isSourceVector;

            public bool CanSavePfa => true;

            public Mapper(WordTokenizingTransformer parent, DataViewSchema inputSchema)
              : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _type = new VectorType(TextDataViewType.Instance);
                _isSourceVector = new bool[_parent._columns.Length];
                for (int i = 0; i < _isSourceVector.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent._columns[i].InputColumnName, out int srcCol);
                    var srcType = inputSchema[srcCol].Type;
                    _isSourceVector[i] = srcType is VectorType;
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, _type, null);
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent._columns.Length);
                disposer = null;

                input.Schema.TryGetColumnIndex(_parent._columns[iinfo].InputColumnName, out int srcCol);
                var srcType = input.Schema[srcCol].Type;
                Host.Assert(srcType.GetItemType() is TextDataViewType);

                if (!(srcType is VectorType))
                    return MakeGetterOne(input, iinfo);
                return MakeGetterVec(input, iinfo);
            }

            private ValueGetter<VBuffer<ReadOnlyMemory<char>>> MakeGetterOne(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(ReadOnlyMemory<char>);
                var terms = new List<ReadOnlyMemory<char>>();
                var separators = _parent._columns[iinfo].SeparatorsArray;

                return
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        getSrc(ref src);
                        terms.Clear();

                        AddTerms(src, separators, terms);

                        var editor = VBufferEditor.Create(ref dst, terms.Count);
                        if (terms.Count > 0)
                        {
                            terms.CopyTo(editor.Values);
                        }

                        dst = editor.Commit();
                    };
            }

            private ValueGetter<VBuffer<ReadOnlyMemory<char>>> MakeGetterVec(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);

                int cv = input.Schema[ColMapNewToOld[iinfo]].Type.GetVectorSize();
                Contracts.Assert(cv >= 0);
                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(VBuffer<ReadOnlyMemory<char>>);
                var terms = new List<ReadOnlyMemory<char>>();
                var separators = _parent._columns[iinfo].SeparatorsArray;

                return
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        getSrc(ref src);
                        terms.Clear();

                        var srcValues = src.GetValues();
                        for (int i = 0; i < srcValues.Length; i++)
                            AddTerms(srcValues[i], separators, terms);

                        var editor = VBufferEditor.Create(ref dst, terms.Count);
                        for (int i = 0; i < terms.Count; i++)
                            editor.Values[i] = terms[i];
                        dst = editor.Commit();
                    };
            }

            private void AddTerms(ReadOnlyMemory<char> txt, char[] separators, List<ReadOnlyMemory<char>> terms)
            {
                Host.AssertNonEmpty(separators);

                var rest = txt;
                if (separators.Length > 1)
                {
                    while (!rest.IsEmpty)
                    {
                        ReadOnlyMemory<char> term;
                        ReadOnlyMemoryUtils.SplitOne(rest, separators, out term, out rest);
                        term = ReadOnlyMemoryUtils.TrimSpaces(term);
                        if (!term.IsEmpty)
                            terms.Add(term);
                    }
                }
                else
                {
                    var separator = separators[0];
                    while (!rest.IsEmpty)
                    {
                        ReadOnlyMemory<char> term;
                        ReadOnlyMemoryUtils.SplitOne(rest, separator, out term, out rest);
                        term = ReadOnlyMemoryUtils.TrimSpaces(term);
                        if (!term.IsEmpty)
                            terms.Add(term);
                    }
                }
            }

            void ISaveAsPfa.SaveAsPfa(BoundPfaContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                var toHide = new List<string>();
                var toDeclare = new List<KeyValuePair<string, JToken>>();

                for (int iinfo = 0; iinfo < _parent._columns.Length; ++iinfo)
                {
                    var info = _parent._columns[iinfo];
                    var srcName = info.InputColumnName;
                    string srcToken = ctx.TokenOrNullForName(srcName);
                    if (srcToken == null)
                    {
                        toHide.Add(info.Name);
                        continue;
                    }
                    var result = SaveAsPfaCore(ctx, iinfo, srcToken);
                    if (result == null)
                    {
                        toHide.Add(info.Name);
                        continue;
                    }
                    toDeclare.Add(new KeyValuePair<string, JToken>(info.Name, result));
                }
                ctx.Hide(toHide.ToArray());
                ctx.DeclareVar(toDeclare.ToArray());
            }

            private JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, JToken srcToken)
            {
                Contracts.AssertValue(ctx);
                Contracts.AssertValue(srcToken);
                Contracts.Assert(CanSavePfa);

                var exInfo = _parent._columns[iinfo];
                var sep = PfaUtils.String("" + exInfo.SeparatorsArray[0]);
                if (_isSourceVector[iinfo])
                {
                    // If it's a vector, we'll concatenate them together.
                    srcToken = PfaUtils.Call("s.join", srcToken, sep);
                }

                if (exInfo.SeparatorsArray.Length > 1)
                {
                    // Due to the intrinsics in PFA, it is much easier if we can do
                    // one split, rather than multiple splits. So, if there are multiple
                    // separators, we first replace them with the first separator, then
                    // split once on that one. This could also have been done with a.flatMap.
                    for (int i = 1; i < exInfo.SeparatorsArray.Length; ++i)
                    {
                        var postSep = PfaUtils.String("" + exInfo.SeparatorsArray[i]);
                        srcToken = PfaUtils.Call("s.replaceall", srcToken, postSep, sep);
                    }
                }
                srcToken = PfaUtils.Call("s.split", srcToken, sep);
                // The TLC word tokenizer does not yield empty strings, but PFA's
                // split does. Filter them out.
                var hasCharsRef = PfaUtils.FuncRef(ctx.Pfa.EnsureHasChars());
                srcToken = PfaUtils.Call("a.filter", srcToken, hasCharsRef);
                return srcToken;
            }
        }

    }

    /// <summary>
    /// Word tokenizer splits text into tokens using the delimiter.
    /// For each text input, the output column is a variable vector of text.
    /// </summary>
    public sealed class WordTokenizingEstimator : TrivialEstimator<WordTokenizingTransformer>
    {
        internal static bool IsColumnTypeValid(DataViewType type) => type.GetItemType() is TextDataViewType;

        internal const string ExpectedColumnType = "Text";

        /// <summary>
        /// Tokenize incoming text in <paramref name="inputColumnName"/> and output the tokens as <paramref name="outputColumnName"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        internal WordTokenizingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null, char[] separators = null)
            : this(env, new[] { (outputColumnName, inputColumnName ?? outputColumnName) }, separators)
        {
        }

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        internal WordTokenizingEstimator(IHostEnvironment env, (string outputColumnName, string inputColumnName)[] columns, char[] separators = null)
            : this(env, columns.Select(x => new ColumnOptions(x.outputColumnName, x.inputColumnName, separators)).ToArray())
        {
        }

        /// <summary>
        ///  Tokenize incoming text in input columns and output the tokens.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        internal WordTokenizingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
          : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordTokenizingEstimator)), new WordTokenizingTransformer(env, columns))
        {
        }
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>
            /// Output column name that will be used to store the tokenization result of <see cref="InputColumnName"/> column.
            /// </summary>
            public readonly string Name;
            /// <summary>
            /// Input column name that will be tokenized into words.
            /// </summary>
            public readonly string InputColumnName;
            /// <summary>
            /// Seperator list used to tokenize input string. If not specified, space will be used.
            /// </summary>
            public IReadOnlyList<char> Separators => SeparatorsArray;
            /// <summary>
            /// State of <see cref="Separators"/>. Since <see langword="char"/>[] is multable, it's not safe to directly expose this field to users.
            /// </summary>
            internal readonly char[] SeparatorsArray;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="separators">Casing text using the rules of the invariant culture. If not specified, space will be used as separator.</param>
            public ColumnOptions(string name, string inputColumnName = null, char[] separators = null)
            {
                Name = name;
                InputColumnName = inputColumnName ?? name;
                SeparatorsArray = separators ?? new[] { ' ' };
            }
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
                if (!IsColumnTypeValid(col.ItemType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, ExpectedColumnType, col.ItemType.ToString());
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.VariableVector, col.ItemType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}