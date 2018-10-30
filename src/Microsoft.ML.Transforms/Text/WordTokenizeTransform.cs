// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Transforms.Text;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

[assembly: LoadableClass(WordTokenizeTransform.Summary, typeof(IDataTransform), typeof(WordTokenizeTransform), typeof(WordTokenizeTransform.Arguments), typeof(SignatureDataTransform),
    "Word Tokenizer Transform", "WordTokenizeTransform", "DelimitedTokenizeTransform", "WordToken", "DelimitedTokenize", "Token")]

[assembly: LoadableClass(WordTokenizeTransform.Summary, typeof(IDataTransform), typeof(WordTokenizeTransform), null, typeof(SignatureLoadDataTransform),
    "Word Tokenizer Transform", WordTokenizeTransform.LoaderSignature)]

[assembly: LoadableClass(WordTokenizeTransform.Summary, typeof(WordTokenizeTransform), null, typeof(SignatureLoadModel),
    "Word Tokenizer Transform", WordTokenizeTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(WordTokenizeTransform), null, typeof(SignatureLoadRowMapper),
   "Word Tokenizer Transform", WordTokenizeTransform.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{

    // The input for this transform is a ReadOnlyMemory or a vector of ReadOnlyMemory, and its output is a vector of ReadOnlyMemory<char>,
    // corresponding to the tokens in the input text, split using a set of user specified separator characters.
    // Empty strings and strings containing only spaces are dropped.
    /// <include file='doc.xml' path='doc/members/member[@name="WordTokenizer"]/*' />
    public sealed class WordTokenizeTransform : OneToOneTransformerBase
    {
        public class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Comma separated set of term separator(s). Commonly: 'space', 'comma', 'semicolon' or other single character.",
                ShortName = "sep")]
            public string TermSeparators;

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
                return TryUnparseCore(sb);
            }
        }

        public abstract class ArgumentsBase : TransformInputBase
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

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        public sealed class TokenizeArguments : ArgumentsBase
        {
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
                loaderAssemblyName: typeof(WordTokenizeTransform).Assembly.FullName);
        }

        private const string RegistrationName = "DelimitedTokenize";

        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly char[] Separators;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="separators">Casing text using the rules of the invariant culture. If not specified, space will be used as separator.</param>
            public ColumnInfo(string input, string output, char[] separators = null)
            {
                Input = input;
                Output = output;
                Separators = separators ?? new[] { ' ' };
            }
        }
        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();
        private readonly ColumnInfo[] _columns;

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckNonEmpty(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        public WordTokenizeTransform(IHostEnvironment env, params ColumnInfo[] columns) :
            base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema.GetColumnType(srcCol);
            if (!WordTokenizeEstimator.IsColumnTypeValid(type))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, WordTokenizeEstimator.ExpectedColumnType, type.ToString());
        }

        private WordTokenizeTransform(IHost host, ModelLoadContext ctx) :
            base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;
            _columns = new ColumnInfo[columnsLength];
            // *** Binary format ***
            // <base>
            // for each added column
            //   charArray: Separators
            for (int i = 0; i < columnsLength; i++)
            {
                var separators = ctx.Reader.ReadCharArray();
                Contracts.CheckDecode(Utils.Size(separators) > 0);
                _columns[i] = new ColumnInfo(ColumnPairs[i].input, ColumnPairs[i].output, separators);
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        public override void Save(ModelSaveContext ctx)
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
                ctx.Writer.WriteCharArray(column.Separators);
        }

        // Factory method for SignatureLoadModel.
        private static WordTokenizeTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new WordTokenizeTransform(host, ctx);
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
                var separators = args.CharArrayTermSeparators ?? PredictionUtil.SeparatorFromString(item.TermSeparators ?? args.TermSeparators);
                cols[i] = new ColumnInfo(item.Source ?? item.Name, item.Name, separators);

            }
            return new WordTokenizeTransform(env, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        protected override IRowMapper MakeRowMapper(ISchema schema) => new Mapper(this, Schema.Create(schema));

        private sealed class Mapper : MapperBase, ISaveAsPfa
        {
            private readonly ColumnType _type;
            private readonly WordTokenizeTransform _parent;
            private readonly bool[] _isSourceVector;

            public bool CanSavePfa => true;

            public Mapper(WordTokenizeTransform parent, Schema inputSchema)
              : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _type = new VectorType(TextType.Instance);
                _isSourceVector = new bool[_parent._columns.Length];
                for (int i = 0; i < _isSourceVector.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent._columns[i].Input, out int srcCol);
                    var srcType = inputSchema.GetColumnType(srcCol);
                    _isSourceVector[i] = srcType.IsVector;
                }
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _type, null);
                }
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent._columns.Length);
                disposer = null;

                input.Schema.TryGetColumnIndex(_parent._columns[iinfo].Input, out int srcCol);
                var srcType = input.Schema.GetColumnType(srcCol);
                Host.Assert(srcType.ItemType.IsText);

                if (!srcType.IsVector)
                    return MakeGetterOne(input, iinfo);
                return MakeGetterVec(input, iinfo);
            }

            private ValueGetter<VBuffer<ReadOnlyMemory<char>>> MakeGetterOne(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(ColMapNewToOld[iinfo]);
                var src = default(ReadOnlyMemory<char>);
                var terms = new List<ReadOnlyMemory<char>>();
                var separators = _parent._columns[iinfo].Separators;

                return
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        getSrc(ref src);
                        terms.Clear();

                        AddTerms(src, separators, terms);

                        var values = dst.Values;
                        if (terms.Count > 0)
                        {
                            if (Utils.Size(values) < terms.Count)
                                values = new ReadOnlyMemory<char>[terms.Count];
                            terms.CopyTo(values);
                        }

                        dst = new VBuffer<ReadOnlyMemory<char>>(terms.Count, values, dst.Indices);
                    };
            }

            private ValueGetter<VBuffer<ReadOnlyMemory<char>>> MakeGetterVec(IRow input, int iinfo)
            {
                Host.AssertValue(input);

                int cv = input.Schema.GetColumnType(ColMapNewToOld[iinfo]).VectorSize;
                Contracts.Assert(cv >= 0);
                var getSrc = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(ColMapNewToOld[iinfo]);
                var src = default(VBuffer<ReadOnlyMemory<char>>);
                var terms = new List<ReadOnlyMemory<char>>();
                var separators = _parent._columns[iinfo].Separators;

                return
                    (ref VBuffer<ReadOnlyMemory<char>> dst) =>
                    {
                        getSrc(ref src);
                        terms.Clear();

                        for (int i = 0; i < src.Count; i++)
                            AddTerms(src.Values[i], separators, terms);

                        var values = dst.Values;
                        if (terms.Count > 0)
                        {
                            if (Utils.Size(values) < terms.Count)
                                values = new ReadOnlyMemory<char>[terms.Count];
                            terms.CopyTo(values);
                        }

                        dst = new VBuffer<ReadOnlyMemory<char>>(terms.Count, values, dst.Indices);
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

            public void SaveAsPfa(BoundPfaContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                var toHide = new List<string>();
                var toDeclare = new List<KeyValuePair<string, JToken>>();

                for (int iinfo = 0; iinfo < _parent._columns.Length; ++iinfo)
                {
                    var info = _parent._columns[iinfo];
                    var srcName = info.Input;
                    string srcToken = ctx.TokenOrNullForName(srcName);
                    if (srcToken == null)
                    {
                        toHide.Add(info.Output);
                        continue;
                    }
                    var result = SaveAsPfaCore(ctx, iinfo, srcToken);
                    if (result == null)
                    {
                        toHide.Add(info.Output);
                        continue;
                    }
                    toDeclare.Add(new KeyValuePair<string, JToken>(info.Output, result));
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
                var sep = PfaUtils.String("" + exInfo.Separators[0]);
                if (_isSourceVector[iinfo])
                {
                    // If it's a vector, we'll concatenate them together.
                    srcToken = PfaUtils.Call("s.join", srcToken, sep);
                }

                if (exInfo.Separators.Length > 1)
                {
                    // Due to the intrinsics in PFA, it is much easier if we can do
                    // one split, rather than multiple splits. So, if there are multiple
                    // separators, we first replace them with the first separator, then
                    // split once on that one. This could also have been done with a.flatMap.
                    for (int i = 1; i < exInfo.Separators.Length; ++i)
                    {
                        var postSep = PfaUtils.String("" + exInfo.Separators[i]);
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
    public sealed class WordTokenizeEstimator : TrivialEstimator<WordTokenizeTransform>
    {
        public static bool IsColumnTypeValid(ColumnType type) => type.ItemType.IsText;

        internal const string ExpectedColumnType = "Text";

        /// <summary>
        /// Tokenize incoming text in <paramref name="inputColumn"/> and output the tokens as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        public WordTokenizeEstimator(IHostEnvironment env, string inputColumn, string outputColumn = null, char[] separators = null)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, separators)
        {
        }

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="separators">The separators to use (uses space character by default).</param>
        public WordTokenizeEstimator(IHostEnvironment env, (string input, string output)[] columns, char[] separators = null)
            : this(env, columns.Select(x => new WordTokenizeTransform.ColumnInfo(x.input, x.output, separators)).ToArray())
        {
        }

        /// <summary>
        ///  Tokenize incoming text in input columns and output the tokens.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        public WordTokenizeEstimator(IHostEnvironment env, params WordTokenizeTransform.ColumnInfo[] columns)
          : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordTokenizeEstimator)), new WordTokenizeTransform(env, columns))
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
                if (!IsColumnTypeValid(col.ItemType))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, ExpectedColumnType, col.ItemType.ToString());
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.VariableVector, col.ItemType, false);
            }

            return new SchemaShape(result.Values);
        }
    }

}