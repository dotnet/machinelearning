// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Newtonsoft.Json.Linq;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(DelimitedTokenizeTransform.Summary, typeof(DelimitedTokenizeTransform), typeof(DelimitedTokenizeTransform.Arguments), typeof(SignatureDataTransform),
    "Word Tokenizer Transform", "WordTokenizeTransform", "DelimitedTokenizeTransform", "WordToken", "DelimitedTokenize", "Token")]

[assembly: LoadableClass(DelimitedTokenizeTransform.Summary, typeof(DelimitedTokenizeTransform), typeof(DelimitedTokenizeTransform.TokenizeArguments), typeof(SignatureTokenizeTransform),
    "Word Tokenizer Transform", "WordTokenizeTransform", "DelimitedTokenizeTransform", "WordToken", "DelimitedTokenize", "Token")]

[assembly: LoadableClass(DelimitedTokenizeTransform.Summary, typeof(DelimitedTokenizeTransform), null, typeof(SignatureLoadDataTransform),
    "Word Tokenizer Transform", DelimitedTokenizeTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Signature for creating an ITokenizeTransform.
    /// </summary>
    public delegate void SignatureTokenizeTransform(IDataView input, OneToOneColumn[] columns);

    public interface ITokenizeTransform : IDataTransform
    {
    }

    // The input for this transform is a ReadOnlyMemory or a vector of ReadOnlyMemory, and its output is a vector of ReadOnlyMemory<char>,
    // corresponding to the tokens in the input text, split using a set of user specified separator characters.
    // Empty strings and strings containing only spaces are dropped.
    /// <include file='doc.xml' path='doc/members/member[@name="WordTokenizer"]/*' />
    public sealed class DelimitedTokenizeTransform : OneToOneTransformBase, ITokenizeTransform
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

        /// <summary>
        /// Extra information for each column (in addition to ColumnInfo).
        /// </summary>
        private sealed class ColInfoEx
        {
            public readonly char[] Separators;

            public ColInfoEx(Arguments args, int iinfo)
            {
                // Use character array separator if any otherwise default to the comma separated string
                Separators = args.CharArrayTermSeparators ?? PredictionUtil.SeparatorFromString(args.Column[iinfo].TermSeparators ?? args.TermSeparators);
                Contracts.CheckUserArg(Utils.Size(Separators) > 0, nameof(args.TermSeparators));
            }

            public ColInfoEx(ArgumentsBase args)
            {
                Separators = args.CharArrayTermSeparators ?? PredictionUtil.SeparatorFromString(args.TermSeparators);
                Contracts.CheckUserArg(Utils.Size(Separators) > 0, nameof(args.TermSeparators));
            }

            public ColInfoEx(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: length of separators
                // char[]: separators
                Separators = ctx.Reader.ReadCharArray();
                Contracts.CheckDecode(Utils.Size(Separators) > 0);
            }

            public void Save(ModelSaveContext ctx)
            {
                // *** Binary format ***
                // int: length of separators
                // char[]: separators
                Contracts.Assert(Utils.Size(Separators) > 0);
                ctx.Writer.WriteCharArray(Separators);
            }
        }

        internal const string Summary = "The input to this transform is text, and the output is a vector of text containing the words (tokens) in the original text. "
            + "The separator is space, but can be specified as any other character (or multiple characters) if needed.";

        public const string LoaderSignature = "TokenizeTextTransform";

        internal const string UserName = "Tokenize Text Transform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "WRDTOKNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DelimitedTokenizeTransform).Assembly.FullName);
        }

        public override bool CanSavePfa => true;

        private readonly ColInfoEx[] _exes;

        // Cached type of the output column(s).
        private readonly ColumnType _columnType;

        private const string RegistrationName = "DelimitedTokenize";

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public DelimitedTokenizeTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column,
                input, TestIsTextItem)
        {
            // REVIEW: Need to decide whether to inject an NA token between slots in ReadOnlyMemory inputs.
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _exes = new ColInfoEx[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                _exes[iinfo] = new ColInfoEx(args, iinfo);

            _columnType = new VectorType(TextType.Instance);
            Metadata.Seal();
        }

        /// <summary>
        /// Public constructor corresponding to SignatureTokenizeTransform. It accepts arguments of type ArgumentsBase,
        /// and a separate array of columns (constructed from the caller -WordBag/WordHashBag- arguments).
        /// </summary>
        public DelimitedTokenizeTransform(IHostEnvironment env, TokenizeArguments args, IDataView input, OneToOneColumn[] columns)
            : base(env, RegistrationName, columns, input, TestIsTextItem)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(columns) > 0, nameof(Arguments.Column));

            // REVIEW: Need to decide whether to inject an NA token between slots in ReadOnlyMemory inputs.
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(columns));

            _exes = new ColInfoEx[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                _exes[iinfo] = new ColInfoEx(args);

            _columnType = new VectorType(TextType.Instance);
            Metadata.Seal();
        }

        private DelimitedTokenizeTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsTextItem)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <base>
            // for each added column
            //   ColInfoEx
            Host.AssertNonEmpty(Infos);
            _exes = new ColInfoEx[Infos.Length];
            for (int iinfo = 0; iinfo < _exes.Length; iinfo++)
                _exes[iinfo] = new ColInfoEx(ctx);

            _columnType = new VectorType(TextType.Instance);
            Metadata.Seal();
        }

        public static DelimitedTokenizeTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new DelimitedTokenizeTransform(h, ctx, input));
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
            for (int i = 0; i < _exes.Length; i++)
                _exes[i].Save(ctx);
        }

        protected override JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColInfo info, JToken srcToken)
        {
            Contracts.AssertValue(ctx);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);
            Contracts.Assert(Infos[iinfo] == info);
            Contracts.AssertValue(srcToken);
            Contracts.Assert(CanSavePfa);

            var exInfo = _exes[iinfo];
            var sep = PfaUtils.String("" + exInfo.Separators[0]);
            if (info.TypeSrc.IsVector)
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

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.AssertValue(_columnType);
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return _columnType;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var info = Infos[iinfo];
            if (!info.TypeSrc.IsVector)
                return MakeGetterOne(input, iinfo);
            return MakeGetterVec(input, iinfo);
        }

        private ValueGetter<VBuffer<ReadOnlyMemory<char>>> MakeGetterOne(IRow input, int iinfo)
        {
            Host.AssertValue(input);
            Host.Assert(Infos[iinfo].TypeSrc.IsText);

            var getSrc = GetSrcGetter<ReadOnlyMemory<char>>(input, iinfo);
            var src = default(ReadOnlyMemory<char>);
            var terms = new List<ReadOnlyMemory<char>>();
            var separators = _exes[iinfo].Separators;

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
            Host.Assert(Infos[iinfo].TypeSrc.IsVector);
            Host.Assert(Infos[iinfo].TypeSrc.ItemType.IsText);

            int cv = Infos[iinfo].TypeSrc.VectorSize;
            Contracts.Assert(cv >= 0);

            var getSrc = GetSrcGetter<VBuffer<ReadOnlyMemory<char>>>(input, iinfo);
            var src = default(VBuffer<ReadOnlyMemory<char>>);
            var terms = new List<ReadOnlyMemory<char>>();
            var separators = _exes[iinfo].Separators;

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
    }
}