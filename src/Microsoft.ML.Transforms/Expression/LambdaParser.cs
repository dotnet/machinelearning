// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    internal sealed class LambdaParser
    {
        public struct SourcePos
        {
            public readonly int IchMin;
            public readonly int IchLim;
            public readonly int LineMin;
            public readonly int ColumnMin;
            public readonly int LineLim;
            public readonly int ColumnLim;

            public SourcePos(List<int> lineMap, TextSpan span, int lineMin = 1)
            {
                Contracts.AssertValue(lineMap);
                Contracts.Assert(span.Min <= span.Lim);

                IchMin = span.Min;
                IchLim = span.Lim;

                if (Utils.Size(lineMap) == 0)
                {
                    LineMin = lineMin;
                    ColumnMin = IchMin + 1;
                    LineLim = lineMin;
                    ColumnLim = IchLim + 1;
                    return;
                }

                int index = FindIndex(lineMap, IchMin, 0);
                LineMin = index + lineMin;
                int ichBase = index == 0 ? 0 : lineMap[index - 1];
                ColumnMin = IchMin - ichBase + 1;

                if (index == lineMap.Count || IchLim < lineMap[index])
                {
                    // Same line.
                    LineLim = LineMin;
                    ColumnLim = IchLim - ichBase + 1;
                }
                else
                {
                    index = FindIndex(lineMap, IchLim, index);
                    Contracts.Assert(index > 0);
                    ichBase = lineMap[index - 1];
                    LineLim = index + lineMin;
                    ColumnLim = IchLim - ichBase + 1;
                }
            }

            private static int FindIndex(List<int> map, int value, int ivMin)
            {
                Contracts.Assert(ivMin <= map.Count);
                int ivLim = map.Count;
                while (ivMin < ivLim)
                {
                    int iv = (ivMin + ivLim) / 2;
                    if (value >= map[iv])
                        ivMin = iv + 1;
                    else
                        ivLim = iv;
                }
                Contracts.Assert(0 <= ivMin && ivMin <= map.Count);
                Contracts.Assert(ivMin == map.Count || value < map[ivMin]);
                Contracts.Assert(ivMin == 0 || value >= map[ivMin - 1]);
                return ivMin;
            }
        }

        // This is re-usable state (if we choose to re-use).
        private readonly NormStr.Pool _pool;
        private readonly KeyWordTable _kwt;
        private readonly Lexer _lex;

        // Created lazily. If we choose to share static state in the future, this
        // should be volatile and set using Interlocked.CompareExchange.
        private Dictionary<TokKind, string> _mapTidStr;

        // This is the parsing state.
        private int[] _perm; // The parameter permutation.
        private DataViewType[] _types;
        private TokenCursor _curs;
        private List<Error> _errors;
        private List<int> _lineMap;

        private LambdaParser()
        {
            _pool = new NormStr.Pool();
            _kwt = new KeyWordTable(_pool);
            InitKeyWordTable();
            _lex = new Lexer(_pool, _kwt);
        }

        private void InitKeyWordTable()
        {
            Action<string, TokKind> p = _kwt.AddPunctuator;

            p("^", TokKind.Car);

            p("*", TokKind.Mul);
            p("/", TokKind.Div);
            p("%", TokKind.Per);
            p("+", TokKind.Add);
            p("-", TokKind.Sub);

            p("&&", TokKind.AmpAmp);
            p("||", TokKind.BarBar);

            p("!", TokKind.Bng);
            p("!=", TokKind.BngEqu);

            p("=", TokKind.Equ);
            p("==", TokKind.EquEqu);
            p("=>", TokKind.EquGrt);
            p("<", TokKind.Lss);
            p("<=", TokKind.LssEqu);
            p("<>", TokKind.LssGrt);
            p(">", TokKind.Grt);
            p(">=", TokKind.GrtEqu);

            p(".", TokKind.Dot);
            p(",", TokKind.Comma);
            p(":", TokKind.Colon);
            p(";", TokKind.Semi);
            p("?", TokKind.Que);
            p("??", TokKind.QueQue);

            p("(", TokKind.OpenParen);
            p(")", TokKind.CloseParen);

            Action<string, TokKind> w = _kwt.AddKeyWord;

            w("false", TokKind.False);
            w("true", TokKind.True);
            w("not", TokKind.Not);
            w("and", TokKind.And);
            w("or", TokKind.Or);
            w("with", TokKind.With);
        }

        public static LambdaNode Parse(out List<Error> errors, out List<int> lineMap, CharCursor chars, int[] perm, params DataViewType[] types)
        {
            Contracts.AssertValue(chars);
            Contracts.AssertNonEmpty(types);
            Contracts.Assert(types.Length <= LambdaCompiler.MaxParams);
            Contracts.Assert(Utils.Size(perm) == types.Length);

            LambdaParser psr = new LambdaParser();
            return psr.ParseCore(out errors, out lineMap, chars, perm, types);
        }

        private LambdaNode ParseCore(out List<Error> errors, out List<int> lineMap, CharCursor chars, int[] perm, DataViewType[] types)
        {
            Contracts.AssertValue(chars);
            Contracts.AssertNonEmpty(types);
            Contracts.Assert(Utils.Size(perm) == types.Length);

            _errors = null;
            _lineMap = new List<int>();
            _curs = new TokenCursor(_lex.LexSource(chars));
            _types = types;
            _perm = perm;

            // Skip over initial comments, new lines, lexing errors, etc.
            SkipJunk();

            LambdaNode node = ParseLambda(TokCur);
            if (TidCur != TokKind.Eof)
                PostError(TokCur, "Expected end of input");

            errors = _errors;
            lineMap = _lineMap;

            _errors = null;
            _lineMap = null;
            _curs = null;

            return node;
        }

        private void AddError(Error err)
        {
            Contracts.Assert(_errors == null || _errors.Count > 0);

            if (Utils.Size(_errors) > 0 && _errors[_errors.Count - 1].Token == err.Token)
            {
                // There's already an error report on this token, so don't issue another.
                return;
            }

            if (_errors == null)
                _errors = new List<Error>();
            _errors.Add(err);
        }

        private void PostError(Token tok, string msg)
        {
            var err = new Error(tok, msg);
            AddError(err);
        }

        private void PostError(Token tok, string msg, params object[] args)
        {
            var err = new Error(tok, msg, args);
            AddError(err);
        }

        private void PostTidError(Token tok, TokKind tidWanted)
        {
            Contracts.Assert(tidWanted != tok.Kind);
            Contracts.Assert(tidWanted != tok.KindContext);
            PostError(tok, "Expected: '{0}', Found: '{1}'", Stringize(tidWanted), Stringize(tok));
        }

        private string Stringize(Token tok)
        {
            Contracts.AssertValue(tok);
            switch (tok.Kind)
            {
                case TokKind.Ident:
                    return tok.As<IdentToken>().Value;
                default:
                    return Stringize(tok.Kind);
            }
        }

        private string Stringize(TokKind tid)
        {
            if (_mapTidStr == null)
            {
                // Build the inverse key word table, mapping token kinds to strings.
                _mapTidStr = new Dictionary<TokKind, string>();
                foreach (var kvp in _kwt.KeyWords)
                {
                    if (!kvp.Value.IsContextKeyWord)
                        _mapTidStr[kvp.Value.Kind] = kvp.Key.Value.ToString();
                }
                foreach (var kvp in _kwt.Punctuators)
                    _mapTidStr[kvp.Value] = kvp.Key.Value.ToString();
            }

            string str;
            if (_mapTidStr.TryGetValue(tid, out str))
                return str;

            return string.Format("<{0}>", tid);
        }

        private TokKind TidCur
        {
            get { return _curs.TidCur; }
        }

        private TokKind CtxCur
        {
            get { return _curs.CtxCur; }
        }

        private Token TokCur
        {
            get { return _curs.TokCur; }
        }

        private Token TokPeek(int cv = 1)
        {
            Contracts.Assert(cv > 0);

            for (int ctok = 0; ;)
            {
                var tok = _curs.TokPeek(++ctok);
                switch (tok.Kind)
                {
                    case TokKind.NewLine:
                    case TokKind.Comment:
                    case TokKind.Error:
                    case TokKind.ErrorInline:
                        break;

                    default:
                        if (--cv <= 0)
                            return tok;
                        break;
                }
            }
        }

        private TokKind TidPeek(int cv = 1)
        {
            return TokPeek(cv).Kind;
        }

        private TokKind TidNext()
        {
            _curs.TidNext();
            SkipJunk();
            return _curs.TidCur;
        }

        private void SkipJunk()
        {
            for (; ; )
            {
                switch (_curs.TidCur)
                {
                    case TokKind.NewLine:
                        _lineMap.Add(_curs.TokCur.Span.Lim);
                        break;
                    case TokKind.Comment:
                        break;

                    case TokKind.Error:
                    case TokKind.ErrorInline:
                        PostError(_curs.TokCur, _curs.TokCur.As<ErrorToken>().ToString());
                        break;

                    default:
                        return;
                }
                _curs.TidNext();
            }
        }

        private Token TokMove()
        {
            Token tok = TokCur;
            TidNext();
            return tok;
        }

        // Eats a token of the given kind. If the token is not the right kind,
        // leaves the current token and reports and returns an error.
        private bool EatTid(TokKind tid)
        {
            if (TidCur == tid || CtxCur == tid)
            {
                TidNext();
                return true;
            }
            PostTidError(TokCur, tid);
            return false;
        }

        // Returns the current token if it's of the given kind and moves to the next token.
        // If the token is not the right kind, reports an error, leaves the token, and returns null.
        private Token TokEat(TokKind tid)
        {
            if (TidCur == tid)
                return TokMove();

            PostTidError(TokCur, tid);
            return null;
        }

        private LambdaNode ParseLambda(Token tokFirst)
        {
            var items = new List<ParamNode>();
            if (TidCur == TokKind.Ident)
            {
                // Single parameter.
                items.Add(ParseParam(0));
            }
            else
            {
                EatTid(TokKind.OpenParen);
                for (; ; )
                {
                    items.Add(ParseParam(items.Count));
                    if (TidCur != TokKind.Comma)
                        break;
                    TidNext();
                }
                EatTid(TokKind.CloseParen);
            }
            if (items.Count != _types.Length)
                PostError(tokFirst, "Wrong number of parameters, expected: {0}", _types.Length);

            // Allow either : or => since the latter is problematic on the command line.
            Token tok = TidCur == TokKind.Colon ? TokMove() : TokEat(TokKind.EquGrt);
            ExprNode expr = ParseExpr();

            return new LambdaNode(tok ?? tokFirst, items.ToArray(), expr);
        }

        private ParamNode ParseParam(int index)
        {
            Contracts.Assert(0 <= index);
            Token tok = TokCur;
            string name;

            if (tok.Kind == TokKind.Ident)
            {
                name = tok.As<IdentToken>().Value;
                TidNext();
            }
            else
            {
                PostTidError(TokCur, TokKind.Ident);
                name = "<missing>";
            }

            DataViewType type;
            if (index < _types.Length)
            {
                type = _types[index];
                for (int i = 0; ; i++)
                {
                    Contracts.Assert(i < _perm.Length);
                    Contracts.Assert(0 <= _perm[i] && _perm[i] < _perm.Length);
                    if (_perm[i] == index)
                    {
                        index = i;
                        break;
                    }
                }
            }
            else
            {
                PostError(tok, "Too many parameters, expected {0}", _types.Length);
                type = null;
            }

            var res = new ParamNode(tok, name, index, type);

            if (res.ExprType == ExprTypeKind.None)
                PostError(tok, "Unsupported type");

            return res;
        }

        private ExprNode ParseExpr()
        {
            return ParseExpr(Precedence.None);
        }

        // Parses the next (maximal) expression with precedence >= precMin.
        private ExprNode ParseExpr(Precedence precMin)
        {
            // ParseOperand may accept PrefixUnary and higher, so ParseExpr should never be called
            // with precMin > Precedence.PrefixUnary - it will not correctly handle those cases.
            Contracts.Assert(Precedence.None <= precMin);
            Contracts.Assert(precMin <= Precedence.PrefixUnary);

            // Get the left operand.
            ExprNode node = ParsePrimary();

            // Process operators and right operands as long as the precedence bound is satisfied.
            for (; ; )
            {
                Contracts.AssertValue(node);
                switch (TidCur)
                {
                    case TokKind.Car:
                        Contracts.Assert(precMin <= Precedence.Power);
                        // Note that the right operand can include unary operators.
                        node = new BinaryOpNode(TokMove(), BinaryOp.Power, node, ParseExpr(Precedence.PrefixUnary));
                        break;

                    case TokKind.Mul:
                        if (precMin > Precedence.Mul)
                            return node;
                        node = new BinaryOpNode(TokMove(), BinaryOp.Mul, node, ParseExpr(Precedence.Mul + 1));
                        break;
                    case TokKind.Div:
                        if (precMin > Precedence.Mul)
                            return node;
                        node = new BinaryOpNode(TokMove(), BinaryOp.Div, node, ParseExpr(Precedence.Mul + 1));
                        break;
                    case TokKind.Per:
                        if (precMin > Precedence.Mul)
                            return node;
                        node = new BinaryOpNode(TokMove(), BinaryOp.Mod, node, ParseExpr(Precedence.Mul + 1));
                        break;

                    case TokKind.Sub:
                        if (precMin > Precedence.Add)
                            return node;
                        node = new BinaryOpNode(TokMove(), BinaryOp.Sub, node, ParseExpr(Precedence.Add + 1));
                        break;
                    case TokKind.Add:
                        if (precMin > Precedence.Add)
                            return node;
                        node = new BinaryOpNode(TokMove(), BinaryOp.Add, node, ParseExpr(Precedence.Add + 1));
                        break;

                    case TokKind.AmpAmp:
                    case TokKind.And:
                        if (precMin > Precedence.And)
                            return node;
                        node = new BinaryOpNode(TokMove(), BinaryOp.And, node, ParseExpr(Precedence.And + 1));
                        break;
                    case TokKind.BarBar:
                    case TokKind.Or:
                        if (precMin > Precedence.Or)
                            return node;
                        node = new BinaryOpNode(TokMove(), BinaryOp.Or, node, ParseExpr(Precedence.Or + 1));
                        break;

                    case TokKind.QueQue:
                        if (precMin > Precedence.Coalesce)
                            return node;
                        // Note that the associativity is different than other binary operators (right instead of left),
                        // so the recursive call accepts Precedence.Coal.
                        node = new BinaryOpNode(TokMove(), BinaryOp.Coalesce, node, ParseExpr(Precedence.Coalesce));
                        break;

                    case TokKind.Que:
                        if (precMin > Precedence.Conditional)
                            return node;
                        node = new ConditionalNode(TokMove(), node, ParseExpr(), TokEat(TokKind.Colon), ParseExpr());
                        break;

                    // Comparison operators
                    // expr = ... = expr
                    // expr <> ... <> expr
                    case TokKind.Equ:
                    case TokKind.EquEqu:
                        if (precMin > Precedence.Compare)
                            return node;
                        node = ParseCompareExpr(node, CompareOp.Equal, TokKind.Equ, TokKind.EquEqu);
                        break;

                    case TokKind.LssGrt:
                    case TokKind.BngEqu:
                        if (precMin > Precedence.Compare)
                            return node;
                        node = ParseCompareExpr(node, CompareOp.NotEqual, TokKind.LssGrt, TokKind.BngEqu);
                        break;

                    // expr < expr
                    // expr <= expr
                    case TokKind.Lss:
                    case TokKind.LssEqu:
                        if (precMin > Precedence.Compare)
                            return node;
                        node = ParseCompareExpr(node, CompareOp.IncrChain, TokKind.LssEqu, TokKind.Lss);
                        break;

                    // expr > expr
                    // expr >= expr
                    case TokKind.Grt:
                    case TokKind.GrtEqu:
                        if (precMin > Precedence.Compare)
                            return node;
                        node = ParseCompareExpr(node, CompareOp.DecrChain, TokKind.GrtEqu, TokKind.Grt);
                        break;

                    case TokKind.True:
                    case TokKind.False:
                    case TokKind.IntLit:
                    case TokKind.FltLit:
                    case TokKind.DblLit:
                    case TokKind.CharLit:
                    case TokKind.StrLit:
                        PostError(TokCur, "Operator expected");
                        node = new BinaryOpNode(TokCur, BinaryOp.Error, node, ParseExpr(Precedence.Error));
                        break;

                    default:
                        return node;
                }
            }
        }

        private ExprNode ParsePrimary()
        {
            switch (TidCur)
            {
                // (Expr)
                case TokKind.OpenParen:
                    return ParseParenExpr();

                // -Expr
                case TokKind.Sub:
                    return new UnaryOpNode(TokMove(), UnaryOp.Minus, ParseExpr(Precedence.PrefixUnary));

                // not Expr
                case TokKind.Not:
                case TokKind.Bng:
                    return new UnaryOpNode(TokMove(), UnaryOp.Not, ParseExpr(Precedence.PrefixUnary));

                // Literals
                case TokKind.IntLit:
                case TokKind.FltLit:
                case TokKind.DblLit:
                    return new NumLitNode(TokMove().As<NumLitToken>());
                case TokKind.True:
                case TokKind.False:
                    return new BoolLitNode(TokMove());
                case TokKind.StrLit:
                    return new StrLitNode(TokMove().As<StrLitToken>());

                // Name
                case TokKind.Ident:
                    if (TidPeek() == TokKind.OpenParen)
                        return ParseInvocation();
                    if (TidPeek() == TokKind.Dot && TidPeek(2) == TokKind.Ident && TidPeek(3) == TokKind.OpenParen)
                        return ParseInvocationWithNameSpace();
                    return ParseIdent();

                case TokKind.CharLit:
                    var result = ParseIdent();
                    TokMove();
                    return result;

                case TokKind.With:
                    return ParseWith();

                default:
                    // Error
                    return ParseIdent();
            }
        }

        private CompareNode ParseCompareExpr(ExprNode node, CompareOp op, TokKind tidLax, TokKind tidStrict)
        {
            Contracts.AssertValue(node);
            Contracts.Assert(TidCur == tidLax || TidCur == tidStrict);

            Token tok = TokCur;
            List<Node> list = new List<Node>();
            List<Token> ops = new List<Token>();
            list.Add(node);
            for (; ; )
            {
                if (TidCur != tidLax && TidCur != tidStrict)
                    break;
                ops.Add(TokMove());
                list.Add(ParseExpr(Precedence.Compare + 1));
            }
            Contracts.Assert(list.Count >= 2);

            // The grammar disallows mixed direction expressions like:
            //   a < b > c <= 4
            // After posting an error, we continue parsing to produce: ((a < b) > c) <= 4.
            // Note that this will also produce a type checking error.
            Contracts.Assert(TidCur != tidLax);
            Contracts.Assert(TidCur != tidStrict);
            switch (TidCur)
            {
                case TokKind.LssGrt:
                case TokKind.BngEqu:
                case TokKind.Equ:
                case TokKind.EquEqu:
                case TokKind.Lss:
                case TokKind.LssEqu:
                case TokKind.Grt:
                case TokKind.GrtEqu:
                    PostError(TokCur, "Mixed direction not allowed");
                    break;
            }

            return new CompareNode(tok, op, new ListNode(tok, list.ToArray(), ops.ToArray()));
        }

        private IdentNode ParseIdent()
        {
            if (TidCur == TokKind.Ident)
                return new IdentNode(TokMove().As<IdentToken>());
            PostTidError(TokCur, TokKind.Ident);
            return new IdentNode(TokCur, "<missing>", true);
        }

        private CallNode ParseInvocation()
        {
            Contracts.Assert(TidCur == TokKind.Ident);
            Contracts.Assert(TidPeek() == TokKind.OpenParen);

            NameNode head = new NameNode(TokMove().As<IdentToken>());
            Contracts.Assert(TidCur == TokKind.OpenParen);

            Token tok = TokMove();
            return new CallNode(tok, head, ParseList(tok, TokKind.CloseParen), TokEat(TokKind.CloseParen));
        }

        private CallNode ParseInvocationWithNameSpace()
        {
            Contracts.Assert(TidCur == TokKind.Ident);
            Contracts.Assert(TidPeek() == TokKind.Dot);
            Contracts.Assert(TidPeek(2) == TokKind.Ident);
            Contracts.Assert(TidPeek(3) == TokKind.OpenParen);

            NameNode ns = new NameNode(TokMove().As<IdentToken>());
            Contracts.Assert(TidCur == TokKind.Dot);
            Token tokDot = TokMove();
            NameNode head = new NameNode(TokMove().As<IdentToken>());
            Contracts.Assert(TidCur == TokKind.OpenParen);
            Token tokParen = TokMove();

            return new CallNode(tokParen, ns, tokDot, head, ParseList(tokParen, TokKind.CloseParen), TokEat(TokKind.CloseParen));
        }

        private ExprNode ParseParenExpr()
        {
            Contracts.Assert(TidCur == TokKind.OpenParen);

            TidNext();
            ExprNode node = ParseExpr(Precedence.None);
            EatTid(TokKind.CloseParen);
            return node;
        }

        private ListNode ParseList(Token tok, TokKind tidEmpty)
        {
            if (TidCur == tidEmpty)
                return new ListNode(tok, new Node[0], null);

            List<Token> commas = null;
            List<Node> list = new List<Node>();
            for (; ; )
            {
                list.Add(ParseExpr());
                if (TidCur != TokKind.Comma)
                    break;
                Utils.Add(ref commas, TokMove());
            }
            return new ListNode(tok, list.ToArray(), Utils.ToArray(commas));
        }

        // Note that the grammar allows multiple assignments per with expression. This is simply syntactic sugar
        // for chained with expressions.
        private WithNode ParseWith(Token tokWith = null)
        {
            Token tok;
            Token tokOpen;
            if (tokWith == null)
            {
                // We're at the beginning of the "with" syntax.
                Contracts.Assert(TidCur == TokKind.With);
                tokWith = TokMove();
                tok = tokWith;
                tokOpen = TokCur;
                EatTid(TokKind.OpenParen);
            }
            else
            {
                // We're at a comma between with assignments.
                Contracts.Assert(TidCur == TokKind.Comma);
                tok = TokMove();
                tokOpen = tok;
            }

            var local = ParseWithLocal();

            ExprNode body;
            if (TidCur == TokKind.Comma)
            {
                // Recurse to get the nested scenario.
                body = ParseWith(tokWith);
            }
            else
            {
                EatTid(TokKind.Semi);
                body = ParseExpr();
                EatTid(TokKind.CloseParen);
            }

            return new WithNode(tok, local, body);
        }

        private WithLocalNode ParseWithLocal()
        {
            Token tok = TokCur;
            string name;

            if (tok.Kind == TokKind.Ident)
            {
                name = tok.As<IdentToken>().Value;
                TidNext();
            }
            else
            {
                PostTidError(TokCur, TokKind.Ident);
                name = "<missing>";
            }

            if (TidCur == TokKind.Equ)
                tok = TokCur;
            EatTid(TokKind.Equ);

            var value = ParseExpr();
            return new WithLocalNode(tok, name, value);
        }
    }
}
