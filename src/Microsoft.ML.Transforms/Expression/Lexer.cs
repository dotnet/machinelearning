// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// The lexer. This is effectively a template. Call LexSource to get an Enumerable of tokens.
    /// </summary>
    [BestFriend]
    internal partial class Lexer
    {
        private readonly NormStr.Pool _pool;
        private readonly KeyWordTable _kwt;

        /// <summary>
        /// The constructor. Caller must provide the name pool and key word table.
        /// </summary>
        public Lexer(NormStr.Pool pool, KeyWordTable kwt)
        {
            Contracts.AssertValue(pool);
            Contracts.AssertValue(kwt);
            _pool = pool;
            _kwt = kwt;
        }

        public IEnumerable<Token> LexSource(CharCursor cursor)
        {
            Contracts.AssertValue(cursor);

            LexerImpl impl = new LexerImpl(this, cursor);
            Token tok;
            while ((tok = impl.GetNextToken()) != null)
                yield return tok;
            yield return impl.GetEof();
        }

        private partial class LexerImpl
        {
            private readonly Lexer _lex;
            private readonly CharCursor _cursor;

            private readonly StringBuilder _sb; // Used while building a token.
            private int _ichMinTok; // The start of the current token.
            private readonly Queue<Token> _queue; // For multiple returns.
#pragma warning disable 414
            // This will be used by any pre-processor, so keep it around.
            private bool _fLineStart;
#pragma warning restore 414

            public LexerImpl(Lexer lex, CharCursor cursor)
            {
                _lex = lex;
                _cursor = cursor;
                _sb = new StringBuilder();
                _queue = new Queue<Token>(4);
                _fLineStart = true;
            }

            /// <summary>
            /// Whether we've hit the end of input yet. If this returns true, ChCur will be zero.
            /// </summary>
            private bool Eof { get { return _cursor.Eof; } }

            /// <summary>
            /// The current character. Zero if we've hit the end of input.
            /// </summary>
            private char ChCur
            {
                get { return _cursor.ChCur; }
            }

            /// <summary>
            /// Advance to the next character and return it.
            /// </summary>
            private char ChNext()
            {
                return _cursor.ChNext();
            }

            private char ChPeek(int ich)
            {
                return _cursor.ChPeek(ich);
            }

            /// <summary>
            /// Marks the beginning of the current token.
            /// </summary>
            private void StartTok()
            {
                _ichMinTok = _cursor.IchCur;
            }

            /// <summary>
            /// Called to embed an error token in the stream.
            /// </summary>
            private void ReportError(ErrId eid)
            {
                ReportError(_ichMinTok, _cursor.IchCur, eid, null);
            }

            private void ReportError(ErrId eid, params object[] args)
            {
                ReportError(_ichMinTok, _cursor.IchCur, eid, args);
            }

            private void ReportError(int ichMin, int ichLim, ErrId eid, params object[] args)
            {
                // REVIEW: Fix this so the error is marked as nested if appropriate!
                ErrorToken err = new ErrorToken(GetTextSpan(ichMin, ichLim), eid, args);
                _queue.Enqueue(err);
            }

            private TextSpan GetSpan()
            {
                var span = new TextSpan(_ichMinTok, _cursor.IchCur);
                StartTok();
                return span;
            }

            private TextSpan GetTextSpan(int ichMin, int ichLim)
            {
                return new TextSpan(ichMin, ichLim);
            }

            /// <summary>
            /// Form and return the next token. Returns null to signal end of input.
            /// </summary>
            public Token GetNextToken()
            {
                // New line tokens and errors can be "nested" inside comments or string literals
                // so this code isn't as simple as lexing a single token and returning it.
                // Note that we return the outer token before nested ones.

                while (_queue.Count == 0)
                {
                    if (Eof)
                        return null;
                    Token tokNew = FetchToken();
                    if (tokNew != null)
                        return tokNew;
                }

                // Only new lines and errors should be enqueued.
                Token tok = _queue.Dequeue();
                Contracts.Assert(tok.Kind == TokKind.NewLine || tok.Kind == TokKind.Error);
                return tok;
            }

            /// <summary>
            /// Call once GetNextToken returns null if you need an Eof token.
            /// </summary>
            public EofToken GetEof()
            {
                Contracts.Assert(Eof);
                return new EofToken(GetTextSpan(_cursor.IchCur, _cursor.IchCur));
            }

            private Token FetchToken()
            {
                Contracts.Assert(!Eof);
                StartTok();

                LexStartKind kind = LexCharUtils.StartKind(ChCur);
                if (kind != LexStartKind.Space && kind != LexStartKind.PreProc)
                    _fLineStart = false;

                switch (kind)
                {
                    case LexStartKind.Punc:
                        return LexPunc();
                    case LexStartKind.NumLit:
                        return LexNumLit();
                    case LexStartKind.StrLit:
                        return LexStrLit();
                    case LexStartKind.Verbatim:
                        if (ChPeek(1) == '"')
                            return LexStrLit();
                        if (LexCharUtils.StartKind(ChPeek(1)) == LexStartKind.Ident)
                            return LexIdent();
                        ChNext();
                        ReportError(ErrId.VerbatimLiteralExpected);
                        return null;
                    case LexStartKind.Ident:
                        return LexIdent();
                    case LexStartKind.Comment:
                        return LexComment();
                    case LexStartKind.Space:
                        return LexSpace();
                    case LexStartKind.LineTerm:
                        LexLineTerm();
                        return null;
                    case LexStartKind.PreProc:
                        return LexPreProc();
                    default:
                        return LexError();
                }
            }

            /// <summary>
            /// Called to lex a punctuator (operator). Asserts the current character lex type
            /// is LexCharType.Punc.
            /// </summary>
            private Token LexPunc()
            {
                int cchPunc = 0;
                TokKind tidPunc = TokKind.None;

                _sb.Length = 0;
                _sb.Append(ChCur);
                for (; ; )
                {
                    TokKind tidCur;
                    NormStr nstr = _lex._pool.Add(_sb);
                    if (!_lex._kwt.IsPunctuator(nstr, out tidCur))
                        break;

                    if (tidCur != TokKind.None)
                    {
                        // This is a real punctuator, not just a prefix.
                        tidPunc = tidCur;
                        cchPunc = _sb.Length;
                    }

                    char ch = ChPeek(_sb.Length);
                    if (!LexCharUtils.IsPunc(ch))
                        break;
                    _sb.Append(ch);
                }
                if (cchPunc == 0)
                    return LexError();
                while (--cchPunc >= 0)
                    ChNext();
                return KeyToken.Create(GetSpan(), tidPunc);
            }

            /// <summary>
            /// Called to lex a numeric literal or a Dot token. Asserts the current
            /// character lex type is LexCharType.NumLit.
            /// </summary>
            private Token LexNumLit()
            {
                Contracts.Assert(LexCharUtils.StartKind(ChCur) == LexStartKind.NumLit);
                Contracts.Assert(LexCharUtils.IsDigit(ChCur) || ChCur == '.');

                // A dot not followed by a digit is just a Dot. This is a very common case (hence first).
                if (ChCur == '.' && !LexCharUtils.IsDigit(ChPeek(1)))
                    return LexPunc();

                // Check for a hex literal. Note that 0x followed by a non-hex-digit is really a 0 followed
                // by an identifier.
                if (ChCur == '0' && (ChPeek(1) == 'x' || ChPeek(1) == 'X') && LexCharUtils.IsHexDigit(ChPeek(2)))
                {
                    // Advance to first hex digit.
                    ChNext();
                    ChNext();
                    return LexHexInt();
                }

                // Decimal literal (possible floating point).
                Contracts.Assert(LexCharUtils.IsDigit(ChCur) || ChCur == '.' && LexCharUtils.IsDigit(ChPeek(1)));
                bool fExp = false;
                bool fDot = ChCur == '.';
                _sb.Length = 0;
                _sb.Append(ChCur);

                for (; ; )
                {
                    if (ChNext() == '.')
                    {
                        if (fDot || !LexCharUtils.IsDigit(ChPeek(1)))
                            break;
                        fDot = true;
                    }
                    else if (!LexCharUtils.IsDigit(ChCur))
                        break;
                    _sb.Append(ChCur);
                }

                // Check for an exponent.
                if (ChCur == 'e' || ChCur == 'E')
                {
                    char chTmp = ChPeek(1);
                    if (LexCharUtils.IsDigit(chTmp) || (chTmp == '+' || chTmp == '-') && LexCharUtils.IsDigit(ChPeek(2)))
                    {
                        fExp = true;
                        _sb.Append(ChCur);
                        _sb.Append(ChNext());
                        while (LexCharUtils.IsDigit(chTmp = ChNext()))
                            _sb.Append(chTmp);
                    }
                }

                bool fReal = fDot || fExp;
                char chSuf = LexRealSuffix(fReal);
                if (fReal || chSuf != '\0')
                    return LexRealNum(chSuf);

                // Integer type.
                return LexDecInt(LexIntSuffix());
            }

            /// <summary>
            /// Lex a hex literal optionally followed by an integer suffix. Asserts the current
            /// character is a hex digit.
            /// </summary>
            private Token LexHexInt()
            {
                Contracts.Assert(LexCharUtils.IsHexDigit(ChCur));

                ulong u = 0;
                bool fOverflow = false;

                do
                {
                    if ((u & 0xF000000000000000) != 0 && !fOverflow)
                    {
                        ReportError(ErrId.IntOverflow);
                        fOverflow = true;
                    }
                    u = (u << 4) + (ulong)LexCharUtils.GetHexVal(ChCur);
                } while (LexCharUtils.IsHexDigit(ChNext()));

                if (fOverflow)
                    u = ulong.MaxValue;

                return new IntLitToken(GetSpan(), u, LexIntSuffix() | IntLitKind.Hex);
            }

            /// <summary>
            /// Lex a decimal integer literal. The digits must be in _sb.
            /// </summary>
            private Token LexDecInt(IntLitKind ilk)
            {
                // Digits are in _sb.
                Contracts.Assert(_sb.Length > 0);
                ulong u = 0;

                try
                {
                    for (int ich = 0; ich < _sb.Length; ich++)
                        u = checked(u * 10 + (ulong)LexCharUtils.GetDecVal(_sb[ich]));
                }
                catch (System.OverflowException)
                {
                    ReportError(ErrId.IntOverflow);
                    u = ulong.MaxValue;
                }
                return new IntLitToken(GetSpan(), u, ilk);
            }

            /// <summary>
            /// Lex a real literal (float, double or decimal). The characters should be in _sb.
            /// </summary>
            private Token LexRealNum(char chSuf)
            {
                // Digits are in _sb.
                Contracts.Assert(_sb.Length > 0);

                TextSpan span = GetSpan();
                switch (chSuf)
                {
                    default:
                        Contracts.Assert(chSuf == '\0' || chSuf == 'D');
                        try
                        {
                            double dbl = double.Parse(_sb.ToString(), NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent);
                            return new DblLitToken(span, dbl, chSuf != 0);
                        }
                        catch (OverflowException)
                        {
                            ReportError(ErrId.FloatOverflow, "double");
                            return new DblLitToken(span, double.PositiveInfinity, chSuf != 0);
                        }
                    case 'F':
                        try
                        {
                            double dbl = double.Parse(_sb.ToString(), NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent);
                            return new FltLitToken(span, (float)dbl);
                        }
                        catch (OverflowException)
                        {
                            ReportError(ErrId.FloatOverflow, "float");
                            return new FltLitToken(span, float.PositiveInfinity);
                        }
                }
            }

            /// <summary>
            /// Lex an optional integer suffix (U and/or L).
            /// </summary>
            private IntLitKind LexIntSuffix()
            {
                IntLitKind ilk = IntLitKind.None;

                for (; ; )
                {
                    if (ChCur == 'U' || ChCur == 'u')
                    {
                        if ((ilk & IntLitKind.Uns) != 0)
                            break;
                        ilk |= IntLitKind.Uns;
                    }
                    else if (ChCur == 'L' || ChCur == 'l')
                    {
                        if ((ilk & IntLitKind.Lng) != 0)
                            break;
                        ilk |= IntLitKind.Lng;
                    }
                    else
                        break;
                    ChNext();
                }
                return ilk;
            }

            /// <summary>
            /// Lex an optional real suffix (F, D, M).
            /// </summary>
            private char LexRealSuffix(bool fKnown)
            {
                char ch;

                switch (ChCur)
                {
                    default:
                        return '\0';
                    case 'd':
                    case 'D':
                        ch = 'D';
                        break;
                    case 'f':
                    case 'F':
                        ch = 'F';
                        break;
                    case 'l':
                    case 'L':
                        if (!fKnown)
                            return '\0';
                        ch = 'L';
                        break;
                }
                ChNext();
                return ch;
            }

            /// <summary>
            /// Lex a string or character literal.
            /// </summary>
            private Token LexStrLit()
            {
                char chQuote;

                _sb.Length = 0;
                if (ChCur == '@')
                {
                    chQuote = '"';
                    ChNext();
                    Contracts.Assert(ChCur == '"');
                    ChNext();
                    for (; ; )
                    {
                        char ch = ChCur;
                        if (ch == '"')
                        {
                            ChNext();
                            if (ChCur != '"')
                                break;
                            ChNext();
                        }
                        else if (LexCharUtils.IsLineTerm(ch))
                            ch = LexLineTerm(_sb);
                        else if (Eof)
                        {
                            ReportError(ErrId.UnterminatedString);
                            break;
                        }
                        else
                            ChNext();
                        _sb.Append(ch);
                    }
                }
                else
                {
                    Contracts.Assert(ChCur == '"' || ChCur == '\'');
                    chQuote = ChCur;

                    ChNext();
                    for (; ; )
                    {
                        char ch = ChCur;
                        if (ch == chQuote || Eof || LexCharUtils.IsLineTerm(ch))
                            break;
                        if (ch == '\\')
                        {
                            uint u;
                            if (!FLexEscChar(false, out u))
                                continue;
                            if (u < 0x10000)
                                ch = (char)u;
                            else
                            {
                                char chT;
                                if (!ConvertToSurrogatePair(u, out chT, out ch))
                                    continue;
                                _sb.Append(chT);
                            }
                        }
                        else
                            ChNext();
                        _sb.Append(ch);
                    }

                    if (ChCur != chQuote)
                        ReportError(ErrId.NewlineInConst);
                    else
                        ChNext();
                }

                if (chQuote == '"')
                    return new StrLitToken(GetSpan(), _sb.ToString());

                if (_sb.Length != 1)
                    ReportError(_sb.Length == 0 ? ErrId.CharConstEmpty : ErrId.CharConstTooLong);
                return new CharLitToken(GetSpan(), _sb.Length > 0 ? _sb[0] : '\0');
            }

            /// <summary>
            /// Lex a character escape. Returns true if successful (ch is valid).
            /// </summary>
            private bool FLexEscChar(bool fUniOnly, out uint u)
            {
                Contracts.Assert(ChCur == '\\');

                int ichErr = _cursor.IchCur;
                bool fUni;
                int cchHex;

                switch (ChNext())
                {
                    case 'u':
                        fUni = true;
                        cchHex = 4;
                        goto LHex;
                    case 'U':
                        fUni = true;
                        cchHex = 8;
                        goto LHex;
                    default:
                        if (!fUniOnly)
                        {
                            switch (ChCur)
                            {
                                default:
                                    goto LBad;
                                case 'x':
                                case 'X':
                                    fUni = false;
                                    cchHex = 4;
                                    goto LHex;
                                case '\'':
                                    u = 0x0027;
                                    break;
                                case '"':
                                    u = 0x0022;
                                    break;
                                case '\\':
                                    u = 0x005C;
                                    break;
                                case '0':
                                    u = 0x0000;
                                    break;
                                case 'a':
                                    u = 0x0007;
                                    break;
                                case 'b':
                                    u = 0x0008;
                                    break;
                                case 'f':
                                    u = 0x000C;
                                    break;
                                case 'n':
                                    u = 0x000A;
                                    break;
                                case 'r':
                                    u = 0x000D;
                                    break;
                                case 't':
                                    u = 0x0009;
                                    break;
                                case 'v':
                                    u = 0x000B;
                                    break;
                            }
                            ChNext();
                            return true;
                        }
LBad:
                        ReportError(ichErr, _cursor.IchCur, ErrId.BadEscape);
                        u = 0;
                        return false;
                }

LHex:
                bool fRet = true;
                ChNext();

                u = 0;
                for (int ich = 0; ich < cchHex; ich++)
                {
                    if (!LexCharUtils.IsHexDigit(ChCur))
                    {
                        fRet = (ich > 0);
                        if (fUni || !fRet)
                            ReportError(ichErr, _cursor.IchCur, ErrId.BadEscape);
                        break;
                    }
                    u = (u << 4) + (uint)LexCharUtils.GetHexVal(ChCur);
                    ChNext();
                }
                return fRet;
            }

            /// <summary>
            /// Convert the pair of characters to a surrogate pair.
            /// </summary>
            private bool ConvertToSurrogatePair(uint u, out char ch1, out char ch2)
            {
                Contracts.Assert(u > 0x0000FFFF);
                if (u > 0x0010FFFF)
                {
                    ReportError(ErrId.BadEscape);
                    ch1 = ch2 = '\0';
                    return false;
                }
                ch1 = (char)((u - 0x10000) / 0x400 + 0xD800);
                ch2 = (char)((u - 0x10000) % 0x400 + 0xDC00);
                return true;
            }

            /// <summary>
            /// Lex an identifier.
            /// </summary>
            private Token LexIdent()
            {
                bool fVerbatim = false;
                if (ChCur == '@')
                {
                    fVerbatim = true;
                    ChNext();
                }

                NormStr nstr = LexIdentCore(ref fVerbatim);
                if (nstr == null)
                {
                    // Error already reported.
                    return null;
                }

                if (!fVerbatim)
                {
                    KeyWordTable.KeyWordKind kind;
                    if (_lex._kwt.IsKeyWord(nstr, out kind))
                        return KeyToken.CreateKeyWord(GetSpan(), nstr.Value.ToString(), kind.Kind, kind.IsContextKeyWord);
                }
                return new IdentToken(GetSpan(), nstr.Value.ToString());
            }

            private NormStr LexIdentCore(ref bool fVerbatim)
            {
                Contracts.Assert(LexCharUtils.IsIdentStart(ChCur));

                _sb.Length = 0;
                for (; ; )
                {
                    char ch;
                    if (ChCur == '\\')
                    {
                        uint u;
                        int ichErr = _cursor.IchCur;

                        if (!FLexEscChar(true, out u))
                            break;
                        if (u > 0xFFFF || !LexCharUtils.IsIdent(ch = (char)u))
                        {
                            ReportError(ichErr, _cursor.IchCur, ErrId.BadChar, LexCharUtils.GetUniEscape(u));
                            break;
                        }
                        fVerbatim = true;
                    }
                    else
                    {
                        if (!LexCharUtils.IsIdent(ChCur))
                            break;
                        ch = ChCur;
                        ChNext();
                    }
                    Contracts.Assert(LexCharUtils.IsIdent(ch));
                    if (!LexCharUtils.IsFormat(ch))
                        _sb.Append(ch);
                }

                if (_sb.Length == 0)
                    return null;

                return _lex._pool.Add(_sb);
            }

            /// <summary>
            /// Lex a comment.
            /// </summary>
            private Token LexComment()
            {
                Contracts.Assert(ChCur == '/');
                int ichErr = _cursor.IchCur;

                switch (ChPeek(1))
                {
                    default:
                        return LexPunc();
                    case '/':
                        // Single line comment.
                        ChNext();
                        _sb.Length = 0;
                        _sb.Append("//");
                        for (; ; )
                        {
                            if (LexCharUtils.IsLineTerm(ChNext()) || Eof)
                                return new CommentToken(GetSpan(), _sb.ToString(), 0);
                            _sb.Append(ChCur);
                        }
                    case '*':
                        /* block comment */
                        ChNext();
                        _sb.Length = 0;
                        _sb.Append("/*");
                        ChNext();
                        int lines = 0;
                        for (; ; )
                        {
                            if (Eof)
                            {
                                ReportError(ichErr, _cursor.IchCur, ErrId.UnterminatedComment);
                                break;
                            }
                            char ch = ChCur;
                            if (LexCharUtils.IsLineTerm(ch))
                            {
                                ch = LexLineTerm(_sb);
                                lines++;
                            }
                            else
                                ChNext();
                            _sb.Append(ch);
                            if (ch == '*' && ChCur == '/')
                            {
                                _sb.Append('/');
                                ChNext();
                                break;
                            }
                        }
                        // We support comment keywords.
                        KeyWordTable.KeyWordKind kind;
                        NormStr nstr = _lex._pool.Add(_sb);
                        if (_lex._kwt.IsKeyWord(nstr, out kind))
                            return KeyToken.CreateKeyWord(GetSpan(), nstr.ToString(), kind.Kind, kind.IsContextKeyWord);
                        return new CommentToken(GetSpan(), _sb.ToString(), lines);
                }
            }

            /// <summary>
            /// Lex a sequence of spacing characters.
            /// Always returns null.
            /// </summary>
            private Token LexSpace()
            {
                Contracts.Assert(LexCharUtils.StartKind(ChCur) == LexStartKind.Space);
                while (LexCharUtils.IsSpace(ChNext()))
                    ;
                return null;
            }

            /// <summary>
            /// Lex a line termination character. Transforms CRLF into a single LF.
            /// Updates the line mapping. When this "drops" a character and sb is not
            /// null, it adds the character to sb. It does NOT add the returned character
            /// to the sb.
            /// </summary>
            private char LexLineTerm(StringBuilder sb = null)
            {
                Contracts.Assert(LexCharUtils.StartKind(ChCur) == LexStartKind.LineTerm);
                int ichMin = _cursor.IchCur;
                if (ChCur == '\xD' && ChPeek(1) == '\xA')
                {
                    if (sb != null)
                        sb.Append(ChCur);
                    ChNext();
                }
                char ch = ChCur;
                ChNext();

                if (_ichMinTok == ichMin)
                {
                    // Not nested.
                    _queue.Enqueue(new NewLineToken(GetSpan(), false));
                }
                else
                {
                    // Is nested.
                    _queue.Enqueue(new NewLineToken(GetTextSpan(ichMin, _cursor.IchCur), true));
                }
                _fLineStart = true;
                return ch;
            }

            private Token LexPreProc()
            {
                // We don't currently support pre-processing.
                return LexError();
            }

            /// <summary>
            /// Skip over an error character. Always returns null.
            /// REVIEW: Should we skip over multiple?
            /// </summary>
            private Token LexError()
            {
                _sb.Length = 0;
                do
                {
                    _sb.AppendFormat("{0}({1})", ChCur, LexCharUtils.GetUniEscape(ChCur));
                } while (LexCharUtils.StartKind(ChNext()) == LexStartKind.None && !Eof);
                return new ErrorToken(GetSpan(), ErrId.BadChar, _sb.ToString());
            }
        }
    }
}
