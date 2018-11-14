// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;

namespace Microsoft.ML.Runtime.CommandLine
{
    [BestFriend]
    internal sealed class CmdLexer
    {
        private CharCursor _curs;

        // Whether \ has a special significance.
        private readonly bool _escapes;

        private bool _error;

        public bool Error { get { return _error; } }

        public CmdLexer(CharCursor curs, bool escapes = true)
        {
            _curs = curs;
            _escapes = escapes;
        }

        /// <summary>
        /// Consume the characters of the next token and append them to the string builder.
        /// </summary>
        public void GetToken(StringBuilder bldr)
        {
            int ichDst = bldr.Length;

            // Skip spaces, comments, etc.
            SkipWhiteSpace();

            while (!_curs.Eof)
            {
                char ch = _curs.ChCur;
                switch (ch)
                {
                case '{':
                    if (bldr.Length == ichDst)
                        GatherCurlyGroup(bldr);
                    return;

                case '}':
                    if (bldr.Length == ichDst)
                    {
                        bldr.Append(ch);
                        _curs.ChNext();
                        // Naked } is an error.
                        _error = true;
                    }
                    return;

                case '=':
                    if (bldr.Length == ichDst)
                    {
                        bldr.Append(ch);
                        _curs.ChNext();
                    }
                    return;

                case '\\':
                    if (_escapes)
                    {
                        GatherSlash(bldr, true);
                        continue;
                    }
                    break;

                case '"':
                    GatherString(bldr, true);
                    continue;

                case '#':
                    // Since we skipped comments, we should only get here if we've collected something.
                    Contracts.Assert(bldr.Length > ichDst);
                    return;

                default:
                    if (char.IsWhiteSpace(ch))
                        return;
                    break;
                }

                bldr.Append(ch);
                _curs.ChNext();
            }
        }

        public void SkipWhiteSpace()
        {
            // Skip comments and whitespace
            for (;;)
            {
                if (char.IsWhiteSpace(_curs.ChCur))
                {
                    while (char.IsWhiteSpace(_curs.ChNext()))
                    {
                    }
                }

                if (_curs.ChCur != '#')
                    return;

                char ch;
                while ((ch = _curs.ChNext()) != '\x0D' && ch != '\x0A' && !_curs.Eof)
                {
                }
            }
        }

        // This collects a curly group, including the curlies and associated escaping.
        // Compare to GatherCurlyContents, that does not keep the curlies or associated escaping.
        private void GatherCurlyGroup(StringBuilder bldr)
        {
            Contracts.Assert(_curs.ChCur == '{');

            int count = 0;
            while (!_curs.Eof)
            {
                char ch = _curs.ChCur;
                switch (ch)
                {
                case '{':
                    count++;
                    break;

                case '}':
                    Contracts.Assert(count > 0);
                    bldr.Append(ch);
                    _curs.ChNext();
                    if (--count <= 0)
                        return;
                    continue;

                case '"':
                    GatherString(bldr, false);
                    continue;

                case '\\':
                    if (_escapes)
                    {
                        GatherSlash(bldr, false);
                        continue;
                    }
                    break;

                default:
                    break;
                }

                bldr.Append(ch);
                _curs.ChNext();
            }

            // Hitting eof is an error.
            _error = true;
        }

        // This collects the contents of a curly group, shedding the curlies and associated escaping.
        // Compare to GatherCurlyGroup, that keeps the curlies and associated escaping.
        internal void GatherCurlyContents(StringBuilder bldr)
        {
            Contracts.Assert(_curs.ChCur == '{');

            _curs.ChNext();
            int count = 0;
            while (!_curs.Eof)
            {
                char ch = _curs.ChCur;
                switch (ch)
                {
                case '{':
                    count++;
                    break;

                case '}':
                    Contracts.Assert(count >= 0);
                    _curs.ChNext();
                    if (--count < 0)
                        return;
                    bldr.Append(ch);
                    continue;

                case '"':
                    GatherString(bldr, false);
                    continue;

                case '\\':
                    if (_escapes)
                    {
                        GatherSlash(bldr, count == 0);
                        continue;
                    }
                    break;

                default:
                    break;
                }

                bldr.Append(ch);
                _curs.ChNext();
            }

            // Hitting eof is an error.
            _error = true;
        }

        private void GatherSlash(StringBuilder bldr, bool reduce)
        {
            Contracts.Assert(_curs.ChCur == '\\');
            Contracts.Assert(_escapes);

            // Count the slashes.
            int cv = 1;
            while (_curs.ChNext() == '\\')
                cv++;

            // This assumes that slash is escaped iff it preceeds a special character
            switch (_curs.ChCur)
            {
            case '"':
            case '{':
            case '}':
            case '#':
                // The escape case. Only keep half the slashes if we're reducing.
                bldr.Append('\\', reduce ? cv / 2 : cv);

                // If there are an odd number of slashes, keep the next char.
                if ((cv & 1) != 0)
                {
                    bldr.Append(_curs.ChCur);
                    _curs.ChNext();
                }
                break;

            default:
                bldr.Append('\\', cv);
                break;
            }
        }

        private void GatherString(StringBuilder bldr, bool reduce)
        {
            Contracts.Assert(_curs.ChCur == '"');

            if (!reduce)
                bldr.Append(_curs.ChCur);
            _curs.ChNext();
            while (!_curs.Eof)
            {
                char ch = _curs.ChCur;
                switch (ch)
                {
                case '"':
                    if (!reduce)
                        bldr.Append(_curs.ChCur);
                    _curs.ChNext();
                    return;

                case '\\':
                    if (_escapes)
                    {
                        GatherSlash(bldr, reduce);
                        continue;
                    }
                    break;

                case '\x0D':
                case '\x0A':
                    // Hitting end of line is an error.
                    _error = true;
                    return;
                }

                bldr.Append(ch);
                _curs.ChNext();
            }

            // Hitting eof is an error.
            _error = true;
        }

        public static string UnquoteValue(string str)
        {
            if (!str.StartsWith("{") || !str.EndsWith("}"))
                return str;

            CharCursor curs = new CharCursor(str);
            CmdLexer lex = new CmdLexer(curs);

            // Gather the curly group contents and make sure it consumes everything.
            StringBuilder sb = new StringBuilder();
            lex.GatherCurlyContents(sb);
            if (lex._error || !curs.Eof)
                return str;

            return sb.ToString();
        }
    }

    public sealed class CmdQuoter
    {
        private readonly string _str;
        private StringBuilder _sb;
        private int _ich;

        private CmdQuoter(string str, StringBuilder sb)
        {
            _str = str;
            _sb = sb;
        }

        public static bool NeedsQuoting(string str)
        {
            Contracts.AssertValueOrNull(str);
            if (string.IsNullOrEmpty(str))
                return true;
            return NeedsQuoting(new StringBuilder(str), 0);
        }

        public static bool NeedsQuoting(StringBuilder sb, int ich)
        {
            Contracts.AssertValue(sb);
            Contracts.Assert(0 <= ich && ich <= sb.Length);

            if (ich >= sb.Length)
                return true;

            if (sb[ich] == '{')
                return true;

            // See if we need to quote. If lexing produces a single token with the exact
            // same value, then we don't need to.
            int ichLim = sb.Length;
            int cch = ichLim - ich;
            var curs = new CharCursor(sb.ToString(ich, cch));
            var lex = new CmdLexer(curs);

            lex.GetToken(sb);
            Contracts.Assert(curs.IchCur > 0 || lex.Error);

            try
            {
                if (!lex.Error && curs.Eof && sb.Length == ichLim + cch)
                {
                    // See if the characters match.
                    for (int ichSrc = ich; ; ichSrc++)
                    {
                        if (ichSrc >= ichLim)
                            return false;
                        if (sb[ichSrc] != sb[ichSrc + cch])
                            break;
                    }
                }
                return true;
            }
            finally
            {
                sb.Length = ich + cch;
            }
        }

        /// <summary>
        /// Returns true if it quoted.
        /// </summary>
        public static bool QuoteValue(string str, StringBuilder sb, bool force = false)
        {
            int ich = sb.Length;
            bool f = QuoteValueCore(str, sb, force);

#if DEBUG // Verify the result.
            string v = sb.ToString(ich, sb.Length - ich);

            var curs = new CharCursor(v);
            Contracts.Assert(f == (force || curs.ChCur == '{'));

            var lex = new CmdLexer(curs);
            var res = new StringBuilder();

            // If it was quoted, gathering curly contents should get us the original. Otherwise,
            // the result should be equivalent to the original and should be a single token.
            if (f)
            {
                Contracts.Assert(v.StartsWith("{") && v.EndsWith("}"));
                lex.GatherCurlyContents(res);
            }
            else
                lex.GetToken(res);

            Contracts.Assert(!lex.Error);
            Contracts.Assert(curs.Eof);
            Contracts.Assert(str == res.ToString());
#endif

            return f;
        }

        private static bool QuoteValueCore(string str, StringBuilder sb, bool force)
        {
            if (string.IsNullOrEmpty(str))
            {
                sb.Append("{}");
                return true;
            }

            int ich = sb.Length;
            if (!force && TryNoQuoting(str, sb))
                return false;
            Contracts.Assert(ich == sb.Length);

            // We need to quote. See if we can just slap curlies around it.
            if (TryNaiveQuoting(str, sb))
                return true;
            Contracts.Assert(ich == sb.Length);

            var quoter = new CmdQuoter(str, sb);
            quoter.QuoteValueCore();
            return true;
        }

        // Determines whether str needs quoting. If not, appends the string to sb and returns try.
        // If so, sb's contents are preserved and returns false.
        private static bool TryNoQuoting(string str, StringBuilder sb)
        {
            Contracts.AssertNonEmpty(str);

            if (str[0] == '{')
                return false;

            int ichDst = sb.Length;

            // See if we need to quote. If lexing produces a single token with the exact
            // same value, then we don't need to.
            var curs = new CharCursor(str);
            var lex = new CmdLexer(curs);

            lex.GetToken(sb);
            Contracts.Assert(curs.IchCur > 0 || lex.Error);

            if (!lex.Error && curs.Eof && sb.Length == ichDst + str.Length)
            {
                // See if the characters match.
                for (int ichSrc = 0; ; ichSrc++)
                {
                    if (ichSrc >= str.Length)
                        return true;
                    if (sb[ichDst + ichSrc] != str[ichSrc])
                        break;
                }
            }

            // Reset the string builder.
            sb.Length = ichDst;
            return false;
        }

        // Try to quote by just slapping curlies around the string. This will normally be sufficient
        // and produces a much more aesthetic result than escaping everything.
        private static bool TryNaiveQuoting(string str, StringBuilder sb)
        {
            Contracts.AssertNonEmpty(str);

            var curs = new CharCursor("{" + str + "}");
            var lex = new CmdLexer(curs);
            var res = new StringBuilder();

            lex.GatherCurlyContents(res);
            if (lex.Error || !curs.Eof || res.Length != str.Length || res.ToString() != str)
                return false;

            sb.Append("{");
            sb.Append(str);
            sb.Append("}");
            return true;
        }

        // Quote with escaping.
        private void QuoteValueCore()
        {
            _sb.Append('{');
            while (_ich < _str.Length)
            {
                char ch = _str[_ich++];
                switch (ch)
                {
                default:
                    _sb.Append(ch);
                    break;

                // REVIEW: We need a proper grammar for quoting/unquoting!
                case '{':
                case '}':
                case '"':
                case '#':
                    _sb.Append('\\');
                    _sb.Append(ch);
                    break;

                case '\\':
                    HandleSlash();
                    break;
                }
            }
            _sb.Append("}");
        }

        private void HandleSlash()
        {
            // ichMin is where the slashes start.
            int ichMin = _ich - 1;
            Contracts.Assert(ichMin >= 0 && _str[ichMin] == '\\');
            Contracts.Assert(ichMin == 0 || _str[ichMin - 1] != '\\');

            // Skip all consecutive slashes.
            while (_ich < _str.Length && _str[_ich] == '\\')
                _ich++;

            int count = _ich - ichMin;
            if (_ich >= _str.Length)
            {
                // Double the slashes.
                _sb.Append('\\', 2 * count);
                return;
            }

            switch (_str[_ich])
            {
            case '{':
            case '}':
            case '"':
            case '#':
                // Double the slashes, add one, and digest the special character.
                _sb.Append('\\', 2 * count + 1);
                _sb.Append(_str[_ich++]);
                break;

            default:
                // Just preserve the slashes.
                _sb.Append('\\', count);
                break;
            }
        }
    }
}
