// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Indicates which lex routine should be called when this character is the first
    /// character of a token. Each character is associated with exactly one of these values.
    /// Some associations may at first be surprising. For example, dot (.) uses NumLit and
    /// slash (/) uses Comment.
    /// </summary>
    internal enum LexStartKind : ushort
    {
        None,
        Punc,
        Ident,
        NumLit,
        StrLit,
        Verbatim,
        Comment,
        PreProc,
        Space,
        LineTerm,
    }

    /// <summary>
    /// Encapsulates information needed to map characters to tokens.
    /// </summary>
    internal static class LexCharUtils
    {
        /// <summary>
        /// Bit masks of the UnicodeCategory enum. A couple extra values are defined
        /// for convenience for the C# lexical grammar.
        /// </summary>
        [Flags]
        private enum UniCatFlags : uint
        {
            ConnectorPunctuation = 1 << UnicodeCategory.ConnectorPunctuation, // Pc
            DecimalDigitNumber = 1 << UnicodeCategory.DecimalDigitNumber, // Nd
            Format = 1 << UnicodeCategory.Format, // Cf
            LetterNumber = 1 << UnicodeCategory.LetterNumber, // Nl
            LowercaseLetter = 1 << UnicodeCategory.LowercaseLetter, // Ll
            ModifierLetter = 1 << UnicodeCategory.ModifierLetter, // Lm
            NonSpacingMark = 1 << UnicodeCategory.NonSpacingMark, // Mn
            OtherLetter = 1 << UnicodeCategory.OtherLetter, // Lo
            SpaceSeparator = 1 << UnicodeCategory.SpaceSeparator, // Zs
            SpacingCombiningMark = 1 << UnicodeCategory.SpacingCombiningMark, // Mc
            TitlecaseLetter = 1 << UnicodeCategory.TitlecaseLetter, // Lt
            UppercaseLetter = 1 << UnicodeCategory.UppercaseLetter, // Lu

            // Useful combinations.
            IdentStartChar = UppercaseLetter | LowercaseLetter | TitlecaseLetter |
              ModifierLetter | OtherLetter | LetterNumber,
            IdentPartChar = IdentStartChar | NonSpacingMark | SpacingCombiningMark |
              DecimalDigitNumber | ConnectorPunctuation | Format,
        }

        /// <summary>
        /// Indicates the different roles a character may have (as non-leading character). This is used for
        /// subsequent (not first) characters in a token. For example, digits all have the Ident flag set.
        /// </summary>
        [Flags]
        private enum LexCharKind : ushort
        {
            None = 0x0000,
            Punc = 0x0001,
            Ident = 0x0002,
            Digit = 0x0004,
            HexDigit = 0x0008,
            Space = 0x0010,
            LineTerm = 0x0020,
        };

        /// <summary>
        /// Information for each character. We have a table of these for all characters less than 0x80.
        /// </summary>
        private struct LexCharInfo
        {
            public readonly LexStartKind StartKind;
            public readonly LexCharKind CharKind;

            public LexCharInfo(LexStartKind sk, LexCharKind ck)
            {
                StartKind = sk;
                CharKind = ck;
            }

            public bool Is(LexCharKind kind)
            {
                return (CharKind & kind) != 0;
            }
        }

        // The mapping from character to CharInfo for characters less than 128.
        private static readonly LexCharInfo[] _rgchi;

        static LexCharUtils()
        {
            // Init the array of CharInfo's.
            _rgchi = new LexCharInfo[128];

            // a - f are Ident and HexDigit
            var info = new LexCharInfo(LexStartKind.Ident, LexCharKind.Ident | LexCharKind.HexDigit);
            for (char ch = 'a'; ch <= 'f'; ch++)
                _rgchi[ch] = info;
            for (char ch = 'A'; ch <= 'F'; ch++)
                _rgchi[ch] = info;

            // g - z are just Ident.
            info = new LexCharInfo(LexStartKind.Ident, LexCharKind.Ident);
            for (char ch = 'g'; ch <= 'z'; ch++)
                _rgchi[ch] = info;
            for (char ch = 'G'; ch <= 'Z'; ch++)
                _rgchi[ch] = info;
            _rgchi['_'] = info;

            // Digits are Digit | HexDigit | Ident.
            info = new LexCharInfo(LexStartKind.NumLit, LexCharKind.Digit | LexCharKind.HexDigit | LexCharKind.Ident);
            for (char ch = '0'; ch <= '9'; ch++)
                _rgchi[ch] = info;
            // Dot can start a numeric literal.
            _rgchi['.'] = new LexCharInfo(LexStartKind.NumLit, LexCharKind.Punc);

            // Space characters.
            info = new LexCharInfo(LexStartKind.Space, LexCharKind.Space);
            foreach (char ch in " \x09\x0B\x0C")
                _rgchi[ch] = info;

            // Line terminators.
            info = new LexCharInfo(LexStartKind.LineTerm, LexCharKind.LineTerm);
            _rgchi['\xA'] = info;
            _rgchi['\xD'] = info;

            // Special lead characters: literals, verbatim, comment, pre-processor.
            info = new LexCharInfo(LexStartKind.StrLit, LexCharKind.None);
            _rgchi['"'] = info;
            _rgchi['\''] = info;
            _rgchi['@'] = new LexCharInfo(LexStartKind.Verbatim, LexCharKind.None);
            _rgchi['/'] = new LexCharInfo(LexStartKind.Comment, LexCharKind.Punc);
            _rgchi['#'] = new LexCharInfo(LexStartKind.PreProc, LexCharKind.None);

            // Punctuators. Some that you might think belong here (like . and /) are handled
            // by other LexStartKinds.
            info = new LexCharInfo(LexStartKind.Punc, LexCharKind.Punc);
            foreach (char ch in "!%&()*+,-:;<=>?[]^{|}~")
                _rgchi[ch] = info;
        }

        private static UniCatFlags GetCatFlags(char ch)
        {
            return (UniCatFlags)(1u << (int)CharUnicodeInfo.GetUnicodeCategory(ch));
        }

        /// <summary>
        /// Returns the lexical character type of the given character.
        /// </summary>
        public static LexStartKind StartKind(char ch)
        {
            if (ch < _rgchi.Length)
                return _rgchi[ch].StartKind;

            UniCatFlags ucf = GetCatFlags(ch);
            if ((ucf & UniCatFlags.IdentStartChar) != 0)
                return LexStartKind.Ident;
            if ((ucf & UniCatFlags.SpaceSeparator) != 0)
                return LexStartKind.Space;
            return LexStartKind.None;
        }

        public static bool IsPunc(char ch)
        {
            return ch < _rgchi.Length && _rgchi[ch].Is(LexCharKind.Punc);
        }
        public static bool IsDigit(char ch)
        {
            return ch < _rgchi.Length && _rgchi[ch].Is(LexCharKind.Digit);
        }
        public static bool IsHexDigit(char ch)
        {
            return ch < _rgchi.Length && _rgchi[ch].Is(LexCharKind.HexDigit);
        }
        public static bool IsIdentStart(char ch)
        {
            if (ch < _rgchi.Length)
                return _rgchi[ch].Is(LexCharKind.Ident) && !_rgchi[ch].Is(LexCharKind.Digit);
            return (GetCatFlags(ch) & UniCatFlags.IdentPartChar) != 0;
        }
        public static bool IsIdent(char ch)
        {
            if (ch < _rgchi.Length)
                return _rgchi[ch].Is(LexCharKind.Ident);
            return (GetCatFlags(ch) & UniCatFlags.IdentPartChar) != 0;
        }
        public static bool IsFormat(char ch)
        {
            return ch >= _rgchi.Length && CharUnicodeInfo.GetUnicodeCategory(ch) == UnicodeCategory.Format;
        }
        public static bool IsSpace(char ch)
        {
            if (ch < _rgchi.Length)
                return _rgchi[ch].Is(LexCharKind.Space);
            return CharUnicodeInfo.GetUnicodeCategory(ch) == UnicodeCategory.SpaceSeparator;
        }
        public static bool IsLineTerm(char ch)
        {
            if (ch < _rgchi.Length)
                return _rgchi[ch].Is(LexCharKind.LineTerm);
            return ch == '\u0085' || ch == '\u2028' || ch == '\u2029';
        }

        public static int GetDecVal(char ch)
        {
            Contracts.Assert('0' <= ch && ch <= '9');
            return ch - '0';
        }

        public static int GetHexVal(char ch)
        {
            Contracts.Assert(IsHexDigit(ch));
            if (ch >= 'a')
            {
                Contracts.Assert(ch <= 'f');
                return ch - ('a' - 10);
            }
            if (ch >= 'A')
            {
                Contracts.Assert(ch <= 'F');
                return ch - ('A' - 10);
            }
            Contracts.Assert('0' <= ch && ch <= '9');
            return ch - '0';
        }

        /// <summary>
        /// Convert the given uint to a unicode escape.
        /// Note that the uint contains raw hex - not a surrogate pair.
        /// </summary>
        public static string GetUniEscape(uint u)
        {
            if (u < 0x00010000)
                return string.Format(@"\u{0:X4}", u);
            return string.Format(@"\U{0:X8}", u);
        }
    }
}
