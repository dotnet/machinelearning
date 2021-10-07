// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.IO;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    internal struct TextSpan
    {
        public readonly int Min;
        public readonly int Lim;

        public TextSpan(int ichMin, int ichLim)
        {
            Contracts.Assert(0 <= ichMin && ichMin <= ichLim);
            Min = ichMin;
            Lim = ichLim;
        }

        public override string ToString()
        {
            return string.Format(CultureInfo.InvariantCulture, "({0},{1})", Min, Lim);
        }
    }

    internal abstract class Token
    {
        public readonly TokKind Kind;
        public readonly TokKind KindContext;
        public readonly TextSpan Span;

        protected Token(TextSpan span, TokKind tid)
        {
            Span = span;
            Kind = tid;
            KindContext = tid;
        }

        protected Token(TextSpan span, TokKind tid, TokKind tidContext)
        {
            // Currently the only contextual variability is that an identifier might double as a keyword.
            Contracts.Assert(tidContext == tid || tid == TokKind.Ident);
            Span = span;
            Kind = tid;
            KindContext = tidContext;
        }

        public T As<T>() where T : Token
        {
            Contracts.Assert(this is T);
            return (T)this;
        }

        public override string ToString()
        {
            return Kind.ToString();
        }
    }

    // Keyword/punctuation token
    internal sealed class KeyToken : Token
    {
        public static KeyToken Create(TextSpan span, TokKind tid)
        {
            return new KeyToken(span, tid);
        }

        public static Token CreateKeyWord(TextSpan span, string str, TokKind tid, bool isContextKeyWord)
        {
            if (isContextKeyWord)
                return new IdentToken(span, str, tid);
            return new KeyToken(span, tid);
        }

        private KeyToken(TextSpan span, TokKind tid)
            : base(span, tid)
        {
        }
    }

    internal sealed class IdentToken : Token
    {
        public readonly string Value;
        public IdentToken(TextSpan span, string val) : base(span, TokKind.Ident) { Value = val; }
        public IdentToken(TextSpan span, string val, TokKind tidContext) : base(span, TokKind.Ident, tidContext) { Value = val; }
        public override string ToString()
        {
            if (KindContext != TokKind.Ident)
                return "ContextKeyword: " + Value;
            return "Ident: " + Value;
        }
    }

    [Flags]
    internal enum IntLitKind : byte
    {
        None = 0x00,
        Uns = 0x01,
        Lng = 0x02,
        Hex = 0x04,

        UnsLng = Uns | Lng,
    }

    internal abstract class NumLitToken : Token
    {
        protected NumLitToken(TextSpan span, TokKind tid)
            : base(span, tid)
        {
        }
    }

    internal sealed class IntLitToken : NumLitToken
    {
        public readonly ulong Value;
        public readonly IntLitKind IntKind; // The kind specified by suffixes.

        public IntLitToken(TextSpan span, ulong val, IntLitKind ilk)
            : base(span, TokKind.IntLit)
        {
            Value = val;
            IntKind = ilk;
        }

        public override string ToString()
        {
            string suff1 = (IntKind & IntLitKind.Uns) != 0 ? "U" : "";
            string suff2 = (IntKind & IntLitKind.Lng) != 0 ? "L" : "";
            string fmt;

            if (IsHex)
                fmt = Value < 0x0000000100000000 ? "0x{0:X8}{1}{2}" : "0x{0:X16}{1}{2}";
            else
                fmt = "{0}{1}{2}";
            return string.Format(fmt, Value, suff1, suff2);
        }

        public bool IsHex { get { return (IntKind & IntLitKind.Hex) != 0; } }
    }

    internal sealed class FltLitToken : NumLitToken
    {
        public readonly float Value;
        public FltLitToken(TextSpan span, float val) : base(span, TokKind.FltLit) { Value = val; }
        public override string ToString()
        {
            if (float.IsPositiveInfinity(Value))
                return "1e1000f";
            return Value.ToString("R") + "f";
        }
    }

    internal sealed class DblLitToken : NumLitToken
    {
        public readonly double Value;
        // Whether this literal has an explicit d/D suffix.
        public readonly bool HasSuffix;

        public DblLitToken(TextSpan span, double val, bool hasSuf)
            : base(span, TokKind.DblLit)
        {
            Value = val;
            HasSuffix = hasSuf;
        }

        public override string ToString()
        {
            string res;
            if (double.IsPositiveInfinity(Value))
                res = "1e1000";
            else
                res = Value.ToString("R");
            if (HasSuffix)
                res += "d";
            else if (AllDigits(res))
                res += ".0";
            return res;
        }

        private static bool AllDigits(string s)
        {
            Contracts.AssertNonEmpty(s);
            for (int i = 0; i < s.Length; i++)
            {
                if ((uint)(s[i] - '0') > 9)
                    return false;
            }
            return true;
        }
    }

    internal sealed class CharLitToken : Token
    {
        public readonly char Value;
        public CharLitToken(TextSpan span, char val) : base(span, TokKind.CharLit) { Value = val; }
        public override string ToString()
        {
            if (Value < ' ' || Value >= 0x7F)
                return string.Format(@"'\u{0:X4}'", (int)Value);
            if (Value == '\'')
                return @"'\''";
            return string.Format("'{0}'", Value);
        }
    }

    internal sealed class StrLitToken : Token
    {
        public readonly string Value;
        public StrLitToken(TextSpan span, string val) : base(span, TokKind.StrLit) { Value = val; }
        public override string ToString()
        {
            StringWriter wrt = new StringWriter();
            Write(wrt, Value);
            return wrt.ToString();
        }

        public static void Write(TextWriter wrt, string str)
        {
            Contracts.AssertValue(str);

            wrt.Write('"');
            int ich = 0;
            for (; ich < str.Length; ich++)
            {
                char ch = str[ich];
                if (ch < ' ' || ch >= 0x7F)
                    wrt.Write(@"\u{0:X4}", (int)ch);
                else
                {
                    if (ch == '\\' || ch == '"')
                        wrt.Write('\\');
                    wrt.Write(ch);
                }
            }
            wrt.Write('"');
        }
    }

    internal sealed class CommentToken : Token
    {
        public readonly string Text;
        public readonly int NewLineCount;
        public CommentToken(TextSpan span, string val, int lines)
            : base(span, TokKind.Comment)
        {
            Text = val;
            NewLineCount = lines;
        }
        public override string ToString() { return "Comment: " + Text; }
    }

    internal sealed class NewLineToken : Token
    {
        public readonly bool Nested;
        public NewLineToken(TextSpan span, bool fNested)
            : base(span, TokKind.NewLine)
        {
            Nested = fNested;
        }
        public override string ToString() { return Nested ? "NewLine<Nested>" : "NewLine"; }
    }

    internal sealed class ErrorToken : Token
    {
        public readonly ErrId Id;
        public readonly object[] Args;

        public ErrorToken(TextSpan span, ErrId eid, params object[] args)
            : base(span, TokKind.Error)
        {
            Id = eid;
            Args = args;
        }

        public override string ToString()
        {
            return Id.GetMsg(Args);
        }
    }

    internal sealed class EofToken : Token
    {
        public EofToken(TextSpan span)
            : base(span, TokKind.Eof)
        {
        }
    }
}
