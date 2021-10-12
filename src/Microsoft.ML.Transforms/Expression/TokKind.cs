// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    internal enum TokKind
    {
        None,

        // Miscellaneous
        Eof,
        Error,
        ErrorInline,

        // Literals
        IntLit,
        FltLit,
        DblLit,
        CharLit,
        StrLit,

        // Noise
        Comment,
        NewLine,

        // Punctuators

        Add, // +
        AddAdd,
        AddEqu,
        Sub, // -
        SubSub,
        SubEqu,
        SubGrt,
        Mul, // *
        MulEqu,
        Div, // /
        DivEqu,
        Per, // %
        PerEqu,

        Car, // ^
        CarEqu,
        Amp,
        AmpAmp, // &&
        AmpEqu,
        Bar,
        BarBar, // ||
        BarEqu,

        Til,
        Bng, // !
        BngEqu, // !=

        Equ, // =
        EquEqu, // ==
        EquGrt, // =>
        Lss, // <
        LssLss,
        LssEqu, // <=
        LssGrt, // <>
        LssLssEqu,
        Grt, // >
        GrtGrt,
        GrtEqu, // >=
        GrtGrtEqu,

        Que, // ?
        QueQue, // ??

        Dot, // .
        Comma, // ,
        Colon, // :
        ColonColon,
        Semi, // ;

        OpenCurly,
        OpenParen, // (
        OpenSquare,

        CloseCurly,
        CloseParen, // )
        CloseSquare,

        // Words - identifier and key words
        Ident,

        False,
        True,
        Not,
        And,
        Or,

        With,
    }

    public enum ErrId
    {
        None,
        BadChar,
        BadEscape,
        CharConstEmpty,
        CharConstTooLong,
        FloatOverflow,
        IntOverflow,
        NewlineInConst,
        UnterminatedComment,
        UnterminatedString,
        VerbatimLiteralExpected,

        // Pre-processing related errors
        BadPreProcPos,
        EndOfPreLineExpected,
        IdentExpected,
        PreProcDirExpected,
        UniEscInPreProc,
    }

    internal static class ErrIdExt
    {
        public static string GetMsgFmt(this ErrId eid, out int carg)
        {
            carg = 0;
            switch (eid)
            {
                case ErrId.BadChar:
                    carg = 1;
                    return "Unexpected character '{0}'";
                case ErrId.BadEscape:
                    return "Unrecognized escape sequence";
                case ErrId.CharConstEmpty:
                    return "Empty character literal";
                case ErrId.CharConstTooLong:
                    return "Too many characters in character literal";
                case ErrId.FloatOverflow:
                    carg = 1;
                    return "Floating-point constant is outside the range of type '{0}'";
                case ErrId.IntOverflow:
                    return "Integral constant is too large";
                case ErrId.NewlineInConst:
                    return "Newline in constant";
                case ErrId.UnterminatedComment:
                    return "End-of-input found in comment";
                case ErrId.UnterminatedString:
                    return "End-of-input found in string or character literal";
                case ErrId.VerbatimLiteralExpected:
                    return @"Keyword, identifier, or string expected after verbatim specifier: @";

                case ErrId.BadPreProcPos:
                    return "Preprocessor directives must be on a new line";
                case ErrId.EndOfPreLineExpected:
                    return "Single-line comment or end-of-line expected";
                case ErrId.IdentExpected:
                    return "Identifier expected";
                case ErrId.PreProcDirExpected:
                    return "Preprocessor directive expected";
                case ErrId.UniEscInPreProc:
                    return "Unicode escapes not permitted in preprocessor directives";

                default:
                    Contracts.Assert(false, "Unknown error id: " + eid);
                    return "Unknown error";
            }
        }

        public static string GetMsg(this ErrId eid, params object[] args)
        {
            int carg;
            string fmt = eid.GetMsgFmt(out carg);
            Contracts.Assert(carg == Utils.Size(args));
            if (carg == 0)
                return fmt;
            string msg = string.Format(fmt, args);
            return msg;
        }
    }
}
