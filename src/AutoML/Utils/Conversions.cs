// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Auto
{
    using BL = Boolean;
    using R4 = Single;
    using TX = ReadOnlyMemory<char>;
    using U1 = Byte;
    using U8 = UInt64;

    internal static class Conversions
    {
        /// <summary>
        /// This produces zero for empty. It returns false if the text is not parsable.
        /// On failure, it sets dst to the NA value.
        /// </summary>
        public static bool TryParse(in TX src, out R4 dst)
        {
            var span = src.Span;
            if (float.TryParse(span.ToString(), out dst))
            {
                return true;
            }
            dst = R4.NaN;
            return IsStdMissing(ref span);
        }

        /// <summary>
        /// Return true if the span contains a standard text representation of NA
        /// other than the standard TX missing representation - callers should
        /// have already dealt with that case and the case of empty.
        /// The standard representations are any casing of:
        ///    ?  NaN  NA  N/A
        /// </summary>
        private static bool IsStdMissing(ref ReadOnlySpan<char> span)
        {
            char ch;
            switch (span.Length)
            {
                default:
                    return false;

                case 1:
                    if (span[0] == '?')
                        return true;
                    return false;
                case 2:
                    if ((ch = span[0]) != 'N' && ch != 'n')
                        return false;
                    if ((ch = span[1]) != 'A' && ch != 'a')
                        return false;
                    return true;
                case 3:
                    if ((ch = span[0]) != 'N' && ch != 'n')
                        return false;
                    if ((ch = span[1]) == '/')
                    {
                        // Check for N/A.
                        if ((ch = span[2]) != 'A' && ch != 'a')
                            return false;
                    }
                    else
                    {
                        // Check for NaN.
                        if (ch != 'a' && ch != 'A')
                            return false;
                        if ((ch = span[2]) != 'N' && ch != 'n')
                            return false;
                    }
                    return true;
            }
        }

        /// <summary>
        /// Try parsing a TX to a BL. This returns false for NA (span.IsMissing).
        /// Otherwise, it trims the span, then succeeds on all casings of the strings:
        /// * false, f, no, n, 0, -1, - => false
        /// * true, t, yes, y, 1, +1, + => true
        /// Empty string (but not missing string) succeeds and maps to false.
        /// </summary>
        public static bool TryParse(in TX src, out BL dst)
        {
            var span = src.Span;

            char ch;
            switch (src.Length)
            {
                case 0:
                    // Empty succeeds and maps to false.
                    dst = false;
                    return true;

                case 1:
                    switch (span[0])
                    {
                        case 'T':
                        case 't':
                        case 'Y':
                        case 'y':
                        case '1':
                        case '+':
                            dst = true;
                            return true;
                        case 'F':
                        case 'f':
                        case 'N':
                        case 'n':
                        case '0':
                        case '-':
                            dst = false;
                            return true;
                    }
                    break;

                case 2:
                    switch (span[0])
                    {
                        case 'N':
                        case 'n':
                            if ((ch = span[1]) != 'O' && ch != 'o')
                                break;
                            dst = false;
                            return true;
                        case '+':
                            if ((ch = span[1]) != '1')
                                break;
                            dst = true;
                            return true;
                        case '-':
                            if ((ch = span[1]) != '1')
                                break;
                            dst = false;
                            return true;
                    }
                    break;

                case 3:
                    switch (span[0])
                    {
                        case 'Y':
                        case 'y':
                            if ((ch = span[1]) != 'E' && ch != 'e')
                                break;
                            if ((ch = span[2]) != 'S' && ch != 's')
                                break;
                            dst = true;
                            return true;
                    }
                    break;

                case 4:
                    switch (span[0])
                    {
                        case 'T':
                        case 't':
                            if ((ch = span[1]) != 'R' && ch != 'r')
                                break;
                            if ((ch = span[2]) != 'U' && ch != 'u')
                                break;
                            if ((ch = span[3]) != 'E' && ch != 'e')
                                break;
                            dst = true;
                            return true;
                    }
                    break;

                case 5:
                    switch (span[0])
                    {
                        case 'F':
                        case 'f':
                            if ((ch = span[1]) != 'A' && ch != 'a')
                                break;
                            if ((ch = span[2]) != 'L' && ch != 'l')
                                break;
                            if ((ch = span[3]) != 'S' && ch != 's')
                                break;
                            if ((ch = span[4]) != 'E' && ch != 'e')
                                break;
                            dst = false;
                            return true;
                    }
                    break;
            }

            dst = false;
            return false;
        }
    }
}
