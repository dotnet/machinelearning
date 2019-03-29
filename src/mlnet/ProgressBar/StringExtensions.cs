// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.CLI.ShellProgressBar
{
    internal static class StringExtensions
    {
        public static string Excerpt(string phrase, int length = 60)
        {
            if (string.IsNullOrEmpty(phrase) || phrase.Length < length)
                return phrase;
            return phrase.Substring(0, length - 3) + "...";
        }
    }
}
