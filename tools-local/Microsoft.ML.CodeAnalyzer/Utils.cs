// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.CodeAnalyzer
{
    internal static class Utils
    {
        /// <summary>
        /// Checks whether a name is properly <c>camelCased</c> or <c>PascalCased</c>.
        /// Also disallows things like <c>HTMLStream<c> while preferring <c>IOStream</c>.
        /// </summary>
        /// <param name="name">The symbol name to analyze</param>
        /// <param name="min">The position in the name to start</param>
        /// <param name="upper">Whether it should be <c>PascalCased</c></param>
        /// <returns>Whether this name is good</returns>
        public static bool NameIsGood(string name, int min, bool upper)
        {
            // C# naming guidelines say, any initialism greater than two characters should not
            // be all upper cased. So: _readIOStream is good, and _readHttpStream is good. You
            // could imagine having two two-letter initialisms, like: _readIOUI, where you use
            // two two character initialism, but I'm going to suppose that never happens since
            // if someone is doing that, that's pretty odd. The upshot is: 
            const int maxConsecutive = 3;
            // Force the first after the _ to be lower case.
            int consecutive = upper ? 0 : maxConsecutive;
            // Specific to numbers. You could imagine counterexamples, like, say, d3js. Should
            // we be even more strict, and say that the numbers should only appear potentially
            // in suffixes?
            for (int i = min; i < name.Length; ++i)
            {
                char c = name[i];
                // Only letters and digits.
                if (!char.IsLetterOrDigit(c))
                    return false;
                if (char.IsDigit(c))
                {
                    // Consider digits as being effectively upper case letters, where they appear.
                    upper = false;
                    consecutive = 0;
                    continue;
                }
                if (char.IsUpper(c))
                {
                    upper = false;
                    if (++consecutive > maxConsecutive)
                        return false;
                    continue;
                }
                if (upper)
                    return false;
                consecutive = 0;
            }
            // Don't allow maxConsecutive on the end. So: IOStream is fine, but IOS is not.
            return consecutive < maxConsecutive;
        }
    }
}
