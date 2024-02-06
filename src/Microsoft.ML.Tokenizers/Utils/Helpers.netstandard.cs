// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Tokenizers
{
    internal static class Helpers
    {
        public static byte[] FromBase64String(string base64String, int offset, int length) => Convert.FromBase64String(base64String.Substring(offset, length));

        // Not support signed number
        internal static bool TryParseInt32(string s, int offset, out int result)
        {
            result = 0;
            if ((uint)offset >= s.Length)
            {
                return false;
            }

            for (int i = offset; i < s.Length; i++)
            {
                if ((uint)(s[i] - '0') > ('9' - '0'))
                {
                    return false;
                }

                result = result * 10 + (s[i] - '0');
            }

            return true;
        }
    }
}

