// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Model
{
    internal static class ModelUtils
    {
        private static string ArgCase(string name)
        {
            if (string.IsNullOrEmpty(name))
                return name;
            if (!char.IsUpper(name[0]))
                return name;

            if (name.Length == 1)
                return name.ToLowerInvariant();
            if (!char.IsUpper(name[1]))
                return name.Substring(0, 1).ToLowerInvariant() + name.Substring(1);

            int firstNonUpper;
            for (firstNonUpper = 0; firstNonUpper < name.Length && char.IsUpper(name[firstNonUpper]); ++firstNonUpper)
                ;
            Contracts.Assert(1 < firstNonUpper && firstNonUpper <= name.Length);
            if (firstNonUpper == name.Length)
                return name.ToLowerInvariant();
            --firstNonUpper;
            return name.Substring(0, firstNonUpper).ToLowerInvariant() + name.Substring(firstNonUpper);
        }

        public static string CreateNameCore(string name, Func<string, bool> contains)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            Contracts.CheckValue(contains, nameof(contains));

            name = ArgCase(name);
            if (!contains(name))
                return name;
            int append = 0;
            while (contains(name + append))
                append++;
            return name + append;
        }
    }
}
