// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Concurrent;
using Microsoft.CodeAnalysis;

namespace Microsoft.ML.InternalCodeAnalyzer
{
    internal static class IMethodSymbolExtensions
    {
        public static bool IsTestMethod(this IMethodSymbol method, ConcurrentDictionary<INamedTypeSymbol, bool> knownTestAttributes, INamedTypeSymbol factAttribute)
        {
            foreach (var attribute in method.GetAttributes())
            {
                if (attribute.AttributeClass.IsTestAttribute(knownTestAttributes, factAttribute))
                    return true;
            }

            return false;
        }
    }
}
