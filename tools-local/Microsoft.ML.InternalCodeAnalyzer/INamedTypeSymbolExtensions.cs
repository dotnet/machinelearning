﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Concurrent;
using System.Diagnostics;
using Microsoft.CodeAnalysis;

namespace Microsoft.ML.InternalCodeAnalyzer
{
    internal static class INamedTypeSymbolExtensions
    {
        public static bool IsTestAttribute(this INamedTypeSymbol attributeClass, ConcurrentDictionary<INamedTypeSymbol, bool> knownTestAttributes, INamedTypeSymbol factAttribute)
        {
            if (knownTestAttributes.TryGetValue(attributeClass, out var isTest))
                return isTest;

            return knownTestAttributes.GetOrAdd(attributeClass, ExtendsFactAttribute(attributeClass, factAttribute));
        }

        private static bool ExtendsFactAttribute(INamedTypeSymbol namedType, INamedTypeSymbol factAttribute)
        {
            Debug.Assert(factAttribute is object);
            for (var current = namedType; current is object; current = current.BaseType)
            {
                if (Equals(current, factAttribute))
                    return true;
            }

            return false;
        }
    }
}
