// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Convenience method to more easily extract an <see cref="IHostEnvironment"/> from an <see cref="IInternalCatalog"/>
    /// implementor without requiring an explicit cast.
    /// </summary>
    [BestFriend]
    internal static class CatalogUtils
    {
        public static IHostEnvironment GetEnvironment(this IInternalCatalog catalog) => Contracts.CheckRef(catalog, nameof(catalog)).Environment;
    }

    /// <summary>
    /// An internal interface for the benefit of those <see cref="IHostEnvironment"/>-bearing objects accessible through
    /// <see cref="MLContext"/>. Because this is meant to consumed by component authors implementations of this interface
    /// should be explicit.
    /// </summary>
    [BestFriend]
    internal interface IInternalCatalog
    {
        IHostEnvironment Environment { get; }
    }
}
