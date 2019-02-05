// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Set of extension methods to extract <see cref="IHostEnvironment"/> from various catalog classes.
    /// </summary>
    [BestFriend]
    internal static class CatalogUtils
    {
        public static IHostEnvironment GetEnvironment(this TransformsCatalog catalog) => Contracts.CheckRef(catalog, nameof(catalog)).Environment;
        public static IHostEnvironment GetEnvironment(this TransformsCatalog.SubCatalogBase subCatalog) => Contracts.CheckRef(subCatalog, nameof(subCatalog)).Environment;
        public static IHostEnvironment GetEnvironment(this ModelOperationsCatalog catalog) => Contracts.CheckRef(catalog, nameof(catalog)).Environment;
        public static IHostEnvironment GetEnvironment(this ModelOperationsCatalog.SubCatalogBase subCatalog) => Contracts.CheckRef(subCatalog, nameof(subCatalog)).Environment;
        public static IHostEnvironment GetEnvironment(this DataOperationsCatalog catalog) => Contracts.CheckRef(catalog, nameof(catalog)).Environment;
        public static IHostEnvironment GetEnvironment(TrainCatalogBase.CatalogInstantiatorBase obj) => Contracts.CheckRef(obj, nameof(obj)).Owner.Environment;
        public static IHostEnvironment GetEnvironment(TrainCatalogBase catalog) => Contracts.CheckRef(catalog, nameof(catalog)).Environment;
    }
}
