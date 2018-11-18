// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Set of extension methods to extract <see cref="IHostEnvironment"/> from various catalog classes.
    /// </summary>
    public static class CatalogUtils
    {
        public static IHostEnvironment GetEnvironment(this TransformsCatalog catalog) => Contracts.CheckRef(catalog, nameof(catalog)).Environment;
        public static IHostEnvironment GetEnvironment(this TransformsCatalog.SubCatalogBase subCatalog) => Contracts.CheckRef(subCatalog, nameof(subCatalog)).Environment;
        public static IHostEnvironment GetEnvironment(this ModelOperationsCatalog catalog) => Contracts.CheckRef(catalog, nameof(catalog)).Environment;
        public static IHostEnvironment GetEnvironment(this DataOperations catalog) => Contracts.CheckRef(catalog, nameof(catalog)).Environment;
        public static IHostEnvironment GetEnvironment(TrainContextBase.ContextInstantiatorBase obj) => Contracts.CheckRef(obj, nameof(obj)).Owner.Environment;
        public static IHostEnvironment GetEnvironment(TrainContextBase ctx) => Contracts.CheckRef(ctx, nameof(ctx)).Environment;

    }
}
