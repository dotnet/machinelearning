// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for the <see cref="DataOperationsCatalog"/> to read from databases.
    /// </summary>
    public static class DatabaseLoaderCatalog
    {
        /// <summary>Create a database loader <see cref="DatabaseLoader"/>.</summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="columns">Array of columns <see cref="DatabaseLoader.Column"/> defining the schema.</param>
        public static DatabaseLoader CreateDatabaseLoader(this DataOperationsCatalog catalog,
            params DatabaseLoader.Column[] columns)
        {
            var options = new DatabaseLoader.Options
            {
                Columns = columns,
            };
            return catalog.CreateDatabaseLoader(options);
        }

        /// <summary>Create a database loader <see cref="DatabaseLoader"/>.</summary>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="options">Defines the settings of the load operation.</param>
        public static DatabaseLoader CreateDatabaseLoader(this DataOperationsCatalog catalog,
            DatabaseLoader.Options options)
            => new DatabaseLoader(CatalogUtils.GetEnvironment(catalog), options);

        /// <summary>Create a database loader <see cref="DatabaseLoader"/>.</summary>
        /// <typeparam name="TInput">Defines the schema of the data to be loaded. Use public fields or properties
        /// decorated with <see cref="LoadColumnAttribute"/> (and possibly other attributes) to specify the column
        /// names and their data types in the schema of the loaded data.</typeparam>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        public static DatabaseLoader CreateDatabaseLoader<TInput>(this DataOperationsCatalog catalog)
            => DatabaseLoader.CreateDatabaseLoader<TInput>(CatalogUtils.GetEnvironment(catalog));
    }
}
