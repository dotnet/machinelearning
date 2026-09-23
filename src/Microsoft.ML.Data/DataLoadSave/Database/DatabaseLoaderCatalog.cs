// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Data.Common;
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

        /// <summary>
        /// Load an <see cref="IDataView"/> from a database using a <see cref="DatabaseLoader"/>.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        /// <typeparam name="TInput">Defines the schema of the data to be loaded. Use public fields or properties
        /// decorated with <see cref="LoadColumnAttribute"/> (and possibly other attributes) to specify the column
        /// names and their data types in the schema of the loaded data.</typeparam>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="source">The <see cref="DatabaseSource"/> containing the connection and command information.</param>
        /// <returns>The data view.</returns>
        public static IDataView LoadFromDatabase<TInput>(this DataOperationsCatalog catalog,
            DatabaseSource source)
            => catalog.CreateDatabaseLoader<TInput>().Load(source);

        /// <summary>
        /// Load an <see cref="IDataView"/> from a database using a <see cref="DatabaseLoader"/>.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        /// <typeparam name="TInput">Defines the schema of the data to be loaded. Use public fields or properties
        /// decorated with <see cref="LoadColumnAttribute"/> (and possibly other attributes) to specify the column
        /// names and their data types in the schema of the loaded data.</typeparam>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="providerFactory">The factory used to create the <see cref="DbConnection"/>.</param>
        /// <param name="connectionString">The string used to open the connection.</param>
        /// <param name="commandText">The text command to run against the data source.</param>
        /// <returns>The data view.</returns>
        public static IDataView LoadFromDatabase<TInput>(this DataOperationsCatalog catalog,
            DbProviderFactory providerFactory, string connectionString, string commandText)
            => catalog.LoadFromDatabase<TInput>(new DatabaseSource(providerFactory, connectionString, commandText));

        /// <summary>
        /// Load an <see cref="IDataView"/> from a database using a <see cref="DatabaseLoader"/>.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual loading happens here, just schema validation.
        /// </summary>
        /// <typeparam name="TInput">Defines the schema of the data to be loaded. Use public fields or properties
        /// decorated with <see cref="LoadColumnAttribute"/> (and possibly other attributes) to specify the column
        /// names and their data types in the schema of the loaded data.</typeparam>
        /// <param name="catalog">The <see cref="DataOperationsCatalog"/> catalog.</param>
        /// <param name="providerFactory">The factory used to create the <see cref="DbConnection"/>.</param>
        /// <param name="connectionString">The string used to open the connection.</param>
        /// <param name="commandText">The text command to run against the data source.</param>
        /// <param name="commandTimeoutInSeconds">The timeout (in seconds) for the database command.</param>
        /// <returns>The data view.</returns>
        public static IDataView LoadFromDatabase<TInput>(this DataOperationsCatalog catalog,
            DbProviderFactory providerFactory, string connectionString, string commandText, int commandTimeoutInSeconds)
            => catalog.LoadFromDatabase<TInput>(new DatabaseSource(providerFactory, connectionString, commandText, commandTimeoutInSeconds));
    }
}
