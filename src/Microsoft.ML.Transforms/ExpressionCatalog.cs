// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Transforms
{
    public static class ExpressionCatalog
    {
        /// <summary>
        ///
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="outputColumnName"></param>
        /// <param name="expression"></param>
        /// <param name="inputColumnNames"></param>
        /// <returns></returns>
        public static ExpressionEstimator Expression(this TransformsCatalog catalog, string outputColumnName, string expression, params string[] inputColumnNames)
            => new ExpressionEstimator(CatalogUtils.GetEnvironment(catalog), new ExpressionEstimator.ColumnOptions(outputColumnName, inputColumnNames, expression));
    }
}
