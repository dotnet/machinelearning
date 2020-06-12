// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    public static class ExpressionCatalog
    {
        /// <summary>
        /// Creates an <see cref="ExpressionEstimator"/>.
        /// </summary>
        /// <param name="catalog">The <see cref="TransformsCatalog"/>.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnNames"/>.
        /// This column's data type will be the same as that of the input column.</param>
        /// <param name="expression">The expression to apply to <paramref name="inputColumnNames"/> to create the column <paramref name="outputColumnName"/>.</param>
        /// <param name="inputColumnNames">The names of the input columns.</param>
        /// <example>
        /// <format type="text/markdown">
        /// <![CDATA[
        /// [!code-csharp[Expression](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Transforms/Expression.cs)]
        /// ]]>
        /// </format>
        /// </example>
        public static ExpressionEstimator Expression(this TransformsCatalog catalog, string outputColumnName, string expression, params string[] inputColumnNames)
            => new ExpressionEstimator(CatalogUtils.GetEnvironment(catalog), new ExpressionEstimator.ColumnOptions(outputColumnName, inputColumnNames, expression));
    }
}
