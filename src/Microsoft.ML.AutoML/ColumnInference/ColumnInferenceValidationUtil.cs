// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.AutoML
{
    internal static class ColumnInferenceValidationUtil
    {
        /// <summary>
        /// Validate all columns specified in column info exist in inferred data view.
        /// </summary>
        public static void ValidateSpecifiedColumnsExist(ColumnInformation columnInfo,
            IDataView dataView)
        {
            var columnNames = ColumnInformationUtil.GetColumnNames(columnInfo);
            foreach (var columnName in columnNames)
            {
                if (dataView.Schema.GetColumnOrNull(columnName) == null)
                {
                    throw new ArgumentException($"Specified column {columnName} " +
                        $"is not found in the dataset.");
                }
            }
        }
    }
}
