// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class DatasetDimensionsApi
    {
        private const long MaxRowsToRead = 1000;

        public static ColumnDimensions[] CalcColumnDimensions(MLContext context, IDataView data)
        {
            data = context.Data.TakeRows(data, MaxRowsToRead);

            // Init array of all column dimensions to return.
            var colDimensions = new ColumnDimensions[data.Schema.Count];

            // Loop through each column & calculate dimensions.
            for (var i = 0; i < data.Schema.Count; i++)
            {
                var columnSchema = data.Schema[i];
                var itemType = columnSchema.Type.GetItemType();
                var isVectorColumn = columnSchema.Type.IsVector();

                // Non-vector text column
                if (itemType.IsText() && !isVectorColumn)
                {
                    colDimensions[i] = DatasetDimensionsUtil.CalcStringColumnDimensions(data, columnSchema);
                }
                // Non-vector numeric column
                else if (itemType == NumberDataViewType.Single && !isVectorColumn)
                {
                    colDimensions[i] = DatasetDimensionsUtil.CalcNumericColumnDimensions(data, columnSchema);
                }
                // Vector numeric column
                else if (itemType == NumberDataViewType.Single)
                {
                    colDimensions[i] = DatasetDimensionsUtil.CalcNumericVectorColumnDimensions(data, columnSchema);
                }
            }

            return colDimensions;
        }
    }
}
