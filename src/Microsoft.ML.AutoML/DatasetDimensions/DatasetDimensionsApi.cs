// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML
{
    internal class DatasetDimensionsApi
    {
        private const long MaxRowsToRead = 1000;

        public static ColumnDimensions[] CalcColumnDimensions(MLContext context, IDataView data, PurposeInference.Column[] purposes)
        {
            data = context.Data.TakeRows(data, MaxRowsToRead);

            var colDimensions = new ColumnDimensions[data.Schema.Count];

            for (var i = 0; i < data.Schema.Count; i++)
            {
                var column = data.Schema[i];
                var purpose = purposes[i];

                // default column dimensions
                int? cardinality = null;
                bool? hasMissing = null;

                var itemType = column.Type.GetItemType();

                // If categorical text feature, calculate cardinality
                if (itemType.IsText() && purpose.Purpose == ColumnPurpose.CategoricalFeature)
                {
                    cardinality = DatasetDimensionsUtil.GetTextColumnCardinality(data, column);
                }

                // If numeric feature, discover missing values
                if (itemType == NumberDataViewType.Single)
                {
                    hasMissing = column.Type.IsVector() ?
                        DatasetDimensionsUtil.HasMissingNumericVector(data, column) :
                        DatasetDimensionsUtil.HasMissingNumericSingleValue(data, column);
                }

                colDimensions[i] = new ColumnDimensions(cardinality, hasMissing);
            }

            return colDimensions;
        }
    }
}
