// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal class DatasetDimensionsApi
    {
        private const int MaxRowsToRead = 1000;

        public static ColumnDimensions[] CalcColumnDimensions(IDataView data, PurposeInference.Column[] purposes)
        {
            data = data.Take(MaxRowsToRead);

            var colDimensions = new ColumnDimensions[data.Schema.Count];

            for (var i = 0; i < data.Schema.Count; i++)
            {
                var column = data.Schema[i];
                var purpose = purposes[i];

                // default column dimensions
                int? cardinality = null;
                bool? hasMissing = null;

                var itemType = column.Type.ItemType();

                // If categorical text feature, calculate cardinality
                if (itemType.IsText() && purpose.Purpose == ColumnPurpose.CategoricalFeature)
                {
                    cardinality = DatasetDimensionsUtil.GetTextColumnCardinality(data, i);
                }

                // If numeric feature, discover missing values
                if (itemType == NumberType.R4)
                {
                    hasMissing = column.Type.IsVector() ? 
                        DatasetDimensionsUtil.HasMissingNumericVector(data, i) : 
                        DatasetDimensionsUtil.HasMissingNumericSingleValue(data, i);
                }

                colDimensions[i] = new ColumnDimensions(cardinality, hasMissing);
            }

            return colDimensions;
        }
    }
}
