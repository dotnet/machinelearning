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
            var sampledData = new ReservoirSampledDataView(data, 1000);

            var colDimensions = new ColumnDimensions[sampledData.Schema.Count];

            for (var i = 0; i < sampledData.Schema.Count; i++)
            {
                var column = sampledData.Schema[i];
                var purpose = purposes[i];

                // default column dimensions
                int? cardinality = null;
                bool? hasMissing = null;

                var itemType = column.Type.GetItemType();

                // If categorical text feature, calculate cardinality
                if (itemType.IsText() && purpose.Purpose == ColumnPurpose.CategoricalFeature)
                {
                    cardinality = DatasetDimensionsUtil.GetTextColumnCardinality(sampledData, column);
                }

                // If numeric feature, discover missing values
                if (itemType == NumberDataViewType.Single)
                {
                    hasMissing = column.Type.IsVector() ? 
                        DatasetDimensionsUtil.HasMissingNumericVector(sampledData, column) : 
                        DatasetDimensionsUtil.HasMissingNumericSingleValue(sampledData, column);
                }

                colDimensions[i] = new ColumnDimensions(cardinality, hasMissing);
            }

            return colDimensions;
        }
    }
}
