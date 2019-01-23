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

                // if categorical text feature, calc cardinality
                if(column.Type.ItemType().IsText() && purpose.Purpose == ColumnPurpose.CategoricalFeature)
                {
                    cardinality = DatasetDimensionsUtil.GetTextColumnCardinality(data, i);
                }

                // if numeric feature, discover missing values
                // todo: upgrade logic to consider R8?
                if (column.Type.ItemType() == NumberType.R4)
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
