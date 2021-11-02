using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples
{
    public class SearchDataPrediction
    {
        [ColumnName("PredictedLabel")]
        public float Prediction;

        public float Score { get; set; }
    }
}
