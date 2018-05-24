using Microsoft.ML.Runtime.Api;

namespace BinaryClassification_SentimentAnalysis
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")] public bool Sentiment;
    }
}