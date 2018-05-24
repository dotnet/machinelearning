using Microsoft.ML.Runtime.Api;

namespace BinaryClassification_SentimentAnalysis
{
    public class SentimentData
    {
        [Column("0")] public string SentimentText;

        [Column("1", name: "Label")] public float Sentiment;
    }
}
