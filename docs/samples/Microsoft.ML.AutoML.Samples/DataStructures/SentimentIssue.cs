using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples
{
    public class SentimentIssue
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
