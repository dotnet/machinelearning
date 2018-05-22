using System.Collections.Generic;

namespace BinaryClassification_SentimentAnalysis
{
    internal class TestSentimentData
    {
        internal static readonly IEnumerable<SentimentData> Sentiments = new[]
        {
            new SentimentData
            {
                SentimentText = "Contoso's 11 is a wonderful experience",
                Sentiment = 0
            },
            new SentimentData
            {
                SentimentText = "The acting in this movie is very bad",
                Sentiment = 0
            },
            new SentimentData
            {
                SentimentText = "Joe versus the Volcano Coffee Company is a great film.",
                Sentiment = 0
            }
        };
    }
}