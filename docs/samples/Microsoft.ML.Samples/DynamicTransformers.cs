using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.StaticPipe;

namespace Microsoft.ML.Samples
{
    public class DynamicTransformers
    {

        public static void KeyToValue()
        {

        }

        public static void Concat()
        {

        }

        public static void Term()
        {

        }

        public static void TextTransform()
        {
            // Create a new environment for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var env = new LocalEnvironment();

            IEnumerable<SamplesUtils.DatasetUtils.SampleSentimentData> data = SamplesUtils.DatasetUtils.GetSentimentData();

            // Preview of the data.
            // Sentiment    SentimentText
            // true         This is the best game I've ever played.
            // false        ==RUDE== Dude, you are rude upload that picture back, or else.
            // true          Until the next game comes out, this game is undisputedly the best Xbox game of all time

            var trainData = env.CreateStreamingDataView(data);

            var learningPipeline = new TextTransform(env, "SentimentText", "TextFeatures");

            var transformedData = learningPipeline.Fit(trainData).Transform(trainData);

            var textFeaturesColumn = transformedData.GetColumn<VBuffer<float>>(env, "TextFeatures").ToArray();

            // Preview of the transformedData.
            Console.WriteLine("TextFeatures column obtained post-transformation.");
            foreach (var featureRow in textFeaturesColumn)
            {
                foreach (var value in featureRow.Values)
                    Console.Write($"{value} ");
                Console.WriteLine("");
            }

            //Transformed data
            // 0.2581989 0.2581989 0.2581989 0.2581989 0.5163978 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.7071068 0.7071068 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0.2581989 0.2581989 0.2581989 0.2581989 0.5163978 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.2581989 0.7071068 0.7071068 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.1924501 0.4472136 0.4472136 0.4472136 0.4472136 0.4472136
            // 0 0.1230915 0.1230915 0.1230915 0.1230915 0.246183 0.246183 0.246183 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.3692745 0.246183 0.246183 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.246183 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.1230915 0.2886751 0 0 0 0 0 0 0.2886751 0.5773503 0.2886751 0.2886751 0.2886751 0.2886751 0.2886751 0.2886751
        }

        public static void MinMaxNormalizer()
        {

        }
    }
}
