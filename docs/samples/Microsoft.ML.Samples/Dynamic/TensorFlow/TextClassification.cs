using System;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic.TensorFlow
{
    class TextClassification
    {
        public const int MaxSentenceLenth = 600;
        /// <summary>
        /// Example use of the TensorFlow sentiment classification model.
        /// Download the model from 
        /// https://github.com/dotnet/machinelearning-testdata/blob/master/Microsoft.ML.TensorFlow.TestModels/sentiment_model
        /// The model is in 'SavedModel' format. For further explanation on how was the `sentiment_model` created 
        /// c.f. https://github.com/dotnet/machinelearning-testdata/blob/master/Microsoft.ML.TensorFlow.TestModels/sentiment_model/README.md
        /// </summary>
        public static void ScoringWithTextClassificationModelSample()
        {
            string modelLocation = @"sentiment_model";

            var mlContext = new MLContext(seed: 1, conc: 1);
            var data = new[] { new IMDBSentiment() { Sentiment_Text = "this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert  is an amazing actor and now the same being director  father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for  and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also  to the two little boy's that played the  of norman and paul they were just brilliant children are often left out of the  list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all" } };
            var dataView = mlContext.Data.ReadFromEnumerable(data);

            // This is the dictionary to convert words into the integer indexes.
            var lookupMap = mlContext.Data.ReadFromTextFile(@"sentiment_model/imdb_word_index.csv",
                   columns: new[]
                   {
                        new TextLoader.Column("Words", DataKind.TX, 0),
                        new TextLoader.Column("Ids", DataKind.I4, 1),
                   },
                   separatorChar: ','
               );

            // The model expects the input feature vector to be a fixed length vector.
            // In this sample, CustomMappingEstimator is used to resize variable length vector to fixed length vector.
            // The following ML.NET pipeline
            //      1. tokenzies the string into words, 
            //      2. maps each word to an integer which is an index in the dictionary ('lookupMap'),
            //      3. Resizes the integer vector to a fixed length vector using CustomMappingEstimator ('ResizeFeaturesAction')
            //      4. Passes the data to TensorFlow for scoring.
            //      5. Retreives the 'Prediction' from TensorFlow and put it into ML.NET Pipeline 

            Action<IMDBSentiment, IntermediateFeatures> ResizeFeaturesAction = (i, j) =>
            {
                j.Sentiment_Text = i.Sentiment_Text;
                j.Features = i.VarLengthFeatures;
                Array.Resize(ref j.Features, MaxSentenceLenth);
            };

            var engine = mlContext.Transforms.Text.TokenizeWords("TokenizedWords", "Sentiment_Text")
                .Append(mlContext.Transforms.Conversion.ValueMap(lookupMap, "Words", "Ids", new[] { ("VarLengthFeatures", "TokenizedWords") }))
                .Append(mlContext.Transforms.CustomMapping(ResizeFeaturesAction, "Resize"))
                .Append(mlContext.Transforms.ScoreTensorFlowModel(modelLocation, new[] { "Prediction/Softmax" }, new[] { "Features" }))
                .Append(mlContext.Transforms.CopyColumns(("Prediction", "Prediction/Softmax")))
                .Fit(dataView)
                .CreatePredictionEngine<IMDBSentiment, OutputScores>(mlContext);

            // Predict with TensorFlow pipeline.
            var prediction = engine.Predict(data[0]);

            Console.WriteLine("Number of classes: {0}", prediction.Prediction.Length); 
            Console.WriteLine("Is sentiment/review positive? {0}", prediction.Prediction[1] > 0.5 ? "Yes." : "No."); 
            Console.WriteLine("Prediction Confidence: {0}", prediction.Prediction[1].ToString("0.00"));

            //// Expected output
            // Number of classes: 2
            // Is sentiment/review positive ? Yes
            // Prediction Confidence: 0.65
        }


        /// <summary>
        /// Class to hold original sentiment data.
        /// </summary>
        public class IMDBSentiment
        {
            public string Sentiment_Text { get; set; }
            [VectorType(0)]
            public int[] VarLengthFeatures { get; set; }
        }

        /// <summary>
        /// Class to hold intermediate data. Mostly used by CustomMapping Estimator
        /// </summary>
        public class IntermediateFeatures
        {
            public string Sentiment_Text { get; set; }
            [VectorType(600)]
            public int[] Features;
        }

        /// <summary>
        /// Class to contain the output values from the transformation.
        /// </summary>
        class OutputScores
        {
            [VectorType(2)]
            public float[] Prediction { get; set; }
        }

    }
}
