using System;
using System.IO;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class TextClassification
    {
        public const int MaxSentenceLenth = 600;
        /// <summary>
        /// Example use of the TensorFlow sentiment classification model.
        /// </summary>
        public static void Example()
        {
            string modelLocation = SamplesUtils.DatasetUtils.DownloadTensorFlowSentimentModel();

            var mlContext = new MLContext();
            var data = new[] { new IMDBSentiment() {
                Sentiment_Text = "this film was just brilliant casting location scenery story direction " +
                "everyone's really suited the part they played and you could just imagine being there robert " +
                "is an amazing actor and now the same being director  father came from the same scottish " +
                "island as myself so i loved the fact there was a real connection with this film the witty " +
                "remarks throughout the film were great it was just brilliant so much that i bought the " +
                "film as soon as it was released for  and would recommend it to everyone to watch and the " +
                "fly fishing was amazing really cried at the end it was so sad and you know what they say " +
                "if you cry at a film it must have been good and this definitely was also  to the two " +
                "little boy's that played the  of norman and paul they were just brilliant children are " +
                "often left out of the  list i think because the stars that play them all grown up are " +
                "such a big profile for the whole film but these children are amazing and should be praised " +
                "for what they have done don't you think the whole story was so lovely because it was true " +
                "and was someone's life after all that was shared with us all" } };
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // This is the dictionary to convert words into the integer indexes.
            var lookupMap = mlContext.Data.LoadFromTextFile(Path.Combine(modelLocation, "imdb_word_index.csv"),
                columns: new[]
                   {
                        new TextLoader.Column("Words", DataKind.String, 0),
                        new TextLoader.Column("Ids", DataKind.Int32, 1),
                   },
                separatorChar: ','
               );

            // Load the TensorFlow model once.
            //      - Use it for quering the schema for input and output in the model
            //      - Use it for prediction in the pipeline.
            var tensorFlowModel = mlContext.Model.LoadTensorFlowModel(modelLocation);
            var schema = tensorFlowModel.GetModelSchema();
            var featuresType = (VectorType)schema["Features"].Type;
            Console.WriteLine("Name: {0}, Type: {1}, Shape: (-1, {2})", "Features", featuresType.ItemType.RawType, featuresType.Dimensions[0]);
            var predictionType = (VectorType)schema["Prediction/Softmax"].Type;
            Console.WriteLine("Name: {0}, Type: {1}, Shape: (-1, {2})", "Prediction/Softmax", predictionType.ItemType.RawType, predictionType.Dimensions[0]);
            
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
                var features = i.VariableLenghtFeatures;
                Array.Resize(ref features, MaxSentenceLenth);
                j.Features = features;
            };

            var model = mlContext.Transforms.Text.TokenizeIntoWords("TokenizedWords", "Sentiment_Text")
                .Append(mlContext.Transforms.Conversion.MapValue("VariableLenghtFeatures", lookupMap,
                    lookupMap.Schema["Words"], lookupMap.Schema["Ids"], "TokenizedWords"))
                .Append(mlContext.Transforms.CustomMapping(ResizeFeaturesAction, "Resize"))
                .Append(tensorFlowModel.ScoreTensorFlowModel(new[] { "Prediction/Softmax" }, new[] { "Features" }))
                .Append(mlContext.Transforms.CopyColumns("Prediction", "Prediction/Softmax"))
                .Fit(dataView);
            var engine = mlContext.Model.CreatePredictionEngine<IMDBSentiment, OutputScores>(model);

            // Predict with TensorFlow pipeline.
            var prediction = engine.Predict(data[0]);

            Console.WriteLine("Number of classes: {0}", prediction.Prediction.Length); 
            Console.WriteLine("Is sentiment/review positive? {0}", prediction.Prediction[1] > 0.5 ? "Yes." : "No."); 
            Console.WriteLine("Prediction Confidence: {0}", prediction.Prediction[1].ToString("0.00"));

            /////////////////////////////////// Expected output ///////////////////////////////////
            // 
            // Name: Features, Type: System.Int32, Shape: (-1, 600)
            // Name: Prediction/Softmax, Type: System.Single, Shape: (-1, 2)
            // 
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

            /// <summary>
            /// This is a variable length vector designated by VectorType attribute.
            /// Variable length vectors are produced by applying operations such as 'TokenizeWords' on strings
            /// resulting in vectors of tokens of variable lengths.
            /// </summary>
            [VectorType]
            public int[] VariableLenghtFeatures { get; set; }
        }

        /// <summary>
        /// Class to hold intermediate data. Mostly used by CustomMapping Estimator
        /// </summary>
        public class IntermediateFeatures
        {
            public string Sentiment_Text { get; set; }

            [VectorType(MaxSentenceLenth)]
            public int[] Features { get; set; }
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
