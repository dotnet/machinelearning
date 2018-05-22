using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace BinaryClassification_SentimentAnalysis
{
    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, @"..\..\..\..\datasets\", "imdb_labelled.txt");
        private static string TestDataPath => Path.Combine(AppPath, @"..\..\..\..\datasets\", "yelp_labelled.txt");
        private static string ModelPath => Path.Combine(AppPath, "Models", "SentimentModel.zip");

        private static async Task Main(string[] args)
        {
            var model = await TrainAsync();

            Evaluate(model);

            var predictions = model.Predict(TestSentimentData.Sentiments);

            var sentimentsAndPredictions =
                TestSentimentData.Sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine(
                    $"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")} sentiment");
            }

            Console.ReadLine();
        }

        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> TrainAsync()
        {
            // LearningPipeline allows us to add steps in order to keep everything together 
            // during the learning process.  
            var pipeline = new LearningPipeline();

            // The TextLoader loads a dataset with comments and corresponding postive or negative sentiment. 
            // When you create a loader you specify the schema by passing a class to the loader containing
            // all the column names and their types. This will be used to create the model, and train it. 
            pipeline.Add(new TextLoader<SentimentData>(TrainDataPath, useHeader: false, separator: "tab"));

            // TextFeaturizer is a transform that will be used to featurize an input column. 
            // This is used to format and clean the data.
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            //add a FastTreeBinaryClassifier, the decision tree learner for this project, and 
            //three hyperparameters to be used for tuning decision tree performance 
            pipeline.Add(new FastTreeBinaryClassifier() {NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2});

            Console.WriteLine("=============== Training model ===============");
            // We train our pipeline based on the dataset that has been loaded and transformed 
            var model = pipeline.Train<SentimentData, SentimentPrediction>();

            await model.WriteAsync(ModelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

        private static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader<SentimentData>(TestDataPath, useHeader: true, separator: "tab");

            // BinaryClassificationEvaluator computes the quality metrics for the PredictionModel
            //using the specified data set.
            var evaluator = new BinaryClassificationEvaluator();

            Console.WriteLine("=============== Evaluating model ===============");

            // BinaryClassificationMetrics contains the overall metrics computed by binary classification evaluators
            var metrics = evaluator.Evaluate(model, testData);

            // The Accuracy metric gets the accuracy of a classifier which is the proportion 
            //of correct predictions in the test set.

            // The Auc metric gets the area under the ROC curve.
            // The area under the ROC curve is equal to the probability that the classifier ranks
            // a randomly chosen positive instance higher than a randomly chosen negative one
            // (assuming 'positive' ranks higher than 'negative').

            // The F1Score metric gets the classifier's F1 score.
            // The F1 score is the harmonic mean of precision and recall:
            //  2 * precision * recall / (precision + recall).

            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End evaluating ===============");
            Console.WriteLine();
        }
    }
}