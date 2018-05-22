using System;
using System.IO;
using System.Threading.Tasks;

using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace GitHubLabeler
{
    internal class Predictor
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string DataPath => Path.Combine(AppPath, "Data", "corefx_issues.tsv");

        private static string ModelPath => Path.Combine(AppPath, "Models", "Model.zip");

        private static PredictionModel<GitHubIssue, GitHubIssuePrediction> _model;

        public static async Task TrainAsync()
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader<GitHubIssue>(DataPath, useHeader: true));

            pipeline.Add(new Dictionarizer(("Area", "Label")));

            pipeline.Add(new TextFeaturizer("Title", "Title"));

            pipeline.Add(new TextFeaturizer("Description", "Description"));
            
            pipeline.Add(new ColumnConcatenator("Features", "Title", "Description"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            Console.WriteLine("=============== Training model ===============");

            var model = pipeline.Train<GitHubIssue, GitHubIssuePrediction>();

            await model.WriteAsync(ModelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);
        }

        public static async Task<string> PredictAsync(GitHubIssue issue)
        {
            if (_model == null)
            {
                _model = await PredictionModel.ReadAsync<GitHubIssue, GitHubIssuePrediction>(ModelPath);
            }

            var prediction = _model.Predict(issue);

            return prediction.Area;
        }
    }
}
