using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.Trainers;
using Tensorflow;
using Microsoft.ML.Data;
using Microsoft.ML.AutoPipeline;

namespace Microsoft.ML.AutoPipeline.Test
{
    public class AutoEstimatorChainTest : BaseTestClass
    {
        private ITestOutputHelper _output;

        public AutoEstimatorChainTest(ITestOutputHelper output) : base(output)
        {
            this._output = output;
        }

        [Fact]
        public void RecommendationE2ETest_RandomSweeper()
        {
            var context = new MLContext();
            var paramaters = new MFOption();
            var dataset = context.Data.LoadFromTextFile<ModelInput>(@".\TestData\recommendation-ratings-train.csv", separatorChar: ',', hasHeader: true);
            var split = context.Data.TrainTestSplit(dataset, 0.3);
            var randomSweeper = new RandomSweeper(context, paramaters.ValueGenerators, 20);
            var pipelines = context.Transforms.Conversion.MapValueToKey("userId", "userId")
                          .Append(context.Transforms.Conversion.MapValueToKey("movieId", "movieId"))
                          .Append(context.Recommendation().Trainers.MatrixFactorization, paramaters, randomSweeper, Data.TransformerScope.Everything)
                          .Append(context.Transforms.CopyColumns("output", "Score"));

            foreach (var (model, sweeper) in pipelines.Fits(split.TrainSet))
            {
                var eval = model.Transform(split.TestSet);
                var metrics = context.Regression.Evaluate(eval, "rating", "Score");
                this._output.WriteLine(sweeper.Current.ToString());
                this._output.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            }
        }

        [Fact]
        public void RecommendationE2ETest_GridSearchSweeper()
        {
            var context = new MLContext();
            var paramaters = new MFOption();
            var dataset = context.Data.LoadFromTextFile<ModelInput>(@".\TestData\recommendation-ratings-train.csv", separatorChar: ',', hasHeader: true);
            var split = context.Data.TrainTestSplit(dataset, 0.3);
            var gridSearchSweeper = new GridSearchSweeper(context, paramaters.ValueGenerators, 20);
            var pipelines = context.Transforms.Conversion.MapValueToKey("userId", "userId")
                          .Append(context.Transforms.Conversion.MapValueToKey("movieId", "movieId"))
                          .Append(context.Recommendation().Trainers.MatrixFactorization, paramaters, gridSearchSweeper, Data.TransformerScope.Everything)
                          .Append(context.Transforms.CopyColumns("output", "Score"));

            foreach (var (model, sweeper) in pipelines.Fits(split.TrainSet))
            {
                var eval = model.Transform(split.TestSet);
                var metrics = context.Regression.Evaluate(eval, "rating", "Score");
                this._output.WriteLine(sweeper.Current.ToString());
                this._output.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            }
        }


        private class MFOption : OptionBuilder<MatrixFactorizationTrainer.Options>
        {
            public string MatrixColumnIndexColumnName = "userId";

            public string MatrixRowIndexColumnName = "movieId";

            public string LabelColumnName = "rating";

            [Parameter("Alpha", 0.0001f, 1f, 0.1f)]
            public float Alpha = 0.0001f;

            [Parameter("Lambda", 0.01f, 1f, 0.1f)]
            public double Lambda = 0.01f;

            [Parameter("LearningRate", 0.001f, 0.1f, 0.01f)]
            public double LearningRate = 0.001f;
        }

        private class ModelInput
        {
            [ColumnName("userId"), LoadColumn(0)]
            public float UserId { get; set; }


            [ColumnName("movieId"), LoadColumn(1)]
            public float MovieId { get; set; }


            [ColumnName("rating"), LoadColumn(2)]
            public float Rating { get; set; }
        }
    }
}
