using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Samples.Dynamic.Trainers.BinaryClassification
{
    public static class PermutationFeatureImportanceLoadFromDisk
    {
        public static void Example()
        {

            var mlContext = new MLContext(seed: 1);
            var samples = GenerateData();
            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Create pipeline
            var featureColumns =
                new string[] { nameof(Data.Feature1), nameof(Data.Feature2) };
            var pipeline = mlContext.Transforms
                .Concatenate("Features", featureColumns)
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression()
                );

            // Create and save model
            var model0 = pipeline.Fit(data);
            var lt = model0.LastTransformer;
            var modelPath = "./model.zip";
            mlContext.Model.Save(model0, data.Schema, modelPath);

            // Load model
            var model = mlContext.Model.Load(modelPath, out var schema);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            var linearPredictor = (model as TransformerChain<ITransformer>).LastTransformer as BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>;

            // Execute PFI with the linearPredictor
            var permutationMetrics = mlContext.BinaryClassification
                .PermutationFeatureImportance(linearPredictor, transformedData,
                permutationCount: 30);

            // Sort indices according to PFI results
            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.AreaUnderRocCurve })
                .OrderByDescending(
                feature => Math.Abs(feature.AreaUnderRocCurve.Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature\tModel Weight\tChange in AUC"
                + "\t95% Confidence in the Mean Change in AUC");
            var auc = permutationMetrics.Select(x => x.AreaUnderRocCurve).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}\t{1:0.00}\t{2:G4}\t{3:G4}",
                    featureColumns[i],
                    linearPredictor.Model.SubModel.Weights[i], // this way we can access the weights inside the submodel
                    auc[i].Mean,
                    1.96 * auc[i].StandardError);
            }

            // Expected output:
            //  Feature     Model Weight Change in AUC  95% Confidence in the Mean Change in AUC
            //  Feature2        35.15     -0.387        0.002015
            //  Feature1        17.94     -0.1514       0.0008963
        }

        private class Data
        {
            public bool Label { get; set; }

            public float Feature1 { get; set; }

            public float Feature2 { get; set; }
        }

        /// Generate Data
        private static IEnumerable<Data> GenerateData(int nExamples = 10000,
            double bias = 0, double weight1 = 1, double weight2 = 2, int seed = 1)
        {
            var rng = new Random(seed);
            for (int i = 0; i < nExamples; i++)
            {
                var data = new Data
                {
                    Feature1 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                    Feature2 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                };

                // Create a noisy label.
                var value = (float)(bias + weight1 * data.Feature1 + weight2 *
                    data.Feature2 + rng.NextDouble() - 0.5);

                data.Label = Sigmoid(value) > 0.5;
                yield return data;
            }
        }

        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-1 * x));
    }
}
