using Microsoft.ML.SamplesUtils;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic.Trainers.MulticlassClassification
{
    public static class StochasticDualCoordinateAscentWithOptions
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            // Setting the seed to a fixed number in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            // Create a list of data examples.
            var examples = DatasetUtils.GenerateRandomMulticlassClassificationExamples(1000);

            // Convert the examples list to an IDataView object, which is consumable by ML.NET API.
            var dataView = mlContext.Data.LoadFromEnumerable(examples);

            //////////////////// Data Preview ////////////////////
            // Label    Features
            // AA       0.7262433,0.8173254,0.7680227,0.5581612,0.2060332,0.5588848,0.9060271,0.4421779,0.9775497,0.2737045
            // BB       0.4919063,0.6673147,0.8326591,0.6695119,1.182151,0.230367,1.06237,1.195347,0.8771811,0.5145918
            // CC       1.216908,1.248052,1.391902,0.4326252,1.099942,0.9262842,1.334019,1.08762,0.9468155,0.4811099
            // DD       0.7871246,1.053327,0.8971719,1.588544,1.242697,1.362964,0.6303943,0.9810045,0.9431419,1.557455

            var options = new SdcaNonCalibratedMulticlassTrainer.Options
            {
                // Add custom loss
                Loss = new HingeLoss(),
                // Make the convergence tolerance tighter.
                ConvergenceTolerance = 0.05f,
                // Increase the maximum number of passes over training data.
                MaximumNumberOfIterations = 30,
            };

            // Create a pipeline. 
            var pipeline =
                    // Convert the string labels into key types.
                    mlContext.Transforms.Conversion.MapValueToKey("Label")
                    // Apply StochasticDualCoordinateAscent multiclass trainer.
                    .Append(mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(options));

            // Split the data into training and test sets. Only training set is used in fitting
            // the created pipeline. Metrics are computed on the test.
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);

            // Train the model.
            var model = pipeline.Fit(split.TrainSet);

            // Do prediction on the test set.
            var dataWithPredictions = model.Transform(split.TestSet);

            // Evaluate the trained model using the test set.
            var metrics = mlContext.MulticlassClassification.Evaluate(dataWithPredictions);
            SamplesUtils.ConsoleUtils.PrintMetrics(metrics);

            // Expected output:
            //   Micro Accuracy: 0.82
            //   Macro Accuracy: 0.81
            //   Log Loss: 0.64
            //   Log Loss Reduction: 52.51
        }
    }
}
