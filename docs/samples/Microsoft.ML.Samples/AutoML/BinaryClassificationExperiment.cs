using Microsoft.ML;
using Microsoft.ML.Auto;

namespace Samples.AutoML
{
    public static class BinaryClassificationExperiment
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Download and featurize the dataset.
            var dataView = Microsoft.ML.SamplesUtils.DatasetUtils.LoadFeaturizedAdultDataset(mlContext);

            // Run an AutoML experiment
            var experimentResult = mlContext.Auto()
                .CreateBinaryClassificationExperiment(60)
                .Execute(dataView, "IsOver50K");
        }
    }
}
