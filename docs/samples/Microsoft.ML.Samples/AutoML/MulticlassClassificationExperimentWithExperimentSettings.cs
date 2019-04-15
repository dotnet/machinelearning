using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.SamplesUtils;

namespace Samples.AutoML
{
    public static class MulticlassClassificationExperimentWithExperimentSettings
    {
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Download and featurize the dataset.
            // Create a list of data examples.
            var examples = DatasetUtils.GenerateRandomMulticlassClassificationExamples(1000);

            // Convert the examples list to an IDataView object, which is consumable by ML.NET API.
            var dataView = mlContext.Data.LoadFromEnumerable(examples);

            // Run an AutoML experiment
            var experimentSettings = new MulticlassExperimentSettings()
            {
                MaxExperimentTimeInSeconds = 60
            };
            var experimentResult = mlContext.Auto()
                .CreateMulticlassClassificationExperiment(experimentSettings)
                .Execute(dataView);
        }
    }
}