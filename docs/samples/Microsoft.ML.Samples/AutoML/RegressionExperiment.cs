using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Data;

namespace Samples.AutoML
{
    public static class RegressionExperiment
    {
        public static void Example()
        {
            // Downloading a regression dataset from github.com/dotnet/machinelearning
            string dataFile = Microsoft.ML.SamplesUtils.DatasetUtils.DownloadHousingRegressionDataset();

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Creating a data loader, based on the format of the data
            // The data is tab separated with all numeric columns.
            // The first column being the label and rest are numeric features
            // Here only seven numeric columns are used as features
            var dataView = mlContext.Data.LoadFromTextFile(dataFile, new TextLoader.Options
            {
                Separators = new[] { '\t' },
                HasHeader = true,
                Columns = new[]
               {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("Features", DataKind.Single, 1, 6)
                }
            });

            // Run an AutoML experiment
            var experimentResult = mlContext.Auto()
                .CreateRegressionExperiment(60)
                .Execute(dataView);
        }
    }
}
