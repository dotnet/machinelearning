using Microsoft.ML.Data;
using Microsoft.ML.SamplesUtils;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class ReplaceMissingValues
    {
        public static void Example()
        {
            // Creating the ML.Net IHostEnvironment object, needed for the pipeline.
            var mlContext = new MLContext();

            // Download the training and validation files.
            string dataFile = DatasetUtils.DownloadMslrWeb10k();

            // Create the loader to load the data.
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column("GroupId", DataKind.String, 1),
                    new TextLoader.Column("Features", DataKind.Single, new[] { new TextLoader.Range(2, 138) })
                }
            );

            // Load the raw dataset.
            var data = loader.Load(dataFile);

            // Create the featurization pipeline. First, hash the GroupId column.
            var pipeline = mlContext.Transforms.Conversion.Hash("GroupId")
                // Replace missing values in Features column with the default replacement value for its type.
                .Append(mlContext.Transforms.ReplaceMissingValues("Features"));

            // Fit the pipeline and transform the dataset.
            var transformedData = pipeline.Fit(data).Transform(data);
        }
    }
}
