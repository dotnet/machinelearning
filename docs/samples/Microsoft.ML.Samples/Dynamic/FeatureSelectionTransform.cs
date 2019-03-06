using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class FeatureSelectionTransform
    {
        public static void Example()
        {
            // Downloading a classification dataset from github.com/dotnet/machinelearning.
            // It will be stored in the same path as the executable
            string dataFilePath = SamplesUtils.DatasetUtils.DownloadBreastCancerDataset();

            // Data Preview
            //    1. Label							0=benign, 1=malignant
            //    2. Clump Thickness               1 - 10
            //    3. Uniformity of Cell Size       1 - 10
            //    4. Uniformity of Cell Shape      1 - 10
            //    5. Marginal Adhesion             1 - 10
            //    6. Single Epithelial Cell Size   1 - 10
            //    7. Bare Nuclei                   1 - 10
            //    8. Bland Chromatin               1 - 10
            //    9. Normal Nucleoli               1 - 10
            //   10. Mitoses                       1 - 10

            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // First, we define the loader: specify the data columns and where to find them in the text file. Notice that we combine entries from
            // all the feature columns into entries of a vector of a single column named "Features".
            var loader = ml.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Label", DataKind.Boolean, 0),
                        new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 9) })
                    },
                hasHeader: true
            );

            // Then, we use the loader to load the data as an IDataView.
            var data = loader.Load(dataFilePath);

            // Second, we define the transformations that we apply on the data. Remember that an Estimator does not transform data
            // directly, but it needs to be trained on data using .Fit(), and it will output a Transformer, which can transform data.

            // In this example we define a CountFeatureSelectingEstimator, that selects slots in a feature vector that have more non-default 
            // values than the specified count. This transformation can be used to remove slots with too many missing values.
            var countSelectEst = ml.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(
                outputColumnName: "FeaturesCountSelect", inputColumnName: "Features", count: 695);

            // We also define a MutualInformationFeatureSelectingEstimator that selects the top k slots in a feature 
            // vector based on highest mutual information between that slot and a specified label. Notice that it is possible to 
            // specify the parameter `numBins', which controls the number of bins used in the approximation of the mutual information
            // between features and label.
            var mutualInfoEst = ml.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation(
                outputColumnName: "FeaturesMISelect", inputColumnName: "FeaturesCountSelect", labelColumnName: "Label", slotsInOutput: 5);

            // Now, we can put the previous two transformations together in a pipeline.
            var pipeline = countSelectEst.Append(mutualInfoEst);

            // The pipeline can then be trained, using .Fit(), and the resulting transformer can be used to transform data. 
            var transformedData = pipeline.Fit(data).Transform(data);

            // Small helper to print the data inside a column, in the console. Only prints the first 10 rows.
            Action<string, IEnumerable<VBuffer<float>>> printHelper = (columnName, column) =>
            {
                Console.WriteLine($"{columnName} column obtained post-transformation.");
                int count = 0;
                foreach (var row in column)
                {
                    foreach (var value in row.GetValues())
                        Console.Write($"{value}\t");
                    Console.WriteLine("");
                    count++;
                    if (count >= 10)
                        break;
                }

                Console.WriteLine("===================================================");
            };

            // Print the data that results from the transformations.
            var countSelectColumn = transformedData.GetColumn<VBuffer<float>>(transformedData.Schema["FeaturesCountSelect"]);
            var MISelectColumn = transformedData.GetColumn<VBuffer<float>>(transformedData.Schema["FeaturesMISelect"]);
            printHelper("FeaturesCountSelect", countSelectColumn);
            printHelper("FeaturesMISelect", MISelectColumn);

            // Below is the output of the this code. We see that some slots habe been dropped by the first transformation.
            // Among the remaining slots, the second transformation only preserves the top 5 slots based on mutualinformation 
            // with the label column.

            // FeaturesCountSelect column obtained post-transformation.
            // 5       4       4       5       7       3       2       1
            // 3       1       1       1       2       3       1       1
            // 6       8       8       1       3       3       7       1
            // 4       1       1       3       2       3       1       1
            // 8       10      10      8       7       9       7       1
            // 1       1       1       1       2       3       1       1
            // 2       1       2       1       2       3       1       1
            // 2       1       1       1       2       1       1       5
            // 4       2       1       1       2       2       1       1
            // 1       1       1       1       1       3       1       1
            // ===================================================
            // FeaturesMISelect column obtained post-transformation.
            // 4       4       7       3       2
            // 1       1       2       3       1
            // 8       8       3       3       7
            // 1       1       2       3       1
            // 10      10      7       9       7
            // 1       1       2       3       1
            // 1       2       2       3       1
            // 1       1       2       1       1
            // 2       1       2       2       1
            // 1       1       1       3       1
            // ===================================================
        }
    }
}
