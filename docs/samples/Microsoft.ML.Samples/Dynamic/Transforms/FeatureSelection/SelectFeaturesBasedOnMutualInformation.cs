using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public static class SelectFeaturesBasedOnMutualInformation
    {
        private static readonly int printRowCount = 4;

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
            var mlContext = new MLContext();

            // First, we define the loader: specify the data columns and where to find them in the text file. Notice that we combine entries from
            // all the feature columns into entries of a vector of a single column named "Features".
            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                    {
                        new TextLoader.Column("Label", DataKind.Boolean, 0),
                        new TextLoader.Column("GroupA", DataKind.Single, new [] { new TextLoader.Range(1, 3) }),
                        new TextLoader.Column("GroupB", DataKind.Single, new [] { new TextLoader.Range(4, 6) }),
                        new TextLoader.Column("GroupC", DataKind.Single, new [] { new TextLoader.Range(7, 9) }),
                    },
                hasHeader: true
            );

            // Then, we use the loader to load the data as an IDataView.
            var data = loader.Load(dataFilePath);

            Console.WriteLine("Contents of column 'GroupB'");
            PrintDataColumn(data, "GroupB");
            // 5 7 10
            // 1 2 2
            // 1 3 4
            // 3 2 1

            Console.WriteLine("Contents of column 'GroupC'");
            PrintDataColumn(data, "GroupC");
            // 3 2 1
            // 3 1 1
            // 3 7 1
            // 3 1 1

            // Second, we define the transformations that we apply on the data. Remember that an Estimator does not transform data
            // directly, but it needs to be trained on data using .Fit(), and it will output a Transformer, which can transform data.

            // We define a MutualInformationFeatureSelectingEstimator that selects the top k slots in a feature 
            // vector based on highest mutual information between that slot and a specified label. Notice that it is possible to 
            // specify the parameter `numBins', which controls the number of bins used in the approximation of the mutual information
            // between features and label.

            var pipeline = mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation(
                outputColumnName: "FeaturesSelectedGroupB", inputColumnName: "GroupB", labelColumnName: "Label", slotsInOutput: 2);

            // The pipeline can then be trained, using .Fit(), and the resulting transformer can be used to transform data. 
            var transformedData = pipeline.Fit(data).Transform(data);

            Console.WriteLine("Contents of column 'FeaturesSelectedGroupB'");
            PrintDataColumn(transformedData, "FeaturesSelectedGroupB");
            // Note, SelectFeaturesBasedOnMutualInformation retained 2 slots (out of 3).  
            // 7 10
            // 2 2
            // 3 4
            // 2 1

            // Multi column example : This pipeline uses two columns for transformation
            pipeline = mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnMutualInformation(
                new InputOutputColumnPair[] { new InputOutputColumnPair("GroupB"), new InputOutputColumnPair("GroupC") },
                labelColumnName: "Label",
                slotsInOutput:4);

            transformedData = pipeline.Fit(data).Transform(data);

            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true).Take(printRowCount);
            Console.WriteLine("Contents of two columns 'GroupB' and 'GroupC'.");
            foreach (var item in convertedData)
                Console.WriteLine("{0}\t\t{1}", string.Join(" ", item.GroupB), string.Join(" ", item.GroupC));

            // 7 10            3 2
            // 2 2             3 1
            // 3 4             3 7
            // 2 1             3 1
        }

        private static void PrintDataColumn(IDataView transformedData, string columnName)
        {
            var countSelectColumn = transformedData.GetColumn<float[]>(transformedData.Schema[columnName]);

            int count = 0;
            foreach (var row in countSelectColumn)
            {
                for (var i = 0; i < row.Length; i++)
                    Console.Write($"{row[i]} ");
                Console.WriteLine();

                count += 1;
                if (count >= printRowCount)
                    break;
            }
        }

        private class TransformedData
        {
            public float[] GroupB { get; set; }

            public float[] GroupC { get; set; }
        }
    }
}
