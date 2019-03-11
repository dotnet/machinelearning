using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.Samples.Dynamic
{
    public class KMeans
    {
        public static void Example()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext(seed: 1);

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.Data.LoadFromEnumerable(data);

            // Preview of the data.
            //
            // Age  Case  Education  Induced     Parity  PooledStratum  RowNum  ...
            // 26   1       0-5yrs      1         6         3             1  ...
            // 42   1       0-5yrs      1         1         1             2  ...
            // 39   1       0-5yrs      2         6         4             3  ...
            // 34   1       0-5yrs      2         4         2             4  ...
            // 35   1       6-11yrs     1         3         32            5  ...

            // A pipeline for concatenating the age, parity and induced columns together in the Features column and training a KMeans model on them.
            string outputColumnName = "Features";
            var pipeline = ml.Transforms.Concatenate(outputColumnName, new[] { "Age", "Parity", "Induced" })
                .Append(ml.Clustering.Trainers.KMeans(outputColumnName, numberOfClusters: 2));

            var model = pipeline.Fit(trainData);

            // Get cluster centroids and the number of clusters k from KMeansModelParameters.
            VBuffer<float>[] centroids = default;
            int k;

            var modelParams = model.LastTransformer.Model;
            modelParams.GetClusterCentroids(ref centroids, out k);

            var centroid = centroids[0].GetValues();
            Console.WriteLine($"The coordinates of centroid 0 are: ({string.Join(", ", centroid.ToArray())})");

            //  Expected output similar to:
            //      The coordinates of centroid 0 are: (26, 6, 1)
            //
            // Note: use the advanced options constructor to set the number of threads to 1 for a deterministic behavior.
        }
    }
}
