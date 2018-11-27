using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.KMeansClustering;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Samples.Dynamic
{
    public class KMeans_example
    {
        public static void KMeans()
        {
            // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            IEnumerable<SamplesUtils.DatasetUtils.SampleInfertData> data = SamplesUtils.DatasetUtils.GetInfertData();
            var trainData = ml.CreateStreamingDataView(data);

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
                .Append(ml.Clustering.Trainers.KMeans(outputColumnName, clustersCount: 2));

            var model = pipeline.Fit(trainData);

            // Get centroids and k from KMeansModelParameters.
            VBuffer<float>[] centroids = default;
            int k;

            var modelParams = model.LastTransformer.Model;
            modelParams.GetClusterCentroids(ref centroids, out k);

            var centroid = centroids[0].GetValues();
            Console.WriteLine("The coordinates of centroid 0 are: " + string.Join(", ", centroid.ToArray()));
        }
    }
}
