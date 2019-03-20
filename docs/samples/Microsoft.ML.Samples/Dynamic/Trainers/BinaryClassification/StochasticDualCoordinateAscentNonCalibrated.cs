using System;
using System.Linq;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.Samples.Dynamic.Trainers.BinaryClassification
{
    public static class StochasticDualCoordinateAscentNonCalibrated
    {
        public static void Example()
        {
            // Generate IEnumerable<BinaryLabelFloatFeatureVectorSample> as training examples.
            var rawData = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorSamples(100);

            // Information in first example.
            // Label: true
            Console.WriteLine("First example's label is {0}", rawData.First().Label);
            // Features is a 10-element float[]:
            //   [0]	1.0173254	float
            //   [1]	0.9680227	float
            //   [2]	0.7581612	float
            //   [3]	0.406033158	float
            //   [4]	0.7588848	float
            //   [5]	1.10602713	float
            //   [6]	0.6421779	float
            //   [7]	1.17754972	float
            //   [8]	0.473704457	float
            //   [9]	0.4919063	float
            Console.WriteLine("First example's feature vector is {0}", rawData.First().Features);

            // Create a new context for ML.NET operations. It can be used for exception tracking and logging, 
            // as a catalog of available operations and as the source of randomness.
            var mlContext = new MLContext();

            // Step 1: Read the data as an IDataView.
            var data = mlContext.Data.LoadFromEnumerable(rawData);

            // ML.NET doesn't cache data set by default. Caching is always recommended when using the
            // StochasticDualCoordinateAscent algorithm because it may incur multiple data passes.
            data = mlContext.Data.Cache(data);

            // Step 2: Create a binary classifier. This trainer may produce a logistic regression model.
            // We set the "Label" column as the label of the dataset, and the "Features" column as the features column.
            var pipeline = mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(
                labelColumnName: "Label", featureColumnName: "Features", loss: new HingeLoss(), l2Regularization: 0.001f);

            // Step 3: Train the pipeline created.
            var model = pipeline.Fit(data);

            // Step 4: Make prediction and evaluate its quality (on training set).
            var prediction = model.Transform(data);

            var rawPrediction = mlContext.Data.CreateEnumerable<SamplesUtils.DatasetUtils.NonCalibratedBinaryClassifierOutput>(prediction, false);

            // Step 5: Inspect the prediction of the first example.
            // Note that positive/negative label may be associated with positive/negative score
            var first = rawPrediction.First();
            Console.WriteLine("The first example actual label is {0}. The trained model assigns it a score {1}.",
                first.Label /*true*/, first.Score /*around 3*/);
        }
    }
}
