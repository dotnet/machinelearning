using System;
using Microsoft.ML;
using Microsoft.ML.SamplesUtils;

namespace Samples.Dynamic
{
    /// This example demonstrates the use of the ValueToKeyMappingEstimator, by mapping KeyType values to the original strings. 
    /// For more on ML.NET KeyTypes see: https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewTypeSystem.md#key-types 
    /// It is possible to have multiple values map to the same category.

    public class MapKeyToValueMultiColumn
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

            // Create a pipeline. 
            var pipeline =
                    // Convert the string labels into key types.
                    mlContext.Transforms.Conversion.MapValueToKey("Label")
                    // Apply StochasticDualCoordinateAscent multiclass trainer.
                    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy());

            // Train the model and do predictions on same data set. 
            // Typically predictions would be in a different, validation set. 
            var dataWithPredictions = pipeline.Fit(dataView).Transform(dataView);

            // at this point, the Label colum is tranformed from strings, to DataViewKeyType and
            // the transformation has added the PredictedLabel column, with 
            var newPipeline = mlContext.Transforms.Conversion.MapKeyToValue(new[] 
            {
                new InputOutputColumnPair("LabelOriginalValue","Label"),
                new InputOutputColumnPair("PredictedLabelOriginalValue","PredictedLabel")
            });

            var transformedData = newPipeline.Fit(dataWithPredictions).Transform(dataWithPredictions);

            var values = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, reuseRowObject: false);

            // Printing the columns of the transformed data. 
            Console.WriteLine($" Label   LabelOriginalValue   PredictedLabel   PredictedLabelOriginalValue");
            foreach (var row in values)
                Console.WriteLine($"{row.Label}\t\t{row.LabelOriginalValue}\t\t\t{row.PredictedLabel}\t\t\t{row.PredictedLabelOriginalValue}");

            // Label LabelOriginalValue   PredictedLabel   PredictedLabelOriginalValue
            //  1           AA                  2                   BB
            //  1           AA                  1                   AA
            //  4           DD                  4                   DD
            //  2           BB                  2                   BB
            //  1           AA                  1                   AA
            //  1           AA                  1                   AA
            //  1           AA                  1                   AA
            //  2           BB                  2                   BB

        }
        private class TransformedData
        {
            public uint Label { get; set; }
            public uint PredictedLabel { get; set; }
            public string LabelOriginalValue { get; set; }
            public string PredictedLabelOriginalValue { get; set; }
        }
    }
}
