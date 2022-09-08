using System;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var context = new MLContext(1);
            context.Log += (o, e) =>
            {
                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    Console.WriteLine(e.RawMessage);
                }
            };
            var trainPath = @"D:/large_csv.csv";
            var dataset = context.Data.LoadFromTextFile<ModelInput>(trainPath, ',', hasHeader: false);
            var experiment = context.Auto().CreateExperiment();
            var label = "Entry(Text)";
            var pipeline = context.Transforms.Conversion.MapValueToKey(label, label)
                                .Append(context.Auto().MultiClassification(label, featureColumnName: "_data", useFastTree: true, useFastForest: false, useLgbm: false, fastTreeOption: new CodeGen.FastTreeOption { DiskTranspose = true, }));

            experiment.SetDataset(context.Data.TrainTestSplit(dataset))
                    .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MacroAccuracy, label)
                    .SetPipeline(pipeline)
                    .SetTrainingTimeInSeconds(10000);

            experiment.Run();
            try
            {
                RecommendationExperiment.Run();
                Console.Clear();

                RegressionExperiment.Run();
                Console.Clear();

                BinaryClassificationExperiment.Run();
                Console.Clear();

                MulticlassClassificationExperiment.Run();
                Console.Clear();

                RankingExperiment.Run();
                Console.Clear();

                Cifar10.Run();
                Console.Clear();

                Console.WriteLine("Done");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception {ex}");
            }

            Console.ReadLine();
        }
    }

    class ModelInput
    {
        [LoadColumn(0), NoColumn]
        public string _data0 { get; set; }

        [LoadColumn(1), NoColumn]
        public float ignoreData1 { get; set; }

        [LoadColumn(2, 4205)]
        public float[] _data { get; set; }

        [LoadColumn(4206), NoColumn]//(4206,4208)]
        public float _ignoreData4206 { get; set; }
        [LoadColumn(4207), NoColumn]//(4206,4208)]
        public float _ignoreData4207 { get; set; }
        [LoadColumn(4208), NoColumn]//(4206,4208)]
        public float _ignoreData4208 { get; set; }

        [LoadColumn(4209), ColumnName("Entry(Text)")]
        public string _label { get; set; }
    }
}
