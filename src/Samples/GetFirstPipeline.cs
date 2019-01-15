using Microsoft.ML;
using Microsoft.ML.Auto;

namespace Samples
{
    public static class GetFirstPipeline
    {
        const string TrainDataPath = @"C:\data\sample_train2.csv";
        const string Label = "Label";

        public static void Run()
        {
            var context = new MLContext();
            var columnInference = context.Data.InferColumns(TrainDataPath, Label, true);
            var textLoader = context.Data.CreateTextReader(columnInference);
            var data = textLoader.Read(TrainDataPath);
            var pipeline = context.BinaryClassification.AutoFit(data, Label);
        }
    }
}
