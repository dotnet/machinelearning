using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Microsoft.ML.AutoML.Samples
{
    public static class Cifar10
    {
        public static string cifar10FolderPath = Path.Combine(Path.GetTempPath(), "cifar10");
        public static string cifar10ZipPath = Path.Combine(Path.GetTempPath(), "cifar10.zip");
        public static string cifar10Url = @"https://github.com/YoongiKim/CIFAR-10-images/archive/refs/heads/master.zip";
        public static string directory = "CIFAR-10-images-master";

        public static void Run()
        {
            var imageInputs = Directory.GetFiles(cifar10FolderPath)
                .Where(p => Path.GetExtension(p) == ".jpg")
                .Select(p => new ModelInput
                {
                    ImagePath = p,
                    Label = p.Split("\\").SkipLast(1).Last(),
                });

            var testImages = imageInputs.Where(f => f.ImagePath.Contains("test"));
            var trainImages = imageInputs.Where(f => f.ImagePath.Contains("train"));
            var context = new MLContext();
            context.Log += (e, o) =>
            {
                if (o.Source.StartsWith("AutoMLExperiment"))
                    Console.WriteLine(o.Message);
            };

            var trainDataset = context.Data.LoadFromEnumerable(trainImages);
            var testDataset = context.Data.LoadFromEnumerable(testImages);
            var experiment = context.Auto().CreateExperiment();
            var pipeline = context.Auto().Featurizer(trainDataset)
                            .Append(context.Auto().MultiClassification());

            experiment.SetDataset(trainDataset, testDataset)
                    .SetPipeline(pipeline)
                    .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MicroAccuracy)
                    .SetTrainingTimeInSeconds(200);

            var result = experiment.Run();
        }

        class ModelInput
        {
            public string ImagePath { get; set; }

            public string Label { get; set; }
        }
    }
}
