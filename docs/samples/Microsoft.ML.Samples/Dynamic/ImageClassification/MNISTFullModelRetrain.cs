using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class MNISTFullModelRetrain
    {
        /// <summary>
        /// Example full model retrain using the MNIST model in a ML.NET pipeline.
        /// </summary>

        private static string sourceDir = Directory.GetCurrentDirectory();

        // Represents the path to the machinelearning directory
        private static string mlDir = @"..\..\..\..\";

        public static void Example()
        {
            var mlContext = new MLContext(seed: 1);

            // Download training data into current directory and load into IDataView
            var trainData = DataDownload("Train-Tiny-28x28.txt", mlContext);

            // Download testing data into current directory and load into IDataView
            var testData = DataDownload("MNIST.Test.tiny.txt", mlContext);

            // Download the MNIST model and its variables into current directory
            ModelDownload();

            // Full model retrain pipeline
            var pipe = mlContext.Transforms.CopyColumns("Features", "Placeholder")
                    .Append(mlContext.Model.RetrainDnnModel(
                        inputColumnNames: new[] { "Features" },
                        outputColumnNames: new[] { "Prediction" },
                        labelColumnName: "TfLabel",
                        dnnLabel: "Label",
                        optimizationOperation: "MomentumOp",
                        lossOperation: "Loss",
                        modelPath: "mnist_conv_model",
                        metricOperation: "Accuracy",
                        epoch: 10,
                        learningRateOperation: "learning_rate",
                        learningRate: 0.01f,
                        batchSize: 20))
                    .Append(mlContext.Transforms.Concatenate("Features",
                        "Prediction"))
                    .AppendCacheCheckpoint(mlContext)
                    .Append(mlContext.MulticlassClassification.Trainers.LightGbm(
                        new Microsoft.ML.Trainers.LightGbm
                        .LightGbmMulticlassTrainer.Options()
                        {
                            LabelColumnName = "Label",
                            FeatureColumnName = "Features",
                            Seed = 1,
                            NumberOfThreads = 1,
                            NumberOfIterations = 1
                        }));

            var trainedModel = pipe.Fit(trainData);
            var predicted = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted);

            // Print out metrics
            Console.WriteLine();
            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}, " +
                $"macro-accuracy = {metrics.MacroAccuracy}");

            // Get one sample for the fully retrained model to predict on
            var sample = GetOneMNISTExample();

            // Create a prediction engine to predict on one sample
            var predictionEngine = mlContext.Model.CreatePredictionEngine<
                MNISTData, MNISTPrediction>(trainedModel);

            var prediction = predictionEngine.Predict(sample);

            // Print predicted labels
            Console.WriteLine("Predicted Labels: ");
            foreach (var pLabel in prediction.PredictedLabels)
            {
                Console.Write(pLabel + "  ");
            }

            // Clean up folder by deleting extra files made during retrain
            CleanUp("mnist_conv_model");
        }

        // Copies data from another location into current directory
        // and loads it into IDataView using a TextLoader
        private static IDataView DataDownload(string fileName, MLContext mlContext)
        {
            string dataPath = Path.Combine(mlDir, "test", "data", fileName);
            if (!File.Exists(fileName))
            {
                System.IO.File.Copy(dataPath, Path.Combine(sourceDir, fileName));
            }

            return mlContext.Data.CreateTextLoader(
                new[]
                 {
                     new TextLoader.Column("Label", DataKind.UInt32,
                         new[] { new TextLoader.Range(0) }, new KeyCount(10)),
                     new TextLoader.Column("TfLabel", DataKind.Int64, 0),
                     new TextLoader.Column("Placeholder", DataKind.Single,
                         new[] { new TextLoader.Range(1, 784) })
                 },
                 allowSparse: true
             ).Load(fileName);
        }

        // Copies MNIST model folder from another location into current directory
        private static void ModelDownload()
        {
            if (!Directory.Exists(Path.Combine(sourceDir, "mnist_conv_model")))
            {
                // The original path to the MNIST model
                var oldModel = Path.Combine(new[] { mlDir, "packages",
                    "microsoft.ml.tensorflow.testmodels", "0.0.11-test",
                    "contentfiles", "any", "any", "mnist_conv_model" });

                // Create a new folder in the current directory for the MNIST model
                string newModel = Directory.CreateDirectory(Path.Combine(sourceDir,
                    "mnist_conv_model")).FullName;

                // Copy the model into the new mnist_conv_model folder
                System.IO.File.Copy(Path.Combine(oldModel, "saved_model.pb"),
                    Path.Combine(newModel, "saved_model.pb"));

                // The original folder that the model variables are in.
                // Because the folder already exists, the "CreateDirectory" method
                // call creates a DirectoryInfo object for the existing folder 
                // rather than making a new directory.
                var oldVariables = Directory.CreateDirectory(Path.Combine(oldModel,
                    "variables"));

                // Create a new folder in the new mnist_conv_model directory to
                // store the model variables
                var newVariables = Directory.CreateDirectory(Path.Combine(newModel,
                    "variables"));

                // Get the files in the original variables folder
                var variableNames = oldVariables.GetFiles();

                foreach (var vName in variableNames)
                {
                    // Copy each file from the original variables folder into the
                    // new variables folder
                    System.IO.File.Copy(vName.FullName, Path.Combine(
                        newVariables.FullName, vName.Name));
                }

            }
        }
        public class MNISTData
        {
            public long Label;

            [VectorType(784)]
            public float[] Placeholder;
        }

        public class MNISTPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }

        // Returns one sample
        private static MNISTData GetOneMNISTExample()
        {
            return new MNISTData()
            {
                Placeholder = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126,
                136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172,
                253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238,
                253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56,
                39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253,
                253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43,
                154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
                190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81,
                240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130,
                183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221,
                253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219,
                253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133,
                11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136,
                253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0 }
            };
        }

        // Deletes extra variable folders produced during retrain
        private static void CleanUp(string model_location)
        {
            var directories = Directory.GetDirectories(model_location,
                "variables-*");
            if (directories != null && directories.Length > 0)
            {
                var varDir = Path.Combine(model_location, "variables");
                if (Directory.Exists(varDir))
                    Directory.Delete(varDir, true);
                Directory.Move(directories[0], varDir);
            }
        }
    }
}