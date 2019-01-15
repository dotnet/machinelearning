using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;

namespace Samples
{
    public class BinaryClassification
    {
        public static void Run()
        {
            const string trainDataPath = @"C:\data\sample_train2.csv";
            const string validationDataPath = @"C:\data\sample_valid2.csv";
            const string testDataPath = @"C:\data\sample_test2.csv";

            var mlContext = new MLContext();

            // load data
            var textLoader = new TextLoader(mlContext,
                new TextLoader.Arguments()
                {
                    Separator = ",",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Age", DataKind.R4, 0),
                        new TextLoader.Column("Workclass", DataKind.TX, 1),
                        new TextLoader.Column("Fnlwgt", DataKind.R4, 2),
                        new TextLoader.Column("Education", DataKind.TX, 3),
                        new TextLoader.Column("EducationNum", DataKind.R4, 4),
                        new TextLoader.Column("MaritalStatus", DataKind.TX, 5),
                        new TextLoader.Column("Occupation", DataKind.TX, 6),
                        new TextLoader.Column("Relationship", DataKind.TX, 7),
                        new TextLoader.Column("Race", DataKind.TX, 8),
                        new TextLoader.Column("Sex", DataKind.TX, 9),
                        new TextLoader.Column("CapitalGain", DataKind.R4, 10),
                        new TextLoader.Column("CapitalLoss", DataKind.R4, 11),
                        new TextLoader.Column("HoursPerWeek", DataKind.R4, 12),
                        new TextLoader.Column("NativeCountry", DataKind.TX, 13),
                        new TextLoader.Column("Label", DataKind.Bool, 14),
                    }
                });

            var trainData = textLoader.Read(trainDataPath);
            var validationData = textLoader.Read(validationDataPath);
            var testData = textLoader.Read(testDataPath);

            //////// SDCA

            //// preprocess
            //var preprocessorEstimator = mlContext.Transforms.Categorical.OneHotEncoding("Workclass", "Workclass")
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding("Education", "Education"))
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding("MaritalStatus", "MaritalStatus"))
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding("Occupation", "Occupation"))
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding("Relationship", "Relationship"))
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding("Race", "Race"))
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding("Sex", "Sex"))
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding("NativeCountry", "NativeCountry"))
            //    .Append(mlContext.Transforms.Concatenate(DefaultColumnNames.Features,
            //        "Age", "Workclass", "Fnlwgt", "Education", "EducationNum", "MaritalStatus", "Occupation", "Relationship",
            //        "Race", "Sex", "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry"));

            //// train model
            //var trainer = mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent();
            //var estimatorChain = preprocessorEstimator.Append(trainer);
            //var model = estimatorChain.Fit(trainData);

            //////// AutoML

            // run AutoML & train model
            var autoMlResult = mlContext.BinaryClassification.AutoFit(trainData, "Label", validationData,  
                settings : new AutoFitSettings()
                {
                    StoppingCriteria = new ExperimentStoppingCriteria() { MaxIterations = 10 }
                });
            // get best AutoML model
            var model = autoMlResult.BestPipeline.Model;

            // run AutoML on test data
            var transformedOutput = model.Transform(testData);
            var results = mlContext.BinaryClassification.Evaluate(transformedOutput);
            Console.WriteLine($"Model Accuracy: {results.Accuracy}\r\n");

            // save model to disk
            var modelPath = $"Model.zip";
            using (var fs = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }
            ITransformer savedModel;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                savedModel = mlContext.Model.Load(stream);
            }

            // create a prediction engine from the loaded model
            var predFunction = savedModel.CreatePredictionEngine<UciAdultInput, UciAdultOutput>(mlContext);
            var prediction = predFunction.Predict(new UciAdultInput()
            {
                Age = 28,
                Workclass = "Local-gov",
                Fnlwgt = 336951,
                Education = "Assoc-acdm",
                EducationNum = 12,
                MaritalStatus = "Married-civ-spouse",
                Occupation = "Protective-serv",
                Relationship = "Husband",
                Race = "White",
                Sex = "Male",
                CapitalGain = 0,
                CapitalLoss = 0,
                HoursPerWeek = 40,
                NativeCountry = "United-States",
            });

            Console.WriteLine($"Predicted label: {prediction.PredictedLabel}");
            Console.WriteLine($"Predicted probability: {prediction.Probability}");

            Console.ReadLine();
        }

        public class UciAdultInput
        {
            public float Age;
            public string Workclass;
            public float Fnlwgt;
            public string Education;
            public float EducationNum;
            public string MaritalStatus;
            public string Occupation;
            public string Relationship;
            public string Race;
            public string Sex;
            public float CapitalGain;
            public float CapitalLoss;
            public float HoursPerWeek;
            public string NativeCountry;
            public bool Label;
        }

        public class UciAdultOutput
        {
            public float Probability;
            public bool PredictedLabel;
        }
    }
}
