// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers.Online;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void OnlineLinearWorkout()
        {
            var dataPath = GetDataPath("breast-cancer.txt");

            var regressionData = TextLoader.CreateReader(ML, ctx => (Label: ctx.LoadFloat(0), Features: ctx.LoadFloat(1, 10)))
                .Read(dataPath);

            var regressionPipe = regressionData.MakeNewEstimator()
                .Append(r => (r.Label, Features: r.Features.Normalize()));

            var regressionTrainData = regressionPipe.Fit(regressionData).Transform(regressionData).AsDynamic;

            var ogdTrainer = new OnlineGradientDescentTrainer(ML, "Label", "Features");
            TestEstimatorCore(ogdTrainer, regressionTrainData);
            var ogdModel = ogdTrainer.Fit(regressionTrainData);
            ogdTrainer.Train(regressionTrainData, ogdModel.Model);

            var binaryData = TextLoader.CreateReader(ML, ctx => (Label: ctx.LoadBool(0), Features: ctx.LoadFloat(1, 10)))
               .Read(dataPath);

            var binaryPipe = binaryData.MakeNewEstimator()
                .Append(r => (r.Label, Features: r.Features.Normalize()));

            var binaryTrainData = binaryPipe.Fit(binaryData).Transform(binaryData).AsDynamic;
            var apTrainer = new AveragedPerceptronTrainer(ML, "Label", "Features", lossFunction: new HingeLoss(), advancedSettings: s =>
            {
                s.LearningRate = 0.5f;
            });
            TestEstimatorCore(apTrainer, binaryTrainData);

            var apModel = apTrainer.Fit(binaryTrainData);
            apTrainer.Train(binaryTrainData, apModel.Model);

            var svmTrainer = new LinearSvmTrainer(ML, "Label", "Features");
            TestEstimatorCore(svmTrainer, binaryTrainData);

            var svmModel = svmTrainer.Fit(binaryTrainData);
            svmTrainer.Train(binaryTrainData, apModel.Model);

            Done();

        }


        [Fact]
        public void OnlineLinearWorkout1()
        {
            // load data
            var textLoader = new TextLoader(ML,
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
            var trainDataPath = "F:/tlc/test/adult.train";
            var trainData = textLoader.Read(trainDataPath);
            var validationData = textLoader.Read(trainDataPath);
            var testData = textLoader.Read(trainDataPath);

            // preprocess
            var preprocessorEstimator = ML.Transforms.Categorical.OneHotEncoding("Workclass", "Workclass")
                .Append(ML.Transforms.Categorical.OneHotEncoding("Education", "Education"))
                .Append(ML.Transforms.Categorical.OneHotEncoding("MaritalStatus", "MaritalStatus"))
                .Append(ML.Transforms.Categorical.OneHotEncoding("Occupation", "Occupation"))
                .Append(ML.Transforms.Categorical.OneHotEncoding("Relationship", "Relationship"))
                .Append(ML.Transforms.Categorical.OneHotEncoding("Race", "Race"))
                .Append(ML.Transforms.Categorical.OneHotEncoding("Sex", "Sex"))
                .Append(ML.Transforms.Categorical.OneHotEncoding("NativeCountry", "NativeCountry"))
                .Append(ML.Transforms.Concatenate(DefaultColumnNames.Features,
                    "Age", "Workclass", "Fnlwgt", "Education", "EducationNum", "MaritalStatus", "Occupation", "Relationship",
                    "Race", "Sex", "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry"));
            // train model
            var trainer = ML.BinaryClassification.Trainers.LinearSupportVectorMachines();
            var estimatorChain = preprocessorEstimator.Append(trainer);
            var model = estimatorChain.Fit(trainData);
        }
    }
}
