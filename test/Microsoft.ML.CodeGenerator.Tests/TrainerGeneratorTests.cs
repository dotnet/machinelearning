// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.CodeGenerator.CSharp;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Xunit;
using Xunit.Abstractions;

namespace mlnet.Tests
{
    public class TrainerGeneratorTests : BaseTestClass
    {
        public TrainerGeneratorTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void CultureInvariantTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"LearningRate", 0.1f },
                {"NumberOfLeaves", 1 },
            };
            PipelineNode node = new PipelineNode("LightGbmBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });

            //Set culture to deutsch.
            var currentCulture = Thread.CurrentThread.CurrentCulture;
            Thread.CurrentThread.CurrentCulture = new CultureInfo("de-DE");

            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();

            Thread.CurrentThread.CurrentCulture = currentCulture;
            string expectedTrainerString = "LightGbm(learningRate:0.1f,numberOfLeaves:1,labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void LightGbmBinaryBasicTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"LearningRate", 0.1f },
                {"NumberOfLeaves", 1 },
            };
            PipelineNode node = new PipelineNode("LightGbmBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LightGbm(learningRate:0.1f,numberOfLeaves:1,labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void LightGbmBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"LearningRate", 0.1f },
                {"NumLeaves", 1 },
                {"UseSoftmax", true }
            };
            PipelineNode node = new PipelineNode("LightGbmBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LightGbm(new LightGbmBinaryTrainer.Options(){LearningRate=0.1f,NumLeaves=1,UseSoftmax=true,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            string expectedUsings = "using Microsoft.ML.Trainers.LightGbm;\r\n";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void SymbolicSgdLogisticRegressionBinaryBasicTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("SymbolicSgdLogisticRegressionBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "SymbolicSgdLogisticRegression(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void SymbolicSgdLogisticRegressionBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"LearningRate", 0.1f },
            };
            PipelineNode node = new PipelineNode("SymbolicSgdLogisticRegressionBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "SymbolicSgdLogisticRegression(new SymbolicSgdLogisticRegressionBinaryTrainer.Options(){LearningRate=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void SgdCalibratedBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("SgdCalibratedBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "SgdCalibrated(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void SgdCalibratedBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"Shuffle", true },
            };
            PipelineNode node = new PipelineNode("SgdCalibratedBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "SgdCalibrated(new SgdCalibratedTrainer.Options(){Shuffle=true,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void SdcaLogisticRegressionBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("SdcaLogisticRegressionBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "SdcaLogisticRegression(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void SdcaLogisticRegressionBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"BiasLearningRate", 0.1f },
            };
            PipelineNode node = new PipelineNode("SdcaLogisticRegressionBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options(){BiasLearningRate=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void SdcaMaximumEntropyMultiBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("SdcaMaximumEntropyMulti", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "SdcaMaximumEntropy(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void SdcaMaximumEntropyMultiAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"BiasLearningRate", 0.1f },
            };
            PipelineNode node = new PipelineNode("SdcaMaximumEntropyMulti", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "SdcaMaximumEntropy(new SdcaMaximumEntropyMulticlassTrainer.Options(){BiasLearningRate=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void SdcaRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("SdcaRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "Sdca(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void SdcaRegressionAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"BiasLearningRate", 0.1f },
            };
            PipelineNode node = new PipelineNode("SdcaRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "Sdca(new SdcaRegressionTrainer.Options(){BiasLearningRate=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void MatrixFactorizationBasicTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("MatrixFactorization", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "MatrixFactorization(labelColumnName:\"Label\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);
        }

        [Fact]
        public void MatrixFactorizationAdvancedTest()
        {
            var context = new MLContext();
            var elementProperties = new Dictionary<string, object>()
            {
                {"MatrixColumnIndexColumnName","userId" },
                {"MatrixRowIndexColumnName","movieId" },
                {"LabelColumnName","rating" },
                {nameof(MatrixFactorizationTrainer.Options.NumberOfIterations), 10 },
                {nameof(MatrixFactorizationTrainer.Options.LearningRate), 0.01f },
                {nameof(MatrixFactorizationTrainer.Options.ApproximationRank), 8 },
                {nameof(MatrixFactorizationTrainer.Options.Lambda), 0.01f },
                {nameof(MatrixFactorizationTrainer.Options.LossFunction), MatrixFactorizationTrainer.LossFunctionType.SquareLossRegression },
                {nameof(MatrixFactorizationTrainer.Options.Alpha), 1f },
                {nameof(MatrixFactorizationTrainer.Options.C), 0.00001f },
            };
            PipelineNode node = new PipelineNode("MatrixFactorization", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "MatrixFactorization(new MatrixFactorizationTrainer.Options(){MatrixColumnIndexColumnName=\"userId\",MatrixRowIndexColumnName=\"movieId\",LabelColumnName=\"rating\",NumberOfIterations=10,LearningRate=0.01f,ApproximationRank=8,Lambda=0.01f,LossFunction=MatrixFactorizationTrainer.LossFunctionType.SquareLossRegression,Alpha=1f,C=1E-05f})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(new string[] { "using Microsoft.ML.Trainers;\r\n" }, actual.Item2);
        }

        [Fact]
        public void LbfgsPoissonRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("LbfgsPoissonRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LbfgsPoissonRegression(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void LbfgsPoissonRegressionAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"MaximumNumberOfIterations", 1 },
            };
            PipelineNode node = new PipelineNode("LbfgsPoissonRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "LbfgsPoissonRegression(new LbfgsPoissonRegressionTrainer.Options(){MaximumNumberOfIterations=1,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void OlsRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("OlsRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "Ols(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void OlsRegressionAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"L2Regularization", 0.1f },
            };
            PipelineNode node = new PipelineNode("OlsRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "Ols(new OlsTrainer.Options(){L2Regularization=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void OnlineGradientDescentRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("OnlineGradientDescentRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "OnlineGradientDescent(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void OnlineGradientDescentRegressionAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"RecencyGainMulti", true },
            };
            PipelineNode node = new PipelineNode("OnlineGradientDescentRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "OnlineGradientDescent(new OnlineGradientDescentTrainer.Options(){RecencyGainMulti=true,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void LbfgsLogisticRegressionBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("LbfgsLogisticRegressionBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LbfgsLogisticRegression(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void LbfgsLogisticRegressionBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"DenseOptimizer", true },
            };
            PipelineNode node = new PipelineNode("LbfgsLogisticRegressionBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "LbfgsLogisticRegression(new LbfgsLogisticRegressionBinaryTrainer.Options(){DenseOptimizer=true,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void LbfgsMaximumEntropyMultiMultiBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("LbfgsMaximumEntropyMulti", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LbfgsMaximumEntropy(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void LbfgsMaximumEntropyMultiAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"DenseOptimizer", true },
            };
            PipelineNode node = new PipelineNode("LbfgsMaximumEntropyMulti", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "LbfgsMaximumEntropy(new LbfgsMaximumEntropyMulticlassTrainer.Options(){DenseOptimizer=true,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void LinearSvmBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("LinearSvmBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LinearSvm(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void LinearSvmBinaryParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"NoBias", true },
            };
            PipelineNode node = new PipelineNode("LinearSvmBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n ";
            string expectedTrainerString = "LinearSvm(new LinearSvmTrainer.Options(){NoBias=true,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }


        [Fact]
        public void FastTreeTweedieRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastTreeTweedieRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastTreeTweedie(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void FastTreeTweedieRegressionAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"Shrinkage", 0.1f },
            };
            PipelineNode node = new PipelineNode("OnlineGradientDescentRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "OnlineGradientDescent(new OnlineGradientDescentTrainer.Options(){Shrinkage=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }


        [Fact]
        public void FastTreeRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastTreeRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastTree(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void FastTreeRegressionAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"Shrinkage", 0.1f },
            };
            PipelineNode node = new PipelineNode("FastTreeRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers.FastTree;\r\n";
            string expectedTrainerString = "FastTree(new FastTreeRegressionTrainer.Options(){Shrinkage=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }


        [Fact]
        public void FastTreeBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastTreeBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastTree(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void FastTreeBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"Shrinkage", 0.1f },
            };
            PipelineNode node = new PipelineNode("FastTreeBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers.FastTree;\r\n";
            string expectedTrainerString = "FastTree(new FastTreeBinaryTrainer.Options(){Shrinkage=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }


        [Fact]
        public void FastForestRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastForestRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastForest(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void FastForestRegressionAdvancedParameterTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"Shrinkage", 0.1f },
            };
            PipelineNode node = new PipelineNode("FastForestRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers.FastTree;\r\n";
            string expectedTrainerString = "FastForest(new FastForestRegression.Options(){Shrinkage=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }


        [Fact]
        public void FastForestBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastForestBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastForest(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void FastForestBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"Shrinkage", 0.1f },
            };
            PipelineNode node = new PipelineNode("FastForestBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers.FastTree;\r\n";
            string expectedTrainerString = "FastForest(new FastForestClassification.Options(){Shrinkage=0.1f,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }


        [Fact]
        public void AveragedPerceptronBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("AveragedPerceptronBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "AveragedPerceptron(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);

        }

        [Fact]
        public void AveragedPerceptronBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"Shuffle", true },
            };
            PipelineNode node = new PipelineNode("AveragedPerceptronBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n ";
            string expectedTrainerString = "AveragedPerceptron(new AveragedPerceptronTrainer.Options(){Shuffle=true,LabelColumnName=\"Label\",FeatureColumnName=\"Features\"})";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Equal(expectedUsings, actual.Item2[0]);

        }

        [Fact]
        public void ImageClassificationTrainerBasicTest()
        {
            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("ImageClassification", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "ImageClassification(LabelColumnName:\"Label\",FeatureColumnName:\"Features\")";
            Assert.Equal(expectedTrainerString, actual.Item1);
            Assert.Null(actual.Item2);
        }
    }
}
