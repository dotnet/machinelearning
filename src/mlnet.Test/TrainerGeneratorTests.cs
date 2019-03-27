using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.CodeGenerator.CSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mlnet.Test
{
    /****************************
     * TODO : Add all trainer tests :
     * **************************/
    [TestClass]
    public class TrainerGeneratorTests
    {
        [TestMethod]
        public void LightGbmBinaryBasicTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"LearningRate", 0.1f },
                {"NumLeaves", 1 },
            };
            PipelineNode node = new PipelineNode("LightGbmBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LightGbm(learningRate:0.1f,numLeaves:1,labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
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
            string expectedTrainerString = "LightGbm(new Options(){LearningRate=0.1f,NumLeaves=1,UseSoftmax=true,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            string expectedUsings = "using Microsoft.ML.LightGBM;\r\n";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void SymSgdBinaryBasicTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("SymSgdBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "SymbolicStochasticGradientDescent(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
        public void SymSgdBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"LearningRate", 0.1f },
            };
            PipelineNode node = new PipelineNode("SymSgdBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers.HalLearners;\r\n";
            string expectedTrainerString = "SymbolicStochasticGradientDescent(new SymSgdClassificationTrainer.Options(){LearningRate=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void StochasticGradientDescentBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("StochasticGradientDescentBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "StochasticGradientDescent(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
        public void StochasticGradientDescentBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"Shuffle", true },
            };
            PipelineNode node = new PipelineNode("StochasticGradientDescentBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "StochasticGradientDescent(new SgdBinaryTrainer.Options(){Shuffle=true,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void SDCABinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("SdcaBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "StochasticDualCoordinateAscent(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
        public void SDCABinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"BiasLearningRate", 0.1f },
            };
            PipelineNode node = new PipelineNode("SdcaBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "StochasticDualCoordinateAscent(new SdcaBinaryTrainer.Options(){BiasLearningRate=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void SDCAMultiBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("SdcaMulti", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "StochasticDualCoordinateAscent(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
        public void SDCAMultiAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"BiasLearningRate", 0.1f },
            };
            PipelineNode node = new PipelineNode("SdcaMulti", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "StochasticDualCoordinateAscent(new SdcaMultiClassTrainer.Options(){BiasLearningRate=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void SDCARegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("SdcaRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "StochasticDualCoordinateAscent(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
        public void SDCARegressionAdvancedParameterTest()
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
            string expectedTrainerString = "StochasticDualCoordinateAscent(new SdcaRegressionTrainer.Options(){BiasLearningRate=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void PoissonRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("PoissonRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "PoissonRegression(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
        public void PoissonRegressionAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"MaxIterations", 1 },
            };
            PipelineNode node = new PipelineNode("PoissonRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "PoissonRegression(new PoissonRegression.Options(){MaxIterations=1,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void OrdinaryLeastSquaresRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("OrdinaryLeastSquaresRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "OrdinaryLeastSquares(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
        public void OrdinaryLeastSquaresRegressionAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"L2Weight", 0.1f },
            };
            PipelineNode node = new PipelineNode("OrdinaryLeastSquaresRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers.HalLearners;\r\n";
            string expectedTrainerString = "OrdinaryLeastSquares(new OlsLinearRegressionTrainer.Options(){L2Weight=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void OnlineGradientDescentRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("OnlineGradientDescentRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "OnlineGradientDescent(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
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
            string expectedTrainerString = "OnlineGradientDescent(new OnlineGradientDescentTrainer.Options(){RecencyGainMulti=true,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void LogisticRegressionBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("LogisticRegressionBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LogisticRegression(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
        public void LogisticRegressionBinaryAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"DenseOptimizer", true },
            };
            PipelineNode node = new PipelineNode("LogisticRegressionBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "LogisticRegression(new LogisticRegression.Options(){DenseOptimizer=true,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void LogisticRegressionMultiBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("LogisticRegressionMulti", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LogisticRegression(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
        public void LogisticRegressionMultiAdvancedParameterTest()
        {

            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>()
            {
                {"DenseOptimizer", true },
            };
            PipelineNode node = new PipelineNode("LogisticRegressionMulti", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            var expectedUsings = "using Microsoft.ML.Trainers;\r\n";
            string expectedTrainerString = "LogisticRegression(new MulticlassLogisticRegression.Options(){DenseOptimizer=true,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }

        [TestMethod]
        public void LinearSvmBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("LinearSvmBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "LinearSupportVectorMachines(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
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
            string expectedTrainerString = "LinearSupportVectorMachines(new LinearSvmTrainer.Options(){NoBias=true,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }


        [TestMethod]
        public void FastTreeTweedieRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastTreeTweedieRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastTreeTweedie(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
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
            string expectedTrainerString = "OnlineGradientDescent(new OnlineGradientDescentTrainer.Options(){Shrinkage=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }


        [TestMethod]
        public void FastTreeRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastTreeRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastTree(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
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
            string expectedTrainerString = "FastTree(new FastTreeRegressionTrainer.Options(){Shrinkage=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }


        [TestMethod]
        public void FastTreeBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastTreeBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastTree(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
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
            string expectedTrainerString = "FastTree(new FastTreeBinaryClassificationTrainer.Options(){Shrinkage=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }


        [TestMethod]
        public void FastForestRegressionBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastForestRegression", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastForest(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
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
            string expectedTrainerString = "FastForest(new FastForestRegression.Options(){Shrinkage=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }


        [TestMethod]
        public void FastForestBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("FastForestBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "FastForest(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
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
            string expectedTrainerString = "FastForest(new FastForestClassification.Options(){Shrinkage=0.1f,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }


        [TestMethod]
        public void AveragedPerceptronBinaryBasicTest()
        {
            var context = new MLContext();

            var elementProperties = new Dictionary<string, object>();
            PipelineNode node = new PipelineNode("AveragedPerceptronBinary", PipelineNodeType.Trainer, default(string[]), default(string), elementProperties);
            Pipeline pipeline = new Pipeline(new PipelineNode[] { node });
            CodeGenerator codeGenerator = new CodeGenerator(pipeline, null, null);
            var actual = codeGenerator.GenerateTrainerAndUsings();
            string expectedTrainerString = "AveragedPerceptron(labelColumnName:\"Label\",featureColumnName:\"Features\")";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.IsNull(actual.Item2);

        }

        [TestMethod]
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
            string expectedTrainerString = "AveragedPerceptron(new AveragedPerceptronTrainer.Options(){Shuffle=true,LabelColumn=\"Label\",FeatureColumn=\"Features\"})";
            Assert.AreEqual(expectedTrainerString, actual.Item1);
            Assert.AreEqual(expectedUsings, actual.Item2[0]);

        }
    }
}
