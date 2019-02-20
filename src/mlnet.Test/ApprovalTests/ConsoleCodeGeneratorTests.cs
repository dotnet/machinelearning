using System.Collections.Generic;
using System.IO;
using ApprovalTests;
using ApprovalTests.Reporters;
using Microsoft.ML;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.CodeGenerator.CSharp;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mlnet.Test
{
    [TestClass]
    [UseReporter(typeof(DiffReporter))]
    public class ConsoleCodeGeneratorTests
    {
        private Pipeline pipeline;
        private ColumnInferenceResults columnInference = default;

        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        public void GeneratedTrainCodeTest()
        {
            (Pipeline pipeline,
            ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorOptions()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = new FileInfo("x:\\dummypath\\dummy_train.csv"),
                TestDataset = new FileInfo("x:\\dummypath\\dummy_test.csv")

            });

            (string trainCode, string projectCode, string helperCode) = consoleCodeGen.GenerateCode();

            Approvals.Verify(trainCode);

        }

        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        public void GeneratedProjectCodeTest()
        {
            (Pipeline pipeline,
            ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorOptions()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = new FileInfo("x:\\dummypath\\dummy_train.csv"),
                TestDataset = new FileInfo("x:\\dummypath\\dummy_test.csv")

            });

            (string trainCode, string projectCode, string helperCode) = consoleCodeGen.GenerateCode();

            Approvals.Verify(projectCode);

        }

        [TestMethod]
        [UseReporter(typeof(DiffReporter))]
        public void GeneratedHelperCodeTest()
        {
            (Pipeline pipeline,
            ColumnInferenceResults columnInference) = GetMockedPipelineAndInference();

            var consoleCodeGen = new CodeGenerator(pipeline, columnInference, new CodeGeneratorOptions()
            {
                MlTask = TaskKind.BinaryClassification,
                OutputBaseDir = null,
                OutputName = "MyNamespace",
                TrainDataset = new FileInfo("x:\\dummypath\\dummy_train.csv"),
                TestDataset = new FileInfo("x:\\dummypath\\dummy_test.csv")

            });

            (string trainCode, string projectCode, string helperCode) = consoleCodeGen.GenerateCode();

            Approvals.Verify(helperCode);

        }

        private (Pipeline, ColumnInferenceResults) GetMockedPipelineAndInference()
        {
            if (pipeline == null)
            {
                MLContext context = new MLContext();
                // same learners with different hyperparams
                var hyperparams1 = new Microsoft.ML.Auto.ParameterSet(new List<Microsoft.ML.Auto.IParameterValue>() { new LongParameterValue("NumLeaves", 2) });
                var hyperparams2 = new Microsoft.ML.Auto.ParameterSet(new List<Microsoft.ML.Auto.IParameterValue>() { new LongParameterValue("NumLeaves", 6) });
                var trainer1 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), hyperparams1);
                var trainer2 = new SuggestedTrainer(context, new LightGbmBinaryExtension(), hyperparams2);
                var transforms1 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var transforms2 = new List<SuggestedTransform>() { ColumnConcatenatingExtension.CreateSuggestedTransform(context, new[] { "In" }, "Out") };
                var inferredPipeline1 = new SuggestedPipeline(transforms1, trainer1);
                var inferredPipeline2 = new SuggestedPipeline(transforms2, trainer2);

                this.pipeline = inferredPipeline1.ToPipeline();
                var textLoaderArgs = new TextLoader.Arguments()
                {
                    Column = new[] {
                new TextLoader.Column("Label", DataKind.BL, 0),
                new TextLoader.Column("col1", DataKind.R4, 1),
                new TextLoader.Column("col2", DataKind.R4, 0),
                new TextLoader.Column("col3", DataKind.Text, 0),
                new TextLoader.Column("col4", DataKind.I4, 0),
                new TextLoader.Column("col5", DataKind.U4, 0),
            },
                    AllowQuoting = true,
                    AllowSparse = true,
                    HasHeader = true,
                    Separators = new[] { ',' }
                };

                this.columnInference = new ColumnInferenceResults()
                {
                    TextLoaderArgs = textLoaderArgs
                };
            }
            return (pipeline, columnInference);
        }
    }
}
