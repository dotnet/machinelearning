using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.IO;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI;
using Microsoft.ML.CLI.Commands;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mlnet.Test
{
    [TestClass]
    public class CommandLineTests
    {
        [TestMethod]
        public void TestCommandLineArgs()
        {
            bool parsingSuccessful = false;

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<FileInfo, FileInfo, FileInfo, TaskKind, string, uint, uint>(
                (trainDataset, validationDataset, testDataset, mlTask, labelColumnName, maxExplorationTime, labelColumnIndex) =>
                {
                    parsingSuccessful = true;
                });

            var parser = new CommandLineBuilder()
                        // Parser
                        .AddCommand(CommandDefinitions.New(handler))
                        .UseDefaults()
                        .Build();

            var file = Path.GetTempFileName();
            string[] args = new[] { "new", "--ml-task", "BinaryClassification", "--train-dataset", file, "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            File.Delete(file);
            Assert.IsTrue(parsingSuccessful);
        }


        [TestMethod]
        public void TestCommandLineArgsFailTest()
        {
            bool parsingSuccessful = false;

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<FileInfo, FileInfo, FileInfo, TaskKind, string, uint, uint>(
                (trainDataset, validationDataset, testDataset, mlTask, labelColumnName, maxExplorationTime, labelColumnIndex) =>
                {
                    parsingSuccessful = true;
                });

            var parser = new CommandLineBuilder()
                        // parser
                        .AddCommand(CommandDefinitions.New(handler))
                        .UseDefaults()
                        .Build();

            // Incorrect mltask test
            var file = Path.GetTempFileName();
            string[] args = new[] { "new", "--ml-task", "BinaryClass", "--train-dataset", file, "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

            // Incorrect invocation
            args = new[] { "new", "BinaryClassification", "--train-dataset", file, "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

            // Non-existent file test
            args = new[] { "new", "--ml-task", "BinaryClassification", "--train-dataset", "blah.csv", "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

            // No label column or index test
            args = new[] { "new", "--ml-task", "BinaryClassification", "--train-dataset", "blah.csv" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

        }

        [TestMethod]
        public void TestCommandLineArgsValuesTest()
        {
            bool parsingSuccessful = false;
            var file1 = Path.GetTempFileName();
            var file2 = Path.GetTempFileName();
            var labelName = "Label";

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<FileInfo, FileInfo, FileInfo, TaskKind, string, uint, uint>(
                (trainDataset, validationDataset, testDataset, mlTask, labelColumnName, maxExplorationTime, labelColumnIndex) =>
                {
                    parsingSuccessful = true;
                    Assert.AreEqual(mlTask, TaskKind.BinaryClassification);
                    Assert.AreEqual(trainDataset, file1);
                    Assert.AreEqual(testDataset, file2);
                    Assert.AreEqual(labelColumnName, labelName);
                    Assert.AreEqual(maxExplorationTime, 5);
                });

            var parser = new CommandLineBuilder()
                        // Parser
                        .AddCommand(CommandDefinitions.New(handler))
                        .UseDefaults()
                        .Build();

            // Incorrect mltask test
            string[] args = new[] { "new", "--ml-task", "BinaryClassification", "--train-dataset", file1, "--label-column-name", labelName, "--test-dataset", file2, "--max-exploration-time", "5" };
            parser.InvokeAsync(args).Wait();
            File.Delete(file1);
            File.Delete(file2);
            Assert.IsTrue(parsingSuccessful);

        }
    }
}
