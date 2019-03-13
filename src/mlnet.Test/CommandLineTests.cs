// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.IO;
using Microsoft.ML.CLI.Commands;
using Microsoft.ML.CLI.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mlnet.Test
{
    [TestClass]
    public class CommandLineTests
    {
        [TestMethod]
        public void TestMinimumCommandLineArgs()
        {
            bool parsingSuccessful = false;

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<NewCommandSettings>(
                (opt) =>
                {
                    parsingSuccessful = true;
                });

            var parser = new CommandLineBuilder()
                        // Parser
                        .AddCommand(CommandDefinitions.New(handler))
                        .UseDefaults()
                        .Build();

            var trainDataset = Path.GetTempFileName();
            var testDataset = Path.GetTempFileName();
            string[] args = new[] { "new", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            File.Delete(trainDataset);
            File.Delete(testDataset);
            Assert.IsTrue(parsingSuccessful);
        }


        [TestMethod]
        public void TestCommandLineArgsFailTest()
        {
            bool parsingSuccessful = false;

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<NewCommandSettings>(
                (opt) =>
                {
                    parsingSuccessful = true;
                });

            var parser = new CommandLineBuilder()
                        // parser
                        .AddCommand(CommandDefinitions.New(handler))
                        .UseDefaults()
                        .Build();

            // Incorrect mltask test
            var trainDataset = Path.GetTempFileName();
            var testDataset = Path.GetTempFileName();

            //wrong value to ml-task
            string[] args = new[] { "new", "--ml-task", "bad-value", "--train-dataset", trainDataset, "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

            // Incorrect invocation
            args = new[] { "new", "binary-classification", "--train-dataset", trainDataset, "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

            // Non-existent file test
            args = new[] { "new", "--ml-task", "binary-classification", "--train-dataset", "nonexistentfile.csv", "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

            // No label column or index test
            args = new[] { "new", "--ml-task", "binary-classification", "--train-dataset", trainDataset, "--test-dataset", testDataset };
            parser.InvokeAsync(args).Wait();
            File.Delete(trainDataset);
            File.Delete(testDataset);
            Assert.IsFalse(parsingSuccessful);
        }

        [TestMethod]
        public void TestCommandLineArgsValuesTest()
        {
            bool parsingSuccessful = false;
            var trainDataset = Path.GetTempFileName();
            var testDataset = Path.GetTempFileName();
            var validDataset = Path.GetTempFileName();
            var labelName = "Label";
            var name = "testname";
            var outputPath = ".";
            var falseString = "false";

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<NewCommandSettings>(
                (opt) =>
                {
                    parsingSuccessful = true;
                    Assert.AreEqual(opt.MlTask, "binary-classification");
                    Assert.AreEqual(opt.Dataset, trainDataset);
                    Assert.AreEqual(opt.TestDataset, testDataset);
                    Assert.AreEqual(opt.ValidationDataset, validDataset);
                    Assert.AreEqual(opt.LabelColumnName, labelName);
                    Assert.AreEqual(opt.MaxExplorationTime, 5);
                    Assert.AreEqual(opt.Name, name);
                    Assert.AreEqual(opt.OutputPath, outputPath);
                    Assert.AreEqual(opt.HasHeader, bool.Parse(falseString));
                });

            var parser = new CommandLineBuilder()
                        // Parser
                        .AddCommand(CommandDefinitions.New(handler))
                        .UseDefaults()
                        .Build();

            // Incorrect mltask test
            string[] args = new[] { "new", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--validation-dataset", validDataset, "--test-dataset", testDataset, "--max-exploration-time", "5", "--name", name, "--output-path", outputPath, "--has-header", falseString };
            parser.InvokeAsync(args).Wait();
            File.Delete(trainDataset);
            File.Delete(testDataset);
            File.Delete(validDataset);
            Assert.IsTrue(parsingSuccessful);

        }

        [TestMethod]
        public void TestCommandLineArgsMutuallyExclusiveArgsTest()
        {
            bool parsingSuccessful = false;
            var dataset = Path.GetTempFileName();
            var trainDataset = Path.GetTempFileName();
            var testDataset = Path.GetTempFileName();
            var labelName = "Label";

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<NewCommandSettings>(
                (opt) =>
                {
                    parsingSuccessful = true;
                });

            var parser = new CommandLineBuilder()
                        // Parser
                        .AddCommand(CommandDefinitions.New(handler))
                        .UseDefaults()
                        .Build();

            // Incorrect arguments : specifying dataset and train-dataset
            string[] args = new[] { "new", "--ml-task", "BinaryClassification", "--dataset", dataset, "--train-dataset", trainDataset, "--label-column-name", labelName, "--test-dataset", testDataset, "--max-exploration-time", "5" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

            // Incorrect arguments : specifying  train-dataset and not specifying test-dataset
            args = new[] { "new", "--ml-task", "BinaryClassification", "--train-dataset", trainDataset, "--label-column-name", labelName, "--max-exploration-time", "5" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

            // Incorrect arguments : specifying  label column name and index
            args = new[] { "new", "--ml-task", "BinaryClassification", "--train-dataset", trainDataset, "--label-column-name", labelName, "--label-column-index", "0", "--test-dataset", testDataset, "--max-exploration-time", "5" };
            parser.InvokeAsync(args).Wait();
            File.Delete(trainDataset);
            File.Delete(testDataset);
            File.Delete(dataset);
            Assert.IsFalse(parsingSuccessful);

        }

        [TestMethod]
        public void CacheArgumentTest()
        {
            bool parsingSuccessful = false;
            var trainDataset = Path.GetTempFileName();
            var testDataset = Path.GetTempFileName();
            var labelName = "Label";
            var cache = "on";

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<NewCommandSettings>(
                (opt) =>
                {
                    parsingSuccessful = true;
                    Assert.AreEqual(opt.MlTask, "binary-classification");
                    Assert.AreEqual(opt.Dataset, trainDataset);
                    Assert.AreEqual(opt.LabelColumnName, labelName);
                    Assert.AreEqual(opt.Cache, cache);
                });

            var parser = new CommandLineBuilder()
                        // Parser
                        .AddCommand(CommandDefinitions.New(handler))
                        .UseDefaults()
                        .Build();

            // valid cache test
            string[] args = new[] { "new", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--cache", cache };
            parser.InvokeAsync(args).Wait();
            Assert.IsTrue(parsingSuccessful);

            parsingSuccessful = false;

            cache = "off";
            // valid cache test
            args = new[] { "new", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--cache", cache };
            parser.InvokeAsync(args).Wait();
            Assert.IsTrue(parsingSuccessful);

            parsingSuccessful = false;

            cache = "auto";
            // valid cache test
            args = new[] { "new", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--cache", cache };
            parser.InvokeAsync(args).Wait();
            Assert.IsTrue(parsingSuccessful);

            parsingSuccessful = false;

            // invalid cache test
            args = new[] { "new", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--cache", "blah" };
            parser.InvokeAsync(args).Wait();
            Assert.IsFalse(parsingSuccessful);

            File.Delete(trainDataset);
            File.Delete(testDataset);
        }
    }
}
