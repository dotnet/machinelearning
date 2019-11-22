// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using Microsoft.ML.CLI.Commands;
using Microsoft.ML.CLI.Data;
using Xunit;

namespace mlnet.Tests
{
    public class CommandLineTests
    {
        [Fact]
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
                        .AddCommand(CommandDefinitions.AutoTrain(handler))
                        .UseDefaults()
                        .UseExceptionHandler((e, ctx) =>
                        {
                            Console.WriteLine(e.ToString());
                        })
                        .Build();

            var trainDataset = Path.GetTempFileName();
            var testDataset = Path.GetTempFileName();
            string[] args = new[] { "auto-train", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            File.Delete(trainDataset);
            File.Delete(testDataset);
            Assert.True(parsingSuccessful);
        }


        [Fact]
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
                        .AddCommand(CommandDefinitions.AutoTrain(handler))
                        .UseDefaults()
                        .UseExceptionHandler((e, ctx) =>
                        {
                            Console.WriteLine(e.ToString());
                        })
                        .Build();

            // Incorrect mltask test
            var trainDataset = Path.GetTempFileName();
            var testDataset = Path.GetTempFileName();

            //wrong value to ml-task
            string[] args = new[] { "auto-train", "--ml-task", "bad-value", "--train-dataset", trainDataset, "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            Assert.False(parsingSuccessful);

            // Incorrect invocation
            args = new[] { "auto-train", "binary-classification", "--train-dataset", trainDataset, "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            Assert.False(parsingSuccessful);

            // Non-existent file test
            args = new[] { "auto-train", "--ml-task", "binary-classification", "--train-dataset", "nonexistentfile.csv", "--label-column-name", "Label" };
            parser.InvokeAsync(args).Wait();
            Assert.False(parsingSuccessful);

            // No label column or index test
            args = new[] { "auto-train", "--ml-task", "binary-classification", "--train-dataset", trainDataset, "--test-dataset", testDataset };
            parser.InvokeAsync(args).Wait();
            File.Delete(trainDataset);
            File.Delete(testDataset);
            Assert.False(parsingSuccessful);
        }

        [Fact]
        public void TestCommandLineArgsValuesTest()
        {
            bool parsingSuccessful = false;
            var trainDataset = Path.GetTempFileName();
            var testDataset = Path.GetTempFileName();
            var validDataset = Path.GetTempFileName();
            var labelName = "Label";
            var name = "testname";
            var outputPath = Path.GetTempPath();
            var falseString = "false";

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<NewCommandSettings>(
                (opt) =>
                {
                    Assert.Equal("binary-classification", opt.MlTask);
                    Assert.Equal(opt.Dataset.FullName, trainDataset);
                    Assert.Equal(opt.TestDataset.FullName, testDataset);
                    Assert.Equal(opt.ValidationDataset.FullName, validDataset);
                    Assert.Equal(opt.LabelColumnName, labelName);
                    Assert.Equal(opt.MaxExplorationTime, (uint)5);
                    Assert.Equal(opt.Name, name);
                    Assert.Equal(opt.OutputPath.FullName, outputPath);
                    Assert.Equal(opt.HasHeader, bool.Parse(falseString));
                    parsingSuccessful = true;
                });

            var parser = new CommandLineBuilder()
                        // Parser
                        .AddCommand(CommandDefinitions.AutoTrain(handler))
                        .UseDefaults()
                        .UseExceptionHandler((e, ctx) =>
                        {
                            Console.WriteLine(e.ToString());
                        })
                        .Build();

            // Incorrect mltask test
            string[] args = new[] { "auto-train", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--validation-dataset", validDataset, "--test-dataset", testDataset, "--max-exploration-time", "5", "--name", name, "--output-path", outputPath, "--has-header", falseString };
            parser.InvokeAsync(args).Wait();
            File.Delete(trainDataset);
            File.Delete(testDataset);
            File.Delete(validDataset);
            Assert.True(parsingSuccessful);

        }

        [Fact]
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
                        .AddCommand(CommandDefinitions.AutoTrain(handler))
                        .UseDefaults()
                        .UseExceptionHandler((e, ctx) =>
                        {
                            Console.WriteLine(e.ToString());
                        })
                        .Build();

            // Incorrect arguments : specifying dataset and train-dataset
            string[] args = new[] { "auto-train", "--ml-task", "BinaryClassification", "--dataset", dataset, "--train-dataset", trainDataset, "--label-column-name", labelName, "--test-dataset", testDataset, "--max-exploration-time", "5" };
            parser.InvokeAsync(args).Wait();
            Assert.False(parsingSuccessful);

            // Incorrect arguments : specifying  train-dataset and not specifying test-dataset
            args = new[] { "auto-train", "--ml-task", "BinaryClassification", "--train-dataset", trainDataset, "--label-column-name", labelName, "--max-exploration-time", "5" };
            parser.InvokeAsync(args).Wait();
            Assert.False(parsingSuccessful);

            // Incorrect arguments : specifying  label column name and index
            args = new[] { "auto-train", "--ml-task", "BinaryClassification", "--train-dataset", trainDataset, "--label-column-name", labelName, "--label-column-index", "0", "--test-dataset", testDataset, "--max-exploration-time", "5" };
            parser.InvokeAsync(args).Wait();
            File.Delete(trainDataset);
            File.Delete(testDataset);
            File.Delete(dataset);
            Assert.False(parsingSuccessful);

        }

        [Fact]
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
                    Assert.Equal("binary-classification", opt.MlTask);
                    Assert.Equal(opt.Dataset.FullName, trainDataset);
                    Assert.Equal(opt.LabelColumnName, labelName);
                    Assert.Equal(opt.Cache, cache);
                    parsingSuccessful = true;
                });

            var parser = new CommandLineBuilder()
                        // Parser
                        .AddCommand(CommandDefinitions.AutoTrain(handler))
                        .UseDefaults()
                        .UseExceptionHandler((e, ctx) =>
                        {
                            Console.WriteLine(e.ToString());
                        })
                        .Build();

            // valid cache test
            string[] args = new[] { "auto-train", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--cache", cache };
            parser.InvokeAsync(args).Wait();
            Assert.True(parsingSuccessful);

            parsingSuccessful = false;

            cache = "off";
            // valid cache test
            args = new[] { "auto-train", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--cache", cache };
            parser.InvokeAsync(args).Wait();
            Assert.True(parsingSuccessful);

            parsingSuccessful = false;

            cache = "auto";
            // valid cache test
            args = new[] { "auto-train", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--cache", cache };
            parser.InvokeAsync(args).Wait();
            Assert.True(parsingSuccessful);

            parsingSuccessful = false;

            // invalid cache test
            args = new[] { "auto-train", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--cache", "blah" };
            parser.InvokeAsync(args).Wait();
            Assert.False(parsingSuccessful);

            File.Delete(trainDataset);
            File.Delete(testDataset);
        }

        [Fact]
        public void IgnoreColumnsArgumentTest()
        {
            bool parsingSuccessful = false;
            var trainDataset = Path.GetTempFileName();
            var testDataset = Path.GetTempFileName();
            var labelName = "Label";
            var ignoreColumns = "a,b,c";

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<NewCommandSettings>(
                (opt) =>
                {
                    Assert.Equal("binary-classification", opt.MlTask);
                    Assert.Equal(opt.Dataset.FullName, trainDataset);
                    Assert.Equal(opt.LabelColumnName, labelName);
                    Assert.True(opt.IgnoreColumns.SequenceEqual(new List<string>() { "a", "b", "c" }));
                    parsingSuccessful = true;
                });

            var parser = new CommandLineBuilder()
                        // Parser
                        .AddCommand(CommandDefinitions.AutoTrain(handler))
                        .UseDefaults()
                        .UseExceptionHandler((e, ctx) =>
                        {
                            Console.WriteLine(e.ToString());
                        })
                        .Build();

            // valid cache test
            string[] args = new[] { "auto-train", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--ignore-columns", ignoreColumns };
            parser.InvokeAsync(args).Wait();
            Assert.True(parsingSuccessful);

            parsingSuccessful = false;

            args = new[] { "auto-train", "--ml-task", "binary-classification", "--dataset", trainDataset, "--label-column-name", labelName, "--ignore-columns", "a b c" };
            parser.InvokeAsync(args).Wait();
            Assert.False(parsingSuccessful);

            File.Delete(trainDataset);
            File.Delete(testDataset);
        }
    }
}
