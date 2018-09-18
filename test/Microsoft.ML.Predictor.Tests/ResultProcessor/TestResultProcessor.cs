// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Collections.Generic;
using System.Reflection;
using Xunit.Abstractions;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    // REVIEW: The data files need to be ported. Are these tests even needed?
    public sealed class TestResultProcessor : BaseTestPredictors
    {
        public static StreamWriter OutFile;
        public const string SubDirectory = "ResultProcessor";
        private const string TestDataPrefix = "Microsoft.ML.Runtime.RunTests.ResultProcessor.TestData.";
        private const string TestDataOutPath = @"ResultProcessor\TestData";

        public TestResultProcessor(ITestOutputHelper helper) : base(helper)
        {
        }

        // Worker method for running the tests
        private void RunTestCore(string name, string fileName, string[] testDataNames, string[] extraArgs = null)
        {
            string outPath = DeleteOutputPath(SubDirectory, fileName);
            string[] resultFilePaths = SaveResourcesAsFiles(testDataNames);

            RunResultProcessorTest(Env, resultFilePaths, outPath, extraArgs);
            CheckEqualityNormalized(SubDirectory, fileName);

            Done();
        }

        private string[] SaveResourcesAsFiles(string[] resourceNames)
        {
            List<string> result = new List<string>();
            for (int i = 0; i < resourceNames.Length; i++)
            {
                string filePath = DeleteOutputPath(TestDataOutPath, resourceNames[i]);
                string resourceName = TestDataPrefix + resourceNames[i];
                Stream resourceStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName);
                using (var reader = new StreamReader(resourceStream))
                {
                    File.WriteAllText(filePath, reader.ReadToEnd());
                }
                result.Add(filePath);
            }
            return result.ToArray();
        }

        /// <summary>
        /// A test for Processing the Result of a single Classfier
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Results Processor")]
        public void RPSingleClassifierTest()
        {
            List<string> testFiles = new List<string>();
            for (int i = 0; i < 16; i++)
            {
                testFiles.Add("SingleClassifier." + i.ToString() + ".out.txt");
            }

            RunTestCore("RPSingleClassifierTest",
                "singleclassifier-sample-output.txt", testFiles.ToArray());
        }

        /// <summary>
        /// A test for Processing the Result of a single Classfier
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Results Processor")]
        public void RPSingleClassifierTestWithSpace()
        {
            List<string> testFiles = new List<string>();
            for (int i = 0; i < 9; i++)
            {
                testFiles.Add("SingleClassifier.WithSpace." + i.ToString() + ".out.txt");
            }

            RunTestCore("RPSingleClassifierTestWithSpace",
                "singleclassifier-withspace-sample-output.txt", testFiles.ToArray());
        }

        /// <summary>
        /// A test for processing the result with empty liens
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Results Processor")]
        public void RPSingleClassifierTestWIthEmptyLines()
        {
            List<string> testFiles = new List<string>();
            for (int i = 0; i < 9; i++)
            {
                testFiles.Add("SingleClassifier.WithEmptyLines." + i.ToString() + ".out.txt");
            }

            RunTestCore("RPSingleClassifierTestWithEmptyLines",
                "singleclassifier-withemptylines-sample-output.txt", testFiles.ToArray());
        }

        /// <summary>
        /// A test for Processing the Result of a Multiple Classfiers
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Results Processor")]
        public void RPMultiClassifierTest()
        {
            List<string> testFiles = new List<string>();
            for (int i = 0; i < 21; i++)
            {
                testFiles.Add("MultiClassifier." + i.ToString() + ".out.txt");
            }

            RunTestCore("RPMultiClassifierTest",
                "multiclassifier-sample-output.txt", testFiles.ToArray());
        }

        /// <summary>
        /// A test for Processing the Result of a Multiple Classfiers
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Results Processor")]
        public void RPProcessClassifierRegressorTest()
        {
            List<string> testFiles = new List<string>();
            for (int i = 55; i < 58; i++)
            {
                testFiles.Add("ClassifierRegressor." + i.ToString() + ".out.txt");
            }
            for (int i = 64; i < 81; i++)
            {
                testFiles.Add("ClassifierRegressor." + i.ToString() + ".out.txt");
            }
            RunTestCore("RPProcessClassifierRegressorTest",
                "classifierregressor-sample-output.txt", testFiles.ToArray());
        }
    }
}
