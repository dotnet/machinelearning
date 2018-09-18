// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class TestConcurrency : BaseTestPredictors
    {
        private const string Category = "Multithreading";

        public TestConcurrency(ITestOutputHelper helper) : base(helper)
        {
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory(Category)]
        public void TestCVWithLRParallel()
        {
            TestParallelRun("CVWithLR", "CV", "LR {nt=2}", TestDatasets.breastCancer.trainFilename);
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory(Category)]
        public void TestBootstrapWithLRParallel()
        {
            var bc = TestDatasets.breastCancer;
            TestParallelRun("BootstrapWithLR", "traintest", "LR {nt=2}", bc.trainFilename, bc.testFilename);
        }

        private void TestParallelRun(string basePrefix, string command, string predictorWithArgs, string trainFile, string testFile = null)
        {
            string cmd = command + " seed=1 tr=" + predictorWithArgs + " data=" + GetDataPath(trainFile);
            if (!string.IsNullOrWhiteSpace(testFile))
                cmd += " test=" + GetDataPath(testFile);

            string consName = basePrefix + "-out.raw";
            string consOutPath = DeleteOutputPath(Category, consName);
            using (var writer = OpenWriter(consOutPath))
            using (var env = new ConsoleEnvironment(42, outWriter: writer, errWriter: writer))
            {
                int res = MainForTest(env, writer, cmd);
                if (res != 0)
                    Log("*** Predictor returned {0}", res);
            }

            var rpName = basePrefix + "-rp.txt";
            RunResultProcessorTest(Env, new string[] { consOutPath }, DeleteOutputPath(Category, rpName), null);
            CheckEqualityNormalized(Category, rpName);
            Done();
        }
    }
}
