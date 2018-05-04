// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.CommandLine;

namespace Microsoft.ML.Runtime.RunTests
{
    using TestLearners = TestLearnersBase;

#if OLD_TESTS // REVIEW: Do these add any value?
    public class TestCrossValidation : BaseTestPredictorsOld
    {
        private const string Category = "CrossValidation";

        [Fact, TestCategory(Category)]
        public void TestRandomBalancedFoldCreation()
        {
            ListInstances li = CreateInstancesWithNKeys(5);
            var foldCreator = new CVFoldCreation();
            var cmd = new TLCArguments();
            cmd.numFolds = 5;
            cmd.command = Command.CrossValidation;
            cmd.stratifyInstances = true;

            int[] folds = foldCreator.CreateFoldIndicesStratified(li, cmd, new Random(1));
            int[] expectedIndices = { 1, 0, 3, 4, 2 };
            for (int i = 0; i < folds.Length; i++)
                Assert.AreEqual<int>(folds[i], expectedIndices[i]);

            li = CreateInstancesWithNKeys(7);
            folds = foldCreator.CreateFoldIndicesStratified(li, cmd, new Random(1));
            expectedIndices = new int[] { 1, 0, 4, 1, 0, 2, 3 };
            for (int i = 0; i < folds.Length; i++)
                Assert.AreEqual<int>(folds[i], expectedIndices[i]);

            li = CreateInstancesWithNKeys(10);
            folds = foldCreator.CreateFoldIndicesStratified(li, cmd, new Random(1));
            expectedIndices = new int[] { 2, 1, 0, 3, 2, 4, 0, 4, 3, 1 };
            for (int i = 0; i < folds.Length; i++)
                Assert.AreEqual<int>(folds[i], expectedIndices[i]);

            Done();
        }

        private ListInstances CreateInstancesWithNKeys(int n)
        {
            var li = new ListInstances();
            var v = new WritableVector(new Float[] { 1, 2, 3 });
            for (int i = 0; i < n; i++)
            {
                li.Add(new Instance(v, 1, i.ToString()));
            }

            return li;
        }

        /// <summary>
        /// This tests that there's no deadlock in CrossValidaton when exception occurs
        /// </summary>
        [Fact, TestCategory(Category)]
        public void TestCrossValidationWithInvalidTester()
        {
            var argsStr = GetDataPath(TestDatasets.breastCancer.trainFilename)
                       + " /ev=MulticlassTester /o z.txt /threads=+ /disableTracking=+";

            var args = new TLCArguments();
            CmdParser.ParseArguments(argsStr, args);

            try
            {
                RunExperiments.Run(args);
            }
            catch (AggregateException ex)
            {
                Log("Caught expected exception: {0}", ex);
                Done();
                return;
            }

            Fail("Expected exception!");
            Done();
        }

        [Fact, TestCategory(Category)]
        public void TestCVWithTrainProportion()
        {
            Run_CV(
                TestLearners.logisticRegression_tlOld,
                TestDatasets.breastCancer.trainFilename,
                "CV-trainProportion.txt",
                new[] { "/tp=0.1" });
            Done();
        }
    }
#endif
}
