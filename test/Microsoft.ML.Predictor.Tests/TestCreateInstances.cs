// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.PCA;

namespace Microsoft.ML.Runtime.Internal.Internallearn.Test
{
#if OLD_TESTS // REVIEW: Does any of this need ported?
    public class CreateInstancesTests : BaseTestBaseline
    {
        private const string Dir = "CreateInstances";

        [Fact, TestCategory("CreateInstances"), TestCategory("FeatureHandler")]
        public void TestCreateTextInstances()
        {
            TestDataset adult = TestDatasets.adult;
            string trainData = GetDataPath(adult.trainFilename);
            string testData = GetDataPath(adult.testFilename);

            var prefix = TestContext.TestName + "-";

            string outName = prefix + "Adult-Train.txt";
            string statsName = prefix + "Adult-Train.stats.txt";
            string outTestName = prefix + "Adult-Test.txt";
            string testStatsName = prefix + "Adult-Test.stats.txt";
            string outValidName = prefix + "Adult-Valid.txt";
            string validStatsName = prefix + "Adult-Valid.stats.txt";

            string outFile = DeleteOutputPath(Dir, outName);
            string statsFile = DeleteOutputPath(Dir, statsName);
            string outTestFile = DeleteOutputPath(Dir, outTestName);
            string testStatsFile = DeleteOutputPath(Dir, testStatsName);
            string outValidFile = DeleteOutputPath(Dir, outValidName);
            string validStatsFile = DeleteOutputPath(Dir, validStatsName);

            var argsStr =
                string.Format(
                    "/c=CreateInstances {0} /test={1} /valid={1} /cacheinst=- {2} " +
                    "/cifile={3} /cistatsfile={4} /citestfile={5} /citeststatsfile={6} /civalidfile={7} /civalidstatsfile={8}",
                    trainData, testData, adult.extraSettings,
                    outFile, statsFile, outTestFile, testStatsFile, outValidFile, validStatsFile);
            argsStr += " /writer TextInstanceWriter{/stats=+} /disableTracking=+";
            var args = new TLCArguments();
            Assert.IsTrue(CmdParser.ParseArguments(argsStr, args));

            RunExperiments.Run(args);

            CheckEquality(Dir, outName);
            CheckEquality(Dir, statsName);
            CheckEquality(Dir, outTestName);
            CheckEquality(Dir, testStatsName);
            CheckEquality(Dir, outValidName);
            CheckEquality(Dir, validStatsName);
            Done();
        }

        [Fact, TestCategory("CreateInstances"), TestCategory("FeatureHandler")]
        public void TestCreateTextInstancesConstant()
        {
            TestDataset breast = TestDatasets.breastCancerConst;
            string trainData = GetDataPath(breast.trainFilename);

            var prefix = TestContext.TestName + "-";

            string outName = prefix + "BreastCancer.txt";
            string statsName = prefix + "BreastCancer.stats.txt";

            string outFile = DeleteOutputPath(Dir, outName);
            string statsFile = DeleteOutputPath(Dir, statsName);

            var argsStr =
                string.Format(
                    "c=CreateInstances {0} {1} cifile={2} cistatsfile={3}",
                    trainData, breast.extraSettings, outFile, statsFile);
            argsStr += " writer=TextInstanceWriter{stats+} disableTracking+";
            var args = new TLCArguments();
            Assert.IsTrue(CmdParser.ParseArguments(argsStr, args));

            RunExperiments.Run(args);

            CheckEquality(Dir, outName);
            CheckEquality(Dir, statsName);
            Done();
        }

        [Fact, TestCategory("CreateInstances"), TestCategory("FeatureTransformer")]
        public void TestCreateTextInstancesWithNormalization()
        {
            TestDataset dataset = TestDatasets.mnistTiny28;
            string trainData = GetDataPath(dataset.trainFilename);
            string testData = GetDataPath(dataset.testFilename);

            var prefix = TestContext.TestName + "-";
            string outFile1 = DeleteOutputPath(Dir, prefix + "Norm-Separate-Train.txt");
            string outTestFile1 = DeleteOutputPath(Dir, prefix + "Norm-Separate-Test.txt");
            string outFile2 = DeleteOutputPath(Dir, prefix + "Norm-Trans-Train.txt");
            string outTestFile2 = DeleteOutputPath(Dir, prefix + "Norm-Trans-Test.txt");

            string transArgs = "inst=Trans{trans=RFF {rng=1}}";

            var argsStr1 = string.Format(
                    "/c=CreateInstances {0} /test={1} /norm=MinMaxNormalizer /{2} /cifile={3} /citestfile={4}",
                    trainData, testData, transArgs, outFile1, outTestFile1);
            var args1 = new TLCArguments();
            Assert.IsTrue(CmdParser.ParseArguments(argsStr1, args1));

            RunExperiments.Run(args1);

            var argsStr2 = string.Format(
                "/c=CreateInstances {0} /test={1} /inst Trans{{trans=MinMaxNormalizer {2}}} /cifile={3} /citestfile={4}",
                trainData, testData, transArgs, outFile2, outTestFile2);
            var args2 = new TLCArguments();
            Assert.IsTrue(CmdParser.ParseArguments(argsStr2, args2));

            RunExperiments.Run(args2);

            var instances1 = new TlcTextInstances(new TlcTextInstances.Arguments(), outFile1);
            var instances2 = new TlcTextInstances(new TlcTextInstances.Arguments(), outFile2);
            CompareInstances(instances1, instances2);

            var testInstances1 = new TlcTextInstances(new TlcTextInstances.Arguments(), outTestFile1);
            var testInstances2 = new TlcTextInstances(new TlcTextInstances.Arguments(), outTestFile2);
            CompareInstances(testInstances1, testInstances2);

            Done();
        }

        private void CompareInstances(TlcTextInstances instances1, TlcTextInstances instances2)
        {
            Assert.IsTrue(instances1.Schema.NumFeatures == instances2.Schema.NumFeatures, "mismatch on schema features");

            using (var e1 = instances1.GetEnumerator())
            using (var e2 = instances2.GetEnumerator())
            {
                for (; ; )
                {
                    bool b1 = e1.MoveNext();
                    bool b2 = e2.MoveNext();
                    Assert.IsTrue(b1 == b2, "different number of instances");
                    if (!b1)
                        break;
                    var inst1 = e1.Current;
                    var inst2 = e2.Current;
                    Assert.IsTrue(inst1.Label == inst2.Label, "mismatch on instance label");
                    Assert.IsTrue(inst1.NumFeatures == inst2.NumFeatures, "mismatch on number of features");
                    Assert.IsTrue(inst1.NumFeatures == instances1.Schema.NumFeatures, "mismatch on number of instance vs. schema features");
                    Assert.IsTrue(Utils.AreEqual(inst1.Features.Values, inst2.Features.Values), "mismatch on feature values");
                    Assert.IsTrue(Utils.AreEqual(inst1.Features.Indices, inst2.Features.Indices), "mismatch on feature indices");
                }
            }
        }

        [Fact, TestCategory("CreateInstances"), TestCategory("FeatureTransformer")]
        public void TestPcaTransform()
        {
            // Force Microsoft.ML.Runtime.PCA assembly to be loaded into the AppDomain so 
            // ReflectionUtils.FindClassCore does not return null when called by ReflectionUtils.CreateInstance
            Assert.AreEqual(typeof(PCAPredictor).Name, "PCAPredictor");

            string trainData = GetDataPath(TestDatasets.mnistTiny28.trainFilename);
            string fileName = TestContext.TestName + "-Train.txt";
            string outFile = DeleteOutputPath(Dir, fileName);

            const int rank = 3;
            string pcaTransformArgs = string.Format("/inst Trans{{trans=pca {{k={0} seed=1}}}}", rank);
            var argsStr1 = string.Format(
                    "/c CreateInstances {0} {1} /rs=1 /cifile={2}",
                    trainData,
                    pcaTransformArgs,
                    outFile);
            var args1 = new TLCArguments();
            Assert.IsTrue(CmdParser.ParseArguments(argsStr1, args1));

            RunExperiments.Run(args1);
            CheckEquality(Dir, fileName);

            // Verify the scales of the transformed features decrease with respect to the feature index
            TlcTextInstances outputInstances = new TlcTextInstances(new TlcTextInstances.Arguments(), outFile);
            Double[] l1norms = new Double[rank];
            foreach (Instance instance in outputInstances)
            {
                Assert.IsTrue(instance.Features.Count == rank);
                for (int i = 0; i < instance.Features.Values.Length; i++)
                    l1norms[i] += (instance.Features.Values[i] < 0 ? -instance.Features.Values[i] : instance.Features.Values[i]);
            }

            for (int i = 0; i < l1norms.Length - 1; i++)
            {
                Assert.IsTrue(l1norms[i] > l1norms[i + 1]);
            }

            Done();
        }

        [Fact, TestCategory("CreateInstances"), TestCategory("FeatureHandler")]
        public void TestFeatureHandlerIncorrectMapping()
        {
            string trainData = GetDataPath(TestDatasets.breastCancer.trainFilename);
            string dataModelFile = DeleteOutputPath(Dir, TestContext.TestName + "-data-model.zip");
            string ciFile = DeleteOutputPath(Dir, TestContext.TestName + "-ci.tsv");
            string argsString = string.Format(
                "/c CreateInstances {0} /inst Text{{text=1,2,3}} /m {1} /cifile {2}",
                trainData,
                dataModelFile,
                ciFile);
            var args = new TLCArguments();
            Assert.IsTrue(CmdParser.ParseArguments(argsString, args));
            RunExperiments.Run(args);

            string ciFailFile = DeleteOutputPath(Dir, TestContext.TestName + "-ci-fail.tsv");
            argsString = string.Format(
                "/c CreateInstances {0} /inst Text{{text=1,2}} /im {1} /cifile {2}",
                trainData,
                dataModelFile,
                ciFailFile);
            args = new TLCArguments();
            Assert.IsTrue(CmdParser.ParseArguments(argsString, args));
            try
            {
                RunExperiments.Run(args);
                Assert.Fail("Expected to throw with different input model format");
            }
            catch (Exception ex)
            {
                Assert.IsTrue(ex.GetBaseException() is InvalidOperationException);
            }

            Done();
        }
    }
#endif
}
