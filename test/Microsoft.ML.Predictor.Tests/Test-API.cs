// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Internal.Internallearn.Test
{
#if OLD_TESTS // REVIEW: Should any of this be ported?
    using TestLearners = TestLearnersBase;

    /// <summary>
    ///This is a test class for TestPredictorMainTest and is intended
    ///to contain all TestPredictorMainTest Unit Tests
    ///</summary>
    public sealed class TestAPI : BaseTestBaseline
    {
        /// <summary>
        ///A test for Main
        ///</summary>
        [Fact, TestCategory("Example"), TestCategory("FastRank")]
        [Ignore] // REVIEW: OVA no longer implements BulkPredict.
        public void MulticlassExampleTest()
        {
            string dataFilename = GetDataPath(TestDatasets.msm.trainFilename);

            ///*********  Training a model *******//
            // assume data is in memory in matrix/vector form. Sparse format also supported.
            Float[][] data;
            Float[] labels;
            // below just reads some actual data into these arrays
            PredictionUtil.ReadInstancesAsArrays(dataFilename, out data, out labels);

            // Create an Instances dataset.
            ListInstances instances = new ListInstances();
            for (int i = 0; i < data.Length; i++)
                instances.AddInst(data[i], labels[i]);
            instances.CopyMetadata(null);

            // Create a predictor and specify some non-default settings
            var args = new OVA.OldArguments();
            PredictionUtil.ParseArguments(args, "p=FR ps=iter:20");
            ITrainer<Instances, IPredictor<Instance, Float[]>> trainer = new OVA(args, new TrainHost(new Random(1), 0));

            // Train a predictor
            trainer.Train(instances);
            var predictor = trainer.CreatePredictor();

            ///*********  Several ways to save models. Only binary can be used to-reload in TLC. *******//

            // Save the model in internal binary format that can be used for loading it.
            string modelFilename = Path.GetTempFileName();
            PredictorUtils.Save(modelFilename, predictor, instances, null);

            // Save the model as a plain-text description
            string modelFilenameText = Path.GetTempFileName();
            PredictorUtils.SaveText(predictor, instances.Schema.FeatureNames, modelFilenameText);

            // Save the model in Bing's INI format
            string modelFilenameIni = Path.GetTempFileName();
            PredictorUtils.SaveIni(predictor, instances.Schema.FeatureNames, modelFilenameIni);

            ///*********  Loading and making predictions with a previously saved model *******//
            // Note:   there are several alternative ways to construct instances
            // E.g., see FactoryExampleTest  below that demonstrates named-feature : value pairs.

            // Load saved model
            IDataModel dataModel;
            IDataStats dataStats;
            var pred = PredictorUtils.LoadPredictor<Float[]>(out dataModel, out dataStats, modelFilename);

            // Get predictions for instances
            Float[][] predictions = new Float[instances.Count][];

            for (int i = 0; i < instances.Count; i++)
            {
                predictions[i] = pred.Predict(instances[i]);
            }

            // REVIEW: This looks like it wasn't doing what was expected - OVA didn't override
            // BulkPredict, so this wasn't using FastRank's BulkPredict.
            Float[][] bulkPredictions = ((IBulkPredictor<Instance, Instances, Float[], Float[][]>)pred).BulkPredict(instances);

            Assert.AreEqual(predictions.Length, bulkPredictions.Length);
            for (int i = 0; i < predictions.Length; i++)
            {
                Assert.AreEqual(predictions[i].Length, bulkPredictions[i].Length);
                for (int j = 0; j < predictions[i].Length; j++)
                {
                    Assert.AreEqual(predictions[i][j], bulkPredictions[i][j]);
                }
            }

            //test new testers
            {
                var results = new MulticlassTester(new MulticlassTesterArguments()).Test(pred, instances);

                // Get metric names and print them alongside numbers
                for (int i = 0; i < results.Length; i++)
                {
                    Log("{0,-30} {1}", results[i].Name, results[i].Value);
                }

                // sanity check vs. original predictor
                var results2 = new MulticlassTester(new MulticlassTesterArguments()).Test(predictor, instances);
                Assert.AreEqual(results.Length, results2.Length);
                for (int i = 0; i < results.Length; i++)
                {
                    Assert.AreEqual(results[i].Name, results2[i].Name);
                    Assert.AreEqual(results[i].Value, results2[i].Value);
                }
            }
            File.Delete(modelFilename);
            File.Delete(modelFilenameText);
            File.Delete(modelFilenameIni);

            Done();
        }

        /// <summary>
        ///A test for Main
        ///</summary>
        [Fact, TestCategory("Example"), TestCategory("FastRank")]
        public void SimpleExampleTest()
        {
            RunMTAThread(() =>
            {
                string dataFilename = GetDataPath(TestDatasets.msm.trainFilename);

                ///*********  Training a model *******//
                // assume data is in memory in matrix/vector form. Sparse format also supported.
                Float[][] data;
                Float[] labels;
                // below just reads some actual data into these arrays
                PredictionUtil.ReadInstancesAsArrays(dataFilename, out data, out labels);

                // Create an Instances dataset.
                ListInstances instances = new ListInstances();
                for (int i = 0; i < data.Length; i++)
                    instances.AddInst(data[i], labels[i]);
                instances.CopyMetadata(null);

                // Create a predictor and specify some non-default settings
                var sub = new SubComponent<ITrainer<Instances, IPredictor<Instance, Float>>, SignatureOldBinaryClassifierTrainer>(
                    "FastRank", "nl=5 lr =0.25 iter= 20");
                var trainer = sub.CreateInstance(new TrainHost(new Random(1), 0));

                // Train a predictor
                trainer.Train(instances);
                var predictor = trainer.CreatePredictor();

                ///*********  Several ways to save models. Only binary can be used to-reload in TLC. *******//

                // Save the model in internal binary format that can be used for loading it.
                string modelFilename = Path.GetTempFileName();
                PredictorUtils.Save(modelFilename, predictor, instances, null);

                // Save the model as a plain-text description
                string modelFilenameText = Path.GetTempFileName();
                PredictorUtils.SaveText(predictor, instances.Schema.FeatureNames, modelFilenameText);

                // Save the model in Bing's INI format
                string modelFilenameIni = Path.GetTempFileName();
                PredictorUtils.SaveIni(predictor, instances.Schema.FeatureNames, modelFilenameIni);

                ///*********  Loading and making predictions with a previously saved model *******//
                // Note:   there are several alternative ways to construct instances
                // E.g., see FactoryExampleTest  below that demonstrates named-feature : value pairs.

                // Load saved model
                IDataModel dataModel;
                IDataStats dataStats;
                var pred = PredictorUtils.LoadPredictor<Float>(out dataModel, out dataStats, modelFilename);
                var dp = pred as IDistributionPredictor<Instance, Float, Float>;

                // Get predictions for instances
                Float[] probPredictions = new Float[instances.Count];
                Float[] rawPredictions = new Float[instances.Count];
                Float[] rawPredictions1 = new Float[instances.Count];
                for (int i = 0; i < instances.Count; i++)
                {
                    probPredictions[i] = dp.PredictDistribution(instances[i], out rawPredictions[i]);
                    rawPredictions1[i] = dp.Predict(new Instance(data[i]));
                }

                Float[] bulkPredictions = ((IBulkPredictor<Instance, Instances, Float, Float[]>)pred).BulkPredict(instances);

                Assert.AreEqual(rawPredictions.Length, bulkPredictions.Length);
                Assert.AreEqual(rawPredictions.Length, rawPredictions1.Length);
                for (int i = 0; i < rawPredictions.Length; i++)
                    Assert.AreEqual(rawPredictions[i], bulkPredictions[i]);
                for (int i = 0; i < rawPredictions.Length; i++)
                    Assert.AreEqual(rawPredictions[i], rawPredictions1[i]);

                //test new testers
                {
                    var results = new ClassifierTester(new ProbabilityPredictorTesterArguments()).Test(pred, instances);

                    // Get metric names and print them alongside numbers
                    for (int i = 0; i < results.Length; i++)
                    {
                        Log("{0,-30} {1}", results[i].Name, results[i].Value);
                    }

                    // sanity check vs. original predictor
                    var results2 = new ClassifierTester(new ProbabilityPredictorTesterArguments()).Test(predictor, instances);
                    Assert.AreEqual(results.Length, results2.Length);
                    for (int i = 0; i < results.Length; i++)
                    {
                        Assert.AreEqual(results[i].Name, results2[i].Name);
                        Assert.AreEqual(results[i].Value, results2[i].Value);
                    }
                }
                File.Delete(modelFilename);
                File.Delete(modelFilenameText);
                File.Delete(modelFilenameIni);
            });
            Done();
        }

        /// <summary>
        ///A test for factory-style, feature-value instance production
        ///</summary>
        [Fact, TestCategory("Example"), TestCategory("FeatureHandler")]
        public void FactoryExampleTest()
        {
            var dataset = TestDatasets.adultText;
            string dataFilename = GetDataPath(dataset.trainFilename);
            string testDataFilename = GetDataPath(dataset.testFilename);

            ///*********  Training a model *******//
            string modelFilename = Path.GetTempFileName();
            TLCArguments cmd = new TLCArguments();
            Assert.IsTrue(CmdParser.ParseArguments(dataset.extraSettings, cmd));
            cmd.command = Command.Train;
            cmd.modelfile = modelFilename;
            cmd.datafile = dataFilename;
            cmd.instancesSettings = dataset.settings;
            cmd.classifierName = TestLearners.linearSVM.Trainer;
            RunExperiments.Run(cmd);

            // Load and make predictions with a previously saved model.

            IDataModel dataModel;
            IDataStats dataStats;
            var predictor = (IDistributionPredictor<Instance, Float, Float>)PredictorUtils.LoadPredictor(
                out dataModel, out dataStats, modelFilename);
            var instanceFactory = ReflectionUtilsOld.CreateInstanceOld<IInstanceFactory, SignatureInstances>(
                cmd.instancesClass, cmd.instancesSettings, null, dataModel);

            bool headerSkip = true;
            List<Float> outputs = new List<Float>();
            List<Float> probabilities = new List<Float>();
            using (StreamReader reader = new StreamReader(testDataFilename))
            {
                List<string> features = new List<string>();
                string text;
                long line = 0;
                while ((text = reader.ReadLine()) != null)
                {
                    ++line;
                    if (string.IsNullOrWhiteSpace(text))
                        continue;

                    string[] cols = text.Split(',');
                    Assert.IsTrue(cols.Length == 15);

                    if (headerSkip)
                    {
                        // skip header line
                        headerSkip = false;
                        continue;
                    }

                    features.Clear();
                    // Add in the "max dimensionality"
                    features.Add("15");
                    for (int col = 0; col < cols.Length; col++)
                    {
                        string s = cols[col].Trim();
                        switch (col)
                        {
                        case 0:
                        case 2:
                        case 4:
                        case 10:
                        case 11:
                        case 12:
                        case 14:
                            // numeric feature or label -- add if non-zero
                            Float val = InstancesUtils.FloatParse(s);
                            if (val == 0) // Beware of NaNs - they should be recorded!
                                continue;
                            break;
                        }
                        features.Add(col + ":" + s);
                    }

                    Instance instance = instanceFactory.ProduceInstance(line, features.ToArray());
                    Float rawOutput, probability;
                    probability = predictor.PredictDistribution(instance, out rawOutput);
                    outputs.Add(rawOutput);
                    probabilities.Add(probability);
                }
            }

            List<Float> originalOutputs = new List<Float>();
            List<Float> originalProbabilities = new List<Float>();
            var env = new LocalEnvironment(SysRandom.Wrap(RunExperiments.GetRandom(cmd)));
            Instances instances = RunExperiments.CreateTestData(cmd, testDataFilename, dataModel, null, env);
            foreach (Instance instance in instances)
            {
                Float rawOutput, probability;
                probability = predictor.PredictDistribution(instance, out rawOutput);
                originalOutputs.Add(rawOutput);
                originalProbabilities.Add(probability);
            }

            CollectionAssert.AreEqual(outputs, originalOutputs);
            CollectionAssert.AreEqual(probabilities, originalProbabilities);

            File.Delete(modelFilename);

            Done();
        }

        [Fact, TestCategory("Example"), TestCategory("FastRank")]
        public void SimpleWeightedTest()
        {
            RunMTAThread(() =>
            {
                string basename = GetDataPath("ArtificiallyWeighted\\breast-cancer");
                WeightedMetricTest(
                    CreateTextInstances(basename + "-noweights.txt", ""),
                    CreateTextInstances(basename + "-weights-one.txt", "weight=0 label=1"),
                    CreateTextInstances(basename + "-weights-quarter.txt", "weight=0 label=1"),
                    "FastRank",
                    () => new ClassifierTester(new ProbabilityPredictorTesterArguments())
                );

                basename = GetDataPath("ArtificiallyWeighted\\housing");
                WeightedMetricTest(
                    CreateTextInstances(basename + "-noweights.txt", ""),
                    CreateTextInstances(basename + "-weights-one.txt", "weight=0 label=1"),
                    CreateTextInstances(basename + "-weights-quarter.txt", "weight=0 label=1"),
                    "FastRankRegression",
                    () => new LinearRegressorTester(new LinearRegressorTester.Arguments())
                );

                basename = GetDataPath("ArtificiallyWeighted\\ranking-sample");
                WeightedMetricTest(
                    CreateExtractInstances(basename + "-noweights.txt"),
                    CreateExtractInstances(basename + "-weights-one.txt", "weight=0 label=1"),
                    null,
                    "FastRankRanking",
                    () => new RankerTester()
                );
            });
            Done();
        }

        private Instances CreateTextInstances(string filename, string settings = null)
        {
            var args = new TlcTextInstances.Arguments();
            CmdParser.ParseArguments(settings, args);
            return new TlcTextInstances(args, filename);
        }

        private Instances CreateExtractInstances(string filename, string settings = null)
        {
            var args = new ExtractInstances.Arguments();
            CmdParser.ParseArguments(settings, args);
            return new ExtractInstances(args, filename);
        }

        private void WeightedMetricTest(Instances noWeights, Instances weights1, Instances weightsQuarter, string predictorName, Func<Tester<Float>> tester)
        {
            Instances[] data = new Instances[3] { noWeights, weights1, weightsQuarter };
            Metric[][] results = new Metric[3][];
            var sub = new SubComponent<ITrainer<Instances, IPredictor<Instance, Float>>, SignatureOldTrainer>(
                predictorName, "nl=5 lr=0.25 iter=20 mil=1");
            for (int i = 0; i < 3; i++)
            {
                Instances instances = data[i];
                if (instances == null)
                    continue;

                // Create the trainer
                var trainer = sub.CreateInstance(new TrainHost(new Random(1), 0));

                // Train a predictor
                trainer.Train(instances);
                var predictor = trainer.CreatePredictor();

                results[i] = tester().Test(predictor, instances);
            }

            //Compare metrics results with unweighted metrics
            for (int i = 1; i < 3; i++)
            {
                if (results[i] == null)
                    continue;
                //The nonweighted result should have half of the metrics
                Assert.AreEqual(results[i].Length, results[0].Length * 2);
                for (int m = 0; m < results[0].Length; m++)
                {
                    Assert.AreEqual(results[0][m].Name, results[i][m].Name);
                    Double diff = Math.Abs(results[0][m].Value - results[i][m].Value);
                    if (diff > 1e-6)
                    {
                        Fail("{0} differ: {1} vs. {2}", results[0][m].Name, results[0][m].Value, results[i][m].Value);
                    }
                }
            }

            //Compare all metrics between weight 1 (with and without explicit weight in the input)
            for (int m = 0; m < results[0].Length; m++)
            {
                Assert.IsTrue(Math.Abs(results[0][m].Value - results[1][m].Value) < 1e-10);
                Assert.IsTrue(Math.Abs(results[0][m].Value - results[1][m + results[0].Length].Value) < 1e-10);
            }
        }
    }
#endif
}