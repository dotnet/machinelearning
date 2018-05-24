// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime.CommandLine;

namespace Microsoft.ML.Runtime.RunTests
{
    using Xunit;
    using Xunit.Abstractions;
    using TestLearners = TestLearnersBase;

    /// <summary>
    /// Tests using maml commands (IDV) functionality.
    /// </summary>
    public sealed partial class TestPredictors : BaseTestPredictors
    {
        /// <summary>
        /// Get a list of datasets for binary classifier base test.
        /// </summary>
        public IList<TestDataset> GetDatasetsForBinaryClassifierBaseTest()
        {
            // MSM dataset is not yet ported.
            return new[] {
                TestDatasets.breastCancer,
                /* TestDatasets.msm */
            };
        }

        public IList<TestDataset> GetDatasetsForMulticlassClassifierTest()
        {
            return new[] {
                TestDatasets.breastCancer,
                TestDatasets.iris
            };
        }

        /// <summary>
        /// Get a list of datasets for regressor test.
        /// </summary>
        public IList<TestDataset> GetDatasetsForRegressorTest()
        {
            return new[] { TestDatasets.housing };
        }

        /// <summary>
        /// Get a list of datasets for ranking test.
        /// </summary>
        public IList<TestDataset> GetDatasetsForRankingTest()
        {
            return new[] { TestDatasets.rankingText };
        }

        /// <summary>
        /// Get a list of datasets for binary classifier base test.
        /// </summary>
        public IList<TestDataset> GetDatasetsForBinaryClassifierMoreTest()
        {
            return new[] {
                TestDatasets.breastCancerBoolLabel,
                TestDatasets.breastCancerPipeMissing,
                TestDatasets.breastCancerPipeMissingFilter,
                TestDatasets.msm
            };
        }

        /// <summary>
        /// Get a list of datasets for the WeightingPredictorsTest test.
        /// </summary>
        public IList<TestDataset> GetDatasetsForClassificationWeightingPredictorsTest()
        {
            return new[] { TestDatasets.breastCancerWeighted };
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact]
        [TestCategory("Binary")]
        public void BinaryClassifierPerceptronTest()
        {
            var binaryPredictors = new[] { TestLearners.perceptron, /*TestLearners.perceptron_reg*/ };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("FastRank")]
        public void EarlyStoppingTest()
        {
            RunMTAThread(() =>
            {
                var dataset = TestDatasets.msm.Clone();
                dataset.validFilename = dataset.testFilename;
                var predictor = TestLearners.fastRankClassificationPruning;
                Run_TrainTest(predictor, dataset);
            });
            Done();
        }

        /// <summary>
        /// Multiclass Logistic Regression test.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Multiclass")]
        [TestCategory("Logistic Regression")]
        public void MulticlassLRTest()
        {
            RunOneAllTests(TestLearners.multiclassLogisticRegression, TestDatasets.iris);
            RunOneAllTests(TestLearners.multiclassLogisticRegression, TestDatasets.irisLabelName);
            Done();
        }

        /// <summary>
        /// Multiclass Logistic Regression with non-negative coefficients test.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Multiclass")]
        [TestCategory("Logistic Regression")]
        public void MulticlassLRNonNegativeTest()
        {
            RunOneAllTests(TestLearners.multiclassLogisticRegressionNonNegative, TestDatasets.iris);
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Multiclass")]
        [TestCategory("SDCA")]
        public void MulticlassSdcaTest()
        {
            var predictors = new[] {
                TestLearners.multiclassSdca, TestLearners.multiclassSdcaL1, TestLearners.multiclassSdcaSmoothedHinge };
            var datasets = GetDatasetsForMulticlassClassifierTest();
            RunAllTests(predictors, datasets);
            Done();
        }

        /// <summary>
        /// Multiclass Logistic Regression test with a tree featurizer.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Multiclass")]
        [TestCategory("Logistic Regression")]
        [TestCategory("FastTree")]
        public void MulticlassTreeFeaturizedLRTest()
        {
            RunMTAThread(() =>
            {
                RunOneAllTests(TestLearners.multiclassLogisticRegression, TestDatasets.irisTreeFeaturized);
                RunOneAllTests(TestLearners.multiclassLogisticRegression, TestDatasets.irisTreeFeaturizedPermuted);
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Multiclass")]
        [TestCategory("Evaluators")]
        public void MultiClassCVTest()
        {
            var predictor = new PredictorAndArgs
            {
                Trainer = new SubComponent("MulticlassLogisticRegression", "ot=1e-3 nt=1"),
                MamlArgs = new[]
                {
                    "loader=text{col=TextLabel:TX:0 col=Features:Num:~}",
                    "prexf=expr{col=Label:TextLabel expr={x=>single(x)>4?na(4):single(x)}}",
                    "prexf=missingvaluefilter{col=Label}",
                    "prexf=Term{col=Strat:TextLabel}",
                    "strat=Strat",
                    "evaluator=multiclass{opcs+}"
                }
            };
            Run_CV(predictor, TestDatasets.mnistTiny28, extraTag: "DifferentClassCounts");
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Clustering")]
        [TestCategory("KMeans")]
        public void KMeansClusteringTest()
        {
            var predictors = new[] { TestLearners.KMeansDefault, TestLearners.KMeansInitPlusPlus, TestLearners.KMeansInitRandom };
            var datasets = new[] { TestDatasets.adult, TestDatasets.mnistTiny28 };
            RunAllTests(predictors, datasets);
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("SDCA")]
        public void LinearClassifierTest()
        {
            var binaryPredictors = new[]
                {
                    TestLearners.binarySdca,
                    TestLearners.binarySdcaL1,
                    TestLearners.binarySdcaSmoothedHinge,
                    TestLearners.binarySgd,
                    TestLearners.binarySgdHinge
                };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact]
        [TestCategory("Binary")]
        public void BinaryClassifierLogisticRegressionTest()
        {
            var binaryPredictors = new[] { TestLearners.logisticRegression };
            RunOneAllTests(TestLearners.logisticRegression, TestDatasets.breastCancer, summary: true);
            // RunOneAllTests(TestLearners.logisticRegression, TestDatasets.msm);
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        public void BinaryClassifierTesterThresholdingTest()
        {
            var binaryPredictors = new[] { TestLearners.logisticRegression };
            var binaryClassificationDatasets = new[] { TestDatasets.breastCancer };
            RunAllTests(binaryPredictors, binaryClassificationDatasets, new[] { "eval=BinaryClassifier{threshold=0.95 useRawScore=-}" }, "withThreshold");
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact]
        [TestCategory("Binary")]
        public void BinaryClassifierLogisticRegressionNormTest()
        {
            var binaryPredictors = new[] { TestLearners.logisticRegressionNorm };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        ///A test for binary classifiers with non-negative coefficients
        ///</summary>
        [Fact]
        [TestCategory("Binary")]
        public void BinaryClassifierLogisticRegressionNonNegativeTest()
        {
            var binaryPredictors = new[] { TestLearners.logisticRegressionNonNegative };
            var binaryClassificationDatasets = new[] { TestDatasets.breastCancer };
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact]
        [TestCategory("Binary")]
        public void BinaryClassifierLogisticRegressionBinNormTest()
        {
            var binaryPredictors = new[] { TestLearners.logisticRegressionBinNorm };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact]
        [TestCategory("Binary")]
        public void BinaryClassifierLogisticRegressionGaussianNormTest()
        {
            var binaryPredictors = new[] { TestLearners.logisticRegressionGaussianNorm };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("FastRank")]
        public void BinaryClassifierFastRankClassificationTest()
        {
            RunMTAThread(() =>
            {
                var learner = TestLearners.fastRankClassification;
                var data = TestDatasets.breastCancer;
                string dir = learner.Trainer.Kind;
                string prName = "prcurve-breast-cancer-prcurve.txt";
                string prPath = DeleteOutputPath(dir, prName);
                string eval = string.Format("eval=Binary{{pr={{{0}}}}}", prPath);
                Run_TrainTest(learner, data, new[] { eval });
                CheckEqualityNormalized(dir, prName);
                Run_CV(learner, data);

                Run_CV(TestLearners.fastRankClassification, TestDatasets.breastCancer);
                RunOneAllTests(TestLearners.fastRankClassification, TestDatasets.msm);
            });
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact]
        [TestCategory("Binary")]
        [TestCategory("FastForest")]
        public void FastForestClassificationTest()
        {
            RunMTAThread(() =>
            {
                var binaryPredictors = new[] { TestLearners.FastForestClassification };
                var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
                RunAllTests(binaryPredictors, binaryClassificationDatasets);
            });
            Done();
        }

        /// <summary>
        ///A test for regressors
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        [TestCategory("FastForest")]
        public void FastForestRegressionTest()
        {
            RunMTAThread(() =>
            {
                var regressionPredictors = new[] {
                    TestLearners.FastForestRegression,
                    TestLearners.QuantileRegressionScorer,
                };
                var regressionDatasets = GetDatasetsForRegressorTest();
                RunAllTests(regressionPredictors, regressionDatasets);
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Weighting Predictors")]
        [TestCategory("FastForest")]
        public void WeightingFastForestClassificationPredictorsTest()
        {
            RunMTAThread(() =>
            {
                RunAllTests(
                    new[] { TestLearners.FastForestClassification },
                    new[] { TestDatasets.breastCancerDifferentlyWeighted });
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Weighting Predictors")]
        [TestCategory("FastForest")]
        public void WeightingFastForestRegressionPredictorsTest()
        {
            RunMTAThread(() =>
            {
                var regressionPredictors = new[] {
                    TestLearners.FastForestRegression,
                    TestLearners.QuantileRegressionScorer,
                };

                RunAllTests(
                    regressionPredictors,
                    new[] { TestDatasets.housingDifferentlyWeightedRep });
            });
            Done();
        }
        
        [Fact]
        [TestCategory("Binary")]
        [TestCategory("FastTree")]
        public void FastTreeBinaryClassificationTest()
        {
            RunMTAThread(() =>
            {
                var learners = new[] { TestLearners.FastTreeClassfier, TestLearners.FastTreeDropoutClassfier,
                    TestLearners.FastTreeBsrClassfier, TestLearners.FastTreeClassfierDisk };
                var binaryClassificationDatasets = new List<TestDataset> { TestDatasets.breastCancerPipe};
                foreach (var learner in learners)
                {
                    foreach (TestDataset dataset in binaryClassificationDatasets)
                        Run_TrainTest(learner, dataset);
                }
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("FastTree")]
        public void FastTreeBinaryClassificationCategoricalSplitTest()
        {
            RunMTAThread(() =>
            {
                var learners = new[] { TestLearners.FastTreeClassfier, TestLearners.FastTreeWithCategoricalClassfier,
                    TestLearners.FastTreeClassfierDisk, TestLearners.FastTreeWithCategoricalClassfierDisk };

                var binaryClassificationDatasets = new List<TestDataset> { TestDatasets.adultOnlyCat, TestDatasets.adult };
                foreach (var learner in learners)
                {
                    foreach (TestDataset dataset in binaryClassificationDatasets)
                        Run_TrainTest(learner, dataset, extraTag: "Cat", summary: true, saveAsIni: true);
                }
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regression")]
        [TestCategory("FastTree")]
        public void FastTreeRegressionCategoricalSplitTest()
        {
            RunMTAThread(() =>
            {
                var learners = new[] { TestLearners.FastTreeRegressor, TestLearners.FastTreeRegressorCategorical };

                var regressionDatasets = new List<TestDataset> { TestDatasets.autosSample };
                foreach (var learner in learners)
                {
                    foreach (TestDataset dataset in regressionDatasets)
                        Run_TrainTest(learner, dataset, extraTag: "Cat", summary: true, saveAsIni: true);
                }
            });
            Done();
        }

        [Fact]
        [TestCategory("Binary")]
        [TestCategory("FastTree")]
        public void FastTreeBinaryClassificationNoOpGroupIdTest()
        {
            RunMTAThread(() =>
            {
                var learners = new[] { TestLearners.FastTreeClassfier };
                // In principle the training with this group ID should be the same as the training without
                // this group ID, since the trainer should not be paying attention to the group ID.
                var binaryClassificationDatasets = new List<TestDataset> { TestDatasets.breastCancerGroupId };
                foreach (var learner in learners)
                {
                    foreach (TestDataset dataset in binaryClassificationDatasets)
                        Run_TrainTest(learner, dataset);
                }
            });
            Done();
        }

        [Fact]
        [TestCategory("Binary")]
        [TestCategory("FastTree")]
        public void FastTreeHighMinDocsTest()
        {
            RunMTAThread(() =>
            {
                var learners = new[] { TestLearners.FastTreeClassfierHighMinDocs };
                var binaryClassificationDatasets = new List<TestDataset> { TestDatasets.breastCancerPipe };
                foreach (var learner in learners)
                {
                    foreach (TestDataset dataset in binaryClassificationDatasets)
                        Run_TrainTest(learner, dataset);
                }
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("FastTree")]
        public void FastTreeRankingTest()
        {
            RunMTAThread(() =>
            {
                var learners = new[] { TestLearners.FastTreeRanker, TestLearners.FastTreeRankerCustomGains };
                var rankingDatasets = GetDatasetsForRankingTest();
                foreach (var learner in learners)
                {
                    foreach (TestDataset dataset in rankingDatasets)
                        Run_TrainTest(learner, dataset);
                }
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("FastTree")]
        public void FastTreeRegressionTest()
        {
            RunMTAThread(() =>
            {
                var learners = new[] {
                    TestLearners.FastTreeRegressor,
                    TestLearners.FastTreeDropoutRegressor,
                    TestLearners.FastTreeTweedieRegressor
                };
                var datasets = GetDatasetsForRegressorTest();
                foreach (var learner in learners)
                {
                    foreach (TestDataset dataset in datasets)
                        Run_TrainTest(learner, dataset);
                }
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("FastTree")]
        public void GamRegressionTest()
        {
            RunMTAThread(() =>
            {
                var learners = new[] { TestLearners.RegressionGamTrainer };
                var datasets = GetDatasetsForRegressorTest();
                foreach (var learner in learners)
                {
                    foreach (TestDataset dataset in datasets)
                    {
                        Run_TrainTest(learner, dataset);
                    }
                }
            });
            Done();
        }

        [Fact]
        [TestCategory("FastTree")]
        public void GamBinaryClassificationTest()
        {
            RunMTAThread(() =>
            {
                var learners = new[] { TestLearners.BinaryClassificationGamTrainer, TestLearners.BinaryClassificationGamTrainerDiskTranspose };
                var datasets = GetDatasetsForBinaryClassifierBaseTest();

                foreach (var learner in learners)
                {
                    foreach (TestDataset dataset in datasets)
                    {
                        Run_TrainTest(learner, dataset);
                    }
                }
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("FastTree")]
        public void FastTreeUnderbuiltRegressionTest()
        {
            // In this test, we specify we want, per tree, 30 splits with a minimum 30 docs per leaf,
            // on a training set with only about 500 examples. This is to test the somewhat unusual
            // case where the number of actual leaves is less than the number of maximum leaves per tree.
            RunMTAThread(() =>
            {
                Run_TrainTest(TestLearners.FastTreeUnderbuiltRegressor, TestDatasets.housing, null, "Underbuilt");
            });
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        public void BinaryClassifierLinearSvmTest()
        {
            var binaryPredictors = new[] { TestLearners.linearSVM };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierMoreTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        /// A test for regressors
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        [TestCategory("FastTree")]
        public void RegressorFastRankTest()
        {
            RunMTAThread(() =>
            {
                var regressionPredictors = new[] { TestLearners.fastRankRegression };
                var regressionDatasets = GetDatasetsForRegressorTest();
                RunAllTests(regressionPredictors, regressionDatasets);
            });
            Done();
        }

        /// <summary>
        /// A test for regressors.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        public void RegressorOgdTest()
        {
            var regressionPredictors = new[] { TestLearners.OGD };
            var regressionDatasets = GetDatasetsForRegressorTest();
            RunAllTests(regressionPredictors, regressionDatasets);
            Done();
        }

        /// <summary>
        /// A test for ordinary least squares regression.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        public void RegressorOlsTest()
        {
            var regressionPredictors = new[] { TestLearners.Ols, TestLearners.OlsNorm, TestLearners.OlsReg };
            var regressionDatasets = GetDatasetsForRegressorTest();
            RunAllTests(regressionPredictors, regressionDatasets);
            Done();
        }

        /// <summary>
        /// A test for ordinary least squares regression.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        public void RegressorOlsTestOne()
        {
            Run_TrainTest(TestLearners.Ols, TestDatasets.housing);
            Done();
        }

        /// <summary>
        /// Test method for SDCA regression.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        [TestCategory("SDCAR")]
        public void RegressorSdcaTest()
        {
            var regressionPredictors = new[] { TestLearners.Sdcar, TestLearners.SdcarNorm, TestLearners.SdcarReg };
            var regressionDatasets = GetDatasetsForRegressorTest();
            RunAllTests(regressionPredictors, regressionDatasets);
            Done();
        }

#region "Regressor"

#if OLD_TESTS // REVIEW: Port these tests?
        /// <summary>
        /// A test for ordinary least squares regression using synthetic data, under various
        /// conditions. Unlike many other learners, OLS is an attempt to solve a problem exactly,
        /// so we can more precisely judge the quality of the solution.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        public void RegressorSyntheticOlsTest()
        {
            const int featureCount = 15;
            const Float scale = 2;
            Float[] model = new Float[featureCount + 1];
            Random rgen = new Random(0);
            for (int i = 0; i < model.Length; ++i)
                model[i] = scale * (2 * rgen.NextFloat() - 1);

            ListInstances instances = new ListInstances();
            for (int id = 0; id < 10 * model.Length; ++id)
            {
                Float label = model[featureCount];
                WritableVector vec;
                if (rgen.Next(2) == 1)
                {
                    // Dense
                    Float[] features = new Float[featureCount];
                    for (int i = 0; i < features.Length; ++i)
                        label += model[i] * (features[i] = scale * (2 * rgen.NextFloat() - 1));
                    vec = WritableVector.CreateDense(features, false);
                }
                else
                {
                    // Sparse
                    int entryCount = rgen.Next(featureCount);
                    int[] indices = Utils.GetRandomPermutation(rgen, featureCount).Take(entryCount).OrderBy(x => x).ToArray();
                    Float[] features = new Float[indices.Length];
                    for (int ii = 0; ii < indices.Length; ++ii)
                        label += model[indices[ii]] * (features[ii] = scale * (2 * rgen.NextFloat() - 1));
                    vec = WritableVector.CreateSparse(featureCount, indices, features, false);
                }
                instances.Add(new Instance(vec, label, "", false) { Id = id });
            }

            const Double tol = 1e-4;
            TrainHost host = new TrainHost(new Random(0));

            var args = new OlsLinearRegressionTrainer.OldArguments();
            {
                // Exactly determined case.
                Log("Train using exactly model.Length examples, so we have an exact solution, but no statistics.");
                ListInstances subinstances = new ListInstances();
                subinstances.AddRange(instances.Take(model.Length));
                var trainer = new OlsLinearRegressionTrainer(args, host);
                trainer.Train(subinstances);
                var pred = trainer.CreatePredictor();
                pred = WriteReloadOlsPredictor(pred);

                Assert.AreEqual(featureCount, pred.InputType.VectorSize, "Unexpected input size");
                Assert.IsFalse(pred.HasStatistics, "Should not have statistics with exact specified model");
                Assert.AreEqual(null, pred.PValues, "Should not have p-values with no-stats model");
                Assert.AreEqual(null, pred.TValues, "Should not have t-values with no-stats model");
                Assert.AreEqual(null, pred.StandardErrors, "Should not have standard errors with no-stats model");
                Assert.IsTrue(Double.IsNaN(pred.RSquaredAdjusted), "R-squared adjusted should be NaN with no-stats model");
                foreach (Instance inst in subinstances)
                    Assert.AreEqual(inst.Label, pred.Predict(inst), tol, "Mismatch on example id {0}", inst.Id);
            }

            Float finalNorm;
            {
                // Overdetermined but still exact case.
                Log("Train using more examples with non-noised label, so we have an exact solution, and statistics.");
                var trainer = new OlsLinearRegressionTrainer(args, host);
                trainer.Train(instances);
                var pred = trainer.CreatePredictor();
                pred = WriteReloadOlsPredictor(pred);
                Assert.AreEqual(featureCount, pred.InputType.VectorSize, "Unexpected input size");
                Assert.IsTrue(pred.HasStatistics, "Should have statistics");
                Assert.AreEqual(1.0, pred.RSquared, 1e-6, "Coefficient of determination should be 1 for exact specified model");
                Assert.IsTrue(FloatUtils.IsFinite(pred.RSquaredAdjusted), "R-squared adjusted should be finite with exact specified model");
                Assert.AreEqual(featureCount, pred.Weights.Count, "Wrong number of weights");
                Assert.AreEqual(featureCount + 1, pred.PValues.Count, "Wrong number of pvalues");
                Assert.AreEqual(featureCount + 1, pred.TValues.Count, "Wrong number of t-values");
                Assert.AreEqual(featureCount + 1, pred.StandardErrors.Count, "Wrong number of standard errors");
                foreach (Instance inst in instances)
                    Assert.AreEqual(inst.Label, pred.Predict(inst), tol, "Mismatch on example id {0}", inst.Id);
                finalNorm = pred.Weights.Sum(x => x * x);

                // Suppress statistics and retrain.
                args.perParameterSignificance = false;
                var trainer2 = new OlsLinearRegressionTrainer(args, host);
                trainer2.Train(instances);
                args.perParameterSignificance = true;
                var pred2 = trainer2.CreatePredictor();
                pred2 = WriteReloadOlsPredictor(pred2);

                Assert.AreEqual(null, pred2.PValues, "P-values present but should be absent");
                Assert.AreEqual(null, pred2.TValues, "T-values present but should be absent");
                Assert.AreEqual(null, pred2.StandardErrors, "Standard errors present but should be absent");
                Assert.AreEqual(pred.RSquared, pred2.RSquared);
                Assert.AreEqual(pred.RSquaredAdjusted, pred2.RSquaredAdjusted);
                Assert.AreEqual(pred.Bias, pred2.Bias);
                var w1 = pred.Weights.ToArray();
                var w2 = pred2.Weights.ToArray();
                Assert.AreEqual(w1.Length, w2.Length);
                for (int i = 0; i < w1.Length; ++i)
                    Assert.AreEqual(w1[i], w2[i]);
            }

            Float[] regularizationParams = new Float[] { 0, (Float)0.01, (Float)0.1 };

            foreach (Float regParam in regularizationParams)
            {
                foreach (bool subdefined in new bool[] { true, false })
                {
                    // Overdetermined and inexact case, for which OLS solution is feasible but inexact.
                    Log("");
                    Log("Train using noised label, reg param {0}, so solution is no longer exact", regParam);
                    ListInstances noisyInstances = new ListInstances();
                    Float boundCost = 0;
                    foreach (Instance inst in instances)
                    {
                        // When we noise the label, we do it on an appreciable but still relatively small scale,
                        // compared to the regular distribution of the labels.
                        Float diff = scale * (2 * rgen.NextFloat() - 1) / 3;
                        boundCost += diff * diff;
                        noisyInstances.Add(new Instance(inst.Features, inst.Label + diff, inst.Name, false) { Id = inst.Id });
                        // Make sure this solver also works, when we have 
                        if (subdefined && 2 * noisyInstances.Count >= model.Length)
                            break;
                    }
                    args.l2Weight = regParam;
                    // Transform the friendlier user-facing parameter into the actual value injected into the solver.
                    var regParam2 = regParam * regParam * noisyInstances.Count;
                    boundCost += regParam2 * finalNorm;
                    var trainer = new OlsLinearRegressionTrainer(args, host);

                    if (subdefined && regParam == 0)
                    {
                        // In the non-ridge regression case, ordinary least squares should fail on a deficient system.
                        bool caught = false;
                        try
                        {
                            trainer.Train(noisyInstances);
                        }
                        catch (InvalidOperationException)
                        {
                            caught = true;
                        }
                        Assert.IsTrue(caught, "Failed to encounter an error, when running OLS on a deficient system");
                        continue;
                    }
                    else
                    {
                        trainer.Train(noisyInstances);
                    }
                    var pred = trainer.CreatePredictor();
                    pred = WriteReloadOlsPredictor(pred);
                    Assert.AreEqual(featureCount, pred.InputType.VectorSize, "Unexpected input size");
                    Assert.IsTrue(0 <= pred.RSquared && pred.RSquared < 1, "R-squared not in expected range");

                    Func<Func<Instance, Float>, Float> getError = p =>
                        noisyInstances.Select(inst => inst.Label - p(inst)).Sum(e => e * e);

                    // In principle there should be no "better" solution with a lower L2 weight. Wiggle the parameters
                    // with a finite difference, and evaluate the change in error.
                    var referenceNorm = pred.Weights.Sum(x => x * x);
                    Float referenceError = getError(pred.Predict);
                    Float referenceCost = referenceError + regParam2 * referenceNorm;
                    Float smoothing = (Float)(referenceCost * 5e-6);
                    Log("Reference cost is {0} + {1} * {2} = {3}, upper bound was {4}", referenceError, regParam2, referenceNorm, referenceCost, boundCost);
                    Assert.IsTrue(boundCost > referenceCost, "Reference cost {0} was above theoretical upper bound {1}", referenceCost, boundCost);
                    Float lastCost = 0;
                    var weights = pred.Weights.Sum(x => x * x);
                    for (int trial = 0; trial < model.Length * 2; ++trial)
                    {
                        int param = trial / 2;
                        bool up = (trial & 1) == 1;
                        Float[] w = pred.Weights.ToArray();
                        Assert.AreEqual(featureCount, w.Length);
                        Float b = pred.Bias;
                        bool isBias = param == featureCount;
                        Float normDelta;
                        Float origValue;
                        Float newValue;
                        if (isBias)
                        {
                            origValue = OlsWiggle(ref b, out normDelta, up);
                            newValue = b;
                            // Bias not included in regularization
                            normDelta = 0;
                        }
                        else
                        {
                            origValue = OlsWiggle(ref w[param], out normDelta, up);
                            newValue = w[param];
                        }
                        Func<Instance, Float> del = inst => b + inst.Features.AllValues.Select((v, i) => w[i] * v).Sum();
                        Float wiggledCost = getError(del) + regParam2 * (referenceNorm + normDelta);
                        string desc = string.Format("after wiggling {0} {1} from {2} to {3}",
                            isBias ? "bias" : string.Format("weight[{0}]", param), up ? "up" : "down", origValue, newValue);
                        Log("Finite difference cost is {0} ({1}), {2}", wiggledCost, wiggledCost - referenceCost, desc);
                        Assert.IsTrue(wiggledCost > referenceCost * (Float)(1 - 5e-7), "Finite difference cost {0} not higher than reference cost {1}, {2}",
                            wiggledCost, referenceCost, desc);
                        if (up)
                        {
                            // If the solution to the problem really does like at the base of the quadratic, then wiggling
                            // equal amounts up and down should lead to *roughly* the same error.
                            Float ratio = 1 - (lastCost - referenceCost + smoothing) / (wiggledCost - referenceCost + smoothing);
                            Log("Wiggled up had a relative difference of {0:0.0%} vs. wiggled down", ratio);
                            Assert.IsTrue(0.1 > Math.Abs(ratio), "Ratio {0} of up/down too high, {1}", ratio, desc);
                        }
                        lastCost = wiggledCost;
                    }
                }
            }

            Done();
        }

        private Float OlsWiggle(ref Float value, out Float deltaNorm, bool up)
        {
            Float origValue = value;
            Float wiggle = (Float)Math.Max(1e-7, Math.Abs(1e-3 * value));
            value += up ? wiggle : -wiggle;
            deltaNorm = value * value - origValue * origValue;
            return origValue;
        }

        private OlsLinearRegressionPredictor WriteReloadOlsPredictor(OlsLinearRegressionPredictor pred)
        {
            using (MemoryStream mem = new MemoryStream())
            {
                PredictorUtils.Save(mem, pred, null, null, null, useFileSystem: true);
                mem.Seek(0, SeekOrigin.Begin);
                Microsoft.ML.Runtime.Model.IDataModel model;
                Microsoft.ML.Runtime.Model.IDataStats stats;
                return (OlsLinearRegressionPredictor)PredictorUtils.LoadPredictor(out model, out stats, mem, false);
            }
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        public void RegressorSyntheticDuplicatedOlsTest()
        {
            // OLS should result in the same predictor if we just simply duplicate data.
            // Make certain that ridge regression works.
            const int featureCount = 10;
            const Float scale = 2;
            Float[] model = new Float[featureCount + 1];
            Random rgen = new Random(1);
            for (int i = 0; i < model.Length; ++i)
                model[i] = scale * (2 * rgen.NextFloat() - 1);

            ListInstances instances = new ListInstances();
            for (int id = 0; id < 2 * model.Length; ++id)
            {
                Float label = model[featureCount];
                WritableVector vec;
                if (rgen.Next(2) == 1)
                {
                    // Dense
                    Float[] features = new Float[featureCount];
                    for (int i = 0; i < features.Length; ++i)
                        label += model[i] * (features[i] = scale * (2 * rgen.NextFloat() - 1));
                    vec = WritableVector.CreateDense(features, false);
                }
                else
                {
                    // Sparse
                    int entryCount = rgen.Next(featureCount);
                    int[] indices = Utils.GetRandomPermutation(rgen, featureCount).Take(entryCount).OrderBy(x => x).ToArray();
                    Float[] features = new Float[indices.Length];
                    for (int ii = 0; ii < indices.Length; ++ii)
                        label += model[indices[ii]] * (features[ii] = scale * (2 * rgen.NextFloat() - 1));
                    vec = WritableVector.CreateSparse(featureCount, indices, features, false);
                }
                Float diff = scale * (2 * rgen.NextFloat() - 1) / 5;
                instances.Add(new Instance(vec, label + diff, "", false) { Id = id });
            }

            ListInstances instances2 = new ListInstances();
            foreach (Instance inst in instances)
            {
                instances2.Add(new Instance(inst.Features, inst.Label, inst.Name, false) { Id = 2 * inst.Id });
                instances2.Add(new Instance(inst.Features, inst.Label, inst.Name, false) { Id = 2 * inst.Id + 1 });
            }
            OlsLinearRegressionTrainer.OldArguments args = new OlsLinearRegressionTrainer.OldArguments();
            args.l2Weight = (Float)1;
            TrainHost host = new TrainHost(new Random(0));
            var trainer = new OlsLinearRegressionTrainer(args, host);
            trainer.Train(instances);
            var pred = trainer.CreatePredictor();
            var trainer2 = new OlsLinearRegressionTrainer(args, host);
            trainer2.Train(instances2);
            var pred2 = trainer2.CreatePredictor();

            var tol = 1e-5;
            Assert.AreEqual(pred.RSquared, pred2.RSquared, tol);
            Assert.AreEqual(pred.Bias, pred2.Bias, tol);
            var w1 = pred.Weights.ToArray();
            var w2 = pred2.Weights.ToArray();
            Assert.AreEqual(w1.Length, w2.Length);
            for (int i = 0; i < w1.Length; ++i)
                Assert.AreEqual(w1[i], w2[i], tol);

            Done();
        }
#endif

#endregion

        /// <summary>
        ///A test for FR ranker
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("FastRank")]
        public void RankingTest()
        {
            RunMTAThread(() =>
            {
                var rankingPredictors = new[] { TestLearners.fastRankRanking };
                var rankingDatasets = GetDatasetsForRankingTest();
                RunAllTests(rankingPredictors, rankingDatasets);
            });
            Done();
        }

        /// <summary>
        ///A test for Poisson regression
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        public void PoissonRegressorTest()
        {
            var regressionPredictors = new[] { TestLearners.poissonRegression };
            //AP: TestDatasets.displayPoisson is broken as it says header+ but training set does not have proper header
            // Discovered when adding strict schema checks bwteen Train/Test
            // I'm not quite sure how to fix train set. Perhaps just adding proper header is be sufficient (but the columns count between train/test differ so I drop this test set at this point and let someone who added those data fix it and possibly reenable unittest
            var datasets = new[] { TestDatasets.childrenPoisson, TestDatasets.autosSample };
            RunAllTests(regressionPredictors, datasets);
            Done();
        }

        /// <summary>
        ///A test for Poisson regression with non-negative coefficients
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Regressor")]
        public void PoissonRegressorNonNegativeTest()
        {
            var regressionPredictors = new[] { TestLearners.poissonRegressionNonNegative };
            //AP: TestDatasets.displayPoisson is broken as it says header+ but training set does not have proper header
            // Discovered when adding strict schema checks bwteen Train/Test
            // I'm not quite sure how to fix train set. Perhaps just adding proper header is be sufficient (but the columns count between train/test differ so I drop this test set at this point and let someone who added those data fix it and possibly reenable unittest
            var datasets = new[] { TestDatasets.childrenPoisson, TestDatasets.autosSample };
            RunAllTests(regressionPredictors, datasets);
            Done();
        }

        /// <summary>
        /// Multiclass Logistic Regression test.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Multiclass")]
        [TestCategory("Logistic Regression Sparse")]
        public void MulticlassLRSparseTest()
        {
            RunAllTests(
                new List<PredictorAndArgs>() { TestLearners.multiclassLogisticRegressionRegularized },
                new List<TestDataset>() { TestDatasets.reutersMaxDim });
            Done();
        }

        /// <summary>
        /// Get a list of datasets for Calibrator test.
        /// </summary>
        public IList<TestDataset> GetDatasetsForCalibratorTest()
        {
            return new[] { TestDatasets.breastCancer };
        }

        /// <summary>
        ///A test for no calibrators
        ///</summary>
        [Fact]
        [TestCategory("Calibrator")]
        public void DefaultCalibratorPerceptronTest()
        {
            var datasets = GetDatasetsForCalibratorTest();
            RunAllTests( new[] { TestLearners.perceptronDefault }, datasets, new string[] { "cali={}" }, "nocalibration");
            Done();
        }

        /// <summary>
        ///A test for PAV calibrators
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Calibrator")]
        public void PAVCalibratorPerceptronTest()
        {
            var datasets = GetDatasetsForCalibratorTest();
            RunAllTests( new[] { TestLearners.perceptronDefault }, datasets, new[] { "cali=PAV" }, "PAVcalibration");
            Done();
        }

        /// <summary>
        ///A test for random calibrators
        ///</summary>
        [Fact]
        [TestCategory("Calibrator")]
        public void RandomCalibratorPerceptronTest()
        {
            var datasets = GetDatasetsForCalibratorTest();
            RunAllTests( new[] { TestLearners.perceptronDefault }, datasets, new string[] { "numcali=200" }, "calibrateRandom");
            Done();
        }

        /// <summary>
        ///A test for calibrators
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Calibrator")]
        public void CalibratorLinearSvmTest()
        {
            var datasets = GetDatasetsForCalibratorTest();
            RunAllTests(
                new[] { TestLearners.linearSVM },
                datasets,
                new string[] { "cali={}" }, "nocalibration");
            RunAllTests(
                new[] { TestLearners.linearSVM },
                datasets,
                new string[] { "cali=PAV" }, "PAVcalibration");
            Done();
        }

        /// <summary>
        ///A test FR weighting predictors
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Weighting Predictors")]
        [TestCategory("FastRank")]
        public void WeightingClassificationFastRankPredictorsTest()
        {
            RunMTAThread(() =>
            {
                var learner = TestLearners.fastRankClassificationWeighted;
                var data = TestDatasets.breastCancerWeighted;
                string dir = learner.Trainer.Kind;
                string prName = "prcurve-breast-cancer-weighted-prcurve.txt";
                string prPath = DeleteOutputPath(dir, prName);
                string eval = string.Format("eval=Binary{{pr={{{0}}}}}", prPath);
                Run_TrainTest(learner, data, new[] { eval });
                CheckEqualityNormalized(dir, prName);
                Run_CV(learner, data);
            });
            Done();
        }

        /// <summary>
        /// Test weighted logistic regression.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Weighting Predictors")]
        [TestCategory("Logistic Regression")]
        public void WeightingClassificationLRPredictorsTest()
        {
            RunAllTests(
                new[] { TestLearners.logisticRegression },
                GetDatasetsForClassificationWeightingPredictorsTest());
            Done();
        }

        /// <summary>
        /// Test weighted neural nets.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Weighting Predictors")]
        [TestCategory("Neural Nets")]
        public void WeightingClassificationNNPredictorsTest()
        {
            RunAllTests(
                new[] { TestLearners.NnBinDefault },
                GetDatasetsForClassificationWeightingPredictorsTest());
            Done();
        }

        /// <summary>
        ///A test FR weighting predictors
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Weighting Predictors")]
        [TestCategory("FastRank")]
        public void WeightingRegressionPredictorsTest()
        {
            RunMTAThread(() =>
            {
                RunOneAllTests(TestLearners.fastRankRegressionWeighted, TestDatasets.housingWeightedRep);
            });
            Done();
        }

        /// <summary>
        ///A test FR weighting predictors
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Weighting Predictors")]
        [TestCategory("FastRank")]
        public void WeightingRankingPredictorsTest()
        {
            RunMTAThread(() =>
            {
                RunOneAllTests(TestLearners.fastRankRankingWeighted, TestDatasets.rankingWeighted);
            });
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Neural Nets")]
        public void NnConfigTests()
        {
            string path;

            // The baseline should show an input mismatch message.
            path = DeleteOutputPath(TestLearners.NnBinDefault.Trainer.Kind, "BcInputMismatch.nn");
            File.WriteAllText(path,
                @"
input Data [8];
hidden H [20] from Data all;
output Out [2] from H all;
");
            RunOneTrain(TestLearners.NnBinCustom(path), TestDatasets.breastCancer, null, "InputMismatch");

            // The baseline should show an output mismatch message.
            path = DeleteOutputPath(TestLearners.NnBinDefault.Trainer.Kind, "BcOutputMismatch.nn");
            File.WriteAllText(path,
                @"
input Data [9];
hidden H [20] from Data all;
output Out [5] from H all;
");
            RunOneTrain(TestLearners.NnBinCustom(path), TestDatasets.breastCancer, null, "OutputMismatch");

            // The data matches the .nn, but the .nn is multi-class, not binary,
            // so BinaryNeuralNetwork.Validate should throw.
            path = DeleteOutputPath(TestLearners.NnBinDefault.Trainer.Kind, "BcNonBinData.nn");
            File.WriteAllText(path,
                @"
input Data [4];
hidden H [20] from Data all;
output Out [3] from H all;
");
            RunOneTrain(TestLearners.NnBinCustom(path), TestDatasets.iris, null, "NonBinData");

            Done();
        }

#if !CORECLR
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Anomaly")]
        public void PcaAnomalyTest()
        {
            Run_TrainTest(TestLearners.PCAAnomalyDefault, TestDatasets.mnistOneClass);
            Run_TrainTest(TestLearners.PCAAnomalyNoNorm, TestDatasets.mnistOneClass);

            // REVIEW: This next test was misbehaving in a strange way that seems to have gone away
            // mysteriously (bad build?).
            Run_TrainTest(TestLearners.PCAAnomalyDefault, TestDatasets.azureCounterUnlabeled, summary: true);

            Done();
        }
#endif
        /// <summary>
        ///A test for one-class svm (libsvm wrapper)
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Anomaly")]
        public void OneClassSvmLibsvmWrapperTest()
        {
            // We don't use the predictor that uses the MKL library, because results can be slightly different depending on the number of threads.
            Run_TrainTest(TestLearners.OneClassSvmLinear, TestDatasets.mnistOneClass, extraTag: "LinearKernel");
            Run_TrainTest(TestLearners.OneClassSvmPoly, TestDatasets.mnistOneClass, extraTag: "PolynomialKernel");
            Run_TrainTest(TestLearners.OneClassSvmRbf, TestDatasets.mnistOneClass, extraTag: "RbfKernel");
            Run_TrainTest(TestLearners.OneClassSvmSigmoid, TestDatasets.mnistOneClass, extraTag: "SigmoidKernel");
            Done();
        }

        /// <summary>
        ///A test for one-class svm (libsvm wrapper)
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Anomaly")]
        public void OneClassSvmLibsvmWrapperDenseTest()
        {
            // We don't use the predictor that uses the MKL library, because results can be slightly different depending on the number of threads.
            Run_TrainTest(TestLearners.OneClassSvmLinear, TestDatasets.breastCancerOneClass, extraTag: "LinearKernel");
            Run_TrainTest(TestLearners.OneClassSvmPoly, TestDatasets.breastCancerOneClass, extraTag: "PolynomialKernel");
            Run_TrainTest(TestLearners.OneClassSvmRbf, TestDatasets.breastCancerOneClass, extraTag: "RbfKernel");
            Run_TrainTest(TestLearners.OneClassSvmSigmoid, TestDatasets.breastCancerOneClass, extraTag: "SigmoidKernel");
            Done();
        }

#if !CORECLR
        /// <summary>
        ///A test for one-class svm (libsvm wrapper)
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Anomaly")]
        public void CompareSvmPredictorResultsToLibSvm()
        {
            using (var env = new TlcEnvironment(1, conc: 1))
            {
                IDataView trainView = new TextLoader(env, new TextLoader.Arguments(), new MultiFileSource(GetDataPath(TestDatasets.mnistOneClass.trainFilename)));
                trainView =
                    NormalizeTransform.Create(env,
                        new NormalizeTransform.MinMaxArguments()
                        {
                            Column = new[] { new NormalizeTransform.AffineColumn() { Name = "Features", Source = "Features" } }
                        },
                     trainView);
                var trainData = TrainUtils.CreateExamples(trainView, "Label", "Features");
                IDataView testView = new TextLoader(env, new TextLoader.Arguments(), new MultiFileSource(GetDataPath(TestDatasets.mnistOneClass.testFilename)));
                ApplyTransformUtils.ApplyAllTransformsToData(env, trainView, testView);
                var testData = TrainUtils.CreateExamples(testView, "Label", "Features");

                CompareSvmToLibSvmCore("linear kernel", "LinearKernel", env, trainData, testData);
                CompareSvmToLibSvmCore("polynomial kernel", "PolynomialKernel{d=2}", env, trainData, testData);
                CompareSvmToLibSvmCore("RBF kernel", "RbfKernel", env, trainData, testData);
                CompareSvmToLibSvmCore("sigmoid kernel", "SigmoidKernel", env, trainData, testData);
            }

            Done();
        }
#endif

        private const float Epsilon = 0.0004f; // Do not use Single.Epsilon as it is not commonly-accepted machine epsilon.
        private const float MaxRelError = 0.000005f;

        public TestPredictors(ITestOutputHelper helper) : base(helper)
        {
        }

#if !CORECLR
        private void CompareSvmToLibSvmCore(string kernelType, string kernel, IHostEnvironment env, RoleMappedData trainData, RoleMappedData testData)
        {
            Contracts.Assert(testData.Schema.Feature != null);

            var args = new OneClassSvmTrainer.Arguments();
            CmdParser.ParseArguments(env, "ker=" + kernel, args);

            var trainer1 = new OneClassSvmTrainer(env, args);
            var trainer2 = new OneClassSvmTrainer(env, args);

            trainer1.Train(trainData);
            var predictor1 = (IValueMapper)trainer1.CreatePredictor();

            LibSvmInterface.ModelHandle predictor2;
            trainer2.TrainCore(trainData, out predictor2);
            LibSvmInterface.ChangeSvmType(predictor2, 4);

            var predictions1 = new List<Float>();
            var predictions2 = new List<Float>();

            int instanceNum = 0;
            int colFeat = testData.Schema.Feature.Index;
            using (var cursor = testData.Data.GetRowCursor(col => col == colFeat))
            {
                Float res1 = 0;
                var buf = default(VBuffer<Float>);
                var getter = cursor.GetGetter<VBuffer<Float>>(colFeat);
                var map1 = predictor1.GetMapper<VBuffer<Float>, Float>();
                while (cursor.MoveNext())
                {
                    getter(ref buf);
                    map1(ref buf, ref res1);

                    Float res2;
                    unsafe
                    {
                        if (buf.IsDense)
                        {
                            fixed (float* pValues = buf.Values)
                                res2 = -LibSvmInterface.SvmPredictDense(predictor2, pValues, buf.Length);
                        }
                        else
                        {
                            fixed (float* pValues = buf.Values)
                            fixed (int* pIndices = buf.Indices)
                                res2 = -LibSvmInterface.SvmPredictSparse(predictor2, pValues, pIndices, buf.Count);
                        }
                    }

                    predictions1.Add(res1);
                    predictions2.Add(res2);
                    Assert.IsTrue(AreEqual(res1, res2, MaxRelError, Epsilon),
                        "Found prediction that does not match the libsvm prediction in line {0}, using {1}",
                        instanceNum, kernelType);
                    instanceNum++;
                }
            }

            LibSvmInterface.FreeSvmModel(ref predictor2);

            var predArray1 = predictions1.ToArray();
            var predArray2 = predictions2.ToArray();
            Array.Sort(predArray2, predArray1);

            for (int i = 0; i < predictions1.Count - 1; i++)
            {
                Assert.IsTrue(IsLessThanOrEqual(predArray1[i], predArray1[i + 1], MaxRelError, Epsilon),
                    "Different ordering of our results and libsvm results");
            }
        }
#endif
        private bool IsLessThanOrEqual(Float a, Float b, Float maxRelError, Float maxAbsError)
        {
            if (a <= b)
                return true;
            Float diff = a - b;
            if (diff <= maxAbsError)
                return true;
            return diff <= maxRelError * a;
        }

        private bool AreEqual(Float a, Float b, Float maxRelError, Float maxAbsError)
        {
            Float diff = Math.Abs(a - b);
            if (diff <= maxAbsError)
                return true;
            Float largest = Math.Max(Math.Abs(a), Math.Abs(b));
            return diff < largest * maxRelError;
        }
    }

#if OLD_TESTS // REVIEW: Some of this should be ported to the new world.
    public sealed partial class TestPredictorsOld
    {
#if OLD_TESTS // REVIEW: Need to port this old time series functionality to the new world.
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Anomaly")]
        [TestCategory("Time Series")]
        public void TimeSeriesAnomalyDetectorTest1()
        {
            const string dir = "Anomaly";
            const string windowDataFile = "AppFailure-unlabeled.windowed.txt";
            const string consName = "LeastSquares.AppFailure-test-out.txt";

            var dataset = TestDatasets.AppFailure;
            var windowDataPath = DeleteOutputPath(dir, windowDataFile);

            //Test window features creation
            var windowsGenerationArgs = "/c CreateInstances " + GetDataPath(dataset.trainFilename) +
                                        " /inst=Text{sep=, name=0 attr=2 nolabel=+} /writer=WindowWriter{size=45 stride=25} /cifile="
                                        + windowDataPath + " /rs=1 /disableTracking=+";
            TestPredictorMain.MainWithArgs(windowsGenerationArgs);
            CheckEquality(dir, windowDataFile);

            //Test Least squares predictor
            ConsoleGrabber consoleGrabber;
            using (consoleGrabber = new ConsoleGrabber())
            {
                var testArgs = "/c Test " + windowDataPath + " /inst=Text{name=0 nolabel=+} /pred=LeastSquaresAnom /rs=1 /disableTracking=+";
                int res = TestPredictorMain.MainWithArgs(testArgs);
                if (res != 0)
                    Log("*** Predictor returned {0}", res);
            }
            string consOutPath = DeleteOutputPath(dir, consName);
            consoleGrabber.Save(consOutPath);
            CheckEqualityNormalized(dir, consName);

            Done();
        }
#endif

#if OLD_TESTS // REVIEW: Figure out what to do with this in the IDV world.
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Anomaly")]
        [TestCategory("Time Series")]
        public void StreamingTimeSeriesAnomalyDetectorTest()
        {
            const string dir = "Anomaly";

            var instArgs = new TlcTextInstances.Arguments();
            CmdParser.ParseArguments("sep=, name=0 nolabel=+", instArgs);
            var dataset = TestDatasets.AppFailure;
            var instances = new TlcTextInstances(instArgs, GetDataPath(dataset.trainFilename));

            var predictor = new OLSAnomalyDetector(45, (Float)0.1);
            var sb = new StringBuilder().AppendLine("Instance\tAnomaly Score\tBad anomaly?");
            foreach (var instance in instances)
            {
                Float score, trend;
                if (predictor.Classify(instance.Features[0], out score, out trend))
                    sb.AppendFormat("{0}\t{1:G4}\t{2}", instance.Name, score, trend > 0).AppendLine(); // trigger alert
            }

            const string outFile = "StreamingLeastSquares-out.txt";
            File.WriteAllText(DeleteOutputPath(dir, outFile), sb.ToString());
            CheckEquality(dir, outFile);

            Done();
        }
#endif

#if OLD_TESTS // REVIEW: Need to port Tremble to the new world.
        /// <summary>
        /// A test for tremble binary classifier using logistic regression 
        /// in leaf and interior nodes
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("TrembleDecisionTree")]
        public void BinaryClassifierTrembleTest()
        {
            var binaryPredictors = new[] { TestLearners.BinaryTrembleDecisionTreeLR };
            var datasets = new[] {
                TestDatasets.breastCancer,
                TestDatasets.adultCatAsAtt,
                TestDatasets.adultSparseWithCatAsAtt,
            };
            RunAllTests(binaryPredictors, datasets);
            Done();
        }

        /// <summary>
        /// A test for tremble multi-class classifier using logistic regression 
        /// in leaf and interior nodes
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("MultiClass")]
        [TestCategory("TrembleDecisionTree")]
        public void MultiClassClassifierTrembleTest()
        {
            var multiClassPredictors = new[] { TestLearners.MultiClassTrembleDecisionTreeLR };
            var multiClassClassificationDatasets = new List<TestDataset>();
            multiClassClassificationDatasets.Add(TestDatasets.iris);
            multiClassClassificationDatasets.Add(TestDatasets.adultCatAsAtt);
            multiClassClassificationDatasets.Add(TestDatasets.adultSparseWithCatAsAtt);
            RunAllTests(multiClassPredictors, multiClassClassificationDatasets);
            Done();
        }

        /// <summary>
        /// A test for tremble default decision tree binary classifier
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("TrembleDecisionTree"), Priority(2)]
        public void BinaryClassifierDecisionTreeTest()
        {
            var binaryPredictors = new[] { TestLearners.BinaryDecisionTreeDefault, TestLearners.BinaryDecisionTreeGini, 
                TestLearners.BinaryDecisionTreePruning, TestLearners.BinaryDecisionTreeModified };
            var binaryClassificationDatasets = new List<TestDataset>();
            binaryClassificationDatasets.Add(TestDatasets.breastCancer);
            binaryClassificationDatasets.Add(TestDatasets.adultCatAsAtt);
            binaryClassificationDatasets.Add(TestDatasets.adultSparseWithCatAsAtt);
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        /// A test for tremble default decision tree binary classifier on weighted data sets
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("Weighting Predictors")]
        [TestCategory("TrembleDecisionTree"), Priority(2)]
        public void BinaryClassifierDecisionTreeWeightingTest()
        {
            var binaryPredictors = new[] { TestLearners.BinaryDecisionTreeDefault, TestLearners.BinaryDecisionTreeGini, 
                TestLearners.BinaryDecisionTreePruning, TestLearners.BinaryDecisionTreeModified, TestLearners.BinaryDecisionTreeRewt };
            var binaryClassificationDatasets = GetDatasetsForClassificationWeightingPredictorsTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        /// A test for tremble default decision tree multi-class classifier
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("MultiClass")]
        [TestCategory("TrembleDecisionTree"), Priority(2)]
        public void MultiClassClassifierDecisionTreeTest()
        {
            var multiClassPredictors = new[] { TestLearners.MultiClassDecisionTreeDefault, TestLearners.MultiClassDecisionTreeGini, 
                TestLearners.MultiClassDecisionTreePruning, TestLearners.MultiClassDecisionTreeModified };
            var multiClassClassificationDatasets = new List<TestDataset>();
            multiClassClassificationDatasets.Add(TestDatasets.iris);
            multiClassClassificationDatasets.Add(TestDatasets.adultCatAsAtt);
            multiClassClassificationDatasets.Add(TestDatasets.adultSparseWithCatAsAtt);
            RunAllTests(multiClassPredictors, multiClassClassificationDatasets);
            Done();
        }

        /// <summary>
        /// A test for tremble default decision tree multi-class classifier on weighted data sets
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("MultiClass")]
        [TestCategory("Weighting Predictors")]
        [TestCategory("TrembleDecisionTree"), Priority(2)]
        public void MultiClassifierDecisionTreeWeightingTest()
        {
            var multiClassPredictors = new[] { TestLearners.MultiClassDecisionTreeDefault, TestLearners.MultiClassDecisionTreeGini, 
                TestLearners.MultiClassDecisionTreePruning, TestLearners.MultiClassDecisionTreeModified };
            var binaryClassificationDatasets = new List<TestDataset>(GetDatasetsForClassificationWeightingPredictorsTest());
            RunAllTests(multiClassPredictors, binaryClassificationDatasets);
            Done();
        }
#endif
    }
#endif

    public sealed partial class TestPredictors
    {
        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("LDSVM")]
        public void BinaryClassifierLDSvmTest()
        {
            var binaryPredictors = new[] { TestLearners.LDSVMDefault };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("LDSVM")]
        public void BinaryClassifierLDSvmNoBiasTest()
        {
            var binaryPredictors = new[] { TestLearners.LDSVMNoBias };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("LDSVM")]
        public void BinaryClassifierLDSvmNoNormTest()
        {
            var binaryPredictors = new[] { TestLearners.LDSvmNoNorm };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        ///A test for binary classifiers
        ///</summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("LDSVM")]
        public void BinaryClassifierLDSvmNoCalibTest()
        {
            var binaryPredictors = new[] { TestLearners.LDSvmNoCalib };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        /// A test for field-aware factorization machine.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Binary")]
        [TestCategory("FieldAwareFactorizationMachine")]
        public void BinaryClassifierFieldAwareFactorizationMachineTest()
        {
            var binaryPredictors = new[] { TestLearners.FieldAwareFactorizationMachine };
            var binaryClassificationDatasets = GetDatasetsForBinaryClassifierBaseTest();
            RunAllTests(binaryPredictors, binaryClassificationDatasets);
            Done();
        }

        /// <summary>
        /// Multiclass Naive Bayes test.
        /// </summary>
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("Multiclass")]
        [TestCategory("Multi Class Naive Bayes Classifier")]
        public void MulticlassNaiveBayes()
        {
            RunOneAllTests(TestLearners.MulticlassNaiveBayesClassifier, TestDatasets.breastCancerSparseBinaryFeatures);
            Done();
        }
    }

#if OLD_TESTS // REVIEW: We should have some tests that verify we can't deserialize old models.
    public sealed partial class TestPredictorsOld
    {
        [Fact(Skip = "Need CoreTLC specific baseline update")]
        [TestCategory("CreateInstances")]
        [TestCategory("FeatureHandler")]
        public void TestFeatureHandlerModelReuse()
        {
            string trainData = GetDataPath(TestDatasets.breastCancer.trainFilename);
            string dataModelFile = DeleteOutputPath(TestContext.TestName + "-data-model.zip");
            string ciFile = DeleteOutputPath(TestContext.TestName + "-ci.tsv");
            string argsString = string.Format(
                "/c CreateInstances {0} /inst Text{{text=1,2,3}} /m {1} /cifile {2}",
                trainData,
                dataModelFile,
                ciFile);
            var args = new TLCArguments();
            Assert.IsTrue(CmdParser.ParseArguments(argsString, args));
            RunExperiments.Run(args);

            // REVIEW: think of a test that would distinguish more dramatically the case when /im works and when it doesn't
            // Right now the only difference is in the output of the feature handler training.
            RunAllTests(
                new[] { TestLearners.logisticRegression_tlOld },
                new[] { TestDatasets.breastCancer },
                new[] { string.Format("/inst Text{{text=1,2,3}} /im {0}", dataModelFile) },
                "feature-handler-reuse"
                );

            Done();
        }
    }
#endif
}
