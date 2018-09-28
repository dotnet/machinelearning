// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.TestFramework;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    [Collection("TestPipelineSweeper and TestAutoInference should not be run at the same time")]
    public sealed class TestAutoInference : BaseTestBaseline
    {
        public TestAutoInference(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        [TestCategory("EntryPoints")]
        public void TestLearn()
        {
            using (var env = new LocalEnvironment()
                .AddStandardComponents()) // AutoInference.InferPipelines uses ComponentCatalog to read text data
            {
                string pathData = GetDataPath("adult.train");
                string pathDataTest = GetDataPath("adult.test");
                int numOfSampleRows = 1000;
                int batchSize = 5;
                int numIterations = 10;
                int numTransformLevels = 3;
                SupportedMetric metric = PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.Auc);

                // Using the simple, uniform random sampling (with replacement) engine
                PipelineOptimizerBase autoMlEngine = new UniformRandomEngine(env);

                // Test initial learning
                var amls = AutoInference.InferPipelines(env, autoMlEngine, pathData, "", out var schema, numTransformLevels, batchSize,
                    metric, out var bestPipeline, numOfSampleRows, new IterationTerminator(numIterations / 2), MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer);
                env.Check(amls.GetAllEvaluatedPipelines().Length == numIterations / 2);

                // Resume learning
                amls.UpdateTerminator(new IterationTerminator(numIterations));
                bestPipeline = amls.InferPipelines(numTransformLevels, batchSize, numOfSampleRows);
                env.Check(amls.GetAllEvaluatedPipelines().Length == numIterations);

                // Use best pipeline for another task
                var inputFileTrain = new SimpleFileHandle(env, pathData, false, false);
#pragma warning disable 0618
                var datasetTrain = ImportTextData.ImportText(env,
                    new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data;
                var inputFileTest = new SimpleFileHandle(env, pathDataTest, false, false);
                var datasetTest = ImportTextData.ImportText(env,
                    new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data;
#pragma warning restore 0618

                // REVIEW: Theoretically, it could be the case that a new, very bad learner is introduced and
                // we get unlucky and only select it every time, such that this test fails. Not
                // likely at all, but a non-zero probability. Should be ok, since all current learners are returning d > .80.
                bestPipeline.RunTrainTestExperiment(datasetTrain, datasetTest, metric, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer,
                    out var testMetricValue, out var trainMtericValue);
                env.Check(testMetricValue > 0.2);
            }
            Done();
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        public void TestTextDatasetLearn()
        {
            using (var env = new LocalEnvironment()
                .AddStandardComponents()) // AutoInference uses ComponentCatalog to find all learners
            {
                string pathData = GetDataPath(@"../UnitTest/tweets_labeled_10k_test_validation.tsv");
                int batchSize = 5;
                int numIterations = 35;
                int numTransformLevels = 1;
                int numSampleRows = 100;
                SupportedMetric metric = PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.AccuracyMicro);

                // Using the simple, uniform random sampling (with replacement) engine
                PipelineOptimizerBase autoMlEngine = new UniformRandomEngine(env);

                // Test initial learning
                var amls = AutoInference.InferPipelines(env, autoMlEngine, pathData, "", out var _, numTransformLevels, batchSize,
                metric, out var _, numSampleRows, new IterationTerminator(numIterations),
                MacroUtils.TrainerKinds.SignatureMultiClassClassifierTrainer);
                env.Check(amls.GetAllEvaluatedPipelines().Length == numIterations);
            }
            Done();
        }

        [Fact]
        public void TestPipelineNodeCloning()
        {
            using (var env = new LocalEnvironment()
                .AddStandardComponents()) // RecipeInference.AllowedLearners uses ComponentCatalog to find all learners
            {
                var lr1 = RecipeInference
                    .AllowedLearners(env, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer)
                    .First(learner => learner.PipelineNode != null && learner.LearnerName.Contains("LogisticRegression"));

                var sdca1 = RecipeInference
                    .AllowedLearners(env, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer)
                    .First(learner => learner.PipelineNode != null && learner.LearnerName.Contains("StochasticDualCoordinateAscent"));

                // Clone and change hyperparam values
                var lr2 = lr1.Clone();
                lr1.PipelineNode.SweepParams[0].RawValue = 1.2f;
                lr2.PipelineNode.SweepParams[0].RawValue = 3.5f;
                var sdca2 = sdca1.Clone();
                sdca1.PipelineNode.SweepParams[0].RawValue = 3;
                sdca2.PipelineNode.SweepParams[0].RawValue = 0;

                // Make sure the changes are propagated to entry point objects
                env.Check(lr1.PipelineNode.UpdateProperties());
                env.Check(lr2.PipelineNode.UpdateProperties());
                env.Check(sdca1.PipelineNode.UpdateProperties());
                env.Check(sdca2.PipelineNode.UpdateProperties());
                env.Check(lr1.PipelineNode.CheckEntryPointStateMatchesParamValues());
                env.Check(lr2.PipelineNode.CheckEntryPointStateMatchesParamValues());
                env.Check(sdca1.PipelineNode.CheckEntryPointStateMatchesParamValues());
                env.Check(sdca2.PipelineNode.CheckEntryPointStateMatchesParamValues());

                // Make sure second object's set of changes didn't overwrite first object's
                env.Check(!lr1.PipelineNode.SweepParams[0].RawValue.Equals(lr2.PipelineNode.SweepParams[0].RawValue));
                env.Check(!sdca2.PipelineNode.SweepParams[0].RawValue.Equals(sdca1.PipelineNode.SweepParams[0].RawValue));
            }
        }

        [Fact]
        public void TestHyperparameterFreezing()
        {
            string pathData = GetDataPath("adult.train");
            int numOfSampleRows = 1000;
            int batchSize = 1;
            int numIterations = 10;
            int numTransformLevels = 3;
            using (var env = new LocalEnvironment()
                .AddStandardComponents()) // AutoInference uses ComponentCatalog to find all learners
            {
                SupportedMetric metric = PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.Auc);

                // Using the simple, uniform random sampling (with replacement) brain
                PipelineOptimizerBase autoMlBrain = new UniformRandomEngine(env);

                // Run initial experiments
                var amls = AutoInference.InferPipelines(env, autoMlBrain, pathData, "", out var _, numTransformLevels, batchSize,
                    metric, out var bestPipeline, numOfSampleRows, new IterationTerminator(numIterations),
                    MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer);

                // Clear results
                amls.ClearEvaluatedPipelines();

                // Get space, remove transforms and all but one learner, freeze hyperparameters on learner.
                var space = amls.GetSearchSpace();
                var transforms = space.Item1.Where(t =>
                    t.ExpertType != typeof(TransformInference.Experts.Categorical)).ToArray();
                var learners = new[] { space.Item2.First() };
                var hyperParam = learners[0].PipelineNode.SweepParams.First();
                var frozenParamValue = hyperParam.RawValue;
                hyperParam.Frozen = true;
                amls.UpdateSearchSpace(learners, transforms);

                // Allow for one more iteration
                amls.UpdateTerminator(new IterationTerminator(numIterations + 1));

                // Do learning. Only retained learner should be left in all pipelines.
                bestPipeline = amls.InferPipelines(numTransformLevels, batchSize, numOfSampleRows);

                // Make sure all pipelines have retained learner
                Assert.True(amls.GetAllEvaluatedPipelines().All(p => p.Learner.LearnerName == learners[0].LearnerName));

                // Make sure hyperparameter value did not change
                Assert.NotNull(bestPipeline);
                Assert.Equal(bestPipeline.Learner.PipelineNode.SweepParams.First().RawValue, frozenParamValue);
            }
        }

        [Fact(Skip = "Dataset not available.")]
        public void TestRegressionPipelineWithMinimizingMetric()
        {
            string pathData = GetDataPath("../Housing (regression)/housing.txt");
            int numOfSampleRows = 100;
            int batchSize = 5;
            int numIterations = 10;
            int numTransformLevels = 1;
            using (var env = new LocalEnvironment()
                .AddStandardComponents()) // AutoInference uses ComponentCatalog to find all learners
            {
                SupportedMetric metric = PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.AccuracyMicro);

                // Using the simple, uniform random sampling (with replacement) brain
                PipelineOptimizerBase autoMlBrain = new UniformRandomEngine(env);

                // Run initial experiments
                var amls = AutoInference.InferPipelines(env, autoMlBrain, pathData, "", out var _, numTransformLevels, batchSize,
                metric, out var bestPipeline, numOfSampleRows, new IterationTerminator(numIterations),
                MacroUtils.TrainerKinds.SignatureRegressorTrainer);

                // Allow for one more iteration
                amls.UpdateTerminator(new IterationTerminator(numIterations + 1));

                // Do learning. Only retained learner should be left in all pipelines.
                bestPipeline = amls.InferPipelines(numTransformLevels, batchSize, numOfSampleRows);

                // Make sure hyperparameter value did not change
                Assert.NotNull(bestPipeline);
                Assert.True(amls.GetAllEvaluatedPipelines().All(
                p => p.PerformanceSummary.MetricValue >= bestPipeline.PerformanceSummary.MetricValue));
            }
        }

        [Fact]
        public void TestLearnerConstrainingByName()
        {
            string pathData = GetDataPath("adult.train");
            int numOfSampleRows = 1000;
            int batchSize = 1;
            int numIterations = 1;
            int numTransformLevels = 2;
            var retainedLearnerNames = new[] { $"LogisticRegressionBinaryClassifier", $"FastTreeBinaryClassifier" };
            using (var env = new LocalEnvironment()
                .AddStandardComponents()) // AutoInference uses ComponentCatalog to find all learners
            {
                SupportedMetric metric = PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.Auc);

                // Using the simple, uniform random sampling (with replacement) brain.
                PipelineOptimizerBase autoMlBrain = new UniformRandomEngine(env);

                // Run initial experiment.
                var amls = AutoInference.InferPipelines(env, autoMlBrain, pathData, "", out var _,
                numTransformLevels, batchSize, metric, out var _, numOfSampleRows,
                new IterationTerminator(numIterations), MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer);

                // Keep only logistic regression and FastTree.
                amls.KeepSelectedLearners(retainedLearnerNames);
                var space = amls.GetSearchSpace();

                // Make sure only learners left are those retained.
                Assert.Equal(retainedLearnerNames.Length, space.Item2.Length);
                Assert.True(space.Item2.All(l => retainedLearnerNames.Any(r => r == l.LearnerName)));
            }
        }

        [Fact]
        public void TestMinimizingMetricTransformations()
        {
            var values = new[] { 100d, 10d, -2d, -1d, 5.8d, -3.1d };
            var maxWeight = values.Max();
            var processed = values.Select(v => AutoMlUtils.ProcessWeight(v, maxWeight, false));
            var expectedResult = new[] { 0d, 90d, 102d, 101d, 94.2d, 103.1d };

            Assert.True(processed.Select((x, idx) => System.Math.Abs(x - expectedResult[idx]) < 0.001).All(r => r));
        }
    }
}
