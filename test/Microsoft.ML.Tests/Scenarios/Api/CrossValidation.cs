using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using System;
using System.Collections.Generic;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Cross-validation: Have a mechanism to do cross validation, that is, you come up with
        /// a data source (optionally with stratification column), come up with an instantiable transform
        /// and trainer pipeline, and it will handle (1) splitting up the data, (2) training the separate
        /// pipelines on in-fold data, (3) scoring on the out-fold data, (4) returning the set of
        /// evaluations and optionally trained pipes. (People always want metrics out of xfold,
        /// they sometimes want the actual models too.)
        /// </summary>
        [Fact]
        void CrossValidation()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            int numFolds = 5;
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline.
                var loader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(dataPath));

                var trans = TextTransform.Create(env, MakeSentimentTextTransformArgs(false), loader);
                var random = new GenerateNumberTransform(env, trans, "StratificationColumn");
                // Train.
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1,
                    ConvergenceTolerance = 1f
                });

                // Auto-caching.
                IDataView trainData = trainer.Info.WantCaching ? (IDataView)new CacheDataView(env, random, prefetch: null) : random;
                var metrics = new List<BinaryClassificationMetrics>();
                for (int fold = 0; fold < numFolds; fold++)
                {
                    var trainFilter = new RangeFilter(env, new RangeFilter.Arguments()
                    {
                        Column = "StratificationColumn",
                        Min = (Double)fold / numFolds,
                        Max = (Double)(fold + 1) / numFolds,
                        Complement = true
                    }, trainData);

                    // Auto-normalization.
                    var trainRoles = new RoleMappedData(trainFilter, label: "Label", feature: "Features");
                    NormalizeTransform.CreateIfNeeded(env, ref trainRoles, trainer);

                    var predictor = trainer.Train(new Runtime.TrainContext(trainRoles));
                    var testFilter = new RangeFilter(env, new RangeFilter.Arguments()
                    {
                        Column = "StratificationColumn",
                        Min = (Double)fold / numFolds,
                        Max = (Double)(fold + 1) / numFolds,
                        Complement = false
                    }, trainData);
                    // Auto-normalization.
                    var testRoles = new RoleMappedData(testFilter, label: "Label", feature: "Features");
                    NormalizeTransform.CreateIfNeeded(env, ref testRoles, trainer);

                    IDataScorerTransform scorer = ScoreUtils.GetScorer(predictor, testRoles, env, testRoles.Schema);

                    BinaryClassifierMamlEvaluator eval = new BinaryClassifierMamlEvaluator(env, new BinaryClassifierMamlEvaluator.Arguments() { });
                    var dataEval = new RoleMappedData(scorer, testRoles.Schema.GetColumnRoleNames(), opt: true);
                    var dict = eval.Evaluate(dataEval);
                    var foldMetrics = BinaryClassificationMetrics.FromMetrics(env, dict["OverallMetrics"], dict["ConfusionMatrix"]);
                    metrics.AddRange(foldMetrics);
                }
            }
        }
    }
}
