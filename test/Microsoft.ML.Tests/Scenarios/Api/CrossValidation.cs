// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using System;
using System.Collections.Generic;
using System.Linq;
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
                var loader = TextLoader.ReadFile(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(dataPath));

                var text = TextTransform.Create(env, MakeSentimentTextTransformArgs(false), loader);
                IDataView trans = new GenerateNumberTransform(env, text, "StratificationColumn");
                // Train.
                var trainer = new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments
                {
                    NumThreads = 1,
                    ConvergenceTolerance = 1f
                });


                var metrics = new List<BinaryClassificationMetrics>();
                for (int fold = 0; fold < numFolds; fold++)
                {
                    IDataView trainPipe = new RangeFilter(env, new RangeFilter.Arguments()
                    {
                        Column = "StratificationColumn",
                        Min = (Double)fold / numFolds,
                        Max = (Double)(fold + 1) / numFolds,
                        Complement = true
                    }, trans);
                    trainPipe = new OpaqueDataView(trainPipe);
                    var trainData = new RoleMappedData(trainPipe, label: "Label", feature: "Features");
                    // Auto-normalization.
                    NormalizeTransform.CreateIfNeeded(env, ref trainData, trainer);
                    var preCachedData = trainData;
                    // Auto-caching.
                    if (trainer.Info.WantCaching)
                    {
                        var prefetch = trainData.Schema.GetColumnRoles().Select(kc => kc.Value.Index).ToArray();
                        var cacheView = new CacheDataView(env, trainData.Data, prefetch);
                        // Because the prefetching worked, we know that these are valid columns.
                        trainData = new RoleMappedData(cacheView, trainData.Schema.GetColumnRoleNames());
                    }

                    var predictor = trainer.Train(new Runtime.TrainContext(trainData));
                    IDataView testPipe = new RangeFilter(env, new RangeFilter.Arguments()
                    {
                        Column = "StratificationColumn",
                        Min = (Double)fold / numFolds,
                        Max = (Double)(fold + 1) / numFolds,
                        Complement = false
                    }, trans);
                    testPipe = new OpaqueDataView(testPipe);
                    var pipe = ApplyTransformUtils.ApplyAllTransformsToData(env, preCachedData.Data, testPipe, trainPipe);

                    var testRoles = new RoleMappedData(pipe, trainData.Schema.GetColumnRoleNames());

                    IDataScorerTransform scorer = ScoreUtils.GetScorer(predictor, testRoles, env, testRoles.Schema);

                    BinaryClassifierMamlEvaluator eval = new BinaryClassifierMamlEvaluator(env, new BinaryClassifierMamlEvaluator.Arguments() { });
                    var dataEval = new RoleMappedData(scorer, testRoles.Schema.GetColumnRoleNames(), opt: true);
                    var dict = eval.Evaluate(dataEval);
                    var foldMetrics = BinaryClassificationMetrics.FromMetrics(env, dict["OverallMetrics"], dict["ConfusionMatrix"]);
                    metrics.Add(foldMetrics.Single());
                }
            }
        }
    }
}
