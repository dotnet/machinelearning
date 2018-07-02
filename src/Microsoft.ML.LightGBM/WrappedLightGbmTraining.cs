// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.LightGBM
{
    /// <summary>
    /// Helpers to train a booster with given parameters.
    /// </summary>
    internal static class WrappedLightGbmTraining
    {
        /// <summary>
        /// Train and return a booster.
        /// </summary>
        public static Booster Train(IChannel ch, IProgressChannel pch,
            Dictionary<string, object> parameters, Dataset dtrain, Dataset dvalid = null, int numIteration = 100,
            bool verboseEval = true, int earlyStoppingRound = 0)
        {
            // create Booster.
            Booster bst = new Booster(parameters, dtrain, dvalid);

            // Disable early stopping if we don't have validation data.
            if (dvalid == null && earlyStoppingRound > 0)
            {
                earlyStoppingRound = 0;
                ch.Warning("Validation dataset not present, early stopping will be disabled.");
            }

            int bestIter = 0;
            double bestScore = double.MaxValue;
            double factorToSmallerBetter = 1.0;

            if (earlyStoppingRound > 0 && ((string)parameters["metric"] == "auc"
                || (string)parameters["metric"] == "ndcg"
                || (string)parameters["metric"] == "map"))
            {
                factorToSmallerBetter = -1.0;
            }

            const int evalFreq = 50;

            var metrics = new List<string>() { "Iteration" };
            var units = new List<string>() { "iterations" };

            if (verboseEval)
            {
                ch.Assert(parameters.ContainsKey("metric"));
                metrics.Add("Training-" + parameters["metric"]);
                if (dvalid != null)
                    metrics.Add("Validation-" + parameters["metric"]);
            }

            var header = new ProgressHeader(metrics.ToArray(), units.ToArray());

            int iter = 0;
            double trainError = double.NaN;
            double validError = double.NaN;
            pch.SetHeader(header, e =>
            {
                e.SetProgress(0, iter, numIteration);
                if (verboseEval)
                {
                    e.SetProgress(1, trainError);
                    if (dvalid != null)
                        e.SetProgress(2, validError);
                }
            });
            for (iter = 0; iter < numIteration; ++iter)
            {
                if (bst.Update())
                    break;

                if (earlyStoppingRound > 0)
                {
                    validError = bst.EvalValid();
                    if (validError * factorToSmallerBetter < bestScore)
                    {
                        bestScore = validError * factorToSmallerBetter;
                        bestIter = iter;
                    }
                    if (iter - bestIter >= earlyStoppingRound)
                    {
                        ch.Info($"Met early stopping, best iteration: {bestIter + 1}, best score: {bestScore / factorToSmallerBetter}");
                        break;
                    }
                }
                if ((iter + 1) % evalFreq == 0)
                {
                    if (verboseEval)
                    {
                        trainError = bst.EvalTrain();
                        if (dvalid == null)
                            pch.Checkpoint(new double?[] { iter + 1, trainError });
                        else
                        {
                            if (earlyStoppingRound == 0)
                                validError = bst.EvalValid();
                            pch.Checkpoint(new double?[] { iter + 1,
                                trainError, validError });
                        }
                    }
                    else
                        pch.Checkpoint(new double?[] { iter + 1 });
                }
            }
            // Set the BestIteration.
            if (iter != numIteration && earlyStoppingRound > 0)
            {
                bst.BestIteration = bestIter + 1;
            }
            return bst;
        }
    }
}
