// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.AutoML
{
    internal static class PipelineSuggester
    {
        private const int TopKTrainers = 3;

        public static Pipeline GetNextPipeline(MLContext context,
            IEnumerable<PipelineScore> history,
            DatasetColumnInfo[] columns,
            TaskKind task,
            IChannel logger,
            bool isMaximizingMetric = true)
        {
            var inferredHistory = history.Select(r => SuggestedPipelineRunDetail.FromPipelineRunResult(context, r));
            var nextInferredPipeline = GetNextInferredPipeline(context, inferredHistory, columns, task, isMaximizingMetric, CacheBeforeTrainer.Auto, logger);
            return nextInferredPipeline?.ToPipeline();
        }

        public static SuggestedPipeline GetNextInferredPipeline(MLContext context,
            IEnumerable<SuggestedPipelineRunDetail> history,
            DatasetColumnInfo[] columns,
            TaskKind task,
            bool isMaximizingMetric,
            CacheBeforeTrainer cacheBeforeTrainer,
            IChannel logger,
            IEnumerable<TrainerName> trainerAllowList = null)
        {
            var availableTrainers = RecipeInference.AllowedTrainers(context, task,
                ColumnInformationUtil.BuildColumnInfo(columns), trainerAllowList);
            var transforms = TransformInferenceApi.InferTransforms(context, task, columns).ToList();
            var transformsPostTrainer = TransformInferenceApi.InferTransformsPostTrainer(context, task, columns).ToList();

            // if we haven't run all pipelines once
            if (history.Count() < availableTrainers.Count())
            {
                return GetNextFirstStagePipeline(context, history, availableTrainers, transforms, transformsPostTrainer, cacheBeforeTrainer);
            }

            // get top trainers from stage 1 runs
            var topTrainers = GetTopTrainers(history, availableTrainers, isMaximizingMetric);

            // sort top trainers by # of times they've been run, from lowest to highest
            var orderedTopTrainers = OrderTrainersByNumTrials(history, topTrainers);

            // keep as hash set of previously visited pipelines
            var visitedPipelines = new HashSet<SuggestedPipeline>(history.Select(h => h.Pipeline));

            // iterate over top trainers (from least run to most run),
            // to find next pipeline
            foreach (var trainer in orderedTopTrainers)
            {
                var newTrainer = trainer.Clone();

                // repeat until passes or runs out of chances
                const int maxNumberAttempts = 10;
                var count = 0;
                do
                {
                    // sample new hyperparameters for the learner
                    if (!SampleHyperparameters(context, newTrainer, history, isMaximizingMetric, logger))
                    {
                        // if unable to sample new hyperparameters for the learner
                        // (ie SMAC returned 0 suggestions), break
                        break;
                    }

                    var suggestedPipeline = SuggestedPipelineBuilder.Build(context, transforms, transformsPostTrainer, newTrainer, cacheBeforeTrainer);

                    // make sure we have not seen pipeline before
                    if (!visitedPipelines.Contains(suggestedPipeline))
                    {
                        return suggestedPipeline;
                    }
                } while (++count <= maxNumberAttempts);
            }

            return null;
        }

        /// <summary>
        /// Get top trainers from first stage
        /// </summary>
        private static IEnumerable<SuggestedTrainer> GetTopTrainers(IEnumerable<SuggestedPipelineRunDetail> history,
            IEnumerable<SuggestedTrainer> availableTrainers,
            bool isMaximizingMetric)
        {
            // narrow history to first stage runs that succeeded
            history = history.Take(availableTrainers.Count()).Where(x => x.RunSucceeded);

            history = history.GroupBy(r => r.Pipeline.Trainer.TrainerName).Select(g => g.First());
            IEnumerable<SuggestedPipelineRunDetail> sortedHistory = history.OrderBy(r => r.Score);
            if (isMaximizingMetric)
            {
                sortedHistory = sortedHistory.Reverse();
            }
            var topTrainers = sortedHistory.Take(TopKTrainers).Select(r => r.Pipeline.Trainer);
            return topTrainers;
        }

        private static IEnumerable<SuggestedTrainer> OrderTrainersByNumTrials(IEnumerable<SuggestedPipelineRunDetail> history,
            IEnumerable<SuggestedTrainer> selectedTrainers)
        {
            var selectedTrainerNames = new HashSet<TrainerName>(selectedTrainers.Select(t => t.TrainerName));
            return history.Where(h => selectedTrainerNames.Contains(h.Pipeline.Trainer.TrainerName))
                .GroupBy(h => h.Pipeline.Trainer.TrainerName)
                .OrderBy(x => x.Count())
                .Select(x => x.First().Pipeline.Trainer);
        }

        private static SuggestedPipeline GetNextFirstStagePipeline(MLContext context,
            IEnumerable<SuggestedPipelineRunDetail> history,
            IEnumerable<SuggestedTrainer> availableTrainers,
            ICollection<SuggestedTransform> transforms,
            ICollection<SuggestedTransform> transformsPostTrainer,
            CacheBeforeTrainer cacheBeforeTrainer)
        {
            var trainer = availableTrainers.ElementAt(history.Count());
            return SuggestedPipelineBuilder.Build(context, transforms, transformsPostTrainer, trainer, cacheBeforeTrainer);
        }

        private static IValueGenerator[] ConvertToValueGenerators(IEnumerable<SweepableParam> hps)
        {
            var results = new IValueGenerator[hps.Count()];

            for (int i = 0; i < hps.Count(); i++)
            {
                switch (hps.ElementAt(i))
                {
                    case SweepableDiscreteParam dp:
                        var dpArgs = new DiscreteParamArguments()
                        {
                            Name = dp.Name,
                            Values = dp.Options.Select(o => o.ToString()).ToArray()
                        };
                        results[i] = new DiscreteValueGenerator(dpArgs);
                        break;

                    case SweepableFloatParam fp:
                        var fpArgs = new FloatParamArguments()
                        {
                            Name = fp.Name,
                            Min = fp.Min,
                            Max = fp.Max,
                            LogBase = fp.IsLogScale,
                        };
                        if (fp.NumSteps.HasValue)
                        {
                            fpArgs.NumSteps = fp.NumSteps.Value;
                        }
                        if (fp.StepSize.HasValue)
                        {
                            fpArgs.StepSize = fp.StepSize.Value;
                        }
                        results[i] = new FloatValueGenerator(fpArgs);
                        break;

                    case SweepableLongParam lp:
                        var lpArgs = new LongParamArguments()
                        {
                            Name = lp.Name,
                            Min = lp.Min,
                            Max = lp.Max,
                            LogBase = lp.IsLogScale
                        };
                        if (lp.NumSteps.HasValue)
                        {
                            lpArgs.NumSteps = lp.NumSteps.Value;
                        }
                        if (lp.StepSize.HasValue)
                        {
                            lpArgs.StepSize = lp.StepSize.Value;
                        }
                        results[i] = new LongValueGenerator(lpArgs);
                        break;
                }
            }
            return results;
        }

        /// <summary>
        /// Samples new hyperparameters for the trainer, and sets them.
        /// Returns true if success (new hyperparameters were suggested and set). Else, returns false.
        /// </summary>
        private static bool SampleHyperparameters(MLContext context, SuggestedTrainer trainer,
            IEnumerable<SuggestedPipelineRunDetail> history, bool isMaximizingMetric, IChannel logger)
        {
            try
            {
                var sps = ConvertToValueGenerators(trainer.SweepParams);
                var sweeper = new SmacSweeper(context,
                    new SmacSweeper.Arguments
                    {
                        SweptParameters = sps
                    });

                IEnumerable<SuggestedPipelineRunDetail> historyToUse = history
                    .Where(r => r.RunSucceeded && r.Pipeline.Trainer.TrainerName == trainer.TrainerName &&
                                r.Pipeline.Trainer.HyperParamSet != null &&
                                r.Pipeline.Trainer.HyperParamSet.Any() &&
                                FloatUtils.IsFinite(r.Score));

                // get new set of hyperparameter values
                var proposedParamSet = sweeper.ProposeSweeps(1, historyToUse.Select(h => h.ToRunResult(isMaximizingMetric))).FirstOrDefault();
                if (!proposedParamSet.Any())
                {
                    return false;
                }

                // associate proposed parameter set with trainer, so that smart hyperparameter
                // sweepers (like KDO) can map them back.
                trainer.SetHyperparamValues(proposedParamSet);

                return true;
            }
            catch (Exception ex)
            {
                logger.Error($"SampleHyperparameters failed with exception: {ex}");
                throw;
            }
        }
    }
}
