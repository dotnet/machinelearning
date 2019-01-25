// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class PipelineSuggester
    {
        private const int TopKTrainers = 3;

        public static Pipeline GetNextPipeline(IEnumerable<PipelineRunResult> history,
            (string, ColumnType, ColumnPurpose, ColumnDimensions)[] columns,
            TaskKind task,
            int iterationsRemaining,
            bool isMaximizingMetric = true)
        {
            var inferredHistory = history.Select(r => InferredPipelineRunResult.FromPipelineRunResult(r));
            var nextInferredPipeline = GetNextInferredPipeline(inferredHistory, columns, task, iterationsRemaining, isMaximizingMetric);
            return nextInferredPipeline.ToPipeline();
        }

        public static InferredPipeline GetNextInferredPipeline(IEnumerable<InferredPipelineRunResult> history,
            (string, ColumnType, ColumnPurpose, ColumnDimensions)[] columns,
            TaskKind task,
            int iterationsRemaining,
            bool isMaximizingMetric = true)
        {
            var context = new MLContext();

            var availableTrainers = RecipeInference.AllowedTrainers(context, task, history.Count() + iterationsRemaining);
            var transforms = CalculateTransforms(context, columns, task);

            // if we haven't run all pipelines once
            if (history.Count() < availableTrainers.Count())
            {
                return GetNextFirstStagePipeline(history, availableTrainers, transforms);
            }

            // get next trainer
            var topTrainers = GetTopTrainers(history, availableTrainers, isMaximizingMetric);
            var nextTrainerIndex = (history.Count() - availableTrainers.Count()) % topTrainers.Count();
            var trainer = topTrainers.ElementAt(nextTrainerIndex).Clone();

            // make sure we have not seen pipeline before.
            // repeat until passes or runs out of chances.
            var visitedPipelines = new HashSet<InferredPipeline>(history.Select(h => h.Pipeline));
            const int maxNumberAttempts = 10;
            var count = 0;
            do
            {
                SampleHyperparameters(trainer, history, isMaximizingMetric);
                var pipeline = new InferredPipeline(transforms, trainer);
                if(!visitedPipelines.Contains(pipeline))
                {
                    return pipeline;
                }
            } while (++count <= maxNumberAttempts);

            return null;
        }
        
        /// <summary>
        /// Get top trainers from first stage
        /// </summary>
        private static IEnumerable<SuggestedTrainer> GetTopTrainers(IEnumerable<InferredPipelineRunResult> history, 
            IEnumerable<SuggestedTrainer> availableTrainers,
            bool isMaximizingMetric)
        {
            // narrow history to first stage runs
            history = history.Take(availableTrainers.Count());

            history = history.GroupBy(r => r.Pipeline.Trainer.TrainerName).Select(g => g.First());
            IEnumerable<InferredPipelineRunResult> sortedHistory = history.OrderBy(r => r.Score);
            if(isMaximizingMetric)
            {
                sortedHistory = sortedHistory.Reverse();
            }
            var topTrainers = sortedHistory.Take(TopKTrainers).Select(r => r.Pipeline.Trainer);
            return topTrainers;
        }

        private static InferredPipeline GetNextFirstStagePipeline(IEnumerable<InferredPipelineRunResult> history,
            IEnumerable<SuggestedTrainer> availableTrainers,
            IEnumerable<SuggestedTransform> transforms)
        {
            var trainer = availableTrainers.ElementAt(history.Count());
            return new InferredPipeline(transforms, trainer);
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

        private static void SampleHyperparameters(SuggestedTrainer trainer, IEnumerable<InferredPipelineRunResult> history, bool isMaximizingMetric)
        {
            var sps = ConvertToValueGenerators(trainer.SweepParams);
            var sweeper = new SmacSweeper(
                new SmacSweeper.Arguments
                {
                    SweptParameters = sps
                });

            IEnumerable<InferredPipelineRunResult> historyToUse = history
                .Where(r => r.RunSucceded && r.Pipeline.Trainer.TrainerName == trainer.TrainerName && r.Pipeline.Trainer.HyperParamSet != null);

            // get new set of hyperparameter values
            var proposedParamSet = sweeper.ProposeSweeps(1, historyToUse.Select(h => h.ToRunResult(isMaximizingMetric))).First();

            // associate proposed param set with trainer, so that smart hyperparam
            // sweepers (like KDO) can map them back.
            trainer.SetHyperparamValues(proposedParamSet);
        }

        private static IEnumerable<SuggestedTransform> CalculateTransforms(MLContext context,
            (string, ColumnType, ColumnPurpose, ColumnDimensions)[] columns,
            TaskKind task)
        {
            var transforms = TransformInferenceApi.InferTransforms(context, columns).ToList();
            // this is a work-around for ML.NET bug tracked by https://github.com/dotnet/machinelearning/issues/1969
            if (task == TaskKind.MulticlassClassification)
            {
                var labelCol = columns.First(c => c.Item3 == ColumnPurpose.Label).Item1;
                var transform = ValueToKeyMappingExtension.CreateSuggestedTransform(context, labelCol, labelCol);
                transforms.Add(transform);
            }
            return transforms;
        }
    }
}