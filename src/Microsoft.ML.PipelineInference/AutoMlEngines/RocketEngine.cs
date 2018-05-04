// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Runtime.Sweeper.Algorithms;

[assembly: EntryPointModule(typeof(RocketEngine.Arguments))]

namespace Microsoft.ML.Runtime.PipelineInference
{
    public class RocketEngine : PipelineOptimizerBase
    {
        private int _currentStage;
        private int _remainingSecondStageTrials;
        private int _remainingThirdStageTrials;
        private readonly int _topK;
        private readonly bool _randomInit;
        private readonly int _numInitPipelines;
        private readonly Dictionary<string, IPipelineOptimizer> _secondaryEngines;
        private readonly Dictionary<string, ISweeper> _hyperSweepers;
        private enum Stages
        {
            First,
            Second,
            Third
        }

        [TlcModule.Component(Name = "Rocket", FriendlyName = "Rocket Engine",
            Desc = "AutoML engine that consists of distinct, hierarchical stages of operation.")]
        public sealed class Arguments : ISupportIPipelineOptimizerFactory
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "topk",
                HelpText = "Number of learners to retain for second stage.", SortOrder = 1)]
            public int TopKLearners = 2;

            [Argument(ArgumentType.AtMostOnce, ShortName = "stage2num",
                HelpText = "Number of trials for retained second stage learners.", SortOrder = 2)]
            public int SecondRoundTrialsPerLearner = 5;

            [Argument(ArgumentType.AtMostOnce, ShortName = "randinit",
                HelpText = "Use random initialization only.", SortOrder = 3)]
            public bool RandomInitialization = false;

            [Argument(ArgumentType.AtMostOnce, ShortName = "numinitseeds",
                HelpText = "Number of initilization pipelines, used for random initialization only.", SortOrder = 4)]
            public int NumInitializationPipelines = new KdoSweeper.Arguments().NumberInitialPopulation;

            public IPipelineOptimizer CreateComponent(IHostEnvironment env) => new RocketEngine(env, this);
        }

        public RocketEngine(IHostEnvironment env, Arguments args)
            : base(env, env.Register("RocketEngine(AutoML)"))
        {
            _currentStage = (int)Stages.First;
            _topK = args.TopKLearners;
            _remainingSecondStageTrials = _topK * args.SecondRoundTrialsPerLearner;
            _remainingThirdStageTrials = 5 * _topK;
            _randomInit = args.RandomInitialization;
            _numInitPipelines = args.NumInitializationPipelines;
            _hyperSweepers = new Dictionary<string, ISweeper>();
            _secondaryEngines = new Dictionary<string, IPipelineOptimizer>
            {
                [nameof(UniformRandomEngine)] = new UniformRandomEngine(env),
                [nameof(DefaultsEngine)] = new DefaultsEngine(env, new DefaultsEngine.Arguments())
            };
        }

        public override void UpdateLearners(RecipeInference.SuggestedRecipe.SuggestedLearner[] availableLearners)
        {
            foreach (var engine in _secondaryEngines.Values)
                engine.UpdateLearners(availableLearners);
            base.UpdateLearners(availableLearners);
        }

        public override void SetSpace(TransformInference.SuggestedTransform[] availableTransforms,
            RecipeInference.SuggestedRecipe.SuggestedLearner[] availableLearners,
            Func<PipelinePattern, long, bool> pipelineVerifier,
            IDataView originalData, IDataView fullyTransformedData, AutoInference.DependencyMap dependencyMapping,
            bool isMaximizingMetric)
        {
            foreach (var engine in _secondaryEngines.Values)
                engine.SetSpace(availableTransforms, availableLearners,
                    pipelineVerifier, originalData, fullyTransformedData, dependencyMapping,
                    isMaximizingMetric);

            base.SetSpace(availableTransforms, availableLearners, pipelineVerifier, originalData, fullyTransformedData,
                dependencyMapping, isMaximizingMetric);
        }

        private void SampleHyperparameters(RecipeInference.SuggestedRecipe.SuggestedLearner learner, PipelinePattern[] history)
        {
            // If first time optimizing hyperparams, create new hyperparameter sweeper.
            if (!_hyperSweepers.ContainsKey(learner.LearnerName))
            {
                var paramTups = AutoMlUtils.ConvertToSweepArgumentStrings(learner.PipelineNode.SweepParams);
                var sps = paramTups.Select(tup =>
                    new SubComponent<IValueGenerator, SignatureSweeperParameter>(tup.Item1, tup.Item2)).ToArray();
                if (sps.Length > 0)
                {
                    _hyperSweepers[learner.LearnerName] = new KdoSweeper(Env,
                        new KdoSweeper.Arguments
                        {
                            SweptParameters = sps,
                            NumberInitialPopulation = Math.Max(_remainingThirdStageTrials, 2)
                        });
                }
                else
                    _hyperSweepers[learner.LearnerName] = new FalseSweeper();
            }
            var sweeper = _hyperSweepers[learner.LearnerName];
            var historyToUse = history.Where(p => p.Learner.LearnerName == learner.LearnerName).ToArray();
            if (_currentStage == (int)Stages.Third)
            {
                _remainingThirdStageTrials--;
                historyToUse = new PipelinePattern[0];
                if (_remainingThirdStageTrials < 1)
                    _currentStage++;
            }
            SampleHyperparameters(learner, sweeper, IsMaximizingMetric, historyToUse);
        }

        private TransformInference.SuggestedTransform[] SampleTransforms(RecipeInference.SuggestedRecipe.SuggestedLearner learner,
            PipelinePattern[] history, out long transformsBitMask, bool uniformRandomSampling = false)
        {
            var sampledTransforms =
                new List<TransformInference.SuggestedTransform>(
                    AutoMlUtils.GetMandatoryTransforms(AvailableTransforms));
            var remainingAvailableTransforms =
                AvailableTransforms.Where(t => !sampledTransforms.Any(t.Equals)).ToArray();
            var mask = AutoMlUtils.TransformsToBitmask(sampledTransforms.ToArray());

            foreach (var transform in remainingAvailableTransforms)
            {
                // Add pseudo-mass to encourage sampling of untried transforms.
                double maxWeight = history.Length > 0 ? history.Max(w => w.PerformanceSummary.MetricValue) : 0d;
                double allWeight = Math.Max(maxWeight, 1d);
                double learnerWeight = Math.Max(maxWeight, 1d);
                int allCounts = 1;
                int learnerCounts = 1;

                // Add mass according to performance.
                foreach (var pipeline in history)
                {
                    if (pipeline.Transforms.Any(transform.Equals))
                    {
                        allWeight +=
                            AutoMlUtils.ProcessWeight(pipeline.PerformanceSummary.MetricValue,
                                maxWeight, IsMaximizingMetric);
                        allCounts++;

                        if (pipeline.Learner.LearnerName == learner.LearnerName)
                        {
                            learnerWeight += pipeline.PerformanceSummary.MetricValue;
                            learnerCounts++;
                        }
                    }
                }

                // Take average mass as weight, and take convex combination of 
                // learner-specific weight and unconditioned weight.
                allWeight /= allCounts > 0 ? allCounts : 1;
                learnerWeight /= learnerCounts > 0 ? learnerCounts : 1;
                var lambda = MathUtils.Sigmoid(learnerCounts - 3);
                var combinedWeight = uniformRandomSampling ?
                    0.5 : lambda * learnerWeight + (1 - lambda) * allWeight;

                // Sample transform according to combined weight.
                if (ProbUtils.SampleUniform() <= combinedWeight / maxWeight)
                    mask |= 1L << transform.AtomicGroupId;
            }

            // Add all chosen transforms.
            sampledTransforms.AddRange(remainingAvailableTransforms.Where(t =>
                AutoMlUtils.AtomicGroupPresent(mask, t.AtomicGroupId)));

            // Add final features concat transform. NOTE: computed bitmask should always 
            // exclude the final features concat. If we forget to exclude that one, will 
            // cause an error in verification, since it isn't included in the original 
            // dependency mapping (i.e., its level isn't in the dictionary).
            sampledTransforms.AddRange(AutoMlUtils.GetFinalFeatureConcat(Env, FullyTransformedData,
                DependencyMapping, sampledTransforms.ToArray(), AvailableTransforms));
            transformsBitMask = mask;

            return sampledTransforms.ToArray();
        }

        private RecipeInference.SuggestedRecipe.SuggestedLearner[] GetTopLearners(IEnumerable<PipelinePattern> history)
        {
            var weights = LearnerHistoryToWeights(history.ToArray(), IsMaximizingMetric);
            var topKTuples = new Tuple<double, int>[_topK];

            for (int i = 0; i < weights.Length; i++)
            {
                if (i < _topK)
                    topKTuples[i] = new Tuple<double, int>(weights[i], i);
                else
                {
                    for (int j = 0; j < topKTuples.Length; j++)
                        if (weights[i] > topKTuples[j].Item1)
                            topKTuples[j] = new Tuple<double, int>(weights[i], i);
                }
            }

            return topKTuples.Select(t => AvailableLearners[t.Item2]).ToArray();
        }

        public override PipelinePattern[] GetNextCandidates(IEnumerable<PipelinePattern> history, int numCandidates)
        {
            var prevCandidates = history.ToArray();

            switch (_currentStage)
            {
                case (int)Stages.First:
                    // First stage: Go through all learners once with default hyperparams and all transforms.
                    // If random initilization is used, generate number of requested initialization trials for
                    // this stage.
                    int numStageOneTrials = _randomInit ? _numInitPipelines : AvailableLearners.Length;
                    var remainingNum = Math.Min(numStageOneTrials - prevCandidates.Length, numCandidates);
                    if (remainingNum < 1)
                    {
                        // Select top k learners, update stage, then get requested 
                        // number of candidates, using second stage logic.
                        UpdateLearners(GetTopLearners(prevCandidates));
                        _currentStage++;
                        return GetNextCandidates(prevCandidates, numCandidates);
                    }
                    else
                        return GetInitialPipelines(prevCandidates, remainingNum);
                case (int)Stages.Second:
                    // Second stage: Using top k learners, try random transform configurations.
                    var candidates = new List<PipelinePattern>();
                    var numSecondStageCandidates = Math.Min(numCandidates, _remainingSecondStageTrials);
                    var numThirdStageCandidates = Math.Max(numCandidates - numSecondStageCandidates, 0);
                    _remainingSecondStageTrials -= numSecondStageCandidates;

                    // Get second stage candidates.
                    if (numSecondStageCandidates > 0)
                        candidates.AddRange(NextCandidates(prevCandidates, numSecondStageCandidates, true, true));

                    // Update stage when no more second stage trials to sample.
                    if (_remainingSecondStageTrials < 1)
                        _currentStage++;

                    // If the number of requested candidates is smaller than remaining second stage candidates,
                    // draw candidates from remaining pool.
                    if (numThirdStageCandidates > 0)
                        candidates.AddRange(NextCandidates(prevCandidates, numThirdStageCandidates));

                    return candidates.ToArray();
                default:
                    // Sample transforms according to weights and use hyperparameter optimization method.
                    // Third stage samples hyperparameters uniform randomly in KDO, fourth and above do not.
                    return NextCandidates(prevCandidates, numCandidates);
            }
        }

        private PipelinePattern[] GetInitialPipelines(IEnumerable<PipelinePattern> history, int numCandidates) =>
            _secondaryEngines[_randomInit ? nameof(UniformRandomEngine) : nameof(DefaultsEngine)]
                .GetNextCandidates(history, numCandidates);

        private PipelinePattern[] NextCandidates(PipelinePattern[] history, int numCandidates,
            bool defaultHyperParams = false, bool uniformRandomTransforms = false)
        {
            const int maxNumberAttempts = 10;
            double[] learnerWeights = LearnerHistoryToWeights(history, IsMaximizingMetric);
            var candidates = new List<PipelinePattern>();
            var sampledLearners = new RecipeInference.SuggestedRecipe.SuggestedLearner[numCandidates];

            if (_currentStage == (int)Stages.Second || _currentStage == (int)Stages.Third)
            {
                // Select remaining learners in round-robin fashion.
                for (int i = 0; i < numCandidates; i++)
                    sampledLearners[i] = AvailableLearners[i % AvailableLearners.Length].Clone();
            }
            else
            {
                // Select learners, based on weights.
                var indices = ProbUtils.SampleCategoricalDistribution(numCandidates, learnerWeights);
                foreach (var item in indices.Select((idx, i) => new { idx, i }))
                    sampledLearners[item.i] = AvailableLearners[item.idx].Clone();
            }

            // Select hyperparameters and transforms based on learner and history.
            foreach (var learner in sampledLearners)
            {
                PipelinePattern pipeline;
                int count = 0;
                bool valid;
                string hashKey;

                if (!defaultHyperParams)
                    SampleHyperparameters(learner, history);
                else
                    AutoMlUtils.PopulateSweepableParams(learner);

                do
                {   // Make sure transforms set is valid and have not seen pipeline before. 
                    // Repeat until passes or runs out of chances.
                    pipeline = new PipelinePattern(SampleTransforms(learner, history,
                        out var transformsBitMask, uniformRandomTransforms), learner, "", Env);
                    hashKey = GetHashKey(transformsBitMask, learner);
                    valid = PipelineVerifier(pipeline, transformsBitMask) && !VisitedPipelines.Contains(hashKey);
                    count++;
                } while (!valid && count <= maxNumberAttempts);

                // If maxed out chances and at second stage, move onto next stage.
                if (count >= maxNumberAttempts && _currentStage == (int)Stages.Second)
                    _currentStage++;

                // Keep only valid pipelines.
                if (valid)
                {
                    VisitedPipelines.Add(hashKey);
                    candidates.Add(pipeline);
                }
            }

            return candidates.ToArray();
        }
    }
}
