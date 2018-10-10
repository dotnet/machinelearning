// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.PipelineInference;

[assembly: EntryPointModule(typeof(UniformRandomEngine.Arguments))]

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Example class of an autoML engine (a pipeline optimizer) that simply tries random enumeration.
    /// If we use a third-party solution for autoML, we can just implement a new wrapper for it as a
    /// PipelineOptimizerBase, and use our existing autoML body code to take advantage of it. This design
    /// should allow for easy development of new autoML methods.
    /// </summary>
    public sealed class UniformRandomEngine : PipelineOptimizerBase
    {
        [TlcModule.Component(Name = "UniformRandom", FriendlyName = "Uniform Random Engine", Desc = "AutoML engine using uniform random sampling.")]
        public sealed class Arguments : ISupportIPipelineOptimizerFactory
        {
            public IPipelineOptimizer CreateComponent(IHostEnvironment env) => new UniformRandomEngine(env);
        }

        public UniformRandomEngine(IHostEnvironment env)
            : base(env, env.Register("UniformRandomEngine(AutoML)"))
        {}

        public override PipelinePattern[] GetNextCandidates(IEnumerable<PipelinePattern> history, int numberOfCandidates, RoleMappedData dataRoles)
        {
            DataRoles = dataRoles;
            return GetRandomPipelines(numberOfCandidates);
        }

        private PipelinePattern[] GetRandomPipelines(int numOfPipelines)
        {
            Host.Check(AvailableLearners.All(l => l.PipelineNode != null));
            Host.Check(AvailableTransforms.All(t => t.PipelineNode != null));
            int atomicGroupLimit = AvailableTransforms.Select(t => t.AtomicGroupId)
                .DefaultIfEmpty(-1).Max() + 1;
            var pipelines = new List<PipelinePattern>();
            int collisions = 0;
            int totalCount = 0;

            while (pipelines.Count < numOfPipelines)
            {
                // Generate random bitmask (set of transform atomic group IDs)
                long transformsBitMask = Host.Rand.Next((int)Math.Pow(2, atomicGroupLimit));

                // Include all "always on" transforms, such as autolabel.
                transformsBitMask |= AutoMlUtils.IncludeMandatoryTransforms(AvailableTransforms.ToList());

                // Get actual learner and transforms for pipeline
                var selectedLearner = AvailableLearners[Host.Rand.Next(AvailableLearners.Length)];
                var selectedTransforms = AvailableTransforms.Where(t =>
                    AutoMlUtils.AtomicGroupPresent(transformsBitMask, t.AtomicGroupId)).ToList();

                // Randomly change transform sweepable hyperparameter settings
                selectedTransforms.ForEach(t => RandomlyPerturbSweepableHyperparameters(t.PipelineNode));

                // Randomly change learner sweepable hyperparameter settings
                RandomlyPerturbSweepableHyperparameters(selectedLearner.PipelineNode);

                // Always include features concat transform
                selectedTransforms.AddRange(AutoMlUtils.GetFinalFeatureConcat(Env, FullyTransformedData,
                    DependencyMapping, selectedTransforms.ToArray(), AvailableTransforms, DataRoles));

                // Compute hash key for checking if we've already seen this pipeline.
                // However, if we keep missing, don't want to get stuck in infinite loop.
                // Try for a good number of times (e.g., numOfPipelines * 4), then just add
                // all generated pipelines to get us out of rut.
                string hashKey = GetHashKey(transformsBitMask, selectedLearner);
                if (collisions < numOfPipelines * 4 && VisitedPipelines.Contains(hashKey))
                {
                    collisions++;
                    continue;
                }

                VisitedPipelines.Add(hashKey);
                collisions = 0;
                totalCount++;

                // Keep pipeline if valid
                var pipeline = new PipelinePattern(selectedTransforms.ToArray(), selectedLearner, "", Env);
                if (!TransformsMaskValidity.ContainsKey(transformsBitMask))
                    TransformsMaskValidity.Add(transformsBitMask, PipelineVerifier(pipeline, transformsBitMask));
                if (TransformsMaskValidity[transformsBitMask])
                    pipelines.Add(pipeline);

                // Only invalid pipelines available, stuck in loop.
                // Break out and return no pipelines.
                if (totalCount > numOfPipelines * 10)
                    break;
            }

            return pipelines.ToArray();
        }

        private void RandomlyPerturbSweepableHyperparameters(TransformPipelineNode transform)
        {
            RandomlyPerturbSweepableHyperparameters(transform.SweepParams);
            transform.UpdateProperties();
        }

        private void RandomlyPerturbSweepableHyperparameters(TrainerPipelineNode learner)
        {
            RandomlyPerturbSweepableHyperparameters(learner.SweepParams);
            learner.UpdateProperties();
        }

        private void RandomlyPerturbSweepableHyperparameters(IEnumerable<TlcModule.SweepableParamAttribute> sweepParams)
        {
            foreach (var param in sweepParams)
            {
                switch (param)
                {
                    case TlcModule.SweepableDiscreteParamAttribute disParam:
                        Env.Assert(disParam.Options.Length > 0, $"Trying to sweep over discrete parameter, {disParam.Name}, with no options.");
                        disParam.RawValue = Host.Rand.Next(disParam.Options.Length);
                        break;
                    case TlcModule.SweepableFloatParamAttribute floParam:
                        var fvg = AutoMlUtils.ToIValueGenerator(floParam);
                        floParam.RawValue = ((IParameterValue<float>)fvg.CreateFromNormalized(Host.Rand.NextSingle())).Value;
                        break;
                    case TlcModule.SweepableLongParamAttribute lonParam:
                        var lvg = AutoMlUtils.ToIValueGenerator(lonParam);
                        lonParam.RawValue = ((IParameterValue<long>)lvg.CreateFromNormalized(Host.Rand.NextSingle())).Value;
                        break;
                    default:
                        throw new NotSupportedException($"Unknown type of sweepable parameter attribute: {param.GetType()}");
                }
            }
        }
    }
}
