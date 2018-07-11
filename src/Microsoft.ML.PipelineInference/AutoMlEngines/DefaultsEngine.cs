// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.PipelineInference;

[assembly: EntryPointModule(typeof(DefaultsEngine.Arguments))]

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// AutoML engine that goes through all lerners using defaults. Need to decide
    /// how to handle which transforms to try.
    /// </summary>
    public sealed class DefaultsEngine : PipelineOptimizerBase
    {
        private int _currentLearnerIndex;

        [TlcModule.Component(Name = "Defaults", FriendlyName = "Defaults Engine",
            Desc = "AutoML engine that returns learners with default settings.")]
        public sealed class Arguments : ISupportIPipelineOptimizerFactory
        {
            public IPipelineOptimizer CreateComponent(IHostEnvironment env) => new DefaultsEngine(env, this);
        }

        public DefaultsEngine(IHostEnvironment env, Arguments args)
            : base(env, env.Register("DefaultsEngine(AutoML)"))
        {
            _currentLearnerIndex = 0;
        }

        public override PipelinePattern[] GetNextCandidates(IEnumerable<PipelinePattern> history, int numCandidates, RoleMappedData dataRoles)
        {
            var candidates = new List<PipelinePattern>();
            DataRoles = dataRoles;

            while (candidates.Count < numCandidates)
            {
                Contracts.Assert(0 <= _currentLearnerIndex && _currentLearnerIndex < AvailableLearners.Length);

                // Select hyperparameters and transforms based on learner and history.
                PipelinePattern pipeline;
                int count = 0;
                bool valid;
                var learner = AvailableLearners[_currentLearnerIndex];

                // Make sure sweep paramater values exist; if not, populate them from learner object.
                if (learner.PipelineNode.SweepParams.Any(p => p.RawValue == null))
                    AutoMlUtils.PopulateSweepableParams(learner);

                do
                {   // Make sure transforms set is valid. Repeat until passes verifier.
                    pipeline = new PipelinePattern(SampleTransforms(out var transformsBitMask), 
                        learner, "", Env);
                    valid = PipelineVerifier(pipeline, transformsBitMask);
                    count++;
                } while (!valid && count <= 1000);

                // Keep only valid pipelines.
                if (valid)
                    candidates.Add(pipeline);

                // Update current index
                _currentLearnerIndex = (_currentLearnerIndex + 1) % AvailableLearners.Length;
            }

            return candidates.ToArray();
        }

        private TransformInference.SuggestedTransform[] SampleTransforms(out long transformsBitMask)
        {
            // For now, return all transforms.
            var sampledTransforms = AvailableTransforms.ToList();
            transformsBitMask = AutoMlUtils.TransformsToBitmask(sampledTransforms.ToArray());

            // Add final features concat transform.
            sampledTransforms.AddRange(AutoMlUtils.GetFinalFeatureConcat(Env, FullyTransformedData,
                DependencyMapping, sampledTransforms.ToArray(), AvailableTransforms, DataRoles));

            return sampledTransforms.ToArray();
        }
    }
}
