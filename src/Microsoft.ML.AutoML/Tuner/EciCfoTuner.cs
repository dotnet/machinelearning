// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// propose hyper parameter using ECI index and <see cref="CostFrugalTuner"/>.
    /// ECI index is a way to measure the importance of a trainer. A higher ECI means a trainer
    /// is more likely to be picked.
    /// </summary>
    internal class EciCostFrugalTuner : ITuner
    {
        private readonly Dictionary<string, ITuner> _tuners;
        private readonly PipelineProposer _pipelineProposer;
        private readonly Parameter _defaultParameter;
        // this dictionary records the schema for each trial.
        // the key is trial id, and value is the schema for that trial.

        public EciCostFrugalTuner(SweepablePipeline sweepablePipeline, AutoMLExperiment.AutoMLExperimentSettings settings, ITrialResultManager trialResultManager = null)
        {
            _tuners = new Dictionary<string, ITuner>();
            _pipelineProposer = new PipelineProposer(sweepablePipeline, settings);
            _defaultParameter = settings.SearchSpace.SampleFromFeatureSpace(settings.SearchSpace.Default)[AutoMLExperiment.PipelineSearchspaceName];
            var pipelineSchemas = sweepablePipeline.Schema.ToTerms().Select(t => t.ToString()).ToArray();
            _tuners = pipelineSchemas.ToDictionary(schema => schema, schema =>
            {
                var searchSpace = sweepablePipeline.BuildSweepableEstimatorPipeline(schema).SearchSpace;
                return new CostFrugalTuner(searchSpace, searchSpace.SampleFromFeatureSpace(searchSpace.Default), seed: settings.Seed) as ITuner;
            });

            if (trialResultManager != null)
            {
                foreach (var trials in trialResultManager.GetAllTrialResults())
                {
                    Update(trials);
                }
            }
        }

        public Parameter Propose(TrialSettings settings)
        {
            var schema = _pipelineProposer.ProposeSearchSpace();

            var tuner = _tuners[schema];
            var parameter = tuner.Propose(settings);
            foreach (var k in _defaultParameter)
            {
                if (!parameter.ContainsKey(k.Key))
                {
                    parameter[k.Key] = _defaultParameter[k.Key];
                }
            }
            settings.Parameter[AutoMLExperiment.PipelineSearchspaceName] = parameter;

            return settings.Parameter;
        }

        public void Update(TrialResult result)
        {
            var originalParameter = result.TrialSettings.Parameter;
            var schema = result.TrialSettings.Parameter[AutoMLExperiment.PipelineSearchspaceName]["_SCHEMA_"].AsType<string>();
            _pipelineProposer.Update(result, schema);
            if (_tuners.TryGetValue(schema, out var tuner))
            {
                var parameter = result.TrialSettings.Parameter[AutoMLExperiment.PipelineSearchspaceName];
                result.TrialSettings.Parameter = parameter;
                tuner.Update(result);
                result.TrialSettings.Parameter = originalParameter;
            }
        }
    }
}
