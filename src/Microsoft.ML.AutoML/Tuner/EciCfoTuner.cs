// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// propose hyper parameter using ECI index and <see cref="CostFrugalTuner"/>.
    /// ECI index is a way to measure the importance of a trainer. A higher ECI means a trainer
    /// is more likely to be picked.
    /// </summary>
    public class EciCostFrugalTuner : ITuner
    {
        private readonly Dictionary<string, ITuner> _tuners;
        private readonly PipelineProposer _pipelineProposer;
        // this dictionary records the schema for each trial.
        // the key is trial id, and value is the schema for that trial.
        private readonly IMetricManager _metricManager;

        public EciCostFrugalTuner(SweepablePipeline sweepablePipeline, IMetricManager metricManager, AutoMLExperiment.AutoMLExperimentSettings settings)
        {
            _tuners = new Dictionary<string, ITuner>();
            _pipelineProposer = new PipelineProposer(sweepablePipeline, settings, metricManager);
            _metricManager = metricManager;
        }

        public Parameter Propose(TrialSettings settings)
        {
            (var searchSpace, var schema) = _pipelineProposer.ProposeSearchSpace();
            if (!_tuners.ContainsKey(schema))
            {
                var t = new CostFrugalTuner(searchSpace, searchSpace.SampleFromFeatureSpace(searchSpace.Default), !_metricManager.IsMaximize);
                _tuners.Add(schema, t);
            }

            var tuner = _tuners[schema];
            settings.Parameter[AutoMLExperiment.PipelineSearchspaceName] = tuner.Propose(settings);

            return settings.Parameter;
        }

        public void Update(TrialResult result)
        {
            var schema = result.TrialSettings.Parameter[AutoMLExperiment.PipelineSearchspaceName]["_SCHEMA_"].AsType<string>();
            if (_tuners.TryGetValue(schema, out var tuner))
            {
                tuner.Update(result);
            }

            _pipelineProposer.Update(result, schema);
        }
    }
}
