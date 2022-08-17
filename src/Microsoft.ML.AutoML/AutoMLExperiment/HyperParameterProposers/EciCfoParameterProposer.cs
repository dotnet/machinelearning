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
    /// propose hyper parameter using ECI index.
    /// ECI index is a way to measure the importance of a trainer. A higher ECI means a trainer
    /// is more likely to be picked.
    /// </summary>
    internal class EciCfoParameterProposer : IHyperParameterProposer
    {
        private readonly Dictionary<string, ITuner> _tuners;
        private readonly PipelineProposer _pipelineProposer;
        // this dictionary records the schema for each trial.
        // the key is trial id, and value is the schema for that trial.
        private readonly IMetricManager _metricManager;
        private readonly ITuner _rootTuner;

        public EciCfoParameterProposer(SweepablePipeline sweepablePipeline, IMetricManager metricManager, AutoMLExperiment.AutoMLExperimentSettings settings, ITunerFactory tunerFactory)
        {
            _rootTuner = tunerFactory.CreateTuner(null);
            _tuners = new Dictionary<string, ITuner>();
            _pipelineProposer = new PipelineProposer(sweepablePipeline, settings, metricManager);
            _metricManager = metricManager;
        }

        public TrialSettings Propose(TrialSettings settings)
        {
            var rootParameter = _rootTuner.Propose(settings);
            (var searchSpace, var schema) = _pipelineProposer.ProposeSearchSpace();
            if (!_tuners.ContainsKey(schema))
            {
                var t = new CostFrugalTuner(searchSpace, searchSpace.SampleFromFeatureSpace(searchSpace.Default), !_metricManager.IsMaximize);
                _tuners.Add(schema, t);
            }

            var tuner = _tuners[schema];
            rootParameter[AutoMLExperiment.PipelineSearchspaceName] = tuner.Propose(settings);
            var parameter = rootParameter;

            settings.Parameter = parameter;
            return settings;
        }

        public void Update(TrialSettings settings, TrialResult result)
        {
            var schema = settings.Parameter[AutoMLExperiment.PipelineSearchspaceName]["_SCHEMA_"].AsType<string>();
            if (_tuners.TryGetValue(schema, out var tuner))
            {
                tuner.Update(result);
            }

            _pipelineProposer.Update(settings, result, schema);
            _rootTuner.Update(result);
        }
    }
}
