// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// propose hyper parameter using ECI index.
    /// ECI index is a way to measure the importance of a trainer. A higher ECI means a trainer
    /// is more likely to be picked.
    /// </summary>
    internal class EciHyperParameterProposer : IHyperParameterProposer
    {
        private readonly Dictionary<string, ITuner> _tuners;
        private readonly PipelineProposer _pipelineProposer;
        // this dictionary records the schema for each trial.
        // the key is trial id, and value is the schema for that trial.
        private readonly IMetricManager _metricManager;

        public EciHyperParameterProposer(SweepablePipeline sweepablePipeline, IMetricManager metricManager, AutoMLExperiment.AutoMLExperimentSettings settings)
        {
            _tuners = new Dictionary<string, ITuner>();
            _pipelineProposer = new PipelineProposer(sweepablePipeline, settings, metricManager);
            _metricManager = metricManager;
        }

        public TrialSettings Propose(TrialSettings settings)
        {
            (var searchSpace, var schema) = _pipelineProposer.ProposeSearchSpace();
            if (!_tuners.ContainsKey(schema))
            {
                var t = new CostFrugalTuner(searchSpace, searchSpace.SampleFromFeatureSpace(searchSpace.Default), !_metricManager.IsMaximize);
                _tuners.Add(schema, t);
            }

            var tuner = _tuners[schema];
            var parameter = tuner.Propose(settings);
            settings.Parameter = parameter;
            return settings;
        }

        public void Update(TrialSettings settings, TrialResult result)
        {
            var schema = settings.Parameter["_SCHEMA_"].AsType<string>();
            if (_tuners.TryGetValue(schema, out var tuner))
            {
                tuner.Update(result);
            }

            _pipelineProposer.Update(settings, result, schema);
        }
    }

    internal class NestedSearchSpaceHyperParameterProposer : IHyperParameterProposer
    {
        private readonly ITuner _tuner;
        private readonly SweepablePipeline _pipeline;

        public NestedSearchSpaceHyperParameterProposer(SweepablePipeline pipeline, ITunerFactory tunerFactory)
        {
            this._tuner = tunerFactory.CreateTuner(null);
            this._pipeline = pipeline;
        }

        public TrialSettings Propose(TrialSettings settings)
        {
            var parameter = _tuner.Propose(settings);
            settings.Parameter = parameter;
            var keys = parameter[AutoMLExperiment.PipelineSearchspaceName]["_SCHEMA_"].AsType<string>().Replace(" ", string.Empty).Split('*');
            var estimators = keys.Select(k => _pipeline.Estimators[k]);

            return settings;
        }

        public void Update(TrialSettings parameter, TrialResult result)
        {
            _tuner.Update(result);
        }
    }
}
