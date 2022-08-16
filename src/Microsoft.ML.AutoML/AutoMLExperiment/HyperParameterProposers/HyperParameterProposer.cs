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
        private readonly IServiceProvider _provider;
        private readonly PipelineProposer _pipelineProposer;
        // this dictionary records the schema for each trial.
        // the key is trial id, and value is the schema for that trial.
        private readonly Dictionary<int, string> _schemasLookupMap;

        public EciHyperParameterProposer(PipelineProposer pipelineProposer, IServiceProvider provider)
        {
            _tuners = new Dictionary<string, ITuner>();
            _pipelineProposer = pipelineProposer;
            _provider = provider;
            _schemasLookupMap = new Dictionary<int, string>();
        }

        public TrialSettings Propose(TrialSettings settings)
        {
            (var pipeline, var schema) = _pipelineProposer.ProposePipeline(settings);
            settings.Pipeline = pipeline;
            var tunerFactory = _provider.GetService<ITunerFactory>();
            if (!_tuners.ContainsKey(schema))
            {
                var t = tunerFactory.CreateTuner(settings);
                _tuners.Add(schema, t);
            }

            var tuner = _tuners[schema];
            var parameter = tuner.Propose(settings);
            settings.Parameter = parameter;
            _schemasLookupMap[settings.TrialId] = schema;
            return settings;
        }

        public void Update(TrialSettings settings, TrialResult result)
        {
            if (_schemasLookupMap.TryGetValue(settings.TrialId, out var schema))
            {
                if (_tuners.TryGetValue(schema, out var tuner))
                {
                    tuner.Update(result);
                }

                _pipelineProposer.Update(settings, result, schema);
            }
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
            settings.Pipeline = new SweepableEstimatorPipeline(estimators);

            return settings;
        }

        public void Update(TrialSettings parameter, TrialResult result)
        {
            _tuner.Update(result);
        }
    }
}
