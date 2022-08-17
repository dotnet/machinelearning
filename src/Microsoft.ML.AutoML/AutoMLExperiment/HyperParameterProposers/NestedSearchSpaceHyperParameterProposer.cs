// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;

namespace Microsoft.ML.AutoML
{
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
