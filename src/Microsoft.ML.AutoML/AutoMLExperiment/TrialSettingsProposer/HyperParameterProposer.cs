// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal class HyperParameterProposer : ITrialSettingsProposer
    {
        private readonly Dictionary<string, ITuner> _tuners;
        private readonly IServiceProvider _provider;

        public HyperParameterProposer(IServiceProvider provider)
        {
            this._tuners = new Dictionary<string, ITuner>();
            this._provider = provider;
        }

        public TrialSettings Propose(TrialSettings settings)
        {
            var tunerFactory = this._provider.GetService<ITunerFactory>();
            if (!this._tuners.ContainsKey(settings.Schema))
            {
                var t = tunerFactory.CreateTuner(settings);
                this._tuners.Add(settings.Schema, t);
            }

            var tuner = this._tuners[settings.Schema];
            var parameter = tuner.Propose(settings);
            settings.Parameter = parameter;

            return settings;
        }

        public void Update(TrialSettings settings, TrialResult result)
        {
            var schema = settings.Schema;
            if (this._tuners.TryGetValue(schema, out var tuner))
            {
                tuner.Update(result);
            }
        }
    }
}
