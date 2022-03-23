// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.SearchSpace.Tuner;

namespace Microsoft.ML.AutoML
{
    internal interface ITunerFactory
    {
        ITuner CreateTuner(TrialSettings settings);
    }

    internal class CfoTunerFactory : ITunerFactory
    {
        private readonly IServiceProvider _provider;

        public CfoTunerFactory(IServiceProvider provider)
        {
            this._provider = provider;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            var experimentSetting = this._provider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
            var searchSpace = settings.Pipeline.SearchSpace;
            var initParameter = settings.Pipeline.Parameter;
            var isMaximize = experimentSetting.EvaluateMetric.IsMaximize;

            return new CfoTuner(searchSpace, initParameter, !isMaximize);
        }
    }

    internal class RandomTunerFactory : ITunerFactory
    {
        private readonly IServiceProvider _provider;

        public RandomTunerFactory(IServiceProvider provider)
        {
            this._provider = provider;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            var searchSpace = settings.Pipeline.SearchSpace;

            return new RandomSearchTuner(searchSpace);
        }
    }

    internal class GridSearchTunerFactory : ITunerFactory
    {
        private readonly IServiceProvider _provider;

        public GridSearchTunerFactory(IServiceProvider provider)
        {
            this._provider = provider;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            var searchSpace = settings.Pipeline.SearchSpace;

            return new GridSearchTuner(searchSpace);
        }
    }
}
