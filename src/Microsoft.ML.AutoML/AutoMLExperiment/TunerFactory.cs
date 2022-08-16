// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Extensions.DependencyInjection;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// interface for all tuner factories.
    /// </summary>
    public interface ITunerFactory
    {
        ITuner CreateTuner(TrialSettings settings);
    }

    internal class CostFrugalTunerFactory : ITunerFactory
    {
        private readonly IServiceProvider _provider;

        public CostFrugalTunerFactory(IServiceProvider provider)
        {
            _provider = provider;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            var experimentSetting = _provider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
            var searchSpace = settings?.Pipeline?.SearchSpace ?? experimentSetting.SearchSpace;
            var initParameter = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            var isMaximize = experimentSetting.IsMaximizeMetric;

            return new CostFrugalTuner(searchSpace, initParameter, !isMaximize);
        }
    }

    internal class RandomTunerFactory : ITunerFactory
    {
        private readonly IServiceProvider _provider;

        public RandomTunerFactory(IServiceProvider provider)
        {
            _provider = provider;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            var experimentSetting = _provider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
            var searchSpace = settings?.Pipeline?.SearchSpace ?? experimentSetting.SearchSpace;

            return new RandomSearchTuner(searchSpace);
        }
    }

    internal class GridSearchTunerFactory : ITunerFactory
    {
        private readonly IServiceProvider _provider;

        public GridSearchTunerFactory(IServiceProvider provider)
        {
            _provider = provider;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            var experimentSetting = _provider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
            var searchSpace = settings?.Pipeline?.SearchSpace ?? experimentSetting.SearchSpace;

            return new GridSearchTuner(searchSpace);
        }
    }
}
