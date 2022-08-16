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
        private readonly SearchSpace.SearchSpace _searchSpace;
        private readonly IMetricManager _metricManager;

        public CostFrugalTunerFactory(AutoMLExperiment.AutoMLExperimentSettings settings, IMetricManager metricManager)
        {
            _searchSpace = settings.SearchSpace;
            _metricManager = metricManager;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            var initParameter = _searchSpace.SampleFromFeatureSpace(_searchSpace.Default);
            var isMaximize = _metricManager.IsMaximize;

            return new CostFrugalTuner(_searchSpace, initParameter, !isMaximize);
        }
    }

    internal class RandomTunerFactory : ITunerFactory
    {
        private readonly SearchSpace.SearchSpace _searchSpace;

        public RandomTunerFactory(AutoMLExperiment.AutoMLExperimentSettings settings)
        {
            _searchSpace = settings.SearchSpace;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            return new RandomSearchTuner(_searchSpace);
        }
    }

    internal class GridSearchTunerFactory : ITunerFactory
    {
        private readonly SearchSpace.SearchSpace _searchSpace;

        public GridSearchTunerFactory(AutoMLExperiment.AutoMLExperimentSettings settings)
        {
            _searchSpace = settings.SearchSpace;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            return new GridSearchTuner(_searchSpace);
        }
    }
}
