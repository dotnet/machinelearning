// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Extensions.DependencyInjection;

#nullable enable
namespace Microsoft.ML.AutoML
{
    public interface ITrialRunnerFactory
    {
        ITrialRunner? CreateTrialRunner(TrialSettings settings);
    }

    internal class TrialRunnerFactory : ITrialRunnerFactory
    {
        private readonly IServiceProvider _provider;

        public TrialRunnerFactory(IServiceProvider provider)
        {
            _provider = provider;
        }

        public ITrialRunner? CreateTrialRunner(TrialSettings settings)
        {
            var datasetManager = this._provider.GetService<IDatasetManager>();
            ITrialRunner? runner = (datasetManager, settings.ExperimentSettings.EvaluateMetric) switch
            {
                (CrossValidateDatasetManager, BinaryMetricSettings) => _provider.GetService<BinaryClassificationCVRunner>(),
                (TrainTestDatasetManager, BinaryMetricSettings) => _provider.GetService<BinaryClassificationTrainTestRunner>(),
                (CrossValidateDatasetManager, MultiClassMetricSettings) => _provider.GetService<MultiClassificationCVRunner>(),
                (TrainTestDatasetManager, MultiClassMetricSettings) => _provider.GetService<MultiClassificationTrainTestRunner>(),
                (CrossValidateDatasetManager, RegressionMetricSettings) => _provider.GetService<RegressionCVRunner>(),
                (TrainTestDatasetManager, RegressionMetricSettings) => _provider.GetService<RegressionTrainTestRunner>(),
                _ => throw new NotImplementedException(),
            };

            return runner;
        }
    }
}
