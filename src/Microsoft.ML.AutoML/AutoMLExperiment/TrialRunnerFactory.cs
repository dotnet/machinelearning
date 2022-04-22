// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Extensions.DependencyInjection;

#nullable enable
namespace Microsoft.ML.AutoML
{
    internal interface ITrialRunnerFactory
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
            ITrialRunner? runner = (settings.ExperimentSettings.DatasetSettings, settings.ExperimentSettings.EvaluateMetric) switch
            {
                (CrossValidateDatasetSettings, BinaryMetricSettings) => _provider.GetService<BinaryClassificationCVRunner>(),
                (TrainTestDatasetSettings, BinaryMetricSettings) => _provider.GetService<BinaryClassificationTrainTestRunner>(),
                (CrossValidateDatasetSettings, MultiClassMetricSettings) => _provider.GetService<MultiClassificationCVRunner>(),
                (TrainTestDatasetSettings, MultiClassMetricSettings) => _provider.GetService<MultiClassificationTrainTestRunner>(),
                (CrossValidateDatasetSettings, RegressionMetricSettings) => _provider.GetService<RegressionCVRunner>(),
                (TrainTestDatasetSettings, RegressionMetricSettings) => _provider.GetService<RegressionTrainTestRunner>(),
                _ => throw new NotImplementedException(),
            };

            return runner;
        }
    }
}
