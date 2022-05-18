// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.Extensions.DependencyInjection;

#nullable enable
namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// interface for trial runner factory.
    /// </summary>
    public interface ITrialRunnerFactory
    {
        ITrialRunner? CreateTrialRunner();
    }

    internal class CustomRunnerFactory : ITrialRunnerFactory
    {
        private readonly ITrialRunner _instance;

        public CustomRunnerFactory(ITrialRunner runner)
        {
            _instance = runner;
        }

        public ITrialRunner? CreateTrialRunner()
        {
            return _instance;
        }
    }

    internal class TrialRunnerFactory : ITrialRunnerFactory
    {
        private readonly IServiceProvider _provider;

        public TrialRunnerFactory(IServiceProvider provider)
        {
            _provider = provider;
        }

        public ITrialRunner? CreateTrialRunner()
        {
            var datasetManager = _provider.GetService<IDatasetManager>();
            var metricManager = _provider.GetService<IMetricManager>();

            ITrialRunner? runner = (datasetManager, metricManager) switch
            {
                (CrossValidateDatasetManager, BinaryMetricManager) => _provider.GetService<BinaryClassificationCVRunner>(),
                (TrainTestDatasetManager, BinaryMetricManager) => _provider.GetService<BinaryClassificationTrainTestRunner>(),
                (CrossValidateDatasetManager, MultiClassMetricManager) => _provider.GetService<MultiClassificationCVRunner>(),
                (TrainTestDatasetManager, MultiClassMetricManager) => _provider.GetService<MultiClassificationTrainTestRunner>(),
                (CrossValidateDatasetManager, RegressionMetricManager) => _provider.GetService<RegressionCVRunner>(),
                (TrainTestDatasetManager, RegressionMetricManager) => _provider.GetService<RegressionTrainTestRunner>(),
                _ => throw new NotImplementedException(),
            };

            return runner;
        }
    }
}
