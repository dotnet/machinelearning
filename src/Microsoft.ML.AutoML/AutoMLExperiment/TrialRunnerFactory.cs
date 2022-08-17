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


    internal class SweepablePipelineTrialRunnerFactory : ITrialRunnerFactory
    {
        private readonly IServiceProvider _provider;

        public SweepablePipelineTrialRunnerFactory(IServiceProvider provider)
        {
            _provider = provider;
        }

        public ITrialRunner? CreateTrialRunner()
        {
            var datasetManager = _provider.GetService<IDatasetManager>();
            var metricManager = _provider.GetService<IMetricManager>();

            ITrialRunner? runner = (datasetManager) switch
            {
                CrossValidateDatasetManager => _provider.GetService<SweepablePipelineCVRunner>(),
                TrainTestDatasetManager => _provider.GetService<SweepablePipelineTrainTestRunner>(),
                _ => throw new NotImplementedException(),
            };

            return runner;
        }
    }
}
