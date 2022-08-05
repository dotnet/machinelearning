// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Fairlearn.reductions;

namespace Microsoft.ML.Fairlearn.AutoML
{
    internal class CostFrugalWithLambdaTunerFactory : ITunerFactory
    {
        private readonly IServiceProvider _provider;
        private readonly ClassificationMoment _moment;
        private readonly MLContext _context;
        private readonly float _gridLimit = 10f;

        public CostFrugalWithLambdaTunerFactory(IServiceProvider provider)
        {
            _provider = provider;
            _moment = provider.GetService<ClassificationMoment>();
            _context = provider.GetService<MLContext>();
            _gridLimit = provider.GetService<GridLimit>().Value;
        }

        public ITuner CreateTuner(TrialSettings settings)
        {
            var experimentSetting = _provider.GetService<AutoMLExperiment.AutoMLExperimentSettings>();
            var searchSpace = settings.Pipeline.SearchSpace;
            var isMaximize = experimentSetting.IsMaximizeMetric;

            var lambdaSearchSpace = Utilities.GenerateBinaryClassificationLambdaSearchSpace(_context, _moment, gridLimit: _gridLimit);
            searchSpace["_lambda_search_space"] = lambdaSearchSpace;
            var initParameter = searchSpace.SampleFromFeatureSpace(searchSpace.Default);

            return new RandomSearchTuner(searchSpace);
        }
    }
}
