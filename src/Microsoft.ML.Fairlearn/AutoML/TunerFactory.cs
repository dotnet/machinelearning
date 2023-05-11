// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.Fairlearn.AutoML
{
    internal class CostFrugalWithLambdaTunerFactory : ITuner
    {
        private readonly IServiceProvider _provider;
        private readonly ClassificationMoment _moment;
        private readonly MLContext _context;
        private readonly float _gridLimit = 10f;
        private readonly SweepablePipeline _pipeline;
        private readonly SearchSpace.SearchSpace _searchSpace;
        private readonly ITuner _tuner;

        public CostFrugalWithLambdaTunerFactory(IServiceProvider provider)
        {
            _provider = provider;
            _moment = provider.GetService<ClassificationMoment>();
            _context = provider.GetService<MLContext>();
            _gridLimit = provider.GetService<GridLimit>().Value;
            _pipeline = provider.GetRequiredService<SweepablePipeline>();
            var lambdaSearchSpace = Utilities.GenerateBinaryClassificationLambdaSearchSpace(_moment, gridLimit: _gridLimit);
            var settings = provider.GetRequiredService<AutoMLExperiment.AutoMLExperimentSettings>();
            _searchSpace = settings.SearchSpace;
            _searchSpace["_lambda_search_space"] = lambdaSearchSpace;
            _tuner = new RandomSearchTuner(_searchSpace, settings.Seed);
        }

        public Parameter Propose(TrialSettings settings)
        {
            return _tuner.Propose(settings);
        }

        public void Update(TrialResult result)
        {
            _tuner.Update(result);
        }
    }
}
