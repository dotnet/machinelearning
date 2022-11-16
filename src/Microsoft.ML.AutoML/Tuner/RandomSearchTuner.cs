// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.SearchSpace;
using Microsoft.ML.SearchSpace.Tuner;

namespace Microsoft.ML.AutoML
{
    internal class RandomSearchTuner : ITuner
    {
        private readonly RandomTuner _tuner;
        private readonly SearchSpace.SearchSpace _searchSpace;

        public RandomSearchTuner(AutoMLExperiment.AutoMLExperimentSettings settings)
            : this(settings.SearchSpace, settings.Seed)
        {
        }

        public RandomSearchTuner(SearchSpace.SearchSpace searchSpace, int? seed = null)
        {
            if (seed == null)
                _tuner = new RandomTuner();
            else
                _tuner = new RandomTuner(seed.Value);
            _searchSpace = searchSpace;
        }

        public Parameter Propose(TrialSettings settings)
        {
            return _tuner.Propose(_searchSpace);
        }

        public void Update(TrialResult result)
        {
            return;
        }
    }
}
