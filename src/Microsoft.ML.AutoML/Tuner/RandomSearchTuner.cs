// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.SearchSpace.Tuner;

namespace Microsoft.ML.AutoML
{
    internal class RandomSearchTuner : ITuner
    {
        private readonly RandomTuner _tuner;
        private readonly SearchSpace.SearchSpace _searchSpace;

        public RandomSearchTuner(SearchSpace.SearchSpace searchSpace)
        {
            this._tuner = new RandomTuner();
            this._searchSpace = searchSpace;
        }
        public Parameter Propose(TrialSettings settings)
        {
            return this._tuner.Propose(this._searchSpace);
        }

        public void Update(TrialResult result)
        {
            return;
        }
    }
}
