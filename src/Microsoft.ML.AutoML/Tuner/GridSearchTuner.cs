// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal class GridSearchTuner : ITuner
    {
        private readonly SearchSpace.Tuner.GridSearchTuner _tuner;
        private IEnumerator<Parameter> _enumerator;

        public GridSearchTuner(SearchSpace.SearchSpace searchSpace, int stepSize = 10)
        {
            _tuner = new SearchSpace.Tuner.GridSearchTuner(searchSpace, stepSize);
            _enumerator = _tuner.Propose().GetEnumerator();
        }
        public Parameter Propose(TrialSettings settings)
        {
            if (!_enumerator.MoveNext())
            {
                _enumerator = _tuner.Propose().GetEnumerator();
                return Propose(settings);
            }
            else
            {
                var res = _enumerator.Current;
                return res;
            }
        }

        public void Update(TrialResult result)
        {
            return;
        }
    }
}
