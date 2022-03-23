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
    internal class GridSearchTuner : ITuner
    {
        private readonly SearchSpace.Tuner.GridSearchTuner _tuner;
        private IEnumerator<Parameter> _enumerator;

        public GridSearchTuner(SearchSpace.SearchSpace searchSpace)
        {
            this._tuner = new SearchSpace.Tuner.GridSearchTuner(searchSpace);
            this._enumerator = this._tuner.Propose().GetEnumerator();
        }
        public Parameter Propose(TrialSettings settings)
        {
            if (!this._enumerator.MoveNext())
            {
                this._enumerator = this._tuner.Propose().GetEnumerator();
                return this.Propose(settings);
            }
            else
            {
                var res = this._enumerator.Current;
                return res;
            }
        }

        public void Update(TrialResult result)
        {
            return;
        }
    }
}
