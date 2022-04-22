// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.SearchSpace.Tuner
{
    internal sealed class RandomTuner
    {
        private readonly Random _rnd;

        public RandomTuner()
        {
            _rnd = new Random();
        }

        public Parameter Propose(SearchSpace searchSpace)
        {
            var d = searchSpace.FeatureSpaceDim;
            var featureVec = Enumerable.Repeat(0, d).Select(i => _rnd.NextDouble()).ToArray();
            return searchSpace.SampleFromFeatureSpace(featureVec);
        }

        public void Update(Parameter param, double metric, bool isMaximize)
        {
            // do nothing
        }
    }
}
