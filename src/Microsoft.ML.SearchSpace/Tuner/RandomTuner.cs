// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;

namespace Microsoft.ML.SearchSpace.Tuner
{
    internal sealed class RandomTuner
    {
        private readonly SearchSpace _searchSpace;
        private readonly Random _rnd;

        public RandomTuner(SearchSpace searchSpace)
        {
            this._searchSpace = searchSpace;
            this._rnd = new Random();
        }

        public RandomTuner(SearchSpace searchSpace, int seed)
        {
            this._searchSpace = searchSpace;
            this._rnd = new Random(seed);
        }

        public Parameter Propose()
        {
            var d = this._searchSpace.FeatureSpaceDim;
            var featureVec = Enumerable.Repeat(0, d).Select(i => this._rnd.NextDouble()).ToArray();
            return this._searchSpace.SampleFromFeatureSpace(featureVec);
        }
    }
}
