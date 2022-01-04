// <copyright file="RandomTuner.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Tuner
{
    public class RandomTuner
    {
        private SearchSpace searchSpace;
        private Random rnd;

        public RandomTuner(SearchSpace ss)
        {
            this.searchSpace = ss;
            this.rnd = new Random();
        }

        public RandomTuner(SearchSpace ss, int seed)
        {
            this.searchSpace = ss;
            this.rnd = new Random(seed);
        }

        public IParameter Propose()
        {
            var d = this.searchSpace.FeatureSpaceDim;
            var featureVec = Enumerable.Repeat(0, d).Select(i => this.rnd.NextDouble()).ToArray();
            return this.searchSpace.SampleFromFeatureSpace(featureVec);
        }
    }
}
