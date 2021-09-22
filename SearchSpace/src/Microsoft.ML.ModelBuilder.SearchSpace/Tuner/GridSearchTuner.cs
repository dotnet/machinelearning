// <copyright file="GridSearchTuner.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Tuner
{
    public class GridSearchTuner
    {
        private SearchSpace searchSpace;
        private int stepSize;

        public GridSearchTuner(SearchSpace ss, int stepSize = 10)
        {
            this.searchSpace = ss;
            this.stepSize = stepSize;
        }

        public IEnumerable<Parameter> Propose()
        {
            var steps = this.searchSpace.Step.Select(x => x ?? this.stepSize)
                                        .Select(x => Enumerable.Range(0, x).Select(i => i * 1.0 / x).ToArray());
            foreach (var featureVec in this.CartesianProduct(steps))
            {
                yield return this.searchSpace.SampleFromFeatureSpace(featureVec);
            }
        }

        private IEnumerable<double[]> CartesianProduct(IEnumerable<double[]> arrays)
        {
            if (arrays.Count() == 1)
            {
                foreach (var i in arrays.First())
                {
                    yield return new[] { i };
                }
            }
            else
            {
                foreach (var i in arrays.First())
                {
                    foreach (var i_s in this.CartesianProduct(arrays.Skip(1)))
                    {
                        yield return new[] { i }.Concat(i_s).ToArray();
                    }
                }
            }
        }
    }
}
