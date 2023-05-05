// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.SearchSpace.Tuner
{
    internal sealed class GridSearchTuner
    {
        private readonly SearchSpace _searchSpace;
        private readonly int _stepSize;

        public GridSearchTuner(SearchSpace searchSpace, int stepSize = 10)
        {
            _searchSpace = searchSpace;
            _stepSize = stepSize;
        }

        public IEnumerable<Parameter> Propose()
        {
            var steps = _searchSpace.Step.Select(x => x ?? _stepSize)
                                        .Select(x => Enumerable.Range(0, x).Select(i => i * 1.0 / x).ToArray());
            foreach (var featureVec in CartesianProduct(steps))
            {
                yield return _searchSpace.SampleFromFeatureSpace(featureVec);
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
                    foreach (var i_s in CartesianProduct(arrays.Skip(1)))
                    {
                        yield return new[] { i }.Concat(i_s).ToArray();
                    }
                }
            }
        }
    }
}
