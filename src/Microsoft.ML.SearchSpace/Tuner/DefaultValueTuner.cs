// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.SearchSpace.Tuner
{
    /// <summary>
    /// A tuner which always return default value.
    /// </summary>
    internal sealed class DefaultValueTuner
    {
        private readonly SearchSpace _searchSpace;

        public DefaultValueTuner(SearchSpace searchSpace)
        {
            _searchSpace = searchSpace;
        }

        public Parameter Propose()
        {
            var defaultFeatureVec = _searchSpace.Default;
            return _searchSpace.SampleFromFeatureSpace(defaultFeatureVec);
        }
    }
}
