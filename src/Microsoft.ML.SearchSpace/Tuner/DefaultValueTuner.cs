// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.ModelBuilder.SearchSpace.Tuner
{
    /// <summary>
    /// A tuner which always return default value.
    /// </summary>
    public class DefaultValueTuner
    {
        private readonly SearchSpace _searchSpace;

        public DefaultValueTuner(SearchSpace ss)
        {
            this._searchSpace = ss;
        }

        public IParameter Propose()
        {
            var defaultFeatureVec = this._searchSpace.Default;
            return this._searchSpace.SampleFromFeatureSpace(defaultFeatureVec);
        }
    }
}
