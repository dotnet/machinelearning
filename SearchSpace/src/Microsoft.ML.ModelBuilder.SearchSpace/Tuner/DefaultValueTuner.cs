// <copyright file="DefaultValueTuner.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.ModelBuilder.SearchSpace.Tuner
{
    /// <summary>
    /// A tuner which always return default value.
    /// </summary>
    public class DefaultValueTuner
    {
        private readonly SearchSpace ss;

        public DefaultValueTuner(SearchSpace ss)
        {
            this.ss = ss;
        }

        public IParameter Propose()
        {
            var defaultFeatureVec = this.ss.Default;
            return this.ss.SampleFromFeatureSpace(defaultFeatureVec);
        }
    }
}
