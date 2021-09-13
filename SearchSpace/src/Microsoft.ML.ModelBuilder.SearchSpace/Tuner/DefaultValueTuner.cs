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
    public class DefaultValueTuner<T>
        where T: class, new()
    {
        private readonly SearchSpace<T> ss;

        public DefaultValueTuner(SearchSpace<T> ss)
        {
            this.ss = ss;
        }

        public T Propose()
        {
            var defaultFeatureVec = this.ss.Default;
            return this.ss.SampleFromFeatureSpace(defaultFeatureVec);
        }
    }
}
