// <copyright file="OptionBase.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

#nullable enable

namespace Microsoft.ML.ModelBuilder.SearchSpace.Option
{
    public abstract class OptionBase
    {
        /// <summary>
        /// mapping value to [0, 1) uniform distribution.
        /// </summary>
        /// <returns>mapping value in [0,1).</returns>
        public abstract double[] MappingToFeatureSpace(Parameter value);

        /// <summary>
        /// sample from [0,1) uniform distribution.
        /// </summary>
        /// <param name="value">value to sample.</param>
        /// <returns>sampled value.</returns>
        public abstract Parameter SampleFromFeatureSpace(double[] values);

        /// <summary>
        /// the dimension of feature space, which is equal to the output length of <see cref="SampleFromFeatureSpace(double[])"/>.
        /// </summary>
        public abstract int FeatureSpaceDim { get; }

        /// <summary>
        /// Gets the default value which is mapping to feature space (if exists).
        /// </summary>
        public virtual double[]? Default { get; protected set; }
    }
}
