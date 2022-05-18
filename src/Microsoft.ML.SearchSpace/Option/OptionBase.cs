// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.


#nullable enable

namespace Microsoft.ML.SearchSpace.Option
{
    /// <summary>
    /// abstrace class for Option.
    /// </summary>
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
        /// <param name="values">value to sample.</param>
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

        /// <summary>
        /// Gets the step of this option. The <see cref="Step"/> is used to determine the number of grid this option should be divided into. In <see cref="ChoiceOption"/>, it's always the length of
        /// <see cref="ChoiceOption.Choices"/>. And in <see cref="UniformNumericOption"/>, it's always [null]. And in <see cref="SearchSpace"/>, it's a combination of all <see cref="Step"/> in its options.
        /// </summary>
        public abstract int?[] Step { get; }
    }
}
