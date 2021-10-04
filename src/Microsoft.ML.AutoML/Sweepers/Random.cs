// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Random sweeper, it generates random values for each of the parameters.
    /// </summary>
    internal sealed class UniformRandomSweeper : SweeperBase
    {
        public UniformRandomSweeper(ArgumentsBase args)
            : base(args, "UniformRandom")
        {
        }

        public UniformRandomSweeper(ArgumentsBase args, IValueGenerator[] sweepParameters)
            : base(args, sweepParameters, "UniformRandom")
        {
        }

        protected override ParameterSet CreateParamSet()
        {
            return new ParameterSet(SweepParameters.Select(sweepParameter => sweepParameter.CreateFromNormalized(AutoMlUtils.Random.Value.NextDouble())));
        }
    }
}
