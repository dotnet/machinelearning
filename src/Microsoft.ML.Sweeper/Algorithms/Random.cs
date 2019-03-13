// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Sweeper;

[assembly: LoadableClass(typeof(UniformRandomSweeper), typeof(SweeperBase.OptionsBase), typeof(SignatureSweeper),
    "Uniform Random Sweeper", "UniformRandomSweeper", "UniformRandom")]
[assembly: LoadableClass(typeof(UniformRandomSweeper), typeof(SweeperBase.OptionsBase), typeof(SignatureSweeperFromParameterList),
    "Uniform Random Sweeper", "UniformRandomSweeperParamList", "UniformRandompl")]

namespace Microsoft.ML.Sweeper
{
    /// <summary>
    /// Random sweeper, it generates random values for each of the parameters.
    /// </summary>
    public sealed class UniformRandomSweeper : SweeperBase
    {
        public UniformRandomSweeper(IHostEnvironment env, OptionsBase options)
            : base(options, env, "UniformRandom")
        {
        }

        public UniformRandomSweeper(IHostEnvironment env, OptionsBase options, IValueGenerator[] sweepParameters)
            : base(options, env, sweepParameters, "UniformRandom")
        {
        }

        protected override ParameterSet CreateParamSet()
        {
            return new ParameterSet(SweepParameters.Select(sweepParameter => sweepParameter.CreateFromNormalized(Host.Rand.NextDouble())));
        }
    }
}
