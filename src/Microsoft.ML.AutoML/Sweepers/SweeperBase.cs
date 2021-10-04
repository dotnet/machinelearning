// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Signature for the GUI loaders of sweepers.
    /// </summary>
    internal delegate void SignatureSweeperFromParameterList(IValueGenerator[] sweepParameters);

    /// <summary>
    /// Base sweeper that ensures the suggestions are different from each other and from the previous runs.
    /// </summary>
    internal abstract class SweeperBase : ISweeper
    {
        internal class ArgumentsBase
        {
            public IValueGenerator[] SweptParameters;

            // Number of tries to generate distinct parameter sets.
            public int Retries;

            public ArgumentsBase()
            {
                Retries = 10;
            }
        }

        private readonly ArgumentsBase _args;
        protected readonly IValueGenerator[] SweepParameters;

        protected SweeperBase(ArgumentsBase args, string name)
        {
            _args = args;

            SweepParameters = args.SweptParameters.ToArray();
        }

        protected SweeperBase(ArgumentsBase args, IValueGenerator[] sweepParameters, string name)
        {
            _args = args;
            SweepParameters = sweepParameters;
        }

        public virtual ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            var prevParamSets = new HashSet<ParameterSet>(previousRuns?.Select(r => r.ParameterSet).ToList() ?? new List<ParameterSet>());
            var result = new HashSet<ParameterSet>();
            for (int i = 0; i < maxSweeps; i++)
            {
                ParameterSet paramSet;
                int retries = 0;
                do
                {
                    paramSet = CreateParamSet();
                    ++retries;
                } while (paramSet != null && retries < _args.Retries &&
                    (AlreadyGenerated(paramSet, prevParamSets) || AlreadyGenerated(paramSet, result)));

                Runtime.Contracts.Assert(paramSet != null);
                result.Add(paramSet);
            }

            return result.ToArray();
        }

        protected abstract ParameterSet CreateParamSet();

        protected static bool AlreadyGenerated(ParameterSet paramSet, ISet<ParameterSet> previousRuns)
        {
            return previousRuns.Contains(paramSet);
        }
    }
}
