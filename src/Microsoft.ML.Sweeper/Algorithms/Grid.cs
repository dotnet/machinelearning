// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Sweeper;

[assembly: LoadableClass(typeof(RandomGridSweeper), typeof(RandomGridSweeper.Arguments), typeof(SignatureSweeper),
    "Random Grid Sweeper", "RandomGridSweeper", "RandomGrid")]
[assembly: LoadableClass(typeof(RandomGridSweeper), typeof(RandomGridSweeper.Arguments), typeof(SignatureSweeperFromParameterList),
    "Random Grid Sweeper", "RandomGridSweeperParamList", "RandomGridpl")]

namespace Microsoft.ML.Runtime.Sweeper
{
    /// <summary>
    /// Signature for the GUI loaders of sweepers.
    /// </summary>
    public delegate void SignatureSweeperFromParameterList(IValueGenerator[] sweepParameters);

    /// <summary>
    /// Base sweeper that ensures the suggestions are different from each other and from the previous runs.
    /// </summary>
    public abstract class SweeperBase : ISweeper
    {
        public class ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Swept parameters", ShortName = "p", SignatureType = typeof(SignatureSweeperParameter))]
            public IComponentFactory<IValueGenerator>[] SweptParameters;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of tries to generate distinct parameter sets.", ShortName = "r")]
            public int Retries = 10;
        }

        private readonly ArgumentsBase _args;
        protected readonly IValueGenerator[] SweepParameters;
        protected readonly IHost Host;

        protected SweeperBase(ArgumentsBase args, IHostEnvironment env, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
            Host.CheckValue(args, nameof(args));
            Host.CheckNonEmpty(args.SweptParameters, nameof(args.SweptParameters));

            _args = args;

            SweepParameters = args.SweptParameters.Select(p => p.CreateComponent(Host)).ToArray();
        }

        protected SweeperBase(ArgumentsBase args, IHostEnvironment env, IValueGenerator[] sweepParameters, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
            Host.CheckValue(args, nameof(args));
            Host.CheckValue(sweepParameters, nameof(sweepParameters));

            _args = args;
            SweepParameters = sweepParameters;
        }

        public virtual ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            var prevParamSets = previousRuns?.Select(r => r.ParameterSet).ToList() ?? new List<ParameterSet>();
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

                Contracts.Assert(paramSet != null);
                result.Add(paramSet);
            }

            return result.ToArray();
        }

        protected abstract ParameterSet CreateParamSet();

        protected static bool AlreadyGenerated(ParameterSet paramSet, IEnumerable<ParameterSet> previousRuns)
        {
            return previousRuns.Any(previousRun => previousRun.Equals(paramSet));
        }
    }

    /// <summary>
    /// Random grid sweeper, it generates random points from the grid.
    /// </summary>
    public sealed class RandomGridSweeper : SweeperBase
    {
        private readonly int _nGridPoints;

        // This stores the order of the grid points that are to be generated
        // Only used when the total number of parameter combinations is less than maxGridPoints
        // Every grid point is stored as an int representing the position it would be in a flattened grid
        // In other words, for D dimensions d1,...dn, a point x1,...xn is represented as
        // Sum(i=1..n, xi * Prod(j=i+1..n, dj))
        private readonly int[] _permutation;
        // This is a parallel array to the _permutation array and stores the (already generated) parameter sets
        private readonly ParameterSet[] _cache;

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Limit for the number of combinations to generate the entire grid.", ShortName = "maxpoints")]
            public int MaxGridPoints = 1000000;
        }

        public RandomGridSweeper(IHostEnvironment env, Arguments args)
            : base(args, env, "RandomGrid")
        {
            _nGridPoints = 1;
            foreach (var sweptParameter in SweepParameters)
            {
                _nGridPoints *= sweptParameter.Count;
                if (_nGridPoints > args.MaxGridPoints)
                    _nGridPoints = 0;
            }
            if (_nGridPoints != 0)
            {
                _permutation = Utils.GetRandomPermutation(Host.Rand, _nGridPoints);
                _cache = new ParameterSet[_nGridPoints];
            }
        }

        public RandomGridSweeper(IHostEnvironment env, Arguments args, IValueGenerator[] sweepParameters)
            : base(args, env, sweepParameters, "RandomGrid")
        {
            _nGridPoints = 1;
            foreach (var sweptParameter in SweepParameters)
            {
                _nGridPoints *= sweptParameter.Count;
                if (_nGridPoints > args.MaxGridPoints)
                    _nGridPoints = 0;
            }
            if (_nGridPoints != 0)
            {
                _permutation = Utils.GetRandomPermutation(Host.Rand, _nGridPoints);
                _cache = new ParameterSet[_nGridPoints];
            }
        }

        public override ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            if (_nGridPoints == 0)
                return base.ProposeSweeps(maxSweeps, previousRuns);

            var result = new HashSet<ParameterSet>();
            var prevParamSets = (previousRuns != null)
                ? previousRuns.Select(r => r.ParameterSet).ToList()
                : new List<ParameterSet>();
            int iPerm = (prevParamSets.Count - 1) % _nGridPoints;
            int tries = 0;
            for (int i = 0; i < maxSweeps; i++)
            {
                for (; ; )
                {
                    iPerm = (iPerm + 1) % _nGridPoints;
                    if (_cache[iPerm] == null)
                        _cache[iPerm] = CreateParamSet(_permutation[iPerm]);
                    if (tries++ >= _nGridPoints)
                        return result.ToArray();
                    if (!AlreadyGenerated(_cache[iPerm], prevParamSets))
                        break;
                }
                result.Add(_cache[iPerm]);
            }
            return result.ToArray();
        }

        protected override ParameterSet CreateParamSet()
        {
            return new ParameterSet(SweepParameters.Select(sweepParameter => sweepParameter[Host.Rand.Next(sweepParameter.Count)]));
        }

        private ParameterSet CreateParamSet(int combination)
        {
            Contracts.Assert(0 <= combination && combination < _nGridPoints);
            int div = _nGridPoints;
            var pset = new List<IParameterValue>();
            foreach (var sweepParameter in SweepParameters)
            {
                Contracts.Assert(sweepParameter.Count > 0);
                Contracts.Assert(div % sweepParameter.Count == 0);
                div /= sweepParameter.Count;
                var pv = sweepParameter[combination / div];
                combination %= div;
                pset.Add(pv);
            }
            return new ParameterSet(pset);
        }
    }
}
