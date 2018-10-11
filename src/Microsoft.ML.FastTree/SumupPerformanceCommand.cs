// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#if USE_SINGLE_PRECISION
using FloatType = System.Single;
#else
using FloatType = System.Double;
#endif

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(typeof(SumupPerformanceCommand), typeof(SumupPerformanceCommand.Arguments), typeof(SignatureCommand),
    "", "FastTreeSumupPerformance", "ftsumup")]

namespace Microsoft.ML.Runtime.FastTree
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    /// <summary>
    /// This is an internal utility command to measure the performance of the IntArray sumup operation.
    /// </summary>
    public sealed class SumupPerformanceCommand : ICommand
    {
        public sealed class Arguments
        {
            // REVIEW: Convert to using subcomponents.
            [Argument(ArgumentType.AtMostOnce, HelpText = "The type of IntArray to construct", ShortName = "type", SortOrder = 0)]
            public IntArrayType Type = IntArrayType.Dense;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The length of each int arrays", ShortName = "len", SortOrder = 1)]
            public int Length;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of int arrays to create", ShortName = "c", SortOrder = 2)]
            public int Count;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of bins to have in the int array", ShortName = "b", SortOrder = 3)]
            public int Bins;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The random parameter, which will differ depending on the type of the feature", ShortName = "p", SortOrder = 4)]
            public double Parameter;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of seconds to run sumups in each trial", ShortName = "s", SortOrder = 5)]
            public double Seconds = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Random seed", ShortName = "seed", SortOrder = 101)]
            public int? RandomSeed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Verbose?", ShortName = "v", Hide = true)]
            public bool? Verbose;

            // This is actually an advisory value. The implementations themselves are responsible for
            // determining what they consider appropriate, and the actual heuristics is a bit more
            // complex than just this.
            [Argument(ArgumentType.LastOccurenceWins,
                HelpText = "Desired degree of parallelism in the data pipeline", ShortName = "n", SortOrder = 6)]
            public int? Parallel;
        }

        private readonly IHost _host;

        private readonly IntArrayType _type;
        private readonly int _len;
        private readonly int _count;
        private readonly int _bins;
        private readonly int _parallel;
        private readonly double _param;
        private readonly double _seconds;

        public SumupPerformanceCommand(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));

            // Capture the environment options from args.
            env.CheckUserArg(!args.Parallel.HasValue || args.Parallel > 0, nameof(args.Parallel), "If defined must be positive");

            _host = env.Register("FastTreeSumupPerformance", args.RandomSeed, args.Verbose, args.Parallel);
            _host.CheckValue(args, nameof(args));

            _host.CheckUserArg(Enum.IsDefined(typeof(IntArrayType), args.Type) && args.Type != IntArrayType.Current, nameof(args.Type), "Value not defined");
            _host.CheckUserArg(args.Length >= 0, nameof(args.Length), "Must be non-negative");
            _host.CheckUserArg(args.Count >= 0, nameof(args.Count), "Must be non-negative");
            _host.CheckUserArg(args.Bins > 0, nameof(args.Bins), "Must be positive");
            _host.CheckUserArg(args.Seconds > 0, nameof(args.Seconds), "Must be positive");

            _type = args.Type;
            _len = args.Length;
            _count = args.Count;
            _bins = args.Bins;
            _parallel = args.Parallel ?? Environment.ProcessorCount;
            _param = args.Parameter;
            _seconds = args.Seconds;
        }

        private IEnumerable<int> CreateDense(IChannel ch, Random rgen)
        {
            for (int i = 0; i < _len; ++i)
                yield return rgen.Next(_bins);
        }

        private IEnumerable<int> CreateSparse(IChannel ch, Random rgen)
        {
            ch.CheckUserArg(0 <= _param && _param < 1, nameof(Arguments.Parameter), "For sparse ararys");
            // The parameter is the level of sparsity. Use the geometric distribution to determine the number of
            // Geometric distribution (with 0 support) would be Math.
            double denom = Math.Log(1 - _param);
            if (double.IsNegativeInfinity(denom))
            {
                // The parameter is so high, it's effectively dense.
                foreach (int v in CreateDense(ch, rgen))
                    yield return 0;
                yield break;
            }
            if (denom == 0)
            {
                // The parameter must have been so small that we effectively will never have an "on" entry.
                for (int i = 0; i < _len; ++i)
                    yield return 0;
                yield break;
            }
            ch.Assert(FloatUtils.IsFinite(denom) && denom < 0);
            int remaining = _len;
            while (remaining > 0)
            {
                // A value being sparse or not we view as a Bernoulli trial, so, we can more efficiently
                // model the number of sparse values as being a geometric distribution. This reduces the
                // number of calls to the random number generator considerable vs. the naive sampling.
                double r = 1 - rgen.NextDouble(); // Has support in [0,1). We subtract 1-r to make support in (0,1].
                int numZeros = (int)Math.Min(Math.Floor(Math.Log(r) / denom), remaining);
                for (int i = 0; i < numZeros; ++i)
                    yield return 0;
                if ((remaining -= numZeros) > 0)
                {
                    --remaining;
                    yield return rgen.Next(_bins);
                }
            }
            ch.Assert(remaining == 0);
        }

        private IntArray[] CreateRandomIntArrays(IChannel ch)
        {
            IntArray[] arrays = new IntArray[_count];
            using (var pch = _host.StartProgressChannel("Create IntArrays"))
            {
                int created = 0;
                pch.SetHeader(new ProgressHeader("arrays"), e => e.SetProgress(0, created, arrays.Length));
                IntArrayBits bits = IntArray.NumBitsNeeded(_bins);
                ch.Info("Bits per item is {0}", bits);
                int salt = _host.Rand.Next();
                Func<IChannel, Random, IEnumerable<int>> createIntArray;
                switch (_type)
                {
                    case IntArrayType.Dense:
                        createIntArray = CreateDense;
                        break;
                    case IntArrayType.Sparse:
                        createIntArray = CreateSparse;
                        if (_param == 1)
                            createIntArray = CreateDense;
                        break;
                    default:
                        throw _host.ExceptNotImpl("Haven't yet wrote a random generator appropriate for {0}", _type);
                }

                ParallelEnumerable.Range(0, arrays.Length).ForAll(i =>
                {
                    Random r = new Random(salt + i);
                    arrays[i] = IntArray.New(_len, _type, bits, createIntArray(ch, r));
                    created++;
                });

                return arrays;
            }
        }

        private IEnumerator<double> Geometric(double p, IRandom rgen)
        {
            double denom = Math.Log(1 - p);

            if (double.IsNegativeInfinity(denom))
            {
                // The parameter is so high, it's effectively dense.
                while (true)
                    yield return 0;
            }
            else if (denom == 0)
            {
                // The parameter must have been so small that we effectively will never have an "on" entry.
                while (true)
                    yield return double.PositiveInfinity;
            }
            else
            {
                while (true)
                {
                    double r = 1 - rgen.NextDouble(); // Has support in [0,1). We subtract 1-r to make support in (0,1].
                    yield return Math.Floor(Math.Log(r) / denom);
                }
            }
        }

        private IEnumerable<int> CreateDocIndicesCore(double sparsity, IRandom rgen)
        {
            _host.Assert(0 < sparsity && sparsity < 1);
            int remaining = _len;
            IEnumerator<double> g = Geometric(sparsity, rgen);
            int currDoc = -1;
            while (remaining > 0)
            {
                g.MoveNext();
                double skippedDocs = g.Current + 1; // Number docs till the next good one.
                if (skippedDocs >= remaining)
                    yield break;
                int sd = (int)skippedDocs;
                remaining -= sd;
                yield return (currDoc += sd);
            }
        }

        private IEnumerable<int> CreateDocIndices(double sparsity, IRandom rgen)
        {
            _host.Assert(0 <= sparsity && sparsity <= 1);
            if (sparsity == 1)
                return Enumerable.Range(0, _len);
            if (sparsity == 0)
                return Enumerable.Empty<int>();
            return CreateDocIndicesCore(sparsity, rgen);
        }

        private void InitSumupInputData(SumupInputData data, double sparsity, IRandom rgen)
        {
            int count = 0;
            foreach (int d in CreateDocIndices(sparsity, rgen))
                data.DocIndices[count++] = d;
            _host.Assert(Utils.IsIncreasing(0, data.DocIndices, count, _len));
            data.TotalCount = count;
            FloatType osum = 0;
            if (data.Weights == null)
            {
                for (int i = 0; i < count; ++i)
                    osum += (data.Outputs[i] = (FloatType)(2 * rgen.NextDouble() - 1));
                data.SumWeights = count;
            }
            else
            {
                FloatType wsum = 0;
                for (int i = 0; i < count; ++i)
                {
                    osum += (data.Outputs[i] = (FloatType)(2 * rgen.NextDouble() - 1));
                    wsum += (data.Weights[i] = (FloatType)(2 * rgen.NextDouble() - 1));
                }
                data.SumWeights = wsum;
            }
            data.SumTargets = osum;
        }

        public void Run()
        {
            using (var ch = _host.Start("Run"))
            {
                IntArray[] arrays = CreateRandomIntArrays(ch);
                FeatureHistogram[] histograms =
                    arrays.Select(bins => new FeatureHistogram(bins, _bins, false)).ToArray(arrays.Length);
                long bytes = arrays.Sum(i => (long)i.SizeInBytes());
                ch.Info("Created {0} int arrays taking {1} bytes", arrays.Length, bytes);

                // Objects for the pool.
                ch.Info("Parallelism = {0}", _parallel);
                AutoResetEvent[] events = Utils.BuildArray(_parallel, i => new AutoResetEvent(false));
                AutoResetEvent[] mainEvents = Utils.BuildArray(_parallel, i => new AutoResetEvent(false));
                SumupInputData data = new SumupInputData(_len, 0, 0, new FloatType[_len], null, new int[_len]);
                Thread[] threadPool = new Thread[_parallel];
                Stopwatch sw = new Stopwatch();
                long ticksPerCycle = (long)(Stopwatch.Frequency * _seconds);
                double[] partitionProportion = { 1, 1, 0.5, 1e-1, 1e-2, 1e-3, 1e-4 };

                long completed = 0;

                for (int t = 0; t < threadPool.Length; ++t)
                {
                    Thread thread = threadPool[t] = Utils.CreateForegroundThread((object io) =>
                    {
                        int w = (int)io;
                        AutoResetEvent ev = events[w];
                        AutoResetEvent mev = mainEvents[w];
                        for (int s = 0; s < partitionProportion.Length; s++)
                        {
                            ev.WaitOne();
                            long localCompleted = 0;
                            for (int f = w; ; f = f + threadPool.Length < arrays.Length ? f + threadPool.Length : w)
                            {
                                // This should repeat till done.
                                arrays[f].Sumup(data, histograms[f]);
                                if (sw.ElapsedTicks > ticksPerCycle)
                                    break;
                                Interlocked.Increment(ref completed);
                                ++localCompleted;
                            }
                            mev.Set();
                        }
                    });
                    thread.Start(t);
                }

                foreach (double partition in partitionProportion)
                {
                    InitSumupInputData(data, partition, _host.Rand);
                    completed = 0;
                    sw.Restart();
                    foreach (var e in events)
                        e.Set();
                    foreach (var e in mainEvents)
                        e.WaitOne();
                    double ticksPerDoc = (double)ticksPerCycle / (completed * data.TotalCount);
                    double nsPerDoc = ticksPerDoc * 1e9 / Stopwatch.Frequency;
                    ch.Info("Partition {0} ({1} of {2}), completed {3} ({4:0.000} ns per doc)",
                        partition, data.TotalCount, _len, completed, nsPerDoc);
                }
            }
        }
    }
}
