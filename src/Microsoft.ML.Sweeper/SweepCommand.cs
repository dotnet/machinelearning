// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Sweeper;

[assembly: LoadableClass(SweepCommand.Summary, typeof(SweepCommand), typeof(SweepCommand.Arguments), typeof(SignatureCommand),
    SweepCommand.LoadName, SweepCommand.LoadName, DocName = "command/Sweep.md")]

namespace Microsoft.ML.Sweeper
{
    [BestFriend]
    internal sealed class SweepCommand : ICommand
    {
#pragma warning disable CS0649 // The fields will still be set via the reflection driven mechanisms.
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Config runner", ShortName = "run,ev,evaluator", SignatureType = typeof(SignatureConfigRunner))]
            public IComponentFactory<IConfigRunner> Runner = ComponentFactoryUtils.CreateFromFunction(
                env => new LocalExeConfigRunner(env, new LocalExeConfigRunner.Options()));

            [Argument(ArgumentType.Multiple, HelpText = "Sweeper", ShortName = "s", SignatureType = typeof(SignatureSweeper))]
            public IComponentFactory<ISweeper> Sweeper;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Initial Sweep batch size (for instantiating sweep algorithm)", ShortName = "isbs")]
            public int? InitialSweepBatchSize;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Sweep batch size", ShortName = "sbs")]
            public int SweepBatchSize = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Sweep number of batches", ShortName = "snb")]
            public int SweepNumBatches = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Random seed", ShortName = "seed")]
            public int? RandomSeed;
        }
#pragma warning restore CS0649

        internal const string Summary = "Given a command line template and sweep ranges, creates and runs a sweep.";

        public static readonly string LocalExePath = Path.GetDirectoryName(typeof(SweepCommand).Module.FullyQualifiedName);

        private readonly IHost _host;
        private readonly int _numBatches;
        private readonly int _initBatchSize;
        private readonly int _batchSize;
        private readonly IConfigRunner _runner;

        private readonly ISweeper _sweeper;
        public const string LoadName = "Sweep";

        public SweepCommand(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));

            _host = env.Register("SweepCommand", args.RandomSeed);

            _host.CheckValue(args.Runner, nameof(args.Runner), Strings.PleaseSpecifyARunner);
            _host.CheckValue(args.Sweeper, nameof(args.Sweeper), Strings.PleaseSpecifyASweeper);
            _host.CheckUserArg(args.SweepNumBatches > 0, nameof(args.SweepNumBatches), Strings.MustBePositive);
            _host.CheckUserArg(!(args.InitialSweepBatchSize <= 0), nameof(args.InitialSweepBatchSize), Strings.MustBePositiveIfSpecified);
            _host.CheckUserArg(args.SweepBatchSize > 0, nameof(args.SweepBatchSize), Strings.MustBePositive);

            _numBatches = args.SweepNumBatches;
            _initBatchSize = args.InitialSweepBatchSize ?? args.SweepBatchSize;
            _batchSize = args.SweepBatchSize;
            _runner = args.Runner.CreateComponent(_host);
            _sweeper = args.Sweeper.CreateComponent(_host);
        }

        public void Run()
        {
            try
            {
                var runs = new List<IRunResult>();

                using (var ch = _host.Start("Generating sweeps"))
                {
                    for (var i = 0; i < _numBatches; ++i)
                    {
                        ch.Info("Starting batch {0}", i + 1);
                        var sweeps = _sweeper.ProposeSweeps(_batchSize, runs);
                        if (Utils.Size(sweeps) == 0)
                        {
                            ch.Info("Could only generate {0} sweeps.", runs.Count);
                            break;
                        }
                        runs.AddRange(_runner.RunConfigs(sweeps, runs.Count));
                    }

                    ch.Info("Outputs of finished runs can be found in the specified output folder");
                    _runner.Finish();
                }
            }
            catch (Exception e)
            {
                throw _host.Except(e, Strings.InternalSweepTrainingThrewException);
            }
            finally
            {
                var d = _sweeper as IDisposable;
                if (d != null)
                    d.Dispose();
            }
        }
    }
}
