// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Command;

[assembly: LoadableClass(SweepCommand.Summary, typeof(SweepCommand), typeof(SweepCommand.Arguments), typeof(SignatureCommand),
    SweepCommand.LoadName, SweepCommand.LoadName, DocName = "command/Sweep.md")]

namespace Microsoft.ML.Runtime.Sweeper
{
    public sealed class SweepCommand : ICommand
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Multiple, HelpText = "Config runner", ShortName = "run,ev,evaluator")]
            public SubComponent<IConfigRunner, SignatureConfigRunner> Runner = new SubComponent<IConfigRunner, SignatureConfigRunner>("Local");

            [Argument(ArgumentType.Multiple, HelpText = "Sweeper", ShortName = "s")]
            public SubComponent<ISweeper, SignatureSweeper> Sweeper;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Initial Sweep batch size (for instantiating sweep algorithm)", ShortName = "isbs")]
            public int? InitialSweepBatchSize;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Sweep batch size", ShortName = "sbs")]
            public int SweepBatchSize = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Sweep number of batches", ShortName = "snb")]
            public int SweepNumBatches = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Random seed", ShortName = "seed")]
            public int? RandomSeed;
        }

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

            _host.CheckUserArg(args.Runner.IsGood(), nameof(args.Runner), "Please specify a runner");
            _host.CheckUserArg(args.Sweeper.IsGood(), nameof(args.Sweeper), "Please specify a sweeper");
            _host.CheckUserArg(args.SweepNumBatches > 0, nameof(args.SweepNumBatches), "Must be positive");
            _host.CheckUserArg(!(args.InitialSweepBatchSize <= 0), nameof(args.InitialSweepBatchSize), "Must be positive if specified");
            _host.CheckUserArg(args.SweepBatchSize > 0, nameof(args.SweepBatchSize), "Must be positive");

            _numBatches = args.SweepNumBatches;
            _initBatchSize = args.InitialSweepBatchSize ?? args.SweepBatchSize;
            _batchSize = args.SweepBatchSize;
            _runner = args.Runner.CreateInstance(_host);
            _sweeper = args.Sweeper.CreateInstance(_host);
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

                    ch.Done();
                }
            }
            catch (Exception e)
            {
                throw _host.Except(e, "Internal sweep training threw exception");
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
