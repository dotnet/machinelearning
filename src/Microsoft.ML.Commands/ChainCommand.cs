// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Globalization;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Tools;

[assembly: LoadableClass(ChainCommand.Summary, typeof(ChainCommand), typeof(ChainCommand.Arguments), typeof(SignatureCommand),
    "Chain Command", "Chain")]

namespace Microsoft.ML.Runtime.Tools
{
    using Stopwatch = System.Diagnostics.Stopwatch;

    public sealed class ChainCommand : ICommand
    {
        public sealed class Arguments
        {
#pragma warning disable 649 // never assigned
            [Argument(ArgumentType.Multiple, HelpText = "Command", ShortName = "cmd")]
            public SubComponent<ICommand, SignatureCommand>[] Command;
#pragma warning restore 649 // never assigned
        }

        internal const string Summary = "A command that chains multiple other commands.";

        private readonly IHost _host;

        private readonly Arguments _args;

        public ChainCommand(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));

            _args = args;
            _host = env.Register("Chain");
        }

        public void Run()
        {
            using (var ch = _host.Start("Run"))
            {
                var sw = new Stopwatch();
                int count = 0;

                sw.Start();
                if (_args.Command != null)
                {
                    for (int i = 0; i < _args.Command.Length; i++)
                    {
                        using (var chCmd = _host.Start(string.Format(CultureInfo.InvariantCulture, "Command[{0}]", i)))
                        {
                            var sub = _args.Command[i];

                            chCmd.Info("=====================================================================================");
                            chCmd.Info("Executing: {0}", sub);
                            chCmd.Info("=====================================================================================");

                            var cmd = sub.CreateInstance(_host);
                            cmd.Run();
                            count++;

                            chCmd.Info(" ");

                            chCmd.Done();
                        }
                    }
                }
                sw.Stop();

                ch.Info("=====================================================================================");
                ch.Info("Executed {0} commands in {1}", count, sw.Elapsed);
                ch.Info("=====================================================================================");

                ch.Done();
            }
        }
    }
}
