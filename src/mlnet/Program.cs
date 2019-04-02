// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using Microsoft.DotNet.Cli.Telemetry;
using Microsoft.ML.CLI.Commands;
using Microsoft.ML.CLI.Commands.New;
using Microsoft.ML.CLI.Data;
using Microsoft.ML.CLI.Utilities;
using NLog;
using NLog.Targets;

namespace Microsoft.ML.CLI
{
    class Program
    {
        public static void Main(string[] args)
        {
            var telemetry = new MlTelemetry();

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<NewCommandSettings>(
                (options) =>
             {
                 // Map the verbosity to internal levels
                 var verbosity = Utils.GetVerbosity(options.Verbosity);

                 // Build the output path
                 string outputBaseDir = string.Empty;
                 if (options.Name == null)
                 {

                     options.Name = "Sample" + Utils.GetTaskKind(options.MlTask).ToString();
                     outputBaseDir = Path.Combine(options.OutputPath.FullName, options.Name);
                 }
                 else
                 {
                     outputBaseDir = Path.Combine(options.OutputPath.FullName, options.Name);
                 }

                 // Override the output path
                 options.OutputPath = new DirectoryInfo(outputBaseDir);

                 // Instantiate the command
                 var command = new NewCommand(options, telemetry);

                 // Override the Logger Configuration
                 var logconsole = LogManager.Configuration.FindTargetByName("logconsole");
                 var logfile = (FileTarget)LogManager.Configuration.FindTargetByName("logfile");
                 logfile.FileName = $"{outputBaseDir}/logs/debug_log.txt";
                 var config = LogManager.Configuration;
                 config.AddRule(verbosity, LogLevel.Fatal, logconsole);

                 // Execute the command
                 command.Execute();
             });

            var parser = new CommandLineBuilder()
                         // parser
                         .AddCommand(CommandDefinitions.New(handler))
                         .UseDefaults()
                         .Build();

            var parseResult = parser.Parse(args);

            if (parseResult.Errors.Count == 0)
            {
                if (parseResult.RootCommandResult.Children.Count > 0)
                {
                    var command = parseResult.RootCommandResult.Children.First();
                    var parsedArguments = command.Children;

                    if (parsedArguments.Count > 0)
                    {
                        var options = parsedArguments.ToList().Where(sr => sr is System.CommandLine.OptionResult).Cast<System.CommandLine.OptionResult>();

                        var explicitlySpecifiedOptions = options.Where(opt => !opt.IsImplicit).Select(opt => opt.Name);

                        telemetry.SetCommandAndParameters(command.Name, explicitlySpecifiedOptions);
                    }
                }
            }

            parser.InvokeAsync(parseResult).Wait();
        }
    }
}
