// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.IO;
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
                 var command = new NewCommand(options);

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


            parser.InvokeAsync(args).Wait();
        }
    }
}
