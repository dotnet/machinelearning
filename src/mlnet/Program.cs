// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.CLI.Commands;
using Microsoft.ML.CLI.Commands.New;
using Microsoft.ML.CLI.Data;
using Microsoft.ML.CLI.Telemetry.Events;
using Microsoft.ML.CLI.Utilities;
using NLog;
using NLog.Targets;

namespace Microsoft.ML.CLI
{
    public class Program
    {
        private static Logger _logger = LogManager.GetCurrentClassLogger();

        public static void Main(string[] args)
        {
            Telemetry.Telemetry.Initialize();
            int exitCode = 1;
            Exception ex = null;
            var stopwatch = Stopwatch.StartNew();

            var mlNetCommandEvent = new MLNetCommandEvent();

            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<NewCommandSettings>(
                (options) =>
             {
                 try
                 {
                     // Send telemetry event for command issued
                     mlNetCommandEvent.AutoTrainCommandSettings = options;
                     mlNetCommandEvent.TrackEvent();

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
                     var logFilePath = Path.Combine(Path.Combine(outputBaseDir, "logs"), "debug_log.txt");
                     logfile.FileName = logFilePath;
                     options.LogFilePath = logFilePath;
                     var config = LogManager.Configuration;
                     config.AddRule(verbosity, LogLevel.Fatal, logconsole);

                     // Execute the command
                     command.Execute();
                     exitCode = 0;
                 }
                 catch (Exception e)
                 {
                     ex = e;
                     _logger.Log(LogLevel.Error, e.Message);
                     _logger.Log(LogLevel.Debug, e.ToString());
                     _logger.Log(LogLevel.Info, Strings.LookIntoLogFile);
                     _logger.Log(LogLevel.Error, Strings.Exiting);
                 }
             });

            var parser = new CommandLineBuilder()
                         // parser
                         .AddCommand(CommandDefinitions.AutoTrain(handler))
                         .UseDefaults()
                         .Build();

            var parseResult = parser.Parse(args);

            var commandParseSucceeded = !parseResult.Errors.Any();
            if (commandParseSucceeded)
            {
                if (parseResult.RootCommandResult.Children.Count > 0)
                {
                    var command = parseResult.RootCommandResult.Children.First();
                    var parsedArguments = command.Children;

                    if (parsedArguments.Count > 0)
                    {
                        var options = parsedArguments.ToList().Where(sr => sr is System.CommandLine.OptionResult).Cast<System.CommandLine.OptionResult>();

                        var explicitlySpecifiedOptions = options.Where(opt => !opt.IsImplicit).Select(opt => opt.Name);

                        mlNetCommandEvent.CommandLineParametersUsed = explicitlySpecifiedOptions;
                    }
                }
            }

            // Send system info telemetry
            SystemInfoEvent.TrackEvent();

            parser.InvokeAsync(parseResult).Wait();
            // Send exit telemetry
            ApplicationExitEvent.TrackEvent(exitCode, commandParseSucceeded, stopwatch.Elapsed, ex);
            // Flush pending telemetry logs
            Telemetry.Telemetry.Flush(TimeSpan.FromSeconds(3));
            Environment.Exit(exitCode);
        }
    }
}