// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.IO;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.Commands;
using Microsoft.ML.CLI.Commands.New;
using Microsoft.ML.CLI.Data;
using NLog;
using NLog.Targets;

namespace Microsoft.ML.CLI
{
    class Program
    {
        public static void Main(string[] args)
        {
            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<FileInfo, FileInfo, FileInfo, TaskKind, string, uint, uint>(
                (trainDataset, validationDataset, testDataset, mlTask, labelColumnName, maxExplorationTime, labelColumnIndex) =>
             {
                 if (mlTask == TaskKind.MulticlassClassification)
                 {
                     Console.WriteLine($"{Strings.UnsupportedMlTask}: {mlTask}");
                     return;
                 }
                 /* The below variables needs to be initialized via command line api. Since there is a 
                    restriction at this moment on the number of args and its bindings. .Net team is working 
                    on making this feature to make it possible to bind directly to a type till them we shall 
                    have this place holder by initializing the fields below . 
                    The PR that addresses this issue : https://github.com/dotnet/command-line-api/pull/408  
                  */
                 var basedir = "."; // This needs to be obtained from command line args.
                 var name = "Sample"; // This needs to be obtained from command line args.

                 // Todo: q,m,diag needs to be mapped into LogLevel here.
                 var verbosity = LogLevel.Info;

                 var command = new NewCommand(new NewCommandOptions()
                 {
                     TrainDataset = trainDataset,
                     ValidationDataset = validationDataset,
                     TestDataset = testDataset,
                     MlTask = mlTask,
                     LabelName = labelColumnName,
                     Timeout = maxExplorationTime,
                     LabelIndex = labelColumnIndex,
                     OutputBaseDir = basedir,
                     OutputName = name
                 });

                 // Override the Logger Configuration
                 var logconsole = LogManager.Configuration.FindTargetByName("logconsole");
                 var logfile = (FileTarget)LogManager.Configuration.FindTargetByName("logfile");
                 logfile.FileName = $"{basedir}/debug_log.txt";
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
