// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.IO;
using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI
{
    class Program
    {
        public static void Main(string[] args)
        {
            // Create handler outside so that commandline and the handler is decoupled and testable.
            var handler = CommandHandler.Create<FileInfo, FileInfo, FileInfo, TaskKind, string, uint, uint>(
                (trainDataset, validationDataset, testDataset, mlTask, labelColumnName, timeout, labelColumnIndex) =>
             {
                 var command = new NewCommand(new Options()
                 {
                     TrainDataset = trainDataset,
                     ValidationDataset = validationDataset,
                     TestDataset = testDataset,
                     MlTask = mlTask,
                     LabelName = labelColumnName,
                     Timeout = timeout,
                     LabelIndex = labelColumnIndex
                 });
                 command.Run();
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
