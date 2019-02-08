// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CommandLine;
using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI
{
    public static class CommandDefinitions
    {
        public static System.CommandLine.Command New()
        {
            var newCommand = new System.CommandLine.Command("new", "ML.NET CLI tool for code generation",

                            handler: CommandHandler.Create<FileInfo, FileInfo, TaskKind, string>((FileInfo trainDataset, FileInfo testDataset, TaskKind mlTask, string labelColumnName) =>
                                 {
                                     NewCommand.Run(new Options()
                                     {
                                         TrainDataset = trainDataset,
                                         TestDataset = testDataset,
                                         MlTask = mlTask,
                                         LabelName = labelColumnName
                                     });

                                 }))
            {
                                TrainDataset(),
                                TestDataset(),
                                MlTask(),
                                LabelName(),
                                //ColumnSeperator(),
                                //ExplorationTimeout(),
                                //Name(),
                                //ShowOutput()
                                //LabelIndex()
            };

            newCommand.Argument.AddValidator((sym) =>
            {
                if (sym.Children["--train-dataset"] == null)
                {
                    return "Option required : --train-dataset";
                }
                if (sym.Children["--ml-task"] == null)
                {
                    return "Option required : --ml-task";
                }
                if (sym.Children["--label-column-name"] == null)
                {
                    return "Option required : --label-column-name";
                }

                return null;
            });

            return newCommand;


            Option TrainDataset() =>
               new Option("--train-dataset", "Train dataset file path.",
                          new Argument<FileInfo>().ExistingOnly());

            Option TestDataset() =>
               new Option("--test-dataset", "Test dataset file path.",
                          new Argument<FileInfo>(defaultValue: default(FileInfo)).ExistingOnly());

            Option MlTask() =>
               new Option("--ml-task", "Type of ML task.",
                          new Argument<TaskKind>().WithSuggestions(GetMlTaskSuggestions()));

            Option LabelName() =>
               new Option("--label-column-name", "Name of the label column.",
                          new Argument<string>());

        }

        private static string[] GetMlTaskSuggestions()
        {
            return Enum.GetValues(typeof(TaskKind)).Cast<TaskKind>().Select(v => v.ToString()).ToArray();
        }
    }
}
