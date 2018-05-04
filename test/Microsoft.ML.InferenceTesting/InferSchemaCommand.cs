// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.MLTesting.Inference;
using Newtonsoft.Json;

[assembly: LoadableClass(typeof(InferSchemaCommand), typeof(InferSchemaCommand.Arguments), typeof(SignatureCommand),
    "Infer Schema", "InferSchema", DocName = "command/InferSchema.md")]

namespace Microsoft.ML.Runtime.MLTesting.Inference
{
    /// <summary>
    /// This command generates a suggested RSP to load the text file and recipes it prior to training.
    /// The results are output to the console and also to the RSP file, if it's specified.
    /// </summary>
    public sealed class InferSchemaCommand : ICommand
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Required, HelpText = "Text file with data to analyze", ShortName = "data")]
            public string DataFile;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Path to the output json file describing the columns", ShortName = "out")]
            public string OutputFile;
        }

        private readonly IHost _host;
        private readonly string _dataFile;
        private readonly string _outFile;

        public InferSchemaCommand(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("InferSchema", seed: 0, verbose: true);
            _host.CheckValue(args, nameof(args));

            var files = new MultiFileSource(args.DataFile);
            _host.CheckUserArg(files.Count > 0, nameof(args.DataFile), "dataFile is required");
            _dataFile = args.DataFile;
            if (!string.IsNullOrWhiteSpace(args.OutputFile))
            {
                Utils.CheckOptionalUserDirectory(args.OutputFile, nameof(args.OutputFile));
                _outFile = args.OutputFile;
            }
        }

        public void Run()
        {
            using (var ch = _host.Start("Running"))
            {
                RunCore(ch);
                ch.Done();
            }
        }

        public void RunCore(IChannel ch)
        {
            _host.AssertValue(ch);

            // Inner env is used to ignore verbose messages from the text loader.
            var envInner = _host.Register("inner host", seed: 0, verbose: false);

            ch.Info("Loading file sample into memory.");
            var sample = TextFileSample.CreateFromFullFile(envInner, _dataFile);

            ch.Info("Detecting separator and columns");
            var splitResult = TextFileContents.TrySplitColumns(envInner, sample, TextFileContents.DefaultSeparators);
            if (!splitResult.IsSuccess)
                throw Contracts.ExceptDecode("Couldn't detect separator.");

            ch.Info("Separator detected as '{0}', there are {1} columns.", splitResult.Separator, splitResult.ColumnCount);
            bool hasHeader;
            ColumnGroupingInference.GroupingColumn[] groupingResult = InferenceUtils.InferColumnPurposes(ch, envInner, sample, splitResult, out hasHeader);

            string json = "";
            try
            {
                json = JsonConvert.SerializeObject(groupingResult, Formatting.Indented);
            }
            catch
            {
                ch.Error("Error serializing the schema file. Check its content.");
            }

            if (!string.IsNullOrEmpty(json))
            {
                if (_outFile != null)
                {
                    using (var sw = new StreamWriter(_outFile))
                        PrintSchema(json, sw, ch);
                }
                else
                    PrintSchema(json, null, ch);
            }
        }

        private void PrintSchema(string schema, StreamWriter sw, IChannel ch)
        {
            ch.Info(MessageSensitivity.Schema, schema);
            sw?.WriteLine(schema);
        }
    }
}
