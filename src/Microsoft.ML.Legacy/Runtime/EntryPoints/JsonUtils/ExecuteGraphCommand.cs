// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.Internal.Utilities;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(ExecuteGraphCommand), typeof(ExecuteGraphCommand.Arguments), typeof(SignatureCommand),
    "", "ExecGraph")]

namespace Microsoft.ML.Runtime.EntryPoints.JsonUtils
{
    public sealed class ExecuteGraphCommand : ICommand
    {
        public sealed class Arguments
        {
            [DefaultArgument(ArgumentType.Required, HelpText = "Path to the graph to run")]
            public string GraphPath;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Random seed")]
            public int? Seed;
        }

        public const string LoadName = "ExecuteGraph";

        private readonly IHost _host;
        private readonly string _path;

        public ExecuteGraphCommand(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName, args.Seed);
            _host.CheckValue(args, nameof(args));

            _host.CheckUserArg(args.GraphPath != null && File.Exists(args.GraphPath), nameof(args.GraphPath), "Graph path does not exist");
            _path = args.GraphPath;
        }

        public void Run()
        {
            JObject graph;
            try
            {
                graph = JObject.Parse(File.ReadAllText(_path));
            }
            catch (JsonReaderException ex)
            {
                throw _host.Except(ex, "Failed to parse experiment graph: {0}", ex.Message);
            }

            var runner = new GraphRunner(_host, graph[FieldNames.Nodes] as JArray);

            // Setting inputs.
            var jInputs = graph[FieldNames.Inputs] as JObject;
            if (graph[FieldNames.Inputs] != null && jInputs == null)
                throw _host.Except("Unexpected value for '{0}': {1}", FieldNames.Inputs, graph[FieldNames.Inputs]);
            if (jInputs != null)
            {
                foreach (var kvp in jInputs)
                {
                    var path = kvp.Value as JValue;
                    if (path == null)
                        throw _host.Except("Invalid value for input: {0}", kvp.Value);
                    var varName = kvp.Key;
                    var type = runner.GetPortDataKind(varName);

                    SetInputFromPath(runner, varName, path.Value<string>(), type);
                }
            }

            runner.RunAll();

            // Reading outputs.
            var jOutputs = graph[FieldNames.Outputs] as JObject;
            if (jOutputs != null)
            {
                foreach (var kvp in jOutputs)
                {
                    var path = kvp.Value as JValue;
                    if (path == null)
                        throw _host.Except("Invalid value for output: {0}", kvp.Value);
                    var varName = kvp.Key;
                    var type = runner.GetPortDataKind(varName);

                    GetOutputToPath(runner, varName, path.Value<string>(), type);
                }
            }
        }

        public void SetInputFromPath(GraphRunner runner, string varName, string path, TlcModule.DataKind kind)
        {
            _host.CheckUserArg(runner != null, nameof(runner), "Provide a GraphRunner instance.");
            _host.CheckUserArg(!string.IsNullOrWhiteSpace(varName), nameof(varName), "Specify a graph variable name.");
            _host.CheckUserArg(!string.IsNullOrWhiteSpace(path), nameof(path), "Specify a valid file path.");

            switch (kind)
            {
                case TlcModule.DataKind.FileHandle:
                    var fh = new SimpleFileHandle(_host, path, false, false);
                    runner.SetInput(varName, fh);
                    break;
                case TlcModule.DataKind.DataView:
                    IDataView loader = new BinaryLoader(_host, new BinaryLoader.Arguments(), path);
                    runner.SetInput(varName, loader);
                    break;
                case TlcModule.DataKind.PredictorModel:
                    PredictorModel pm;
                    using (var fs = File.OpenRead(path))
                        pm = new PredictorModel(_host, fs);
                    runner.SetInput(varName, pm);
                    break;
                case TlcModule.DataKind.TransformModel:
                    TransformModel tm;
                    using (var fs = File.OpenRead(path))
                        tm = new TransformModel(_host, fs);
                    runner.SetInput(varName, tm);
                    break;
                default:
                    throw _host.Except("Port type {0} not supported", kind);
            }
        }

        public void GetOutputToPath(GraphRunner runner, string varName, string path, TlcModule.DataKind kind)
        {
            _host.CheckUserArg(runner != null, nameof(runner), "Provide a GraphRunner instance.");
            _host.CheckUserArg(!string.IsNullOrWhiteSpace(varName), nameof(varName), "Specify a graph variable name.");
            _host.CheckUserArg(!string.IsNullOrWhiteSpace(path), nameof(path), "Specify a valid file path.");

            string extension = Path.GetExtension(path);
            switch (kind)
            {
                case TlcModule.DataKind.FileHandle:
                    var fh = runner.GetOutput<IFileHandle>(varName);
                    throw _host.ExceptNotSupp("File handle outputs not yet supported.");
                case TlcModule.DataKind.DataView:
                    var idv = runner.GetOutput<IDataView>(varName);
                    SaveDataView(idv, path, extension);
                    break;
                case TlcModule.DataKind.PredictorModel:
                    var pm = runner.GetOutput<IPredictorModel>(varName);
                    SavePredictorModel(pm, path);
                    break;
                case TlcModule.DataKind.TransformModel:
                    var tm = runner.GetOutput<ITransformModel>(varName);
                    using (var handle = _host.CreateOutputFile(path))
                    using (var fs = handle.CreateWriteStream())
                        tm.Save(_host, fs);
                    break;
                case TlcModule.DataKind.Array:
                    string partialPath = path.Substring(0, path.Length - extension.Length);

                    var ipmArray = runner.GetOutputOrDefault<IPredictorModel[]>(varName);
                    if (ipmArray != null && !ipmArray.GetType().IsValueType)
                    {
                        SaveArrayToFile(ipmArray.ToList(), TlcModule.DataKind.PredictorModel, partialPath, extension);
                        break;
                    }

                    var idvArray = runner.GetOutputOrDefault<IDataView[]>(varName);
                    if (idvArray != null && !idvArray.GetType().IsValueType)
                    {
                        SaveArrayToFile(idvArray.ToList(), TlcModule.DataKind.DataView, partialPath, extension);
                        break;
                    }
                    goto default;
                default:
                    throw _host.Except("Port type {0} not supported", kind);
            }

        }

        private void SaveArrayToFile<T>(List<T> array, TlcModule.DataKind kind, string partialPath, string extension)
           where T : class
        {
            for (int i = 0; i < array.Count; i++)
            {
                string path = $"{partialPath}_{i}{extension}";
                switch (kind)
                {
                    case TlcModule.DataKind.DataView:
                        SaveDataView((IDataView)array[i], path, extension);
                        break;
                    case TlcModule.DataKind.PredictorModel:
                        SavePredictorModel((IPredictorModel)array[i], path);
                        break;
                }
            }
        }

        /// <summary>
        /// Saves the PredictorModel to the given path
        /// </summary>
        private void SavePredictorModel(IPredictorModel pm, string path)
        {
            Contracts.CheckValue(pm, nameof(pm));

            using (var handle = _host.CreateOutputFile(path))
            using (var fs = handle.CreateWriteStream())
                pm.Save(_host, fs);

        }

        /// <summary>
        /// Saves the IDV to file based on its extension
        /// </summary>
        private void SaveDataView(IDataView idv, string path, string extension)
        {
            Contracts.CheckValue(idv, nameof(idv));

            IDataSaver saver;
            if (extension != ".csv" && extension != ".tsv" && extension != ".txt")
                saver = new BinarySaver(_host, new BinarySaver.Arguments());
            else
            {
                var saverArgs = new TextSaver.Arguments
                {
                    OutputHeader = true,
                    OutputSchema = false,
                    Separator = extension == ".csv" ? "comma" : "tab"
                };
                saver = new TextSaver(_host, saverArgs);
            }
            using (var handle = _host.CreateOutputFile(path))
            using (var fs = handle.CreateWriteStream())
            {
                saver.SaveData(fs, idv, Utils.GetIdentityPermutation(idv.Schema.ColumnCount)
                    .Where(x => saver.IsColumnSavable(idv.Schema.GetColumnType(x))).ToArray());
            }
        }
    }

}
