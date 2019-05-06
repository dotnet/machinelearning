// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(ExecuteGraphCommand), typeof(ExecuteGraphCommand.Arguments), typeof(SignatureCommand),
    "", "ExecGraph")]

namespace Microsoft.ML.EntryPoints
{
    internal sealed class ExecuteGraphCommand : ICommand
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
                    PredictorModelImpl pm;
                    using (var fs = File.OpenRead(path))
                        pm = new PredictorModelImpl(_host, fs);
                    runner.SetInput(varName, pm);
                    break;
                case TlcModule.DataKind.TransformModel:
                    TransformModelImpl tm;
                    using (var fs = File.OpenRead(path))
                        tm = new TransformModelImpl(_host, fs);
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
                    if (idv != null)
                        SaveDataView(idv, path, extension);
                    else
                        using (var ch = _host.Start("Get outputs from executed graph"))
                        {
                            string msg = string.Format("Ignoring empty graph output (output name: {0}, type: {1}, expected output's file: {2})",
                                varName, nameof(idv), path + extension);
                            ch.Warning(msg);
                        }
                    break;
                case TlcModule.DataKind.PredictorModel:
                    var pm = runner.GetOutput<PredictorModel>(varName);
                    SavePredictorModel(pm, path);
                    break;
                case TlcModule.DataKind.TransformModel:
                    var tm = runner.GetOutput<TransformModel>(varName);
                    using (var handle = _host.CreateOutputFile(path))
                    using (var fs = handle.CreateWriteStream())
                        tm.Save(_host, fs);
                    break;
                case TlcModule.DataKind.Array:
                    string partialPath = path.Substring(0, path.Length - extension.Length);

                    var ipmArray = runner.GetOutputOrDefault<PredictorModel[]>(varName);
                    if (ipmArray != null && !ipmArray.GetType().IsValueType)
                    {
                        SaveArrayToFile(ipmArray, partialPath, extension);
                        break;
                    }

                    var idvArray = runner.GetOutputOrDefault<IDataView[]>(varName);
                    if (idvArray != null && !idvArray.GetType().IsValueType)
                    {
                        SaveArrayToFile(idvArray, partialPath, extension);
                        break;
                    }
                    goto default;
                default:
                    throw _host.Except("Port type {0} not supported", kind);
            }

        }

        private void SaveArrayToFile(PredictorModel[] array, string partialPath, string extension)
        {
            for (int i = 0; i < array.Length; i++)
            {
                string path = $"{partialPath}_{i}{extension}";
                SavePredictorModel(array[i], path);
            }
        }

        private void SaveArrayToFile(IDataView[] array, string partialPath, string extension)
        {
            for (int i = 0; i < array.Length; i++)
            {
                string path = $"{partialPath}_{i}{extension}";
                SaveDataView(array[i], path, extension);
            }
        }

        /// <summary>
        /// Saves the PredictorModel to the given path
        /// </summary>
        private void SavePredictorModel(PredictorModel pm, string path)
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
                saver.SaveData(fs, idv, Utils.GetIdentityPermutation(idv.Schema.Count)
                    .Where(x => saver.IsColumnSavable(idv.Schema[x].Type)).ToArray());
            }
        }
    }

}
