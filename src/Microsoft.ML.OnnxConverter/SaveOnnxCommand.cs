// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using Google.Protobuf;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Newtonsoft.Json;
using static Microsoft.ML.Model.OnnxConverter.OnnxCSharpToProtoWrapper;

[assembly: LoadableClass(SaveOnnxCommand.Summary, typeof(SaveOnnxCommand), typeof(SaveOnnxCommand.Arguments), typeof(SignatureCommand),
    "Save ONNX", "SaveOnnx", DocName = "command/SaveOnnx.md")]

[assembly: LoadableClass(typeof(void), typeof(SaveOnnxCommand), null, typeof(SignatureEntryPointModule), "SaveOnnx")]

namespace Microsoft.ML.Model.OnnxConverter
{
    internal sealed class SaveOnnxCommand : DataCommand.ImplBase<SaveOnnxCommand.Arguments>
    {
        public const string Summary = "Given a data model, write out the corresponding ONNX.";
        public const string LoadName = "SaveOnnx";

        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.Required, HelpText = "The path to write the output ONNX to.", SortOrder = 1)]
            public string Onnx;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output JSON to.", SortOrder = 2)]
            public string Json;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'name' property in the output ONNX. By default this will be the ONNX extension-less name.", NullName = "<Auto>", SortOrder = 3)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'domain' property in the output ONNX.", NullName = "<Auto>", SortOrder = 4)]
            public string Domain;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Comma delimited list of input column names to drop", ShortName = "idrop", SortOrder = 5)]
            public string InputsToDrop;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Array of input column names to drop", Name = nameof(InputsToDrop), SortOrder = 6)]
            public string[] InputsToDropArray;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Comma delimited list of output column names to drop", ShortName = "odrop", SortOrder = 7)]
            public string OutputsToDrop;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Array of output column names to drop", Name = nameof(OutputsToDrop), SortOrder = 8)]
            public string[] OutputsToDropArray;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Whether we should attempt to load the predictor and attach the scorer to the pipeline if one is present.", ShortName = "pred", SortOrder = 9)]
            public bool? LoadPredictor;

            [Argument(ArgumentType.Required, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Model that needs to be converted to ONNX format.", SortOrder = 10)]
            public TransformModel Model;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The targeted ONNX version. It can be either \"Stable\" or \"Experimental\". If \"Experimental\" is used, produced model can contain components that is not officially supported in ONNX standard.", SortOrder = 11)]
            public OnnxVersion OnnxVersion;
        }

        private readonly string _outputModelPath;
        private readonly string _outputJsonModelPath;
        private readonly string _name;
        private readonly string _domain;
        private readonly bool? _loadPredictor;
        private readonly HashSet<string> _inputsToDrop;
        private readonly HashSet<string> _outputsToDrop;
        private readonly TransformModel _model;
        private const string ProducerName = "ML.NET";
        private const long ModelVersion = 0;

        public SaveOnnxCommand(IHostEnvironment env, Arguments args)
                : base(env, args, LoadName)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckNonWhiteSpace(args.Onnx, nameof(args.Onnx));

            Utils.CheckOptionalUserDirectory(args.Onnx, nameof(args.Onnx));
            _outputModelPath = args.Onnx;
            _outputJsonModelPath = string.IsNullOrWhiteSpace(args.Json) ? null : args.Json;
            if (args.Name == null)
                _name = Path.GetFileNameWithoutExtension(_outputModelPath);
            else
            {
                Host.CheckNonWhiteSpace(args.Name, nameof(args.Name));
                _name = args.Name;
            }

            _loadPredictor = args.LoadPredictor;
            _inputsToDrop = CreateDropMap(args.InputsToDropArray ?? args.InputsToDrop?.Split(','));
            _outputsToDrop = CreateDropMap(args.OutputsToDropArray ?? args.OutputsToDrop?.Split(','));
            _domain = args.Domain;
            _model = args.Model;
        }

        private static HashSet<string> CreateDropMap(string[] toDrop)
        {
            if (toDrop == null)
                return new HashSet<string>();

            return new HashSet<string>(toDrop);
        }

        public override void Run()
        {
            using (var ch = Host.Start("Run"))
            {
                Run(ch);
            }
        }

        internal static void GetPipe(OnnxContextImpl ctx, IChannel ch, IDataView end, out IDataView source, out IDataView trueEnd, out LinkedList<ITransformCanSaveOnnx> transforms)
        {
            ch.AssertValue(end);

            source = trueEnd = (end as CompositeDataLoader)?.View ?? end;
            IDataTransform transform = source as IDataTransform;
            transforms = new LinkedList<ITransformCanSaveOnnx>();
            while (transform != null)
            {
                ITransformCanSaveOnnx onnxTransform = transform as ITransformCanSaveOnnx;
                if (onnxTransform == null || !onnxTransform.CanSaveOnnx(ctx))
                {
                    ch.Warning("Had to stop walkback of pipeline at {0} since it cannot save itself as ONNX.", transform.GetType().Name);
                    while (source as IDataTransform != null)
                        source = (source as IDataTransform).Source;

                    return;
                }
                transforms.AddFirst(onnxTransform);
                transform = (source = transform.Source) as IDataTransform;
            }

            ch.AssertValue(source);
        }

        internal static ModelProto ConvertTransformListToOnnxModel(OnnxContextImpl ctx, IChannel ch, IDataView inputData, IDataView outputData,
            LinkedList<ITransformCanSaveOnnx> transforms, HashSet<string> inputColumnNamesToDrop=null, HashSet<string> outputColumnNamesToDrop=null)
        {
            inputColumnNamesToDrop = inputColumnNamesToDrop ?? new HashSet<string>();
            outputColumnNamesToDrop = outputColumnNamesToDrop ?? new HashSet<string>();
            HashSet<string> inputColumns = new HashSet<string>();
            // Create graph inputs.
            for (int i = 0; i < inputData.Schema.Count; i++)
            {
                string colName = inputData.Schema[i].Name;
                if(inputColumnNamesToDrop.Contains(colName))
                    continue;

                ctx.AddInputVariable(inputData.Schema[i].Type, colName);
                inputColumns.Add(colName);
            }

            // Create graph nodes, outputs and intermediate values.
            foreach (var trans in transforms)
            {
                ch.Assert(trans.CanSaveOnnx(ctx));
                trans.SaveAsOnnx(ctx);
            }

            // Add graph outputs.
            for (int i = 0; i < outputData.Schema.Count; ++i)
            {
                if (outputData.Schema[i].IsHidden)
                    continue;

                var idataviewColumnName = outputData.Schema[i].Name;

                // Since the last IDataView also contains columns of the initial IDataView, last IDataView's columns found in
                // _inputToDrop should be removed too.
                if (inputColumnNamesToDrop.Contains(idataviewColumnName) || outputColumnNamesToDrop.Contains(idataviewColumnName))
                    continue;

                var variableName = ctx.TryGetVariableName(idataviewColumnName);
                // Null variable name occurs when an unsupported transform produces an output and a downsteam step consumes that output.
                // or user accidently removes a transform whose output is used by other transforms.
                ch.Check(variableName != null, "The targeted pipeline can not be fully converted into a well-defined ONNX model. " +
                    "Please check if all steps in that pipeline are convertible to ONNX " +
                    "and all necessary variables are not dropped (via command line arguments).");
                var trueVariableName = ctx.AddIntermediateVariable(null, idataviewColumnName, true);
                ctx.CreateNode("Identity", variableName, trueVariableName, ctx.GetNodeName("Identity"), "");
                ctx.AddOutputVariable(outputData.Schema[i].Type, trueVariableName);
            }

            return ctx.MakeModel();
        }

        private void Run(IChannel ch)
        {
            IDataLoader loader = null;
            IPredictor rawPred = null;
            IDataView view;
            RoleMappedSchema trainSchema = null;

            if (_model == null)
            {
                if (string.IsNullOrEmpty(ImplOptions.InputModelFile))
                {
                    loader = CreateLoader();
                    rawPred = null;
                    trainSchema = null;
                    Host.CheckUserArg(ImplOptions.LoadPredictor != true, nameof(ImplOptions.LoadPredictor),
                        "Cannot be set to true unless " + nameof(ImplOptions.InputModelFile) + " is also specifified.");
                }
                else
                    LoadModelObjects(ch, _loadPredictor, out rawPred, true, out trainSchema, out loader);

                view = loader;
            }
            else
                view = _model.Apply(Host, new EmptyDataView(Host, _model.InputSchema));

            // Create the ONNX context for storing global information
            var assembly = System.Reflection.Assembly.GetExecutingAssembly();
            var versionInfo = System.Diagnostics.FileVersionInfo.GetVersionInfo(assembly.Location);
            var ctx = new OnnxContextImpl(Host, _name, ProducerName, versionInfo.FileVersion,
                ModelVersion, _domain, ImplOptions.OnnxVersion);

            // Get the transform chain.
            IDataView source;
            IDataView end;
            LinkedList<ITransformCanSaveOnnx> transforms;
            GetPipe(ctx, ch, view, out source, out end, out transforms);
            Host.Assert(transforms.Count == 0 || transforms.Last.Value == end);

            // If we have a predictor, try to get the scorer for it.
            if (rawPred != null)
            {
                RoleMappedData data;
                if (trainSchema != null)
                    data = new RoleMappedData(end, trainSchema.GetColumnRoleNames());
                else
                {
                    // We had a predictor, but no roles stored in the model. Just suppose
                    // default column names are OK, if present.
                    data = new RoleMappedData(end, DefaultColumnNames.Label,
                        DefaultColumnNames.Features, DefaultColumnNames.GroupId, DefaultColumnNames.Weight, DefaultColumnNames.Name, opt: true);
                }

                var scorePipe = ScoreUtils.GetScorer(rawPred, data, Host, trainSchema);
                var scoreOnnx = scorePipe as ITransformCanSaveOnnx;
                if (scoreOnnx?.CanSaveOnnx(ctx) == true)
                {
                    Host.Assert(scorePipe.Source == end);
                    end = scorePipe;
                    transforms.AddLast(scoreOnnx);
                }
                else
                {
                    Contracts.CheckUserArg(_loadPredictor != true,
                        nameof(Arguments.LoadPredictor), "We were explicitly told to load the predictor but we do not know how to save it as ONNX.");
                    ch.Warning("We do not know how to save the predictor as ONNX. Ignoring.");
                }
            }
            else
            {
                Contracts.CheckUserArg(_loadPredictor != true,
                    nameof(Arguments.LoadPredictor), "We were explicitly told to load the predictor but one was not present.");
            }

            var model = ConvertTransformListToOnnxModel(ctx, ch, source, end, transforms, _inputsToDrop, _outputsToDrop);

            using (var file = Host.CreateOutputFile(_outputModelPath))
            using (var stream = file.CreateWriteStream())
                model.WriteTo(stream);

            if (_outputJsonModelPath != null)
            {
                using (var file = Host.CreateOutputFile(_outputJsonModelPath))
                using (var stream = file.CreateWriteStream())
                using (var writer = new StreamWriter(stream))
                {
                    var parsedJson = JsonConvert.DeserializeObject(model.ToString());
                    writer.Write(JsonConvert.SerializeObject(parsedJson, Formatting.Indented));
                }
            }

            if (!string.IsNullOrWhiteSpace(ImplOptions.OutputModelFile))
            {
                Contracts.Assert(loader != null);

                ch.Trace("Saving the data pipe");
                // Should probably include "end"?
                SaveLoader(loader, ImplOptions.OutputModelFile);
            }
        }

        public sealed class Output
        {
        }

        [TlcModule.EntryPoint(Name = "Models.OnnxConverter", Desc = "Converts the model to ONNX format.", UserName = "ONNX Converter.")]
        public static Output Apply(IHostEnvironment env, Arguments input)
        {
            new SaveOnnxCommand(env, input).Run();
            return new Output();
        }

    }
}
