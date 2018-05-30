// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using Google.Protobuf;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.UniversalModelFormat.Onnx;
using Newtonsoft.Json;

[assembly: LoadableClass(SaveOnnxCommand.Summary, typeof(SaveOnnxCommand), typeof(SaveOnnxCommand.Arguments), typeof(SignatureCommand),
    "Save ONNX", "SaveOnnx", DocName = "command/SaveOnnx.md")]

[assembly: LoadableClass(typeof(void), typeof(SaveOnnxCommand), null, typeof(SignatureEntryPointModule), "SaveOnnxCommand")]

namespace Microsoft.ML.Runtime.Model.Onnx
{
    public sealed class SaveOnnxCommand : DataCommand.ImplBase<SaveOnnxCommand.Arguments>
    {
        public const string Summary = "Given a data model, write out the corresponding ONNX.";
        public const string LoadName = "SaveOnnx";

        public sealed class Arguments : DataCommand.ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output ONNX to.", SortOrder = 1)]
            public string Onnx;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output JSON to.", SortOrder = 2)]
            public string Json;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'name' property in the output ONNX. By default this will be the ONNX extension-less name.", NullName = "<Auto>", SortOrder = 3)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'domain' property in the output ONNX.", NullName = "<Auto>", SortOrder = 4)]
            public string Domain;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma delimited list of input column names to drop", ShortName = "idrop", SortOrder = 5)]
            public string InputsToDrop;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Array of input column names to drop", SortOrder = 6)]
            public string[] InputsToDropArray;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma delimited list of output column names to drop", ShortName = "odrop", SortOrder = 7)]
            public string OutputsToDrop;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Array of output column names to drop", SortOrder = 8)]
            public string[] OutputsToDropArray;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether we should attempt to load the predictor and attach the scorer to the pipeline if one is present.", ShortName = "pred", SortOrder = 9)]
            public bool? LoadPredictor;

            [Argument(ArgumentType.Required, HelpText = "Model that needs to be converted to ONNX format.", SortOrder = 10)]

            public IPredictorModel Model;
        }

        private readonly string _outputModelPath;
        private readonly string _outputJsonModelPath;
        private readonly string _name;
        private readonly string _domain;
        private readonly bool? _loadPredictor;
        private readonly HashSet<string> _inputsToDrop;
        private readonly HashSet<string> _outputsToDrop;
        private readonly IPredictorModel _model;

        public SaveOnnxCommand(IHostEnvironment env, Arguments args)
                : base(env, args, LoadName)
        {
            Host.CheckValue(args, nameof(args));
            Utils.CheckOptionalUserDirectory(args.Onnx, nameof(args.Onnx));
            _outputModelPath = string.IsNullOrWhiteSpace(args.Onnx) ? null : args.Onnx;
            _outputJsonModelPath = string.IsNullOrWhiteSpace(args.Json) ? null : args.Json;
            if (args.Name == null && _outputModelPath != null)
                _name = Path.GetFileNameWithoutExtension(_outputModelPath);
            else if (!string.IsNullOrWhiteSpace(args.Name))
                _name = args.Name;

            _loadPredictor = args.LoadPredictor;
            _inputsToDrop = args.InputsToDropArray != null ? CreateDropMap(args.InputsToDropArray) : CreateDropMap(args.InputsToDrop);
            _outputsToDrop = args.OutputsToDropArray != null ? CreateDropMap(args.OutputsToDropArray) : CreateDropMap(args.OutputsToDrop);
            _domain = args.Domain;
            _model = args.Model;
        }

        private static HashSet<string> CreateDropMap(string toDrop)
        {
            if (string.IsNullOrWhiteSpace(toDrop))
                return new HashSet<string>();

            return new HashSet<string>(toDrop.Split(','));
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
                ch.Done();
            }
        }

        private void GetPipe(IChannel ch, IDataView end, out IDataView source, out IDataView trueEnd, out LinkedList<ITransformCanSaveOnnx> transforms)
        {
            Host.AssertValue(end);
            source = trueEnd = (end as CompositeDataLoader)?.View ?? end;
            IDataTransform transform = source as IDataTransform;
            transforms = new LinkedList<ITransformCanSaveOnnx>();
            while (transform != null)
            {
                ITransformCanSaveOnnx onnxTransform = transform as ITransformCanSaveOnnx;
                if (onnxTransform == null || !onnxTransform.CanSaveOnnx)
                {
                    ch.Warning("Had to stop walkback of pipeline at {0} since it cannot save itself as ONNX.", transform.GetType().Name);
                    while (source as IDataTransform != null)
                        source = (source as IDataTransform).Source;

                    return;
                }
                transforms.AddFirst(onnxTransform);
                transform = (source = transform.Source) as IDataTransform;
            }

            Host.AssertValue(source);
        }

        private void Run(IChannel ch)
        {
            IDataLoader loader = null; ;
            IPredictor rawPred;
            IDataView view;
            RoleMappedSchema trainSchema = null;

            if (_model == null)
            {
                if (string.IsNullOrEmpty(Args.InputModelFile))
                {
                    loader = CreateLoader();
                    rawPred = null;
                    trainSchema = null;
                    Host.CheckUserArg(Args.LoadPredictor != true, nameof(Args.LoadPredictor),
                        "Cannot be set to true unless " + nameof(Args.InputModelFile) + " is also specifified.");
                }
                else
                    LoadModelObjects(ch, _loadPredictor, out rawPred, true, out trainSchema, out loader);

                view = loader;
            }
            else
            {
                view = _model.TransformModel.View;
                rawPred = _model?.Predictor;
                if (rawPred != null)
                    trainSchema = _model.GetTrainingSchema(Host);
            }

            // Get the transform chain.
            IDataView source;
            IDataView end;
            LinkedList<ITransformCanSaveOnnx> transforms;
            GetPipe(ch, view, out source, out end, out transforms);
            Host.Assert(transforms.Count == 0 || transforms.Last.Value == end);

            var ctx = new OnnxContext(Host, _name, _domain);
            // If we have a predictor, try to get the scorer for it.
            if (rawPred != null)
            {
                RoleMappedData data;
                if (trainSchema != null)
                    data = RoleMappedData.Create(end, trainSchema.GetColumnRoleNames());
                else
                {
                    // We had a predictor, but no roles stored in the model. Just suppose
                    // default column names are OK, if present.
                    data = TrainUtils.CreateExamplesOpt(end, DefaultColumnNames.Label,
                        DefaultColumnNames.Features, DefaultColumnNames.GroupId, DefaultColumnNames.Weight, DefaultColumnNames.Name);
                }

                var scorePipe = ScoreUtils.GetScorer(rawPred, data, Host, trainSchema);
                var scoreOnnx = scorePipe as ITransformCanSaveOnnx;
                if (scoreOnnx?.CanSaveOnnx == true)
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

            HashSet<string> inputColumns = new HashSet<string>();
            //Create graph inputs.
            for (int i = 0; i < source.Schema.ColumnCount; i++)
            {
                string colName = source.Schema.GetColumnName(i);
                if(_inputsToDrop.Contains(colName))
                    continue;

                ctx.AddInputVariable(source.Schema.GetColumnType(i), colName);
                inputColumns.Add(colName);
            }

            //Create graph nodes, outputs and intermediate values.
            foreach (var trans in transforms)
            {
                Host.Assert(trans.CanSaveOnnx);
                trans.SaveAsOnnx(ctx);
            }

            //Add graph outputs.
            for (int i = 0; i < end.Schema.ColumnCount; ++i)
            {
                if (end.Schema.IsHidden(i))
                    continue;

                var idataviewColumnName = end.Schema.GetColumnName(i);;
                if (_outputsToDrop.Contains(idataviewColumnName) || _inputsToDrop.Contains(idataviewColumnName))
                    continue;

                var variableName = ctx.TryGetVariableName(idataviewColumnName);
                if (variableName != null)
                    ctx.AddOutputVariable(end.Schema.GetColumnType(i), variableName);
            }

            var model = ctx.MakeModel();
            if (_outputModelPath != null)
            {
                using (var file = Host.CreateOutputFile(_outputModelPath))
                using (var stream = file.CreateWriteStream())
                    model.WriteTo(stream);
            }

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

            if (!string.IsNullOrWhiteSpace(Args.OutputModelFile))
            {
                Contracts.Assert(loader != null);

                ch.Trace("Saving the data pipe");
                // Should probably include "end"?
                SaveLoader(loader, Args.OutputModelFile);
            }
        }

        public sealed class Output
        {
            //REVIEW: Would be nice to include ONNX protobuf model here but code generator needs an upgrade.
        }

        //REVIEW: Ideally there is no need to define this input class and just reuse the Argument class from SaveONNX command
        //but the code generator cannot parse certain complicated data types in the base class that Argument class extends.
        //We should fix the code generator and use the Argument class.
        public sealed class Input
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output ONNX to.", SortOrder = 1)]
            public string Onnx;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The path to write the output JSON to.", SortOrder = 2)]
            public string Json;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'name' property in the output ONNX. By default this will be the ONNX extension-less name.", NullName = "<Auto>", SortOrder = 3)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The 'domain' property in the output ONNX.", NullName = "<Auto>", SortOrder = 4)]
            public string Domain;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Array of input column names to drop", SortOrder = 5)]
            public string[] InputsToDrop;
      
            [Argument(ArgumentType.AtMostOnce, HelpText = "Array of output column names to drop", SortOrder = 6)]
            public string[] OutputsToDrop;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether we should attempt to load the predictor and attach the scorer to the pipeline if one is present.", ShortName = "pred", SortOrder = 7)]
            public bool? LoadPredictor;

            [Argument(ArgumentType.Required, HelpText = "Model that needs to be converted to ONNX format.", SortOrder = 8)]

            public IPredictorModel Model;
        }


        [TlcModule.EntryPoint(Name = "Models.OnnxConverter", Desc = "Converts the model to ONNX format.", UserName = "ONNX Converter.")]
        public static Output Apply(IHostEnvironment env, Input input)
        {
            Arguments args = new Arguments();
            args.Onnx = input.Onnx;
            args.Json = input.Json;
            args.Name = input.Name;
            args.Domain = input.Domain;
            args.InputsToDropArray = input.InputsToDrop;
            args.OutputsToDropArray = input.OutputsToDrop;
            args.LoadPredictor = input.LoadPredictor;
            args.Model = input.Model;

            var cmd = new SaveOnnxCommand(env, args);
            cmd.Run();
            return new Output();
        }

    }
}
