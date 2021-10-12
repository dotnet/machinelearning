// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Microsoft.ML;
using Microsoft.ML.Command;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
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

            /// <summary>
            /// Entry point API can save either <see cref="TransformModel"/> or <see cref="PredictorModel"/>.
            /// <see cref="Model"/> is used when the saved model is typed to <see cref="TransformModel"/>.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Model that needs to be converted to ONNX format.", SortOrder = 10)]
            public TransformModel Model;

            /// <summary>
            /// Entry point API can save either <see cref="TransformModel"/> or <see cref="PredictorModel"/>.
            /// <see cref="PredictiveModel"/> is used when the saved model is typed to <see cref="PredictorModel"/>.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Predictor model that needs to be converted to ONNX format.", SortOrder = 12)]
            public PredictorModel PredictiveModel;

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
        private readonly PredictorModel _predictiveModel;
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

            if (args.Model != null && args.PredictiveModel != null)
                throw env.Except(nameof(args.Model) + " and " + nameof(args.PredictiveModel) +
                    " cannot be specified at the same time when calling ONNX converter. Please check the content of " + nameof(args) + ".");

            _model = args.Model;
            _predictiveModel = args.PredictiveModel;
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

            source = trueEnd = (end as LegacyCompositeDataLoader)?.View ?? end;
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
            LinkedList<ITransformCanSaveOnnx> transforms, HashSet<string> inputColumnNamesToDrop = null, HashSet<string> outputColumnNamesToDrop = null)
        {
            inputColumnNamesToDrop = inputColumnNamesToDrop ?? new HashSet<string>();
            outputColumnNamesToDrop = outputColumnNamesToDrop ?? new HashSet<string>();
            HashSet<string> inputColumns = new HashSet<string>();
            // Create graph inputs.
            for (int i = 0; i < inputData.Schema.Count; i++)
            {
                string colName = inputData.Schema[i].Name;
                if (inputColumnNamesToDrop.Contains(colName))
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

                var column = outputData.Schema[i];

                var idataviewColumnName = column.Name;

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
                var trueVariableName = ctx.AddIntermediateVariable(outputData.Schema[i].Type, idataviewColumnName + ".output");
                ctx.CreateNode("Identity", variableName, trueVariableName, ctx.GetNodeName("Identity"), "");
                ctx.AddOutputVariable(outputData.Schema[i].Type, trueVariableName);

                if (column.HasSlotNames())
                    AddSlotNames(ctx, column);
            }

            // Add metadata graph outputs

            return ctx.MakeModel();
        }

        private static void AddSlotNames(OnnxContextImpl ctx, DataViewSchema.Column column)
        {
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            column.GetSlotNames(ref slotNames);
            IEnumerable<string> slotNamesAsStrings = slotNames.DenseValues().Select(name => name.ToString());

            string opType = "LabelEncoder";
            string labelEncoderInputName = $"mlnet.{column.Name}.unusedInput";
            string labelEncoderOutputName = $"mlnet.{column.Name}.unusedOutput";
            string labelEncoderNodeName = $"mlnet.{column.Name}.SlotNames";

            string[] oneVals = new string[] { "one" };
            long[] dims = new long[] { 1, 1 };
            var one = ctx.AddInitializer(oneVals, dims, labelEncoderNodeName);

            var labelEncoderOutput = ctx.AddIntermediateVariable(NumberDataViewType.Int64, labelEncoderOutputName, true);
            var node = ctx.CreateNode(opType, one, labelEncoderOutput, labelEncoderNodeName);
            node.AddAttribute("keys_strings", slotNamesAsStrings);
            node.AddAttribute("values_int64s", Enumerable.Range(0, slotNames.Length).Select(x => (long)x));

            ctx.AddOutputVariable(NumberDataViewType.Int64, labelEncoderOutput);
        }

        // Checks if a column has KeyValues Annotations of any type,
        // So to know if it is safe to use KeyToValue Transformer on it.
        private bool HasKeyValues(DataViewSchema.Column column)
        {
            if (column.Type.GetItemType() is KeyDataViewType keyType)
            {
                var metaColumn = column.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues);
                return metaColumn != null &&
                    metaColumn.Value.Type is VectorDataViewType vectorType &&
                    keyType.Count == (ulong)vectorType.Size;
            }

            return false;
        }

        // Get the names of the KeyDataViewType columns that aren't affected by the pipeline that is being exported to ONNX.
        private HashSet<string> GetPassThroughKeyDataViewTypeColumnsNames(IDataView source, IDataView end)
        {
            var inputKeyDataViewTypeColumnsNames = new HashSet<string>();
            foreach (var col in source.Schema)
                if (col.IsHidden == false && HasKeyValues(col))
                    inputKeyDataViewTypeColumnsNames.Add(col.Name);

            var passThroughColumnNames = new HashSet<string>();
            var outputColumnNames = new HashSet<string>();
            foreach (var col in end.Schema)
            {
                if (outputColumnNames.Contains(col.Name))
                {
                    // "Pass through" column names appear only once in the output schema
                    passThroughColumnNames.Remove(col.Name);
                }
                else
                {
                    // We are only interested in the KeyDataViewType outpus columns
                    if (col.IsHidden == false && HasKeyValues(col))
                        passThroughColumnNames.Add(col.Name);
                }
                outputColumnNames.Add(col.Name);
            }

            // Only count those columns that were in the input of the pipeline
            passThroughColumnNames.IntersectWith(inputKeyDataViewTypeColumnsNames);
            return passThroughColumnNames;
        }

        private void Run(IChannel ch)
        {
            ILegacyDataLoader loader = null;
            IPredictor rawPred = null;
            IDataView view;
            RoleMappedSchema trainSchema = null;

            if (_model == null && _predictiveModel == null)
            {
                if (string.IsNullOrEmpty(ImplOptions.InputModelFile))
                {
                    loader = CreateLoader();
                    rawPred = null;
                    trainSchema = null;
                    Host.CheckUserArg(ImplOptions.LoadPredictor != true, nameof(ImplOptions.LoadPredictor),
                        "Cannot be set to true unless " + nameof(ImplOptions.InputModelFile) + " is also specified.");
                }
                else
                    LoadModelObjects(ch, _loadPredictor, out rawPred, true, out trainSchema, out loader);

                view = loader;
            }
            else if (_model != null)
            {
                view = _model.Apply(Host, new EmptyDataView(Host, _model.InputSchema));
            }
            else
            {
                view = _predictiveModel.TransformModel.Apply(Host, new EmptyDataView(Host, _predictiveModel.TransformModel.InputSchema));
                rawPred = _predictiveModel.Predictor;
                trainSchema = _predictiveModel.GetTrainingSchema(Host);
            }

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

                    if (rawPred.PredictionKind == PredictionKind.BinaryClassification || rawPred.PredictionKind == PredictionKind.MulticlassClassification)
                    {
                        // Check if the PredictedLabel Column is a KeyDataViewType and has KeyValue Annotations.
                        // If it does, add a KeyToValueMappingTransformer, to enable NimbusML to get the values back
                        // when using an ONNX model, as described in https://github.com/dotnet/machinelearning/pull/4841
                        var predictedLabelColumn = scorePipe.Schema.GetColumnOrNull(DefaultColumnNames.PredictedLabel);
                        if (predictedLabelColumn.HasValue && HasKeyValues(predictedLabelColumn.Value))
                        {
                            var outputData = new KeyToValueMappingTransformer(Host, DefaultColumnNames.PredictedLabel).Transform(scorePipe);
                            end = outputData;
                            transforms.AddLast(outputData as ITransformCanSaveOnnx);
                        }
                    }
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

            // Convert back to values the KeyDataViewType "pass-through" columns
            // (i.e those that remained untouched by the model). This is done to enable NimbusML to get these values
            // as described in https://github.com/dotnet/machinelearning/pull/4841

            var passThroughColumnNames = GetPassThroughKeyDataViewTypeColumnsNames(source, end);
            foreach (var name in passThroughColumnNames)
            {
                var outputData = new KeyToValueMappingTransformer(Host, name).Transform(end);
                end = outputData;
                transforms.AddLast(end as ITransformCanSaveOnnx);
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
