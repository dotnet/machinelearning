// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;

[assembly: EntryPointModule(typeof(MacroUtils))]

namespace Microsoft.ML.EntryPoints
{
    internal static class MacroUtils
    {
        /// <summary>
        /// Lists the types of trainer signatures. Used by entry points and autoML system
        /// to know what types of evaluators to use for the train test / pipeline sweeper.
        /// </summary>
        public enum TrainerKinds
        {
            SignatureBinaryClassifierTrainer,
            SignatureMulticlassClassificationTrainer,
            SignatureRankerTrainer,
            SignatureRegressorTrainer,
            SignatureMultiOutputRegressorTrainer,
            SignatureAnomalyDetectorTrainer,
            SignatureClusteringTrainer,
        }

        public sealed class EvaluatorSettings
        {
            public string LabelColumn { get; set; }
            public string NameColumn { get; set; }
            public string WeightColumn { get; set; }
            public string GroupColumn { get; set; }
            public string FeatureColumn { get; set; }

            public EvaluatorSettings()
            {
                LabelColumn = DefaultColumnNames.Label;
            }
        }

        public static EvaluateInputBase GetEvaluatorArgs(TrainerKinds kind, out string entryPointName, EvaluatorSettings settings = null)
        {
            switch (kind)
            {
            case TrainerKinds.SignatureBinaryClassifierTrainer:
                entryPointName = "Models.BinaryClassificationEvaluator";
                return new BinaryClassifierMamlEvaluator.Arguments() { LabelColumn = settings.LabelColumn, WeightColumn = settings.WeightColumn, NameColumn = settings.NameColumn };
            case TrainerKinds.SignatureMulticlassClassificationTrainer:
                entryPointName = "Models.ClassificationEvaluator";
                return new MulticlassClassificationMamlEvaluator.Arguments() { LabelColumn = settings.LabelColumn, WeightColumn = settings.WeightColumn, NameColumn = settings.NameColumn };
            case TrainerKinds.SignatureRankerTrainer:
                entryPointName = "Models.RankingEvaluator";
                return new RankingMamlEvaluator.Arguments() { LabelColumn = settings.LabelColumn, WeightColumn = settings.WeightColumn, NameColumn = settings.NameColumn, GroupIdColumn = settings.GroupColumn };
            case TrainerKinds.SignatureRegressorTrainer:
                entryPointName = "Models.RegressionEvaluator";
                return new RegressionMamlEvaluator.Arguments() { LabelColumn = settings.LabelColumn, WeightColumn = settings.WeightColumn, NameColumn = settings.NameColumn };
            case TrainerKinds.SignatureMultiOutputRegressorTrainer:
                entryPointName = "Models.MultiOutputRegressionEvaluator";
                return new MultiOutputRegressionMamlEvaluator.Arguments() { LabelColumn = settings.LabelColumn, WeightColumn = settings.WeightColumn, NameColumn = settings.NameColumn };
            case TrainerKinds.SignatureAnomalyDetectorTrainer:
                entryPointName = "Models.AnomalyDetectionEvaluator";
                return new AnomalyDetectionMamlEvaluator.Arguments() { LabelColumn = settings.LabelColumn, WeightColumn = settings.WeightColumn, NameColumn = settings.NameColumn };
            case TrainerKinds.SignatureClusteringTrainer:
                entryPointName = "Models.ClusterEvaluator";
                return new ClusteringMamlEvaluator.Arguments() { LabelColumn = settings.LabelColumn, WeightColumn = settings.WeightColumn, NameColumn = settings.NameColumn };
            default:
                throw Contracts.Except("Trainer kind not supported");
            }
        }

        public sealed class ArrayIPredictorModelInput
        {
            [Argument(ArgumentType.Required, HelpText = "The models", SortOrder = 1)]
            public PredictorModel[] Models;
        }

        public sealed class ArrayIPredictorModelOutput
        {
            [TlcModule.Output(Desc = "The model array", SortOrder = 1)]
            public PredictorModel[] OutputModels;
        }

        [TlcModule.EntryPoint(Desc = "Create an array variable of " + nameof(PredictorModel), Name = "Data.PredictorModelArrayConverter")]
        public static ArrayIPredictorModelOutput MakeArray(IHostEnvironment env, ArrayIPredictorModelInput input)
        {
            var result = new ArrayIPredictorModelOutput
            {
                OutputModels = input.Models
            };
            return result;
        }

        public sealed class ArrayIDataViewInput
        {
            [Argument(ArgumentType.Required, HelpText = "The data sets", SortOrder = 1)]
            public IDataView[] Data;
        }

        public sealed class ArrayIDataViewOutput
        {
            [TlcModule.Output(Desc = "The data set array", SortOrder = 1)]
            public IDataView[] OutputData;
        }

        [TlcModule.EntryPoint(Desc = "Create an array variable of IDataView", Name = "Data.IDataViewArrayConverter")]
        public static ArrayIDataViewOutput MakeArray(IHostEnvironment env, ArrayIDataViewInput input)
        {
            var result = new ArrayIDataViewOutput
            {
                OutputData = input.Data
            };
            return result;
        }

        internal static void ConvertIPredictorModelsToArray(IHostEnvironment env, RunContext context, List<EntryPointNode> subGraphNodes,
            Var<PredictorModel>[] predModelVars, string outputVarName)
        {
            var predictorArrayConverterArgs = new ArrayIPredictorModelInput();
            var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            var inputMap = new Dictionary<ParameterBinding, VariableBinding>();

            var argName = nameof(predictorArrayConverterArgs.Models);
            inputBindingMap.Add(argName, new List<ParameterBinding>());
            for (int i = 0; i < predModelVars.Length; i++)
            {
                var paramBinding = new ArrayIndexParameterBinding(argName, i);
                inputBindingMap[argName].Add(paramBinding);
                inputMap[paramBinding] = new SimpleVariableBinding(predModelVars[i].VarName);
            }
            var outputMap = new Dictionary<string, string>();
            var output = new ArrayVar<PredictorModel>();
            outputMap.Add(nameof(MacroUtils.ArrayIPredictorModelOutput.OutputModels), outputVarName);
            var arrayConvertNode = EntryPointNode.Create(env, "Data.PredictorModelArrayConverter", predictorArrayConverterArgs,
                context, inputBindingMap, inputMap, outputMap);
            subGraphNodes.Add(arrayConvertNode);
        }

        internal static void ConvertIdataViewsToArray(IHostEnvironment env, RunContext context, List<EntryPointNode> subGraphNodes,
            Var<IDataView>[] vars, string outputVarName)
        {
            var dataviewArrayConverterArgs = new ArrayIDataViewInput();
            var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            var inputMap = new Dictionary<ParameterBinding, VariableBinding>();

            var argName = nameof(dataviewArrayConverterArgs.Data);
            inputBindingMap.Add(argName, new List<ParameterBinding>());
            for (int i = 0; i < vars.Length; i++)
            {
                var paramBinding = new ArrayIndexParameterBinding(argName, i);
                inputBindingMap[argName].Add(paramBinding);
                inputMap[paramBinding] = new SimpleVariableBinding(vars[i].VarName);
            }
            var outputMap = new Dictionary<string, string>();
            outputMap.Add(nameof(ArrayIDataViewOutput.OutputData), outputVarName);
            var arrayConvertNode = EntryPointNode.Create(env, "Data.IDataViewArrayConverter", dataviewArrayConverterArgs,
                context, inputBindingMap, inputMap, outputMap);
            subGraphNodes.Add(arrayConvertNode);
        }
    }
}
