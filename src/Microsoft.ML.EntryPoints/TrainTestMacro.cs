// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(void), typeof(TrainTestMacro), null, typeof(SignatureEntryPointModule), "TrainTestMacro")]

namespace Microsoft.ML.EntryPoints
{
    internal static class TrainTestMacro
    {
        public sealed class SubGraphInput
        {
            [Argument(ArgumentType.Required, HelpText = "The data to be used for training", SortOrder = 1)]
            public Var<IDataView> Data;
        }

        public sealed class SubGraphOutput
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The predictor model", SortOrder = 1)]
            public Var<PredictorModel> PredictorModel;
        }

        public sealed class Arguments
        {
            [TlcModule.OptionalInput]
            [Argument(ArgumentType.Required, ShortName = "train", HelpText = "The data to be used for training", SortOrder = 1)]
            public IDataView TrainingData;

            [TlcModule.OptionalInput]
            [Argument(ArgumentType.Required, ShortName = "test", HelpText = "The data to be used for testing", SortOrder = 2)]
            public IDataView TestingData;

            [TlcModule.OptionalInput]
            [Argument(ArgumentType.AtMostOnce, HelpText = "The aggregated transform model from the pipeline before this command, to apply to the test data, and also include in the final model, together with the predictor model.", SortOrder = 3)]
            public Var<TransformModel> TransformModel = null;

            [Argument(ArgumentType.Required, HelpText = "The training subgraph", SortOrder = 4)]
            public JArray Nodes;

            [Argument(ArgumentType.Required, HelpText = "The training subgraph inputs", SortOrder = 5)]
            public SubGraphInput Inputs = new SubGraphInput();

            [Argument(ArgumentType.Required, HelpText = "The training subgraph outputs", SortOrder = 6)]
            public SubGraphOutput Outputs = new SubGraphOutput();

            [Argument(ArgumentType.AtMostOnce, HelpText = "Specifies the trainer kind, which determines the evaluator to be used.", SortOrder = 7)]
            public MacroUtils.TrainerKinds Kind = MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Identifies which pipeline was run for this train test.", SortOrder = 8)]
            public string PipelineId;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Indicates whether to include and output training dataset metrics.", SortOrder = 9)]
            public Boolean IncludeTrainingMetrics = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for labels", ShortName = "lab", SortOrder = 10)]
            public string LabelColumn = DefaultColumnNames.Label;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for example weight", ShortName = "weight", SortOrder = 11)]
            public Optional<string> WeightColumn = Optional<string>.Implicit(DefaultColumnNames.Weight);

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for grouping", ShortName = "group", SortOrder = 12)]
            public Optional<string> GroupColumn = Optional<string>.Implicit(DefaultColumnNames.GroupId);

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name column name", ShortName = "name", SortOrder = 13)]
            public Optional<string> NameColumn = Optional<string>.Implicit(DefaultColumnNames.Name);
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "The final model including the trained predictor model and the model from the transforms, " +
                "provided as the Input.TransformModel.", SortOrder = 1)]
            public PredictorModel PredictorModel;

            [TlcModule.Output(Desc = "Warning dataset", SortOrder = 3)]
            public IDataView Warnings;

            [TlcModule.Output(Desc = "Overall metrics dataset", SortOrder = 4)]
            public IDataView OverallMetrics;

            [TlcModule.Output(Desc = "Per instance metrics dataset", SortOrder = 5)]
            public IDataView PerInstanceMetrics;

            [TlcModule.Output(Desc = "Confusion matrix dataset", SortOrder = 6)]
            public IDataView ConfusionMatrix;

            [TlcModule.Output(Desc = "Warning dataset for training", SortOrder = 7)]
            public IDataView TrainingWarnings;

            [TlcModule.Output(Desc = "Overall metrics dataset for training", SortOrder = 8)]
            public IDataView TrainingOverallMetrics;

            [TlcModule.Output(Desc = "Per instance metrics dataset for training", SortOrder = 9)]
            public IDataView TrainingPerInstanceMetrics;

            [TlcModule.Output(Desc = "Confusion matrix dataset for training", SortOrder = 10)]
            public IDataView TrainingConfusionMatrix;
        }

        [TlcModule.EntryPoint(Desc = "General train test for any supported evaluator", Name = "Models.TrainTestEvaluator")]
        public static CommonOutputs.MacroOutput<Output> TrainTest(
            IHostEnvironment env,
            Arguments input,
            EntryPointNode node)
        {
            // Create default pipeline ID if one not given.
            input.PipelineId = input.PipelineId ?? Guid.NewGuid().ToString("N");

            // Parse the subgraph.
            var subGraphRunContext = new RunContext(env);
            var subGraphNodes = EntryPointNode.ValidateNodes(env, subGraphRunContext, input.Nodes, label: input.LabelColumn,
                input.GroupColumn.IsExplicit ? input.GroupColumn.Value : null,
                input.WeightColumn.IsExplicit ? input.WeightColumn.Value : null,
                input.NameColumn.IsExplicit ? input.NameColumn.Value : null);

            // Change the subgraph to use the training data as input.
            var varName = input.Inputs.Data.VarName;
            VariableBinding transformModelVarName = null;
            if (input.TransformModel != null)
                transformModelVarName = node.GetInputVariable(nameof(input.TransformModel));

            if (!subGraphRunContext.TryGetVariable(varName, out var dataVariable))
                throw env.Except($"Invalid variable name '{varName}'.");
            var trainingVar = node.GetInputVariable(nameof(input.TrainingData));
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.RenameInputVariable(dataVariable.Name, trainingVar);
            subGraphRunContext.RemoveVariable(dataVariable);

            // Change the subgraph to use the model variable as output.
            varName = input.Outputs.PredictorModel.VarName;
            if (!subGraphRunContext.TryGetVariable(varName, out dataVariable))
                throw env.Except($"Invalid variable name '{varName}'.");

            string predictorModelVarName = node.GetOutputVariableName(nameof(Output.PredictorModel));

            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.RenameOutputVariable(dataVariable.Name, predictorModelVarName);
            subGraphRunContext.RemoveVariable(dataVariable);

            // Move the variables from the subcontext to the main context.
            node.Context.AddContextVariables(subGraphRunContext);

            // Change all the subgraph nodes to use the main context.
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.SetContext(node.Context);

            // Testing using test data set
            var testingVar = node.GetInputVariable(nameof(input.TestingData));
            //var exp = new Experiment(env);

            Dictionary<string, List<ParameterBinding>> inputBindingMap;
            Dictionary<ParameterBinding, VariableBinding> inputMap;
            ParameterBinding paramBinding;
            Dictionary<string, string> outputMap;

            //combine the predictor model with any potential transfrom model passed from the outer graph
            if (transformModelVarName != null && transformModelVarName.VariableName != null)
            {
                var combineArgs = new ModelOperations.SimplePredictorModelInput();
                inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
                inputMap = new Dictionary<ParameterBinding, VariableBinding>();

                var inputTransformModel = new SimpleVariableBinding(transformModelVarName.VariableName);
                var inputPredictorModel = new SimpleVariableBinding(predictorModelVarName);
                paramBinding = new SimpleParameterBinding(nameof(combineArgs.TransformModel));
                inputBindingMap.Add(nameof(combineArgs.TransformModel), new List<ParameterBinding>() { paramBinding });
                inputMap.Add(paramBinding, inputTransformModel);
                paramBinding = new SimpleParameterBinding(nameof(combineArgs.PredictorModel));
                inputBindingMap.Add(nameof(combineArgs.PredictorModel), new List<ParameterBinding>() { paramBinding });
                inputMap.Add(paramBinding, inputPredictorModel);
                outputMap = new Dictionary<string, string>();

                var combineNodeOutputPredictorModel = new Var<PredictorModel>();
                predictorModelVarName = combineNodeOutputPredictorModel.VarName;
                outputMap.Add(nameof(ModelOperations.PredictorModelOutput.PredictorModel), combineNodeOutputPredictorModel.VarName);
                EntryPointNode combineNode = EntryPointNode.Create(env, "Transforms.TwoHeterogeneousModelCombiner", combineArgs,
                    node.Context, inputBindingMap, inputMap, outputMap);
                subGraphNodes.Add(combineNode);
            }

            // Add the scoring node for testing.
            var args = new ScoreModel.Input();
            inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            inputMap = new Dictionary<ParameterBinding, VariableBinding>();
            paramBinding = new SimpleParameterBinding(nameof(args.Data));
            inputBindingMap.Add(nameof(args.Data), new List<ParameterBinding>() { paramBinding });
            inputMap.Add(paramBinding, testingVar);
            var scoreNodeInputPredictorModel = new SimpleVariableBinding(predictorModelVarName);
            paramBinding = new SimpleParameterBinding(nameof(args.PredictorModel));
            inputBindingMap.Add(nameof(args.PredictorModel), new List<ParameterBinding>() { paramBinding });
            inputMap.Add(paramBinding, scoreNodeInputPredictorModel);

            var scoreNodeOutputScoredData = new Var<IDataView>();
            var scoreNodeOutputScoringTransform = new Var<TransformModel>();
            outputMap = new Dictionary<string, string>();
            outputMap.Add(nameof(ScoreModel.Output.ScoredData), scoreNodeOutputScoredData.VarName);
            outputMap.Add(nameof(ScoreModel.Output.ScoringTransform), scoreNodeOutputScoringTransform.VarName);

            EntryPointNode scoreNode = EntryPointNode.Create(env, "Transforms.DatasetScorer", args,
                node.Context, inputBindingMap, inputMap, outputMap);
            subGraphNodes.Add(scoreNode);
            var evalDataVarName = scoreNodeOutputScoredData.VarName;

            // REVIEW: add similar support for FeatureColumnName.
            var settings = new MacroUtils.EvaluatorSettings
            {
                LabelColumn = input.LabelColumn,
                WeightColumn = input.WeightColumn.IsExplicit ? input.WeightColumn.Value : null,
                GroupColumn = input.GroupColumn.IsExplicit ? input.GroupColumn.Value : null,
                NameColumn = input.NameColumn.IsExplicit ? input.NameColumn.Value : null
            };

            if (input.IncludeTrainingMetrics)
            {
                string evalTrainingDataVarName;
                args = new ScoreModel.Input();
                inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
                inputMap = new Dictionary<ParameterBinding, VariableBinding>();
                paramBinding = new SimpleParameterBinding(nameof(args.Data));
                inputBindingMap.Add(nameof(args.Data), new List<ParameterBinding>() { paramBinding });
                inputMap.Add(paramBinding, trainingVar);
                scoreNodeInputPredictorModel = new SimpleVariableBinding(predictorModelVarName);
                paramBinding = new SimpleParameterBinding(nameof(args.PredictorModel));
                inputBindingMap.Add(nameof(args.PredictorModel), new List<ParameterBinding>() { paramBinding });
                inputMap.Add(paramBinding, scoreNodeInputPredictorModel);

                scoreNodeOutputScoredData = new Var<IDataView>();
                scoreNodeOutputScoringTransform = new Var<TransformModel>();
                outputMap = new Dictionary<string, string>();
                outputMap.Add(nameof(ScoreModel.Output.ScoredData), scoreNodeOutputScoredData.VarName);
                outputMap.Add(nameof(ScoreModel.Output.ScoringTransform), scoreNodeOutputScoringTransform.VarName);

                scoreNode = EntryPointNode.Create(env, "Transforms.DatasetScorer", args,
                    node.Context, inputBindingMap, inputMap, outputMap);
                subGraphNodes.Add(scoreNode);
                evalTrainingDataVarName = scoreNodeOutputScoredData.VarName;

                // Add the evaluator node for training.
                var evalTrainingArgs = MacroUtils.GetEvaluatorArgs(input.Kind, out var evalTrainingEntryPointName, settings);
                inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
                inputMap = new Dictionary<ParameterBinding, VariableBinding>();
                var evalTrainingNodeInputData = new SimpleVariableBinding(evalTrainingDataVarName);
                paramBinding = new SimpleParameterBinding(nameof(evalTrainingArgs.Data));
                inputBindingMap.Add(nameof(evalTrainingArgs.Data), new List<ParameterBinding>() { paramBinding });
                inputMap.Add(paramBinding, evalTrainingNodeInputData);

                outputMap = new Dictionary<string, string>();
                if (node.OutputMap.TryGetValue(nameof(Output.TrainingWarnings), out var outTrainingVariableName))
                    outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.Warnings), outTrainingVariableName);
                if (node.OutputMap.TryGetValue(nameof(Output.TrainingOverallMetrics), out outTrainingVariableName))
                    outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.OverallMetrics), outTrainingVariableName);
                if (node.OutputMap.TryGetValue(nameof(Output.TrainingPerInstanceMetrics), out outTrainingVariableName))
                    outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.PerInstanceMetrics), outTrainingVariableName);
                if (node.OutputMap.TryGetValue(nameof(Output.TrainingConfusionMatrix), out outTrainingVariableName))
                    outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.ConfusionMatrix), outTrainingVariableName);
                EntryPointNode evalTrainingNode = EntryPointNode.Create(env, evalTrainingEntryPointName, evalTrainingArgs, node.Context, inputBindingMap, inputMap, outputMap);
                subGraphNodes.Add(evalTrainingNode);
            }

            // Add the evaluator node for testing.
            var evalArgs = MacroUtils.GetEvaluatorArgs(input.Kind, out var evalEntryPointName, settings);
            inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            inputMap = new Dictionary<ParameterBinding, VariableBinding>();
            var evalNodeInputData = new SimpleVariableBinding(evalDataVarName);
            paramBinding = new SimpleParameterBinding(nameof(evalArgs.Data));
            inputBindingMap.Add(nameof(evalArgs.Data), new List<ParameterBinding>() { paramBinding });
            inputMap.Add(paramBinding, evalNodeInputData);

            outputMap = new Dictionary<string, string>();
            if (node.OutputMap.TryGetValue(nameof(Output.Warnings), out var outVariableName))
                outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.Warnings), outVariableName);
            if (node.OutputMap.TryGetValue(nameof(Output.OverallMetrics), out outVariableName))
                outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.OverallMetrics), outVariableName);
            if (node.OutputMap.TryGetValue(nameof(Output.PerInstanceMetrics), out outVariableName))
                outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.PerInstanceMetrics), outVariableName);
            if (node.OutputMap.TryGetValue(nameof(Output.ConfusionMatrix), out outVariableName))
                outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.ConfusionMatrix), outVariableName);
            EntryPointNode evalNode = EntryPointNode.Create(env, evalEntryPointName, evalArgs, node.Context, inputBindingMap, inputMap, outputMap);
            subGraphNodes.Add(evalNode);

            // Marks as an atomic unit that can be run in
            // a distributed fashion.
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.StageId = input.PipelineId;

            return new CommonOutputs.MacroOutput<Output>() { Nodes = subGraphNodes };
        }
    }
}
