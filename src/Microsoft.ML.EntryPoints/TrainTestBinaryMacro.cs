// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(void), typeof(TrainTestBinaryMacro), null, typeof(SignatureEntryPointModule), "TrainTestBinaryMacro")]

// The warning #612 is disabled because the following code uses a lot of things in Legacy.Models and Legacy.Transforms while Legacy is marked as obsolete.
// Because that dependency will be removed form ML.NET, one needs to rewrite all places where legacy APIs are used.
#pragma warning disable 612
namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class TrainTestBinaryMacro
    {
        public sealed class SubGraphInput
        {
            [Argument(ArgumentType.Required, HelpText = "The data to be used for training", SortOrder = 1)]
            public Var<IDataView> Data;
        }

        public sealed class SubGraphOutput
        {
            [Argument(ArgumentType.Required, HelpText = "The model", SortOrder = 1)]
            public Var<PredictorModel> Model;
        }

        public sealed class Arguments
        {
            [TlcModule.OptionalInput]
            [Argument(ArgumentType.Required, ShortName = "train", HelpText = "The data to be used for training", SortOrder = 1)]
            public IDataView TrainingData;

            [TlcModule.OptionalInput]
            [Argument(ArgumentType.Required, ShortName = "test", HelpText = "The data to be used for testing", SortOrder = 2)]
            public IDataView TestingData;

            [Argument(ArgumentType.Required, HelpText = "The training subgraph", SortOrder = 3)]
            public JArray Nodes;

            [Argument(ArgumentType.Required, HelpText = "The training subgraph inputs", SortOrder = 4)]
            public SubGraphInput Inputs = new SubGraphInput();

            [Argument(ArgumentType.Required, HelpText = "The training subgraph outputs", SortOrder = 5)]
            public SubGraphOutput Outputs = new SubGraphOutput();
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "The trained model", SortOrder = 1)]
            public PredictorModel PredictorModel;

            [TlcModule.Output(Desc = "Warning dataset", SortOrder = 2)]
            public IDataView Warnings;

            [TlcModule.Output(Desc = "Overall metrics dataset", SortOrder = 3)]
            public IDataView OverallMetrics;

            [TlcModule.Output(Desc = "Per instance metrics dataset", SortOrder = 4)]
            public IDataView PerInstanceMetrics;

            [TlcModule.Output(Desc = "Confusion matrix dataset", SortOrder = 5)]
            public IDataView ConfusionMatrix;
        }

        [TlcModule.EntryPoint(Desc = "Train test for binary classification", Name = "Models.TrainTestBinaryEvaluator")]
        public static CommonOutputs.MacroOutput<Output> TrainTestBinary(
            IHostEnvironment env,
            Arguments input,
            EntryPointNode node)
        {
            // Parse the subgraph.
            var subGraphRunContext = new RunContext(env);
            var subGraphNodes = EntryPointNode.ValidateNodes(env, subGraphRunContext, input.Nodes);

            // Change the subgraph to use the training data as input.
            var varName = input.Inputs.Data.VarName;
            EntryPointVariable variable;
            if (!subGraphRunContext.TryGetVariable(varName, out variable))
                throw env.Except($"Invalid variable name '{varName}'.");
            var trainingVar = node.GetInputVariable("TrainingData");
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.RenameInputVariable(variable.Name, trainingVar);
            subGraphRunContext.RemoveVariable(variable);

            // Change the subgraph to use the model variable as output.
            varName = input.Outputs.Model.VarName;
            if (!subGraphRunContext.TryGetVariable(varName, out variable))
                throw env.Except($"Invalid variable name '{varName}'.");
            string outputVarName = node.GetOutputVariableName("PredictorModel");
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.RenameOutputVariable(variable.Name, outputVarName);
            subGraphRunContext.RemoveVariable(variable);

            // Move the variables from the subcontext to the main context.
            node.Context.AddContextVariables(subGraphRunContext);

            // Change all the subgraph nodes to use the main context.
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.SetContext(node.Context);

            // Add the scoring node.
            var testingVar = node.GetInputVariable("TestingData");
            var args = new ScoreModel.Input();
            var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            var inputMap = new Dictionary<ParameterBinding, VariableBinding>();
            var paramBinding = new SimpleParameterBinding(nameof(args.Data));
            inputBindingMap.Add(nameof(args.Data), new List<ParameterBinding>() { paramBinding });
            inputMap.Add(paramBinding, testingVar);
            var scoreNodeInputPredictorModel = new SimpleVariableBinding(outputVarName);
            paramBinding = new SimpleParameterBinding(nameof(args.PredictorModel));
            inputBindingMap.Add(nameof(args.PredictorModel), new List<ParameterBinding>() { paramBinding });
            inputMap.Add(paramBinding, scoreNodeInputPredictorModel);

            var scoreNodeOutputScoredData = new Var<IDataView>();
            var scoreNodeOutputScoringTransform = new Var<TransformModel>();
            var outputMap = new Dictionary<string, string>();
            outputMap.Add(nameof(ScoreModel.Output.ScoredData), scoreNodeOutputScoredData.VarName);
            outputMap.Add(nameof(ScoreModel.Output.ScoringTransform), scoreNodeOutputScoringTransform.VarName);

            EntryPointNode scoreNode = EntryPointNode.Create(env, "Transforms.DatasetScorer", args,
                node.Context, inputBindingMap, inputMap, outputMap);
            subGraphNodes.Add(scoreNode);

            // Add the evaluator node.
            var evalArgs = new BinaryClassifierMamlEvaluator.Arguments();
            inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            inputMap = new Dictionary<ParameterBinding, VariableBinding>();
            var evalNodeInputData = new SimpleVariableBinding(scoreNodeOutputScoredData.VarName);
            paramBinding = new SimpleParameterBinding(nameof(evalArgs.Data));
            inputBindingMap.Add(nameof(evalArgs.Data), new List<ParameterBinding>() { paramBinding });
            inputMap.Add(paramBinding, evalNodeInputData);

            outputMap = new Dictionary<string, string>();
            if (node.OutputMap.TryGetValue("Warnings", out var outVariableName))
                outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.Warnings), outVariableName);
            if (node.OutputMap.TryGetValue("OverallMetrics", out outVariableName))
                outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.OverallMetrics), outVariableName);
            if (node.OutputMap.TryGetValue("PerInstanceMetrics", out outVariableName))
                outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.PerInstanceMetrics), outVariableName);
            if (node.OutputMap.TryGetValue("ConfusionMatrix", out outVariableName))
                outputMap.Add(nameof(CommonOutputs.ClassificationEvaluateOutput.ConfusionMatrix), outVariableName);
            EntryPointNode evalNode = EntryPointNode.Create(env, "Models.BinaryClassificationEvaluator", evalArgs,
                node.Context, inputBindingMap, inputMap, outputMap);
            subGraphNodes.Add(evalNode);

            var stageId = Guid.NewGuid().ToString("N");
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.StageId = stageId;

            return new CommonOutputs.MacroOutput<Output>() { Nodes = subGraphNodes };
        }
    }
}
#pragma warning restore 612
