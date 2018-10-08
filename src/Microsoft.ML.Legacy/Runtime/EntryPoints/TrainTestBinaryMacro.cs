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
            public Var<IPredictorModel> Model;
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
            public IPredictorModel PredictorModel;

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
            var exp = new Experiment(env);
            var scoreNode = new Legacy.Transforms.DatasetScorer();
            scoreNode.Data.VarName = testingVar.ToJson();
            scoreNode.PredictorModel.VarName = outputVarName;
            var scoreNodeOutput = exp.Add(scoreNode);
            subGraphNodes.AddRange(EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes()));

            // Add the evaluator node.
            exp.Reset();
            var evalNode = new Legacy.Models.BinaryClassificationEvaluator();
            evalNode.Data.VarName = scoreNodeOutput.ScoredData.VarName;
            var evalOutput = new Legacy.Models.BinaryClassificationEvaluator.Output();
            string outVariableName;
            if (node.OutputMap.TryGetValue("Warnings", out outVariableName))
                evalOutput.Warnings.VarName = outVariableName;
            if (node.OutputMap.TryGetValue("OverallMetrics", out outVariableName))
                evalOutput.OverallMetrics.VarName = outVariableName;
            if (node.OutputMap.TryGetValue("PerInstanceMetrics", out outVariableName))
                evalOutput.PerInstanceMetrics.VarName = outVariableName;
            if (node.OutputMap.TryGetValue("ConfusionMatrix", out outVariableName))
                evalOutput.ConfusionMatrix.VarName = outVariableName;
            exp.Add(evalNode, evalOutput);
            subGraphNodes.AddRange(EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes()));

            var stageId = Guid.NewGuid().ToString("N");
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.StageId = stageId;

            return new CommonOutputs.MacroOutput<Output>() { Nodes = subGraphNodes };
        }
    }
}
