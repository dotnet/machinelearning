// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(void), typeof(TrainTestMacro), null, typeof(SignatureEntryPointModule), "TrainTestMacro")]

namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class TrainTestMacro
    {
        public sealed class SubGraphInput
        {
            [Argument(ArgumentType.Required, HelpText = "The data to be used for training", SortOrder = 1)]
            public Var<IDataView> Data;
        }

        public sealed class SubGraphOutput
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The predictor model", SortOrder = 1)]
            public Var<IPredictorModel> PredictorModel;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Transform model", SortOrder = 2)]
            public Var<ITransformModel> TransformModel;
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
            public Var<ITransformModel> TransformModel = null;

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
            public IPredictorModel PredictorModel;

            [TlcModule.Output(Desc = "The final model including the trained predictor model and the model from the transforms, " +
                "provided as the Input.TransformModel.", SortOrder = 2)]
            public ITransformModel TransformModel;

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
            var subGraphNodes = EntryPointNode.ValidateNodes(env, subGraphRunContext, input.Nodes, node.Catalog, input.LabelColumn,
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
            varName = input.Outputs.PredictorModel == null ? input.Outputs.TransformModel.VarName : input.Outputs.PredictorModel.VarName;
            if (!subGraphRunContext.TryGetVariable(varName, out dataVariable))
                throw env.Except($"Invalid variable name '{varName}'.");

            string outputVarName = input.Outputs.PredictorModel == null ? node.GetOutputVariableName(nameof(Output.TransformModel)) :
                node.GetOutputVariableName(nameof(Output.PredictorModel));

            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.RenameOutputVariable(dataVariable.Name, outputVarName);
            subGraphRunContext.RemoveVariable(dataVariable);

            // Move the variables from the subcontext to the main context.
            node.Context.AddContextVariables(subGraphRunContext);

            // Change all the subgraph nodes to use the main context.
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.SetContext(node.Context);

            // Testing using test data set
            var testingVar = node.GetInputVariable(nameof(input.TestingData));
            var exp = new Experiment(env);

            DatasetScorer.Output scoreNodeOutput = null;
            ML.Models.DatasetTransformer.Output datasetTransformNodeOutput = null;
            if (input.Outputs.PredictorModel == null)
            {
                //combine the predictor model with any potential transfrom model passed from the outer graph
                if (transformModelVarName != null && transformModelVarName.VariableName != null)
                {
                    var modelCombine = new ML.Transforms.ModelCombiner
                    {
                        Models = new ArrayVar<ITransformModel>(
                                new Var<ITransformModel>[] {
                                    new Var<ITransformModel> { VarName = transformModelVarName.VariableName },
                                    new Var<ITransformModel> { VarName = outputVarName} }
                                )
                    };

                    var modelCombineOutput = exp.Add(modelCombine);
                    outputVarName = modelCombineOutput.OutputModel.VarName;
                }

                var datasetTransformerNode = new Models.DatasetTransformer
                {
                    Data = { VarName = testingVar.ToJson() },
                    TransformModel = { VarName = outputVarName }
                };

                datasetTransformNodeOutput = exp.Add(datasetTransformerNode);
            }
            else
            {
                //combine the predictor model with any potential transfrom model passed from the outer graph
                if (transformModelVarName != null && transformModelVarName.VariableName != null)
                {
                    var modelCombine = new TwoHeterogeneousModelCombiner
                    {
                        TransformModel = { VarName = transformModelVarName.VariableName },
                        PredictorModel = { VarName = outputVarName }
                    };

                    var modelCombineOutput = exp.Add(modelCombine);
                    outputVarName = modelCombineOutput.PredictorModel.VarName;
                }

                // Add the scoring node for testing.
                var scoreNode = new DatasetScorer
                {
                    Data = { VarName = testingVar.ToJson() },
                    PredictorModel = { VarName = outputVarName }
                };

                scoreNodeOutput = exp.Add(scoreNode);
            }

            subGraphNodes.AddRange(EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes(), node.Catalog));

            // Do not double-add previous nodes.
            exp.Reset();

            // REVIEW: add similar support for NameColumn and FeatureColumn.
            var settings = new MacroUtils.EvaluatorSettings
            {
                LabelColumn = input.LabelColumn,
                WeightColumn = input.WeightColumn.IsExplicit ? input.WeightColumn.Value : null,
                GroupColumn = input.GroupColumn.IsExplicit ? input.GroupColumn.Value : null,
                NameColumn = input.NameColumn.IsExplicit ? input.NameColumn.Value : null
            };

            string outVariableName;

            if (input.IncludeTrainingMetrics)
            {
                DatasetScorer.Output scoreNodeTrainingOutput = null;
                ML.Models.DatasetTransformer.Output datasetTransformNodeTrainingOutput = null;
                if (input.Outputs.PredictorModel == null)
                {
                    var datasetTransformerNode = new Models.DatasetTransformer
                    {
                        Data = { VarName = testingVar.ToJson() },
                        TransformModel = { VarName = outputVarName }
                    };

                    datasetTransformNodeTrainingOutput = exp.Add(datasetTransformerNode);
                }
                else
                {
                    // Add the scoring node for training.
                    var scoreNodeTraining = new DatasetScorer
                    {
                        Data = { VarName = trainingVar.ToJson() },
                        PredictorModel = { VarName = outputVarName }
                    };
                    scoreNodeTrainingOutput = exp.Add(scoreNodeTraining);
                }

                subGraphNodes.AddRange(EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes(), node.Catalog));

                // Do not double-add previous nodes.
                exp.Reset();

                // Add the evaluator node for training.
                var evalInputOutputTraining = MacroUtils.GetEvaluatorInputOutput(input.Kind, settings);
                var evalNodeTraining = evalInputOutputTraining.Item1;
                var evalOutputTraining = evalInputOutputTraining.Item2;
                evalNodeTraining.Data.VarName = input.Outputs.PredictorModel == null ? datasetTransformNodeTrainingOutput.OutputData.VarName :
                    scoreNodeTrainingOutput.ScoredData.VarName;

                if (node.OutputMap.TryGetValue(nameof(Output.TrainingWarnings), out outVariableName))
                    evalOutputTraining.Warnings.VarName = outVariableName;
                if (node.OutputMap.TryGetValue(nameof(Output.TrainingOverallMetrics), out outVariableName))
                    evalOutputTraining.OverallMetrics.VarName = outVariableName;
                if (node.OutputMap.TryGetValue(nameof(Output.TrainingPerInstanceMetrics), out outVariableName))
                    evalOutputTraining.PerInstanceMetrics.VarName = outVariableName;
                if (node.OutputMap.TryGetValue(nameof(Output.TrainingConfusionMatrix), out outVariableName)
                    && evalOutputTraining is CommonOutputs.IClassificationEvaluatorOutput eoTraining)
                    eoTraining.ConfusionMatrix.VarName = outVariableName;

                exp.Add(evalNodeTraining, evalOutputTraining);
                subGraphNodes.AddRange(EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes(), node.Catalog));
            }

            // Do not double-add previous nodes.
            exp.Reset();

            // Add the evaluator node for testing.
            var evalInputOutput = MacroUtils.GetEvaluatorInputOutput(input.Kind, settings);
            var evalNode = evalInputOutput.Item1;
            var evalOutput = evalInputOutput.Item2;
            evalNode.Data.VarName = input.Outputs.PredictorModel == null ? datasetTransformNodeOutput.OutputData.VarName : scoreNodeOutput.ScoredData.VarName;

            if (node.OutputMap.TryGetValue(nameof(Output.Warnings), out outVariableName))
                evalOutput.Warnings.VarName = outVariableName;
            if (node.OutputMap.TryGetValue(nameof(Output.OverallMetrics), out outVariableName))
                evalOutput.OverallMetrics.VarName = outVariableName;
            if (node.OutputMap.TryGetValue(nameof(Output.PerInstanceMetrics), out outVariableName))
                evalOutput.PerInstanceMetrics.VarName = outVariableName;
            if (node.OutputMap.TryGetValue(nameof(Output.ConfusionMatrix), out outVariableName)
                && evalOutput is CommonOutputs.IClassificationEvaluatorOutput eo)
                eo.ConfusionMatrix.VarName = outVariableName;

            exp.Add(evalNode, evalOutput);
            subGraphNodes.AddRange(EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes(), node.Catalog));

            // Marks as an atomic unit that can be run in 
            // a distributed fashion.
            foreach (var subGraphNode in subGraphNodes)
                subGraphNode.StageId = input.PipelineId;

            return new CommonOutputs.MacroOutput<Output>() { Nodes = subGraphNodes };
        }
    }
}
