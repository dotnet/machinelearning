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
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(void), typeof(OneVersusAllMacro), null, typeof(SignatureEntryPointModule), "OneVersusAllMacro")]

namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// This macro entrypoint implements OVA.
    /// </summary>
    internal static class OneVersusAllMacro
    {
        public sealed class SubGraphOutput
        {
            [Argument(ArgumentType.Required, HelpText = "The predictor model for the subgraph exemplar.", SortOrder = 1)]
            public Var<PredictorModel> Model;
        }

        public sealed class Arguments : TrainerInputBaseWithWeight
        {
            // This is the subgraph that describes how to train a model for submodel. It should
            // accept one IDataView input and output one IPredictorModel output.
            [Argument(ArgumentType.Required, HelpText = "The subgraph for the binary trainer used to construct the OVA learner. This should be a TrainBinary node.", SortOrder = 1)]
            public JArray Nodes;

            [Argument(ArgumentType.Required, HelpText = "The training subgraph output.", SortOrder = 2)]
            public SubGraphOutput OutputForSubGraph = new SubGraphOutput();

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use probabilities in OVA combiner", SortOrder = 3)]
            public bool UseProbabilities = true;
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "The trained multiclass model", SortOrder = 1)]
            public PredictorModel PredictorModel;
        }

        private static Var<PredictorModel> ProcessClass(IHostEnvironment env, List<EntryPointNode> macroNodes, int k, string label, Arguments input, EntryPointNode node)
        {
            Contracts.AssertValue(macroNodes);

            // Convert label into T,F based on k.
            var labelIndicatorArgs = new LabelIndicatorTransform.Options();
            labelIndicatorArgs.ClassIndex = k;
            labelIndicatorArgs.Columns = new[] { new LabelIndicatorTransform.Column() { Name = label, Source = label } };

            var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            var inputMap = new Dictionary<ParameterBinding, VariableBinding>();
            var paramBinding = new SimpleParameterBinding(nameof(labelIndicatorArgs.Data));
            inputBindingMap.Add(nameof(labelIndicatorArgs.Data), new List<ParameterBinding>() { paramBinding });
            inputMap.Add(paramBinding, node.GetInputVariable(nameof(input.TrainingData)));

            var outputMap = new Dictionary<string, string>();
            var remappedLabelVar = new Var<IDataView>();
            outputMap.Add(nameof(CommonOutputs.TransformOutput.OutputData), remappedLabelVar.VarName);
            var labelIndicatorNode = EntryPointNode.Create(env, "Transforms.LabelIndicator", labelIndicatorArgs, node.Context,
                inputBindingMap, inputMap, outputMap);
            macroNodes.Add(labelIndicatorNode);

            // Parse the nodes in input.Nodes into a temporary run context.
            var subGraphRunContext = new RunContext(env);
            var subGraphNodes = EntryPointNode.ValidateNodes(env, subGraphRunContext, input.Nodes);

            // Rename all the variables such that they don't conflict with the ones in the outer run context.
            var mapping = new Dictionary<string, string>();
            bool foundOutput = false;
            Var<PredictorModel> predModelVar = null;
            foreach (var entryPointNode in subGraphNodes)
            {
                // Rename variables in input/output maps, and in subgraph context.
                entryPointNode.RenameAllVariables(mapping);
                foreach (var kvp in mapping)
                    subGraphRunContext.RenameContextVariable(kvp.Key, kvp.Value);

                // Grab a hold of output model from this subgraph.
                if (entryPointNode.GetOutputVariableName("PredictorModel") is string mvn)
                {
                    predModelVar = new Var<PredictorModel> { VarName = mvn };
                    foundOutput = true;
                }

                // Connect label remapper output to wherever training data was expected within the input graph.
                if (entryPointNode.GetInputVariable(nameof(input.TrainingData)) is VariableBinding vb)
                    vb.Rename(remappedLabelVar.VarName);

                // Change node to use the main context.
                entryPointNode.SetContext(node.Context);
            }

            // Move the variables from the subcontext to the main context.
            node.Context.AddContextVariables(subGraphRunContext);

            // Make sure we found the output variable for this model.
            if (!foundOutput)
                throw new Exception("Invalid input graph. Does not output predictor model.");

            // Add training subgraph to our context.
            macroNodes.AddRange(subGraphNodes);
            return predModelVar;
        }

        private static int GetNumberOfClasses(IHostEnvironment env, Arguments input, out string label)
        {
            var host = env.Register("OVA Macro GetNumberOfClasses");
            using (var ch = host.Start("OVA Macro GetNumberOfClasses"))
            {
                // RoleMappedData creation
                var schema = input.TrainingData.Schema;
                label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.LabelColumnName),
                    input.LabelColumnName,
                    DefaultColumnNames.Label);
                var feature = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.FeatureColumnName),
                    input.FeatureColumnName, DefaultColumnNames.Features);
                var weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.ExampleWeightColumnName),
                    input.ExampleWeightColumnName, DefaultColumnNames.Weight);

                // Get number of classes
                var data = new RoleMappedData(input.TrainingData, label, feature, null, weight);
                data.CheckMulticlassLabel(out var numClasses);
                return numClasses;
            }
        }

        [TlcModule.EntryPoint(Desc = "One-vs-All macro (OVA)",
            Name = "Models.OneVersusAll")]
        public static CommonOutputs.MacroOutput<Output> OneVersusAll(
            IHostEnvironment env,
            Arguments input,
            EntryPointNode node)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));
            env.Assert(input.Nodes.Count > 0);

            var numClasses = GetNumberOfClasses(env, input, out var label);
            var predModelVars = new Var<PredictorModel>[numClasses];

            // This will be the final resulting list of nodes that is returned from the macro.
            var macroNodes = new List<EntryPointNode>();

            // Instantiate the subgraph for each label value.
            for (int k = 0; k < numClasses; k++)
                predModelVars[k] = ProcessClass(env, macroNodes, k, label, input, node);

            // Convert the predictor models to an array of predictor models.
            var modelsArray = new Var<PredictorModel[]>();
            MacroUtils.ConvertIPredictorModelsToArray(env, node.Context, macroNodes, predModelVars, modelsArray.VarName);

            // Use OVA model combiner to combine these models into one.
            // Takes in array of models that are binary predictor models and
            // produces single multiclass predictor model.
            var combineArgs = new ModelOperations.CombineOvaPredictorModelsInput();
            combineArgs.Caching = input.Caching;
            combineArgs.FeatureColumnName = input.FeatureColumnName;
            combineArgs.LabelColumnName = input.LabelColumnName;
            combineArgs.NormalizeFeatures = input.NormalizeFeatures;
            combineArgs.UseProbabilities = input.UseProbabilities;

            var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            var inputMap = new Dictionary<ParameterBinding, VariableBinding>();
            var combineNodeModelArrayInput = new SimpleVariableBinding(modelsArray.VarName);
            var paramBinding = new SimpleParameterBinding(nameof(combineArgs.ModelArray));
            inputBindingMap.Add(nameof(combineArgs.ModelArray), new List<ParameterBinding>() { paramBinding });
            inputMap.Add(paramBinding, combineNodeModelArrayInput);
            paramBinding = new SimpleParameterBinding(nameof(combineArgs.TrainingData));
            inputBindingMap.Add(nameof(combineArgs.TrainingData), new List<ParameterBinding>() { paramBinding });
            inputMap.Add(paramBinding, node.GetInputVariable(nameof(input.TrainingData)));

            var outputMap = new Dictionary<string, string>();
            outputMap.Add(nameof(Output.PredictorModel), node.GetOutputVariableName(nameof(Output.PredictorModel)));
            var combineModelsNode = EntryPointNode.Create(env, "Models.OvaModelCombiner",
                combineArgs, node.Context, inputBindingMap, inputMap, outputMap);
            macroNodes.Add(combineModelsNode);

            return new CommonOutputs.MacroOutput<Output>() { Nodes = macroNodes };
        }
    }
}
