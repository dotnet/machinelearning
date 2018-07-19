// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Training;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(void), typeof(OneVersusAllMacro), null, typeof(SignatureEntryPointModule), "OneVersusAllMacro")]
namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// This macro entrypoint implements OVA.
    /// </summary>
    public static class OneVersusAllMacro
    {
        public sealed class SubGraphOutput
        {
            [Argument(ArgumentType.Required, HelpText = "The predictor model for the subgraph exemplar.", SortOrder = 1)]
            public Var<IPredictorModel> Model;
        }

        public sealed class Arguments : LearnerInputBaseWithWeight
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
            public IPredictorModel PredictorModel;
        }

        private static Tuple<List<EntryPointNode>, Var<IPredictorModel>> ProcessClass(IHostEnvironment env, int k, string label, Arguments input, EntryPointNode node)
        {
            var macroNodes = new List<EntryPointNode>();

            // Convert label into T,F based on k.
            var remapper = new ML.Transforms.LabelIndicator
            {
                ClassIndex = k,
                Column = new[]
                {
                    new ML.Transforms.LabelIndicatorTransformColumn
                    {
                        ClassIndex = k,
                        Name = label,
                        Source = label
                    }
                },
                Data = { VarName = node.GetInputVariable(nameof(input.TrainingData)).ToJson() }
            };
            var exp = new Experiment(env);
            var remapperOutNode = exp.Add(remapper);
            var subNodes = EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes(), node.Catalog);
            macroNodes.AddRange(subNodes);

            // Parse the nodes in input.Nodes into a temporary run context.
            var subGraphRunContext = new RunContext(env);
            var subGraphNodes = EntryPointNode.ValidateNodes(env, subGraphRunContext, input.Nodes, node.Catalog);

            // Rename all the variables such that they don't conflict with the ones in the outer run context.
            var mapping = new Dictionary<string, string>();
            bool foundOutput = false;
            Var<IPredictorModel> predModelVar = null;
            foreach (var entryPointNode in subGraphNodes)
            {
                // Rename variables in input/output maps, and in subgraph context.
                entryPointNode.RenameAllVariables(mapping);
                foreach (var kvp in mapping)
                    subGraphRunContext.RenameContextVariable(kvp.Key, kvp.Value);

                // Grab a hold of output model from this subgraph.
                if (entryPointNode.GetOutputVariableName("PredictorModel") is string mvn)
                {
                    predModelVar = new Var<IPredictorModel> { VarName = mvn };
                    foundOutput = true;
                }

                // Connect label remapper output to wherever training data was expected within the input graph.
                if (entryPointNode.GetInputVariable(nameof(input.TrainingData)) is VariableBinding vb)
                    vb.Rename(remapperOutNode.OutputData.VarName);

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

            return new Tuple<List<EntryPointNode>, Var<IPredictorModel>>(macroNodes, predModelVar);
        }

        private static int GetNumberOfClasses(IHostEnvironment env, Arguments input, out string label)
        {
            var host = env.Register("OVA Macro GetNumberOfClasses");
            using (var ch = host.Start("OVA Macro GetNumberOfClasses"))
            {
                // RoleMappedData creation
                ISchema schema = input.TrainingData.Schema;
                label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.LabelColumn),
                    input.LabelColumn,
                    DefaultColumnNames.Label);
                var feature = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.FeatureColumn),
                    input.FeatureColumn, DefaultColumnNames.Features);
                var weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(Arguments.WeightColumn),
                    input.WeightColumn, DefaultColumnNames.Weight);

                // Get number of classes
                var data = new RoleMappedData(input.TrainingData, label, feature, null, weight);
                data.CheckMultiClassLabel(out var numClasses);
                return numClasses;
            }
        }

        [TlcModule.EntryPoint(Desc = "One-vs-All macro (OVA)",
            Name = "Models.OneVersusAll",
            XmlInclude = new[] { @"<include file='../Microsoft.ML.StandardLearners/Standard/MultiClass/doc.xml' path='doc/members/member[@name=""OVA""]'/>" })]
        public static CommonOutputs.MacroOutput<Output> OneVersusAll(
            IHostEnvironment env,
            Arguments input,
            EntryPointNode node)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));
            env.Assert(input.Nodes.Count > 0);

            var numClasses = GetNumberOfClasses(env, input, out var label);
            var predModelVars = new Var<IPredictorModel>[numClasses];

            // This will be the final resulting list of nodes that is returned from the macro.
            var macroNodes = new List<EntryPointNode>();

            // Instantiate the subgraph for each label value.
            for (int k = 0; k < numClasses; k++)
            {
                var result = ProcessClass(env, k, label, input, node);
                predModelVars[k] = result.Item2;
                macroNodes.AddRange(result.Item1);
            }

            // Use OVA model combiner to combine these models into one. 
            // Takes in array of models that are binary predictor models and
            // produces single multiclass predictor model.
            var macroExperiment = new Experiment(env);
            var combinerNode = new Models.OvaModelCombiner
            {
                ModelArray = new ArrayVar<IPredictorModel>(predModelVars),
                TrainingData = new Var<IDataView> { VarName = node.GetInputVariable(nameof(input.TrainingData)).VariableName },
                Caching = (Models.CachingOptions)input.Caching,
                FeatureColumn = input.FeatureColumn,
                NormalizeFeatures = (Models.NormalizeOption)input.NormalizeFeatures,
                LabelColumn = input.LabelColumn,
                UseProbabilities = input.UseProbabilities
            };

            // Get output model variable.
            if (!node.OutputMap.TryGetValue(nameof(Output.PredictorModel), out var outVariableName))
                throw new Exception("Cannot find OVA model output.");

            // Map macro's output back to OVA combiner (so OVA combiner will set the value on our output variable).
            var combinerOutput = new Models.OvaModelCombiner.Output { PredictorModel = new Var<IPredictorModel> { VarName = outVariableName } };

            // Add to experiment (must be done AFTER we assign variable name to output).
            macroExperiment.Add(combinerNode, combinerOutput);

            // Add nodes to main experiment.
            var nodes = macroExperiment.GetNodes();
            var expNodes = EntryPointNode.ValidateNodes(env, node.Context, nodes, node.Catalog);
            macroNodes.AddRange(expNodes);

            return new CommonOutputs.MacroOutput<Output>() { Nodes = macroNodes };
        }
    }
}
