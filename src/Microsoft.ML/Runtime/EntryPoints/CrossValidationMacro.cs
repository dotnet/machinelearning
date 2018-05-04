// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(void), typeof(CrossValidationMacro), null, typeof(SignatureEntryPointModule), "CrossValidationMacro")]

namespace Microsoft.ML.Runtime.EntryPoints
{

    /// <summary>
    /// This macro entry point implements cross validation.
    /// </summary>
    public static class CrossValidationMacro
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
            // This is the data used in the cross validataion. It will be split into k folds
            // and a model will be trained and evaluated for each fold.
            [TlcModule.OptionalInput]
            [Argument(ArgumentType.Required, HelpText = "The data set", SortOrder = 1)]
            public IDataView Data;

            [TlcModule.OptionalInput]
            [Argument(ArgumentType.AtMostOnce, HelpText = "The transform model from the pipeline before this command. It gets included in the Output.PredictorModel.", SortOrder = 2)]
            public ITransformModel TransformModel;

            // This is the subgraph that describes how to train a model for each fold. It should
            // accept one IDataView input and output one IPredictorModel output (see Inputs and Outputs).
            [Argument(ArgumentType.Required, HelpText = "The training subgraph", SortOrder = 3)]
            public JArray Nodes;

            // This is the subgraph input, that shows that the subgraph should only require one 
            // IDataView as input and indicates the variable name (in the subgraph) for it.
            [Argument(ArgumentType.Required, HelpText = "The training subgraph inputs", SortOrder = 4)]
            public SubGraphInput Inputs = new SubGraphInput();

            // This is the subgraph output, that shows that the subgraph should produce one 
            // IPredictorModel as output and indicates the variable name (in the subgraph) for it.
            [Argument(ArgumentType.Required, HelpText = "The training subgraph outputs", SortOrder = 5)]
            public SubGraphOutput Outputs = new SubGraphOutput();

            // For splitting the data into folds, this column is used for grouping rows and makes sure
            // that a group of rows is not split among folds.
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Column to use for stratification", ShortName = "strat", SortOrder = 7)]
            public string StratificationColumn;

            // The number of folds to generate.
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of folds in k-fold cross-validation", ShortName = "k", SortOrder = 8)]
            public int NumFolds = 2;

            // REVIEW: suggest moving to subcomponents for evaluators, to allow for different parameters on the evaluators
            // (and the same for the TrainTest macro). I currently do not know how to do this, so this should be revisited in the future.
            [Argument(ArgumentType.Required, HelpText = "Specifies the trainer kind, which determines the evaluator to be used.", SortOrder = 9)]
            public MacroUtils.TrainerKinds Kind = MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer;
        }

        // REVIEW: This output would be much better as an array of CommonOutputs.ClassificationEvaluateOutput,
        // but that requires changes in the entry points infrastructure to support structs in the output classes.
        public sealed class Output
        {
            [TlcModule.Output(Desc = "The final model including the trained predictor model and the model from the transforms, provided as the Input.TransformModel.", SortOrder = 1)]
            public IPredictorModel[] PredictorModel;

            [TlcModule.Output(Desc = "Warning dataset", SortOrder = 2)]
            public IDataView[] Warnings;

            [TlcModule.Output(Desc = "Overall metrics dataset", SortOrder = 3)]
            public IDataView[] OverallMetrics;

            [TlcModule.Output(Desc = "Per instance metrics dataset", SortOrder = 4)]
            public IDataView[] PerInstanceMetrics;

            [TlcModule.Output(Desc = "Confusion matrix dataset", SortOrder = 5)]
            public IDataView[] ConfusionMatrix;
        }

        [TlcModule.EntryPoint(Desc = "Cross validation for general learning", Name = "Models.CrossValidator")]
        public static CommonOutputs.MacroOutput<Output> CrossValidate(
            IHostEnvironment env,
            Arguments input,
            EntryPointNode node)
        {
            env.CheckValue(input, nameof(input));

            // This will be the final resulting list of nodes that is returned from the macro.
            var subGraphNodes = new List<EntryPointNode>();

            //the input transform model 
            VariableBinding transformModelVarName = null;
            if (input.TransformModel != null)
                transformModelVarName = node.GetInputVariable(nameof(input.TransformModel));

            // Split the input data into folds.
            var exp = new Experiment(env);
            var cvSplit = new Models.CrossValidatorDatasetSplitter();
            cvSplit.Data.VarName = node.GetInputVariable("Data").ToJson();
            cvSplit.NumFolds = input.NumFolds;
            cvSplit.StratificationColumn = input.StratificationColumn;
            var cvSplitOutput = exp.Add(cvSplit);
            subGraphNodes.AddRange(EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes(), node.Catalog));

            var predModelVars = new Var<IPredictorModel>[input.NumFolds];
            var inputTransformModelVars = new Var<IPredictorModel>[input.NumFolds];
            var warningsVars = new Var<IDataView>[input.NumFolds];
            var overallMetricsVars = new Var<IDataView>[input.NumFolds];
            var instanceMetricsVars = new Var<IDataView>[input.NumFolds];
            var confusionMatrixVars = new Var<IDataView>[input.NumFolds];
            // Instantiate the subgraph for each fold.
            for (int k = 0; k < input.NumFolds; k++)
            {
                // Parse the nodes in input.Nodes into a temporary run context.
                var context = new RunContext(env);
                var graph = EntryPointNode.ValidateNodes(env, context, input.Nodes, node.Catalog);

                // Rename all the variables such that they don't conflict with the ones in the outer run context.
                var mapping = new Dictionary<string, string>();
                foreach (var entryPointNode in graph)
                    entryPointNode.RenameAllVariables(mapping);

                // Instantiate a TrainTest entry point for this fold.
                var args = new TrainTestMacro.Arguments
                {
                    Nodes = new JArray(graph.Select(n => n.ToJson()).ToArray()),
                    TransformModel = null
                };

                if (transformModelVarName != null)
                    args.TransformModel = new Var<ITransformModel> { VarName = transformModelVarName.VariableName };

                args.Inputs.Data = new Var<IDataView>
                {
                    VarName = mapping[input.Inputs.Data.VarName]
                };
                args.Outputs.Model = new Var<IPredictorModel>
                {
                    VarName = mapping[input.Outputs.Model.VarName]
                };

                // Set train/test trainer kind to match.
                args.Kind = input.Kind;

                // Set the input bindings for the TrainTest entry point.
                var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
                var inputMap = new Dictionary<ParameterBinding, VariableBinding>();
                var trainingData = new SimpleParameterBinding(nameof(args.TrainingData));
                inputBindingMap.Add(nameof(args.TrainingData), new List<ParameterBinding> { trainingData });
                inputMap.Add(trainingData, new ArrayIndexVariableBinding(cvSplitOutput.TrainData.VarName, k));
                var testingData = new SimpleParameterBinding(nameof(args.TestingData));
                inputBindingMap.Add(nameof(args.TestingData), new List<ParameterBinding> { testingData });
                inputMap.Add(testingData, new ArrayIndexVariableBinding(cvSplitOutput.TestData.VarName, k));
                var outputMap = new Dictionary<string, string>();
                var predModelVar = new Var<IPredictorModel>();
                outputMap.Add(nameof(TrainTestMacro.Output.PredictorModel), predModelVar.VarName);
                predModelVars[k] = predModelVar;

                ML.Transforms.TwoHeterogeneousModelCombiner.Output modelCombineOutput = null;
                if (transformModelVarName != null && transformModelVarName.VariableName != null)
                {
                    var modelCombine = new ML.Transforms.TwoHeterogeneousModelCombiner
                    {
                        TransformModel = { VarName = transformModelVarName.VariableName },
                        PredictorModel = predModelVar
                    };

                    exp.Reset();
                    modelCombineOutput = exp.Add(modelCombine);
                    subGraphNodes.AddRange(EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes(), node.Catalog));
                    predModelVars[k] = modelCombineOutput.PredictorModel;
                }

                var warningVar = new Var<IDataView>();
                outputMap.Add(nameof(TrainTestMacro.Output.Warnings), warningVar.VarName);
                warningsVars[k] = warningVar;
                var overallMetric = new Var<IDataView>();
                outputMap.Add(nameof(TrainTestMacro.Output.OverallMetrics), overallMetric.VarName);
                overallMetricsVars[k] = overallMetric;
                var instanceMetric = new Var<IDataView>();
                outputMap.Add(nameof(TrainTestMacro.Output.PerInstanceMetrics), instanceMetric.VarName);
                instanceMetricsVars[k] = instanceMetric;
                var confusionMatrix = new Var<IDataView>();
                outputMap.Add(nameof(TrainTestMacro.Output.ConfusionMatrix), confusionMatrix.VarName);
                confusionMatrixVars[k] = confusionMatrix;
                subGraphNodes.Add(EntryPointNode.Create(env, "Models.TrainTestEvaluator", args, node.Catalog, node.Context, inputBindingMap, inputMap, outputMap));
            }

            exp.Reset();

            var outModels = new ML.Data.PredictorModelArrayConverter
            {
                Model = new ArrayVar<IPredictorModel>(predModelVars)
            };
            var outModelsOutput = new ML.Data.PredictorModelArrayConverter.Output();
            outModelsOutput.OutputModel.VarName = node.GetOutputVariableName(nameof(Output.PredictorModel));
            exp.Add(outModels, outModelsOutput);

            var warnings = new ML.Data.IDataViewArrayConverter
            {
                Data = new ArrayVar<IDataView>(warningsVars)
            };
            var warningsOutput = new ML.Data.IDataViewArrayConverter.Output();
            warningsOutput.OutputData.VarName = node.GetOutputVariableName(nameof(Output.Warnings));
            exp.Add(warnings, warningsOutput);

            var overallMetrics = new ML.Data.IDataViewArrayConverter
            {
                Data = new ArrayVar<IDataView>(overallMetricsVars)
            };
            var overallMetricsOutput = new ML.Data.IDataViewArrayConverter.Output();
            overallMetricsOutput.OutputData.VarName = node.GetOutputVariableName(nameof(Output.OverallMetrics));
            exp.Add(overallMetrics, overallMetricsOutput);

            var instanceMetrics = new ML.Data.IDataViewArrayConverter
            {
                Data = new ArrayVar<IDataView>(instanceMetricsVars)
            };
            var instanceMetricsOutput = new ML.Data.IDataViewArrayConverter.Output();
            instanceMetricsOutput.OutputData.VarName = node.GetOutputVariableName(nameof(Output.PerInstanceMetrics));
            exp.Add(instanceMetrics, instanceMetricsOutput);

            if (input.Kind == MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer ||
                input.Kind == MacroUtils.TrainerKinds.SignatureMultiClassClassifierTrainer)
            {
                var confusionMatrices = new ML.Data.IDataViewArrayConverter
                {
                    Data = new ArrayVar<IDataView>(confusionMatrixVars)
                };
                var confusionMatricesOutput = new ML.Data.IDataViewArrayConverter.Output();
                confusionMatricesOutput.OutputData.VarName = node.GetOutputVariableName(nameof(Output.ConfusionMatrix));
                exp.Add(confusionMatrices, confusionMatricesOutput);
            }

            subGraphNodes.AddRange(EntryPointNode.ValidateNodes(env, node.Context, exp.GetNodes(), node.Catalog));

            return new CommonOutputs.MacroOutput<Output>() { Nodes = subGraphNodes };
        }
    }
}
