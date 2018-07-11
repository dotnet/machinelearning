// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.PipelineInference;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(void), typeof(PipelineSweeperMacro), null, typeof(SignatureEntryPointModule), "PipelineSweeperMacro")]

namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class PipelineSweeperMacro
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.Required, ShortName = "train", HelpText = "The data to be used for training.", SortOrder = 1)]
            public IDataView TrainingData;

            [Argument(ArgumentType.Required, ShortName = "test", HelpText = "The data to be used for testing.", SortOrder = 2)]
            public IDataView TestingData;

            [Argument(ArgumentType.AtMostOnce, ShortName = "args", HelpText = "The arguments for creating an AutoMlState component.", SortOrder = 3)]
            public AutoInference.ISupportAutoMlStateFactory StateArguments;

            [Argument(ArgumentType.AtMostOnce, ShortName = "state", HelpText = "The stateful object conducting of the autoML search.", SortOrder = 3)]
            public IMlState State;

            [Argument(ArgumentType.Required, ShortName = "bsize", HelpText = "Number of candidate pipelines to retrieve each round.", SortOrder = 4)]
            public int BatchSize;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output datasets from previous iteration of sweep.", SortOrder = 7, Hide = true)]
            public IDataView[] CandidateOutputs;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Column(s) to use as purpose 'Ignore'", SortOrder = 8)]
            public string[] IgnoreColumns;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Column(s) to use as purpose 'Name'", SortOrder = 9)]
            public string[] NameColumns;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Column(s) to use as purpose 'Label'", SortOrder = 10)]
            public string[] LabelColumns;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Column(s) to use as purpose 'NumericFeature'", SortOrder = 11)]
            public string[] NumericFeatureColumns;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Column(s) to use as purpose 'CategoricalFeature'", SortOrder = 12)]
            public string[] CategoricalFeatureColumns;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Column(s) to use as purpose 'TextFeature'", SortOrder = 13)]
            public string[] TextFeatureColumns;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Column(s) to use as purpose 'Weight'", SortOrder = 14)]
            public string[] WeightColumns;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Column(s) to use as purpose 'GroupId'", SortOrder = 15)]
            public string[] GroupIdColumns;

            [Argument(ArgumentType.MultipleUnique, HelpText = "Column(s) to use as purpose 'ImagePath'", SortOrder = 16)]
            public string[] ImagePathColumns;
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "Stateful autoML object, keeps track of where the search in progress.", SortOrder = 1)]
            public IMlState State;

            [TlcModule.Output(Desc = "Results of the sweep, including pipelines (as graph strings), IDs, and metric values.", SortOrder = 1)]
            public IDataView Results;
        }

        public sealed class ResultInput
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "state", HelpText = "The stateful object conducting of the autoML search.", SortOrder = 1)]
            public IMlState State;
        }

        [TlcModule.EntryPoint(Desc = "Extracts the sweep result.", Name = "Models.SweepResultExtractor")]
        public static Output ExtractSweepResult(IHostEnvironment env, ResultInput input)
        {
            var autoMlState = input.State as AutoInference.AutoMlMlState;
            if (autoMlState == null)
                throw env.Except("The state must be a valid AutoMlState.");
            // Create results output dataview
            var rows = autoMlState.GetAllEvaluatedPipelines().Select(p => p.ToResultRow()).ToList();
            IDataView outputView;
            var col1 = new KeyValuePair<string, ColumnType>("Graph", TextType.Instance);
            var col2 = new KeyValuePair<string, ColumnType>("MetricValue", PrimitiveType.FromKind(DataKind.R8));
            var col3 = new KeyValuePair<string, ColumnType>("PipelineId", TextType.Instance);
            var col4 = new KeyValuePair<string, ColumnType>("TrainingMetricValue", PrimitiveType.FromKind(DataKind.R8));
            var col5 = new KeyValuePair<string, ColumnType>("FirstInput", TextType.Instance);
            var col6 = new KeyValuePair<string, ColumnType>("PredictorModel", TextType.Instance);

            if (rows.Count == 0)
            {
                var host = env.Register("ExtractSweepResult");
                outputView = new EmptyDataView(host, new SimpleSchema(host, col1, col2, col3, col4, col5, col6));
            }
            else
            {
                var builder = new ArrayDataViewBuilder(env);
                builder.AddColumn(col1.Key, (PrimitiveType)col1.Value, rows.Select(r => new DvText(r.GraphJson)).ToArray());
                builder.AddColumn(col2.Key, (PrimitiveType)col2.Value, rows.Select(r => r.MetricValue).ToArray());
                builder.AddColumn(col3.Key, (PrimitiveType)col3.Value, rows.Select(r => new DvText(r.PipelineId)).ToArray());
                builder.AddColumn(col4.Key, (PrimitiveType)col4.Value, rows.Select(r => r.TrainingMetricValue).ToArray());
                builder.AddColumn(col5.Key, (PrimitiveType)col5.Value, rows.Select(r => new DvText(r.FirstInput)).ToArray());
                builder.AddColumn(col6.Key, (PrimitiveType)col6.Value, rows.Select(r => new DvText(r.PredictorModel)).ToArray());
                outputView = builder.GetDataView();
            }
            return new Output { Results = outputView, State = autoMlState };
        }

        private static Dictionary<string, ColumnPurpose> GetColumnPurpose(IHostEnvironment env, Arguments input)
        {
            var columnPurpose = new Dictionary<string, ColumnPurpose>();
            if (input.IgnoreColumns != null)
            {
                foreach (var colName in input.IgnoreColumns)
                {
                    columnPurpose.Add(colName, ColumnPurpose.Ignore);
                }
            }
            if (input.NameColumns != null)
            {
                env.Check(input.NameColumns.Length == 1, "NameColumn expected one column name to be specified.");
                columnPurpose.Add(input.NameColumns[0], ColumnPurpose.Name);
            }
            if (input.LabelColumns != null)
            {
                env.Check(input.LabelColumns.Length == 1, "LabelColumn expected one column name to be specified.");
                columnPurpose.Add(input.LabelColumns[0], ColumnPurpose.Label);
            }
            if (input.NumericFeatureColumns != null)
            {
                foreach (var colName in input.NumericFeatureColumns)
                {
                    columnPurpose.Add(colName, ColumnPurpose.NumericFeature);
                }
            }
            if (input.CategoricalFeatureColumns != null)
            {
                foreach (var colName in input.CategoricalFeatureColumns)
                {
                    columnPurpose.Add(colName, ColumnPurpose.CategoricalFeature);
                }
            }
            if (input.TextFeatureColumns != null)
            {
                foreach (var colName in input.TextFeatureColumns)
                {
                    columnPurpose.Add(colName, ColumnPurpose.TextFeature);
                }
            }
            if (input.WeightColumns != null)
            {
                env.Check(input.WeightColumns.Length == 1, "WeightColumn expected one column name to be specified.");
                columnPurpose.Add(input.WeightColumns[0], ColumnPurpose.Weight);
            }
            if (input.GroupIdColumns != null)
            {
                env.Check(input.GroupIdColumns.Length == 1, "GroupIdColumn expected one column name to be specified.");
                columnPurpose.Add(input.GroupIdColumns[0], ColumnPurpose.GroupId);
            }
            if (input.ImagePathColumns != null)
            {
                foreach (var colName in input.ImagePathColumns)
                {
                    columnPurpose.Add(colName, ColumnPurpose.ImagePath);
                }
            }
            return columnPurpose;
        }

        [TlcModule.EntryPoint(Desc = "AutoML pipeline sweeping optimzation macro.", Name = "Models.PipelineSweeper")]
        public static CommonOutputs.MacroOutput<Output> PipelineSweep(
            IHostEnvironment env,
            Arguments input,
            EntryPointNode node)
        {
            env.Check(input.StateArguments != null || input.State is AutoInference.AutoMlMlState,
                "Must have a valid AutoML State, or pass arguments to create one.");
            env.Check(input.BatchSize > 0, "Batch size must be > 0.");

            // Get the user-defined column purposes (if any)
            var columnPurpose = GetColumnPurpose(env, input);

            // If no current state, create object and set data.
            if (input.State == null)
            {
                input.State = input.StateArguments?.CreateComponent(env);

                if (input.State is AutoInference.AutoMlMlState inState)
                    inState.SetTrainTestData(input.TrainingData, input.TestingData);
                else
                    throw env.Except($"Incompatible type. Expecting type {typeof(AutoInference.AutoMlMlState)}, received type {input.State?.GetType()}.");

                var result = node.AddNewVariable("State", input.State);
                node.Context.AddInputVariable(result.Item2, typeof(IMlState));
            }
            var autoMlState = (AutoInference.AutoMlMlState)input.State;

            // The indicators are just so the macro knows those pipelines need to 
            // be run before performing next expansion. If we add them as inputs
            // to the next iteration, the next iteration cannot run until they have
            // their values set. Thus, indicators are needed.
            var pipelineIndicators = new List<Var<IDataView>>();

            var expNodes = new List<EntryPointNode>();

            // Keep versions of the training and testing var names
            var training = new Var<IDataView> { VarName = node.GetInputVariable("TrainingData").VariableName };
            var testing = new Var<IDataView> { VarName = node.GetInputVariable("TestingData").VariableName };
            var amlsVarObj =
                new Var<IMlState>()
                {
                    VarName = node.GetInputVariable(nameof(input.State)).VariableName
                };

            // Make sure search space is defined. If not, infer,
            // with default number of transform levels.
            if (!autoMlState.IsSearchSpaceDefined())
                autoMlState.InferSearchSpace(numTransformLevels: 1);

            // Extract performance summaries and assign to previous candidate pipelines.
            foreach (var pipeline in autoMlState.BatchCandidates)
            {
                if (node.Context.TryGetVariable(ExperimentUtils.GenerateOverallMetricVarName(pipeline.UniqueId), out var v) &&
                    node.Context.TryGetVariable(AutoMlUtils.GenerateOverallTrainingMetricVarName(pipeline.UniqueId), out var v2))
                {
                    pipeline.PerformanceSummary =
                        AutoMlUtils.ExtractRunSummary(env, (IDataView)v.Value, autoMlState.Metric.Name, (IDataView)v2.Value);
                    autoMlState.AddEvaluated(pipeline);
                }
            }

            node.OutputMap.TryGetValue("Results", out string outDvName);
            var outDvVar = new Var<IDataView>() { VarName = outDvName };
            node.OutputMap.TryGetValue("State", out string outStateName);
            var outStateVar = new Var<IMlState>() { VarName = outStateName };

            // Get next set of candidates.
            var candidatePipelines = autoMlState.GetNextCandidates(input.BatchSize);

            // Check if termination condition was met, i.e. no more candidates were returned.
            // If so, end expansion and add a node to extract the sweep result.
            if (candidatePipelines == null || candidatePipelines.Length == 0)
            {
                // Add a node to extract the sweep result.
                var resultSubgraph = new Experiment(env);
                var resultNode = new Microsoft.ML.Models.SweepResultExtractor() { State = amlsVarObj };
                var resultOutput = new Models.SweepResultExtractor.Output() { State = outStateVar, Results = outDvVar };
                resultSubgraph.Add(resultNode, resultOutput);
                var resultSubgraphNodes = EntryPointNode.ValidateNodes(env, node.Context, resultSubgraph.GetNodes(), node.Catalog);
                expNodes.AddRange(resultSubgraphNodes);
                return new CommonOutputs.MacroOutput<Output>() { Nodes = expNodes };
            }

            // Prep all returned candidates
            foreach (var p in candidatePipelines)
            {
                // Add train test experiments to current graph for candidate pipeline
                var subgraph = new Experiment(env);
                var trainTestOutput = p.AddAsTrainTest(training, testing, autoMlState.TrainerKind, subgraph, true);

                // Change variable name to reference pipeline ID in output map, context and entrypoint output.
                var uniqueName = ExperimentUtils.GenerateOverallMetricVarName(p.UniqueId);
                var uniqueNameTraining = AutoMlUtils.GenerateOverallTrainingMetricVarName(p.UniqueId);
                var sgNode = EntryPointNode.ValidateNodes(env, node.Context,
                    new JArray(subgraph.GetNodes().Last()), node.Catalog).Last();
                sgNode.RenameOutputVariable(trainTestOutput.OverallMetrics.VarName, uniqueName, cascadeChanges: true);
                sgNode.RenameOutputVariable(trainTestOutput.TrainingOverallMetrics.VarName, uniqueNameTraining, cascadeChanges: true);
                trainTestOutput.OverallMetrics.VarName = uniqueName;
                trainTestOutput.TrainingOverallMetrics.VarName = uniqueNameTraining;
                expNodes.Add(sgNode);

                // Store indicators, to pass to next iteration of macro.
                pipelineIndicators.Add(trainTestOutput.OverallMetrics);
            }

            // Add recursive macro node
            var macroSubgraph = new Experiment(env);
            var macroNode = new Models.PipelineSweeper()
            {
                BatchSize = input.BatchSize,
                CandidateOutputs = new ArrayVar<IDataView>(pipelineIndicators.ToArray()),
                TrainingData = training,
                TestingData = testing,
                State = amlsVarObj
            };
            var output = new Models.PipelineSweeper.Output() { Results = outDvVar, State = outStateVar };
            macroSubgraph.Add(macroNode, output);

            var subgraphNodes = EntryPointNode.ValidateNodes(env, node.Context, macroSubgraph.GetNodes(), node.Catalog);
            expNodes.AddRange(subgraphNodes);

            return new CommonOutputs.MacroOutput<Output>() { Nodes = expNodes };
        }
    }
}
