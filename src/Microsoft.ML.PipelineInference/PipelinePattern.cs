// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// A runnable pipeline. Contains a learner and set of transforms,
    /// along with a RunSummary if it has already been exectued.
    /// </summary>
    public sealed class PipelinePattern : IEquatable<PipelinePattern>
    {
        /// <summary>
        /// Class for encapsulating the information returned in the output IDataView for a pipeline
        /// that has been run through the TrainTest macro.
        /// </summary>
        public sealed class PipelineResultRow
        {
            public string GraphJson { get; }
            ///<summary>
            /// The metric value of the test dataset result (always needed).
            ///</summary>
            public double MetricValue { get; }
            ///<summary>
            /// The metric value of the training dataset result (not always used or set).
            ///</summary>
            public double TrainingMetricValue { get; }
            public string PipelineId { get; }
            public string FirstInput { get; }
            public string PredictorModel { get; }

            public PipelineResultRow()
            { }

            public PipelineResultRow(string graphJson, double metricValue,
                string pipelineId, double trainingMetricValue, string firstInput,
                string predictorModel)
            {
                GraphJson = graphJson;
                MetricValue = metricValue;
                PipelineId = pipelineId;
                TrainingMetricValue = trainingMetricValue;
                FirstInput = firstInput;
                PredictorModel = predictorModel;
            }
        }

        private readonly IHostEnvironment _env;
        public readonly TransformInference.SuggestedTransform[] Transforms;
        public readonly RecipeInference.SuggestedRecipe.SuggestedLearner Learner;
        public PipelineSweeperRunSummary PerformanceSummary { get; set; }
        public string LoaderSettings { get; set; }
        public Guid UniqueId { get; }

        public PipelinePattern(TransformInference.SuggestedTransform[] transforms,
            RecipeInference.SuggestedRecipe.SuggestedLearner learner,
            string loaderSettings, IHostEnvironment env, PipelineSweeperRunSummary summary = null)
        {
            // Make sure internal pipeline nodes and sweep params are cloned, not shared.
            // Cloning the transforms and learner rather than assigning outright
            // ensures that this will be the case. Doing this here allows us to not
            // worry about changing hyperparameter values in candidate pipelines
            // possibly overwritting other pipelines.
            Transforms = transforms.Select(t => t.Clone()).ToArray();
            Learner = learner.Clone();
            LoaderSettings = loaderSettings;
            _env = env;
            PerformanceSummary = summary;
            UniqueId = Guid.NewGuid();
        }

        /// <summary>
        /// Constructs an entrypoint graph from the current pipeline.
        /// </summary>
        public AutoInference.EntryPointGraphDef ToEntryPointGraph(Experiment experiment = null)
        {
            _env.CheckValue(Learner.PipelineNode, nameof(Learner.PipelineNode));
            var subGraph = experiment ?? _env.CreateExperiment();

            // Insert first node
            Var<IDataView> lastOutput = new Var<IDataView>();

            // Chain transforms
            var transformsModels = new List<Var<ITransformModel>>();
            var viableTransforms = Transforms.ToList().Where(transform => transform.PipelineNode != null);
            foreach (var transform in viableTransforms)
            {
                transform.PipelineNode.SetInputData(lastOutput);
                var returnedDataAndModel1 = transform.PipelineNode.Add(subGraph);
                transformsModels.Add(returnedDataAndModel1.Model);
                lastOutput = returnedDataAndModel1.OutData;
            }

            // Add learner
            Learner.PipelineNode?.SetInputData(lastOutput);
            var returnedDataAndModel2 = Learner.PipelineNode?.Add(subGraph);

            // Create single model for featurizing and scoring data,
            // if transforms present.
            if (Transforms.Length > 0)
            {
                var modelCombine = new ML.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(transformsModels.ToArray()),
                    PredictorModel = returnedDataAndModel2?.Model
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                return new AutoInference.EntryPointGraphDef(subGraph, modelCombineOutput.PredictorModel, lastOutput);
            }

            // No transforms present, so just return predictor's model.
            return new AutoInference.EntryPointGraphDef(subGraph, returnedDataAndModel2?.Model, lastOutput);
        }

        /// <summary>
        /// This method will return some indentifying string for the pipeline,
        /// based on transforms, learner, and (eventually) hyperparameters.
        /// </summary>
        public override string ToString() => $"{Learner}+{string.Join("+", Transforms.Select(t => t.ToString()))}";

        public bool Equals(PipelinePattern obj) => obj != null && UniqueId == obj.UniqueId;

        // REVIEW: We may want to allow for sweeping with CV in the future, so we will need to add new methods like this, or refactor these in that case.
        public Experiment CreateTrainTestExperiment(IDataView trainData, IDataView testData, MacroUtils.TrainerKinds trainerKind,
                bool includeTrainingMetrics, out Models.TrainTestEvaluator.Output resultsOutput)
        {
            var graphDef = ToEntryPointGraph();
            var subGraph = graphDef.Graph;
            var nodes = graphDef.Graph.GetNodes();

            _env.CheckNonEmpty(nodes, nameof(nodes), "Empty Subgraph on TrainTest Experiment.");

            Var<IDataView> firstInput = new Var<IDataView> { VarName = graphDef.GetSubgraphFirstNodeDataVarName(_env) };
            var finalOutput = graphDef.ModelOutput;

            // TrainTestMacro
            var trainTestInput = new Models.TrainTestEvaluator
            {
                TransformModel = null,
                Nodes = subGraph,
                Inputs =
                    {
                        Data = firstInput
                    },
                Outputs =
                    {
                        PredictorModel = finalOutput
                    },
                PipelineId = UniqueId.ToString("N"),
                Kind = MacroUtils.TrainerKindApiValue<Models.MacroUtilsTrainerKinds>(trainerKind),
                IncludeTrainingMetrics = includeTrainingMetrics
            };

            var experiment = _env.CreateExperiment();
            var trainTestOutput = experiment.Add(trainTestInput);

            experiment.Compile();
            experiment.SetInput(trainTestInput.TrainingData, trainData);
            experiment.SetInput(trainTestInput.TestingData, testData);
            resultsOutput = trainTestOutput;
            return experiment;
        }

        public Models.TrainTestEvaluator.Output AddAsTrainTest(Var<IDataView> trainData, Var<IDataView> testData,
            MacroUtils.TrainerKinds trainerKind, Experiment experiment = null, bool includeTrainingMetrics = false)
        {
            experiment = experiment ?? _env.CreateExperiment();
            var graphDef = ToEntryPointGraph(experiment);
            var subGraph = graphDef.Graph;
            var firstInput = new Var<IDataView> { VarName = graphDef.GetSubgraphFirstNodeDataVarName(_env) };
            var finalOutput = graphDef.ModelOutput;

            // TrainTestMacro
            var trainTestInput = new Models.TrainTestEvaluator
            {
                Nodes = subGraph,
                TransformModel = null,
                Inputs =
                    {
                        Data = firstInput
                    },
                Outputs =
                    {
                        PredictorModel = finalOutput
                    },
                TrainingData = trainData,
                TestingData = testData,
                Kind = MacroUtils.TrainerKindApiValue<Models.MacroUtilsTrainerKinds>(trainerKind),
                PipelineId = UniqueId.ToString("N"),
                IncludeTrainingMetrics = includeTrainingMetrics
            };
            var trainTestOutput = experiment.Add(trainTestInput);
            return trainTestOutput;
        }

        /// <summary>
        /// Runs a train-test experiment on the current pipeline, through entrypoints.
        /// </summary>
        public void RunTrainTestExperiment(IDataView trainData, IDataView testData,
            SupportedMetric metric, MacroUtils.TrainerKinds trainerKind, out double testMetricValue,
            out double trainMetricValue)
        {
            var experiment = CreateTrainTestExperiment(trainData, testData, trainerKind, true, out var trainTestOutput);
            experiment.Run();

            var dataOut = experiment.GetOutput(trainTestOutput.OverallMetrics);
            var dataOutTraining = experiment.GetOutput(trainTestOutput.TrainingOverallMetrics);
            testMetricValue = AutoMlUtils.ExtractValueFromIdv(_env, dataOut, metric.Name);
            trainMetricValue = AutoMlUtils.ExtractValueFromIdv(_env, dataOutTraining, metric.Name);
        }

        public static PipelineResultRow[] ExtractResults(IHostEnvironment env, IDataView data,
            string graphColName, string metricColName, string idColName, string trainingMetricColName,
            string firstInputColName, string predictorModelColName)
        {
            var results = new List<PipelineResultRow>();
            var schema = data.Schema;
            if (!schema.TryGetColumnIndex(graphColName, out var graphCol))
                throw env.ExceptParam(nameof(graphColName), $"Column name {graphColName} not found");
            if (!schema.TryGetColumnIndex(metricColName, out var metricCol))
                throw env.ExceptParam(nameof(metricColName), $"Column name {metricColName} not found");
            if (!schema.TryGetColumnIndex(trainingMetricColName, out var trainingMetricCol))
                throw env.ExceptParam(nameof(trainingMetricColName), $"Column name {trainingMetricColName} not found");
            if (!schema.TryGetColumnIndex(idColName, out var pipelineIdCol))
                throw env.ExceptParam(nameof(idColName), $"Column name {idColName} not found");
            if (!schema.TryGetColumnIndex(firstInputColName, out var firstInputCol))
                throw env.ExceptParam(nameof(firstInputColName), $"Column name {firstInputColName} not found");
            if (!schema.TryGetColumnIndex(predictorModelColName, out var predictorModelCol))
                throw env.ExceptParam(nameof(predictorModelColName), $"Column name {predictorModelColName} not found");

            using (var cursor = data.GetRowCursor(col => true))
            {
                var getter1 = cursor.GetGetter<double>(metricCol);
                var getter2 = cursor.GetGetter<DvText>(graphCol);
                var getter3 = cursor.GetGetter<DvText>(pipelineIdCol);
                var getter4 = cursor.GetGetter<double>(trainingMetricCol);
                var getter5 = cursor.GetGetter<DvText>(firstInputCol);
                var getter6 = cursor.GetGetter<DvText>(predictorModelCol);
                double metricValue = 0;
                double trainingMetricValue = 0;
                DvText graphJson = new DvText();
                DvText pipelineId = new DvText();
                DvText firstInput = new DvText();
                DvText predictorModel = new DvText();

                while (cursor.MoveNext())
                {
                    getter1(ref metricValue);
                    getter2(ref graphJson);
                    getter3(ref pipelineId);
                    getter4(ref trainingMetricValue);
                    getter5(ref firstInput);
                    getter6(ref predictorModel);

                    results.Add(new PipelineResultRow(graphJson.ToString(),
                        metricValue, pipelineId.ToString(), trainingMetricValue,
                        firstInput.ToString(), predictorModel.ToString()));
                }
            }

            return results.ToArray();
        }

        public PipelineResultRow ToResultRow()
        {
            var graphDef = ToEntryPointGraph();

            return new PipelineResultRow($"{{\"Nodes\" : [{graphDef.Graph.ToJsonString()}]}}",
                PerformanceSummary?.MetricValue ?? -1d, UniqueId.ToString("N"),
                PerformanceSummary?.TrainingMetricValue ?? -1d,
                graphDef.GetSubgraphFirstNodeDataVarName(_env),
                graphDef.ModelOutput.VarName);
        }
    }
}
