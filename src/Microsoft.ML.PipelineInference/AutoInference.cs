// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Newtonsoft.Json.Linq;

[assembly: EntryPointModule(typeof(AutoInference.AutoMlMlState.Arguments))]
[assembly: EntryPointModule(typeof(AutoInference.ISupportAutoMlStateFactory))]

namespace Microsoft.ML.Runtime.PipelineInference
{
    /// <summary>
    /// Class for generating potential recipes/pipelines, testing them, and zeroing in on the best ones.
    /// For now, only works with maximizing metrics (AUC, Accuracy, etc.).
    /// </summary>
    public class AutoInference
    {
        public struct ColumnInfo
        {
            public string Name { get; set; }
            public ColumnType ItemType { get; set; }
            public bool IsHidden { get; set; }
            public override string ToString() => Name;
        }

        private sealed class ReversedComparer<T> : IComparer<T>
        {
            public int Compare(T x, T y)
            {
                return Comparer<T>.Default.Compare(y, x);
            }
        }

        /// <summary>
        /// Alias to refer to this construct by an easy name.
        /// </summary>
        public class LevelDependencyMap : Dictionary<ColumnInfo, List<TransformInference.SuggestedTransform>> { }

        /// <summary>
        /// Alias to refer to this construct by an easy name.
        /// </summary>
        public class DependencyMap : Dictionary<int, LevelDependencyMap> { }

        /// <summary>
        /// Class for encapsulating an entrypoint experiment graph
        /// and keeping track of the input and output nodes.
        /// </summary>
        public class EntryPointGraphDef
        {
            public Experiment Graph { get; }
            public Var<IPredictorModel> ModelOutput { get; }

            /// <summary>
            /// Get the name of the variable asssigned to the Data or Training Data input, based on what is the first node of the subgraph.
            /// A better way to do this would be with a ICanBeSubGraphFirstNode common interface between ITransformInput and ITrainerInputs
            /// and a custom deserializer.
            /// </summary>
            public string GetSubgraphFirstNodeDataVarName(IExceptionContext ectx)
            {
                var nodes = Graph.GetNodes();

                ectx.Check(nodes != null || nodes.Count == 0, "Empty Subgraph");
                ectx.Check(nodes[0] != null, "Subgraph's first note is empty");
                ectx.Check(nodes[0][FieldNames.Inputs] != null, "Empty subgraph node inputs.");

                string variableName;
                if (!GetDataVariableName(ectx, "Data", nodes[0][FieldNames.Inputs], out variableName))
                    GetDataVariableName(ectx, "TrainingData", nodes[0][FieldNames.Inputs], out variableName);

                ectx.CheckNonEmpty(variableName, nameof(variableName), "Subgraph needs to start with an" +
                    nameof(CommonInputs.ITransformInput) + ", or an " + nameof(CommonInputs.ITrainerInput) +
                    ". Check your subgraph, or account for variation of the name of the Data input here.");
                return variableName;
            }

            public Var<IDataView> TransformsOutputData { get; }

            public EntryPointGraphDef(Experiment experiment, Var<IPredictorModel> model, Var<IDataView> transformsOutputData)
            {
                Graph = experiment;
                ModelOutput = model;
                TransformsOutputData = transformsOutputData;
            }

            private bool GetDataVariableName(IExceptionContext ectx, string nameOfData, JToken firstNodeInputs, out string variableName)
            {
                variableName = null;

                if (firstNodeInputs[nameOfData] == null)
                    return false;

                string dataVar = firstNodeInputs.Value<String>(nameOfData);
                if (!VariableBinding.IsValidVariableName(ectx, dataVar))
                    throw ectx.ExceptParam(nameof(nameOfData), $"Invalid variable name {dataVar}.");

                variableName = dataVar.Substring(1);
                return true;
            }
        }

        [TlcModule.ComponentKind("AutoMlStateBase")]
        public interface ISupportAutoMlStateFactory : IComponentFactory<IMlState>
        { }

        /// <summary>
        /// Class that holds state for an autoML search-in-progress. Should be able to resume search, given this object.
        /// </summary>
        public sealed class AutoMlMlState : IMlState
        {
            private readonly SortedList<double, PipelinePattern> _sortedSampledElements;
            private readonly List<PipelinePattern> _history;
            private readonly IHostEnvironment _env;
            private readonly IHost _host;
            private IDataView _trainData;
            private IDataView _testData;
            private IDataView _transformedData;
            private ITerminator _terminator;
            private string[] _requestedLearners;
            private TransformInference.SuggestedTransform[] _availableTransforms;
            private RecipeInference.SuggestedRecipe.SuggestedLearner[] _availableLearners;
            private DependencyMap _dependencyMapping;
            private RoleMappedData _dataRoles;
            public IPipelineOptimizer AutoMlEngine { get; set; }
            public PipelinePattern[] BatchCandidates { get; set; }
            public SupportedMetric Metric { get; }
            public MacroUtils.TrainerKinds TrainerKind { get; }

            [TlcModule.Component(Name = "AutoMlState", FriendlyName = "AutoML State", Alias = "automlst",
                Desc = "State of an AutoML search and search space.")]
            public sealed class Arguments : ISupportAutoMlStateFactory
            {
                [Argument(ArgumentType.Required, HelpText = "Supported metric for evaluator.", ShortName = "metric")]
                public PipelineSweeperSupportedMetrics.Metrics Metric;

                [Argument(ArgumentType.Required, HelpText = "AutoML engine (pipeline optimizer) that generates next candidates.", ShortName = "engine")]
                public ISupportIPipelineOptimizerFactory Engine;

                [Argument(ArgumentType.Required, HelpText = "Kind of trainer for task, such as binary classification trainer, multiclass trainer, etc.", ShortName = "tk")]
                public MacroUtils.TrainerKinds TrainerKind;

                [Argument(ArgumentType.Required, HelpText = "Arguments for creating terminator, which determines when to stop search.", ShortName = "term")]
                public ISupportITerminatorFactory TerminatorArgs;

                [Argument(ArgumentType.AtMostOnce, HelpText = "Learner set to sweep over (if available).", ShortName = "learners")]
                public string[] RequestedLearners;

                public IMlState CreateComponent(IHostEnvironment env) => new AutoMlMlState(env, this);
            }

            public AutoMlMlState(IHostEnvironment env, Arguments args)
                : this(env,
                      PipelineSweeperSupportedMetrics.GetSupportedMetric(args.Metric),
                      args.Engine.CreateComponent(env),
                      args.TerminatorArgs.CreateComponent(env), args.TrainerKind, requestedLearners: args.RequestedLearners)
            {
            }

            public AutoMlMlState(IHostEnvironment env, SupportedMetric metric, IPipelineOptimizer autoMlEngine,
                ITerminator terminator, MacroUtils.TrainerKinds trainerKind, IDataView trainData = null, IDataView testData = null,
                string[] requestedLearners = null)
            {
                Contracts.CheckValue(env, nameof(env));
                _sortedSampledElements =
                    metric.IsMaximizing ? new SortedList<double, PipelinePattern>(new ReversedComparer<double>()) :
                        new SortedList<double, PipelinePattern>();
                _history = new List<PipelinePattern>();
                _env = env;
                _host = _env.Register("AutoMlState");
                _trainData = trainData;
                _testData = testData;
                _terminator = terminator;
                _requestedLearners = requestedLearners;
                AutoMlEngine = autoMlEngine;
                BatchCandidates = new PipelinePattern[] { };
                Metric = metric;
                TrainerKind = trainerKind;
            }

            public void SetTrainTestData(IDataView trainData, IDataView testData)
            {
                _trainData = trainData;
                _testData = testData;
            }

            private void MainLearningLoop(int batchSize, int numOfTrainingRows)
            {
                var stopwatch = new Stopwatch();
                var probabilityUtils = new Sweeper.Algorithms.SweeperProbabilityUtils(_host);

                while (!_terminator.ShouldTerminate(_history))
                {
                    // Get next set of candidates
                    var currentBatchSize = batchSize;
                    if (_terminator is IterationTerminator itr)
                        currentBatchSize = Math.Min(itr.RemainingIterations(_history), batchSize);
                    var candidates = AutoMlEngine.GetNextCandidates(_sortedSampledElements.Values, currentBatchSize, _dataRoles);

                    // Break if no candidates returned, means no valid pipeline available.
                    if (candidates.Length == 0)
                        break;

                    // Evaluate them on subset of data
                    foreach (var candidate in candidates)
                    {
                        try
                        {
                            ProcessPipeline(probabilityUtils, stopwatch, candidate, numOfTrainingRows);
                        }
                        catch (Exception)
                        {
                            stopwatch.Stop();
                            return;
                        }
                    }
                }
            }

            private void ProcessPipeline(Sweeper.Algorithms.SweeperProbabilityUtils utils, Stopwatch stopwatch, PipelinePattern candidate, int numOfTrainingRows)
            {
                // Create a randomized numer of rows to do train/test with.
                int randomizedNumberOfRows =
                    (int)Math.Floor(utils.NormalRVs(1, numOfTrainingRows, (double)numOfTrainingRows / 10).First());
                if (randomizedNumberOfRows > numOfTrainingRows)
                    randomizedNumberOfRows = numOfTrainingRows - (randomizedNumberOfRows - numOfTrainingRows);

                // Run pipeline, and time how long it takes
                stopwatch.Restart();
                candidate.RunTrainTestExperiment(_trainData.Take(randomizedNumberOfRows),
                    _testData, Metric, TrainerKind, out var testMetricVal, out var trainMetricVal);
                stopwatch.Stop();

                // Handle key collisions on sorted list
                while (_sortedSampledElements.ContainsKey(testMetricVal))
                    testMetricVal += 1e-10;

                // Save performance score
                candidate.PerformanceSummary = new PipelineSweeperRunSummary(testMetricVal, randomizedNumberOfRows, stopwatch.ElapsedMilliseconds, trainMetricVal);
                _sortedSampledElements.Add(candidate.PerformanceSummary.MetricValue, candidate);
                _history.Add(candidate);
            }

            public void UpdateTerminator(ITerminator terminator)
            {
                if (terminator != null)
                    _terminator = terminator;
            }

            private TransformInference.SuggestedTransform[] InferAndFilter(IDataView data, TransformInference.Arguments args,
                TransformInference.SuggestedTransform[] existingTransforms = null)
            {
                // Infer transforms using experts
                var levelTransforms = TransformInference.InferTransforms(_env, data, args, _dataRoles);

                // Retain only those transforms inferred which were also passed in.
                if (existingTransforms != null)
                    return levelTransforms.Where(t => existingTransforms.Any(t2 => t2.Equals(t))).ToArray();
                return levelTransforms;
            }

            public void InferSearchSpace(int numTransformLevels, RoleMappedData dataRoles = null)
            {
                var learners = RecipeInference.AllowedLearners(_env, TrainerKind).ToArray();
                if (_requestedLearners != null && _requestedLearners.Length > 0)
                    learners = learners.Where(l => _requestedLearners.Contains(l.LearnerName)).ToArray();

                _dataRoles = dataRoles;
                ComputeSearchSpace(numTransformLevels, learners, (b, c) => InferAndFilter(b, c));
            }

            public void UpdateSearchSpace(RecipeInference.SuggestedRecipe.SuggestedLearner[] learners,
                TransformInference.SuggestedTransform[] transforms)
            {
                _env.Check(learners != null);
                _env.Check(transforms != null);
                _env.Check(transforms.Length > 0 && learners.Length > 0);
                int numTransformLevels = transforms.Max(t => t.RoutingStructure.Level);
                ComputeSearchSpace(numTransformLevels, learners, (b, c) => InferAndFilter(b, c, transforms));
            }

            public Tuple<TransformInference.SuggestedTransform[], RecipeInference.SuggestedRecipe.SuggestedLearner[]> GetSearchSpace()
            {
                return new Tuple<TransformInference.SuggestedTransform[], RecipeInference.SuggestedRecipe.SuggestedLearner[]>(
                    _availableTransforms.ToArray(), _availableLearners.ToArray());
            }

            public PipelinePattern InferPipelines(int numTransformLevels, int batchSize, int numOfTrainingRows)
            {
                _env.AssertValue(_trainData, nameof(_trainData), "Must set training data prior to calling method.");
                _env.AssertValue(_testData, nameof(_testData), "Must set test data prior to calling method.");

                var h = _env.Register("InferPipelines");
                using (var ch = h.Start("InferPipelines"))
                {
                    // Check if search space has not been initialized. If not,
                    // run method to define it usign inference.
                    if (!IsSearchSpaceDefined())
                        InferSearchSpace(numTransformLevels);

                    // Learn for a given number of iterations
                    MainLearningLoop(batchSize, numOfTrainingRows);

                    // Return best pipeline seen
                    ch.Done();
                    return _sortedSampledElements.Count > 0 ? _sortedSampledElements.First().Value : null;
                }
            }

            private bool IsValidLearnerSet(RecipeInference.SuggestedRecipe.SuggestedLearner[] learners)
            {
                var inferredLearners = RecipeInference.AllowedLearners(_env, TrainerKind);
                return learners.All(l => inferredLearners.Any(i => i.LearnerName == l.LearnerName));
            }

            public void KeepSelectedLearners(IEnumerable<string> learnersToKeep)
            {
                var allLearners = RecipeInference.AllowedLearners(_env, TrainerKind);
                _env.AssertNonEmpty(allLearners);
                _availableLearners = allLearners.Where(l => learnersToKeep.Contains(l.LearnerName)).ToArray();
                AutoMlEngine.UpdateLearners(_availableLearners);
            }

            /// <summary>
            /// Search space is transforms X learners X hyperparameters.
            /// </summary>
            private void ComputeSearchSpace(int numTransformLevels, RecipeInference.SuggestedRecipe.SuggestedLearner[] learners,
                Func<IDataView, TransformInference.Arguments, TransformInference.SuggestedTransform[]> transformInferenceFunction)
            {
                _env.AssertValue(_trainData, nameof(_trainData), "Must set training data prior to inferring search space.");

                var h = _env.Register("ComputeSearchSpace");

                using (var ch = h.Start("ComputeSearchSpace"))
                {
                    _env.Check(IsValidLearnerSet(learners), "Unsupported learner encountered, cannot update search space.");

                    var dataSample = _trainData;
                    var inferenceArgs = new TransformInference.Arguments
                    {
                        EstimatedSampleFraction = 1.0,
                        ExcludeFeaturesConcatTransforms = true
                    };

                    // Initialize structure for mapping columns back to specific transforms
                    var dependencyMapping = new DependencyMap
                    {
                        {0, AutoMlUtils.ComputeColumnResponsibilities(dataSample, new TransformInference.SuggestedTransform[0])}
                    };

                    // Get suggested transforms for all levels. Defines another part of search space.
                    var transformsList = new List<TransformInference.SuggestedTransform>();
                    for (int i = 0; i < numTransformLevels; i++)
                    {
                        // Update level for transforms
                        inferenceArgs.Level = i + 1;

                        // Infer transforms using experts
                        var levelTransforms = transformInferenceFunction(dataSample, inferenceArgs);

                        // If no more transforms to apply, dataSample won't change. So end loop.
                        if (levelTransforms.Length == 0)
                            break;

                        // Make sure we don't overflow our bitmask
                        if (levelTransforms.Max(t => t.AtomicGroupId) > 64)
                            break;

                        // Level-up atomic group id offset.
                        inferenceArgs.AtomicIdOffset = levelTransforms.Max(t => t.AtomicGroupId) + 1;

                        // Apply transforms to dataview for this level.
                        dataSample = AutoMlUtils.ApplyTransformSet(_env, dataSample, levelTransforms);

                        // Keep list of which transforms can be responsible for which output columns
                        dependencyMapping.Add(inferenceArgs.Level,
                            AutoMlUtils.ComputeColumnResponsibilities(dataSample, levelTransforms));
                        transformsList.AddRange(levelTransforms);
                    }

                    var transforms = transformsList.ToArray();
                    Func<PipelinePattern, long, bool> verifier = AutoMlUtils.ValidationWrapper(transforms, dependencyMapping);

                    // Save state, for resuming learning
                    _availableTransforms = transforms;
                    _availableLearners = learners;
                    _dependencyMapping = dependencyMapping;
                    _transformedData = dataSample;

                    // Update autoML engine to know what the search space looks like
                    AutoMlEngine.SetSpace(_availableTransforms, _availableLearners, verifier,
                        _trainData, _transformedData, _dependencyMapping, Metric.IsMaximizing);

                    ch.Done();
                }
            }

            public void AddEvaluated(PipelinePattern pipeline)
            {
                if (pipeline.PerformanceSummary == null)
                    throw new Exception("Candidate pipeline missing run summary.");
                var d = pipeline.PerformanceSummary.MetricValue;
                while (_sortedSampledElements.ContainsKey(d))
                    d += 1e-3;
                _sortedSampledElements.Add(d, pipeline);
                _history.Add(pipeline);

                using (var ch = _host.Start("Suggested Pipeline"))
                {
                    ch.Info($"PipelineSweeper Iteration Number : {_history.Count}");
                    ch.Info($"PipelineSweeper Pipeline Id : {pipeline.UniqueId}");

                    foreach (var transform in pipeline.Transforms)
                    {
                        ch.Info($"PipelineSweeper Transform : {transform.Transform}");
                    }

                    ch.Info($"PipelineSweeper Learner : {pipeline.Learner}");
                    ch.Info($"PipelineSweeper Train Metric Value : {pipeline.PerformanceSummary.TrainingMetricValue}");
                    ch.Info($"PipelineSweeper Test Metric Value : {pipeline.PerformanceSummary.MetricValue}");
                }
            }

            public void AddEvaluated(PipelinePattern[] pipelines)
            {
                foreach (var pipeline in pipelines)
                    AddEvaluated(pipeline);
            }

            public PipelinePattern[] GetNextCandidates(int numberOfCandidates)
            {
                if (_terminator.ShouldTerminate(_history))
                    return new PipelinePattern[] { };
                var currentBatchSize = numberOfCandidates;
                if (_terminator is IterationTerminator itr)
                    currentBatchSize = Math.Min(itr.RemainingIterations(_history), numberOfCandidates);
                BatchCandidates = AutoMlEngine.GetNextCandidates(_sortedSampledElements.Select(kvp => kvp.Value), currentBatchSize, _dataRoles);

                return BatchCandidates;
            }

            public PipelinePattern[] GetAllEvaluatedPipelines() =>
                _sortedSampledElements.Where(kvp => kvp.Value.PerformanceSummary != null).Select(p => p.Value).ToArray();

            public PipelinePattern GetBestPipeline() => _sortedSampledElements.Values[0];

            public void ClearEvaluatedPipelines()
            {
                _sortedSampledElements.Clear();
                BatchCandidates = new PipelinePattern[0];
            }

            public bool IsSearchSpaceDefined() => _availableLearners != null && _availableTransforms != null;
        }

        /// <summary>
        /// The InferPipelines methods are just public portals to the internal function that handle different
        /// types of data being passed in: training IDataView, path to training file, or train and test files.
        /// </summary>
        public static AutoMlMlState InferPipelines(IHostEnvironment env, PipelineOptimizerBase autoMlEngine,
            IDataView trainData, IDataView testData, int numTransformLevels, int batchSize, SupportedMetric metric,
            out PipelinePattern bestPipeline, ITerminator terminator, MacroUtils.TrainerKinds trainerKind)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(trainData, nameof(trainData));
            env.CheckValue(testData, nameof(testData));

            int numOfRows = (int)(trainData.GetRowCount(false) ?? 1000);
            AutoMlMlState amls = new AutoMlMlState(env, metric, autoMlEngine, terminator, trainerKind, trainData, testData);
            bestPipeline = amls.InferPipelines(numTransformLevels, batchSize, numOfRows);
            return amls;
        }

        public static AutoMlMlState InferPipelines(IHostEnvironment env, PipelineOptimizerBase autoMlEngine, string trainDataPath,
            string schemaDefinitionFile, out string schemaDefinition, int numTransformLevels, int batchSize, SupportedMetric metric,
            out PipelinePattern bestPipeline, int numOfSampleRows, ITerminator terminator, MacroUtils.TrainerKinds trainerKind)
        {
            Contracts.CheckValue(env, nameof(env));

            // REVIEW: Should be able to infer schema by itself, without having to
            // infer recipes. Look into this.
            // Set loader settings through inference
            RecipeInference.InferRecipesFromData(env, trainDataPath, schemaDefinitionFile,
                out var _, out schemaDefinition, out var _, true);

#pragma warning disable 0618
            var data = ImportTextData.ImportText(env, new ImportTextData.Input
            {
                InputFile = new SimpleFileHandle(env, trainDataPath, false, false),
                CustomSchema = schemaDefinition
            }).Data;
#pragma warning restore 0618
            var splitOutput = TrainTestSplit.Split(env, new TrainTestSplit.Input { Data = data, Fraction = 0.8f });
            AutoMlMlState amls = new AutoMlMlState(env, metric, autoMlEngine, terminator, trainerKind,
                splitOutput.TrainData.Take(numOfSampleRows), splitOutput.TestData.Take(numOfSampleRows));
            bestPipeline = amls.InferPipelines(numTransformLevels, batchSize, numOfSampleRows);
            return amls;
        }

        public static AutoMlMlState InferPipelines(IHostEnvironment env, PipelineOptimizerBase autoMlEngine, IDataView data, int numTransformLevels,
            int batchSize, SupportedMetric metric, out PipelinePattern bestPipeline, int numOfSampleRows,
            ITerminator terminator, MacroUtils.TrainerKinds trainerKind)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(data, nameof(data));

            var splitOutput = TrainTestSplit.Split(env, new TrainTestSplit.Input { Data = data, Fraction = 0.8f });
            AutoMlMlState amls = new AutoMlMlState(env, metric, autoMlEngine, terminator, trainerKind,
                splitOutput.TrainData.Take(numOfSampleRows), splitOutput.TestData.Take(numOfSampleRows));
            bestPipeline = amls.InferPipelines(numTransformLevels, batchSize, numOfSampleRows);
            return amls;
        }
    }
}
