// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.PipelineInference;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class TestAutoInference : BaseTestBaseline
    {
        public TestAutoInference(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        [TestCategory("EntryPoints")]
        public void TestLearn()
        {
            using (var env = new TlcEnvironment())
            {
                string pathData = GetDataPath("adult.train");
                string pathDataTest = GetDataPath("adult.test");
                int numOfSampleRows = 1000;
                int batchSize = 5;
                int numIterations = 10;
                int numTransformLevels = 3;
                AutoInference.SupportedMetric metric = AutoInference.SupportedMetric.Auc;

                // Using the simple, uniform random sampling (with replacement) engine
                PipelineOptimizerBase autoMlEngine = new UniformRandomEngine(env);

                // Test initial learning
                var amls = AutoInference.InferPipelines(env, autoMlEngine, pathData, "", out var schema, numTransformLevels, batchSize,
                    metric, out var bestPipeline, numOfSampleRows, new IterationTerminator(numIterations / 2), MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer);
                env.Check(amls.GetAllEvaluatedPipelines().Length == numIterations / 2);

                // Resume learning
                amls.UpdateTerminator(new IterationTerminator(numIterations));
                bestPipeline = amls.InferPipelines(numTransformLevels, batchSize, numOfSampleRows);
                env.Check(amls.GetAllEvaluatedPipelines().Length == numIterations);

                // Use best pipeline for another task
                var inputFileTrain = new SimpleFileHandle(env, pathData, false, false);
#pragma warning disable 0618
                var datasetTrain = ImportTextData.ImportText(env,
                    new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data;
                var inputFileTest = new SimpleFileHandle(env, pathDataTest, false, false);
                var datasetTest = ImportTextData.ImportText(env,
                    new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data;
#pragma warning restore 0618

                // REVIEW: Theoretically, it could be the case that a new, very bad learner is introduced and 
                // we get unlucky and only select it every time, such that this test fails. Not
                // likely at all, but a non-zero probability. Should be ok, since all current learners are returning d > .80.
                bestPipeline.RunTrainTestExperiment(datasetTrain, datasetTest, metric, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer,
                    out var testMetricValue, out var trainMtericValue);
                env.Check(testMetricValue > 0.2);
            }
            Done();
        }

        [Fact]
        [TestCategory("EntryPoints")]
        public void TestPipelineSweeperMacroNoTransforms()
        {
            // Set up inputs for experiment
            string pathData = GetDataPath("adult.train");
            string pathDataTest = GetDataPath("adult.test");
            const int numOfSampleRows = 1000;
            const string schema = "sep=, col=Features:R4:0,2,4,10-12 col=Label:R4:14 header=+";

            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows);
#pragma warning restore 0618
            const int batchSize = 5;
            const int numIterations = 20;
            const int numTransformLevels = 2;
            AutoInference.SupportedMetric metric = AutoInference.SupportedMetric.Auc;

            // Using the simple, uniform random sampling (with replacement) engine
            PipelineOptimizerBase autoMlEngine = new UniformRandomEngine(Env);

            // Create search object
            var amls = new AutoInference.AutoMlMlState(Env, metric, autoMlEngine, new IterationTerminator(numIterations),
                MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer, datasetTrain, datasetTest);

            // Infer search space
            amls.InferSearchSpace(numTransformLevels);

            // Create macro object
            var pipelineSweepInput = new Microsoft.ML.Models.PipelineSweeper()
            {
                BatchSize = batchSize,
            };

            var exp = new Experiment(Env);
            var output = exp.Add(pipelineSweepInput);
            exp.Compile();
            exp.SetInput(pipelineSweepInput.TrainingData, datasetTrain);
            exp.SetInput(pipelineSweepInput.TestingData, datasetTest);
            exp.SetInput(pipelineSweepInput.State, amls);
            exp.SetInput(pipelineSweepInput.CandidateOutputs, new IDataView[0]);
            exp.Run();

            // Make sure you get back an AutoMlState, and that it ran for correct number of iterations
            // with at least minimal performance values (i.e., best should have AUC better than 0.1 on this dataset).
            AutoInference.AutoMlMlState amlsOut = (AutoInference.AutoMlMlState)exp.GetOutput(output.State);
            Assert.NotNull(amlsOut);
            Assert.Equal(amlsOut.GetAllEvaluatedPipelines().Length, numIterations);
            Assert.True(amlsOut.GetBestPipeline().PerformanceSummary.MetricValue > 0.1);
        }

        [Fact]
        [TestCategory("EntryPoints")]
        public void EntryPointPipelineSweepSerialization()
        {
            // Get datasets
            var pathData = GetDataPath("adult.train");
            var pathDataTest = GetDataPath("adult.test");
            const int numOfSampleRows = 1000;
            int numIterations = 10;
            const string schema =
                "sep=, col=Features:R4:0,2,4,10-12 col=workclass:TX:1 col=education:TX:3 col=marital_status:TX:5 col=occupation:TX:6 " +
                "col=relationship:TX:7 col=ethnicity:TX:8 col=sex:TX:9 col=native_country:TX:13 col=label_IsOver50K_:R4:14 header=+";
            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows);
#pragma warning restore 0618

            // Define entrypoint graph
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Models.PipelineSweeper',
                      'Inputs': {
                        'TrainingData': '$TrainingData',
                        'TestingData': '$TestingData',
                        'StateArguments': {
                            'Name': 'AutoMlState',
                            'Settings': {
                                'Metric': 'Auc',
                                'Engine': {
                                    'Name': 'UniformRandom'
                                },
                                'TerminatorArgs': {
                                    'Name': 'IterationLimited',
                                    'Settings': {
                                        'FinalHistoryLength': 10
                                    }
                                },
                                'TrainerKind': 'SignatureBinaryClassifierTrainer'
                            }
                        },
                        'BatchSize': 5
                      },
                      'Outputs': {
                        'State': '$StateOut',
                        'Results': '$ResultsOut'
                      }
                    },
                  ]
                }";

            JObject graphJson = JObject.Parse(inputGraph);
            var catalog = ModuleCatalog.CreateInstance(Env);
            var graph = new EntryPointGraph(Env, catalog, graphJson[FieldNames.Nodes] as JArray);
            // Test if ToJson() works properly.
            var nodes = new JArray(graph.AllNodes.Select(node => node.ToJson()));
            var runner = new GraphRunner(Env, catalog, nodes);
            runner.SetInput("TrainingData", datasetTrain);
            runner.SetInput("TestingData", datasetTest);
            runner.RunAll();

            var results = runner.GetOutput<IDataView>("ResultsOut");
            Assert.NotNull(results);
            var rows = PipelinePattern.ExtractResults(Env, results,
                "Graph", "MetricValue", "PipelineId", "TrainingMetricValue", "FirstInput", "PredictorModel");
            Assert.True(rows.Length == numIterations);
        }

        [Fact]
        public void EntryPointPipelineSweep()
        {
            // Get datasets
            var pathData = GetDataPath("adult.tiny.with-schema.txt");
            var pathDataTest = GetDataPath("adult.tiny.with-schema.txt");
            const int numOfSampleRows = 1000;
            int numIterations = 4;
            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTrain }).Data.Take(numOfSampleRows);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest }).Data.Take(numOfSampleRows);
#pragma warning restore 0618
            // Define entrypoint graph
            string inputGraph = @"
                {
                  'Nodes': [                                
                    {
                      'Name': 'Models.PipelineSweeper',
                      'Inputs': {
                        'TrainingData': '$TrainingData',
                        'TestingData': '$TestingData',
                        'StateArguments': {
                            'Name': 'AutoMlState',
                            'Settings': {
                                'Metric': 'Auc',
                                'Engine': {
                                    'Name': 'UniformRandom'
                                },
                                'TerminatorArgs': {
                                    'Name': 'IterationLimited',
                                    'Settings': {
                                        'FinalHistoryLength': 4
                                    }
                                },
                                'TrainerKind': 'SignatureBinaryClassifierTrainer'
                            }
                        },
                        'BatchSize': 2
                      },
                      'Outputs': {
                        'State': '$StateOut',
                        'Results': '$ResultsOut'
                      }
                    },
                  ]
                }";

            JObject graph = JObject.Parse(inputGraph);
            var catalog = ModuleCatalog.CreateInstance(Env);

            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);
            runner.SetInput("TrainingData", datasetTrain);
            runner.SetInput("TestingData", datasetTest);
            runner.RunAll();

            var autoMlState = runner.GetOutput<AutoInference.AutoMlMlState>("StateOut");
            Assert.NotNull(autoMlState);
            var allPipelines = autoMlState.GetAllEvaluatedPipelines();
            var bestPipeline = autoMlState.GetBestPipeline();
            Assert.Equal(allPipelines.Length, numIterations);
            Assert.True(bestPipeline.PerformanceSummary.MetricValue > 0.1);

            var results = runner.GetOutput<IDataView>("ResultsOut");
            Assert.NotNull(results);
            var rows = PipelinePattern.ExtractResults(Env, results,
                "Graph", "MetricValue", "PipelineId", "TrainingMetricValue", "FirstInput", "PredictorModel");
            Assert.True(rows.Length == numIterations);
            Assert.True(rows.All(r => r.TrainingMetricValue > 0.1));
        }

        [Fact]
        [TestCategory("EntryPoints")]
        public void EntryPointPipelineSweepRoles()
        {
            // Get datasets
            var pathData = GetDataPath("adult.train");
            var pathDataTest = GetDataPath("adult.test");
            const int numOfSampleRows = 100;
            int numIterations = 2;
            const string schema =
                "sep=, col=age:R4:0 col=workclass:TX:1 col=fnlwgt:R4:2 col=education:TX:3 col=education_num:R4:4 col=marital_status:TX:5 col=occupation:TX:6 " +
                "col=relationship:TX:7 col=ethnicity:TX:8 col=sex:TX:9 col=Features:R4:10-12 col=native_country:TX:13 col=IsOver50K_:R4:14 header=+";
            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows);
#pragma warning restore 0618

            // Define entrypoint graph
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Models.PipelineSweeper',
                      'Inputs': {
                        'TrainingData': '$TrainingData',
                        'TestingData': '$TestingData',
                        'LabelColumns': ['IsOver50K_'],
                        'WeightColumns': ['education_num'],
                        'NameColumns': ['education'],
                        'TextFeatureColumns': ['workclass', 'marital_status', 'occupation'],
                        'StateArguments': {
                            'Name': 'AutoMlState',
                            'Settings': {
                                'Metric': 'Auc',
                                'Engine': {
                                    'Name': 'Defaults'
                                },
                                'TerminatorArgs': {
                                    'Name': 'IterationLimited',
                                    'Settings': {
                                        'FinalHistoryLength': 2
                                    }
                                },
                                'TrainerKind': 'SignatureBinaryClassifierTrainer',
                                'RequestedLearners' : [
                                    'LogisticRegressionBinaryClassifier',
                                    'FastTreeBinaryClassifier'
                                ]
                            }
                        },
                        'BatchSize': 1
                      },
                      'Outputs': {
                        'State': '$StateOut',
                        'Results': '$ResultsOut'
                      }
                    },
                  ]
                }";

            JObject graphJson = JObject.Parse(inputGraph);
            var catalog = ModuleCatalog.CreateInstance(Env);
            var runner = new GraphRunner(Env, catalog, graphJson[FieldNames.Nodes] as JArray);
            runner.SetInput("TrainingData", datasetTrain);
            runner.SetInput("TestingData", datasetTest);
            runner.RunAll();

            var autoMlState = runner.GetOutput<AutoInference.AutoMlMlState>("StateOut");
            Assert.NotNull(autoMlState);
            var allPipelines = autoMlState.GetAllEvaluatedPipelines();
            var bestPipeline = autoMlState.GetBestPipeline();
            Assert.Equal(allPipelines.Length, numIterations);

            var trainAuc = bestPipeline.PerformanceSummary.TrainingMetricValue;
            var testAuc = bestPipeline.PerformanceSummary.MetricValue;
            Assert.True((0.94 < trainAuc) && (trainAuc < 0.95));
            Assert.True((0.83 < testAuc) && (testAuc < 0.84));

            var results = runner.GetOutput<IDataView>("ResultsOut");
            Assert.NotNull(results);
            var rows = PipelinePattern.ExtractResults(Env, results,
                "Graph", "MetricValue", "PipelineId", "TrainingMetricValue", "FirstInput", "PredictorModel");
            Assert.True(rows.Length == numIterations);
            Assert.True(rows.All(r => r.TrainingMetricValue > 0.1));
        }

        [Fact]
        [TestCategory("EntryPoints")]
        public void EntryPointPipelineSweepMultiClass()
        {
            // Get datasets
            var pathData = GetDataPath("adult.train");
            var pathDataTest = GetDataPath("adult.test");
            const int numOfSampleRows = 100;
            const string schema =
                "sep=, col=age:R4:0 col=workclass:TX:1 col=fnlwgt:R4:2 col=education:TX:3 col=education_num:R4:4 col=marital_status:TX:5 col=occupation:TX:6 " +
                "col=relationship:TX:7 col=ethnicity:TX:8 col=sex:TX:9 col=Features:R4:10-12 col=native_country:TX:13 col=IsOver50K_:R4:14 header=+";
            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows);
#pragma warning restore 0618

            // Define entrypoint graph
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Models.PipelineSweeper',
                      'Inputs': {
                        'TrainingData': '$TrainingData',
                        'TestingData': '$TestingData',
                        'LabelColumns': ['IsOver50K_'],
                        'WeightColumns': ['education_num'],
                        'NameColumns': ['education'],
                        'TextFeatureColumns': ['workclass', 'marital_status', 'occupation'],
                        'StateArguments': {
                            'Name': 'AutoMlState',
                            'Settings': {
                                'Metric': 'Accuracy(micro-avg)',
                                'Engine': {
                                    'Name': 'Defaults'
                                },
                                'TerminatorArgs': {
                                    'Name': 'IterationLimited',
                                    'Settings': {
                                        'FinalHistoryLength': 2
                                    }
                                },
                                'TrainerKind': 'SignatureMultiClassClassifierTrainer',
                            }
                        },
                        'BatchSize': 1
                      },
                      'Outputs': {
                        'State': '$StateOut',
                        'Results': '$ResultsOut'
                      }
                    },
                  ]
                }";

            JObject graphJson = JObject.Parse(inputGraph);
            var catalog = ModuleCatalog.CreateInstance(Env);
            var runner = new GraphRunner(Env, catalog, graphJson[FieldNames.Nodes] as JArray);
            runner.SetInput("TrainingData", datasetTrain);
            runner.SetInput("TestingData", datasetTest);
            runner.RunAll();
        }

        [Fact]
        public void TestRocketPipelineEngine()
        {
            // Get datasets
            var pathData = GetDataPath("adult.train");
            var pathDataTest = GetDataPath("adult.test");
            const int numOfSampleRows = 1000;
            int numIterations = 35;
            const string schema =
                "sep=, col=Features:R4:0,2,4,10-12 col=workclass:TX:1 col=education:TX:3 col=marital_status:TX:5 col=occupation:TX:6 " +
                "col=relationship:TX:7 col=ethnicity:TX:8 col=sex:TX:9 col=native_country:TX:13 col=label_IsOver50K_:R4:14 header=+";
            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows);
#pragma warning restore 0618
            // Define entrypoint graph
            string inputGraph = @"
                {
                  'Nodes': [                                
                    {
                      'Name': 'Models.PipelineSweeper',
                      'Inputs': {
                        'TrainingData': '$TrainingData',
                        'TestingData': '$TestingData',
                        'StateArguments': {
                            'Name': 'AutoMlState',
                            'Settings': {
                                'Metric': 'Auc',
                                'Engine': {
                                    'Name': 'Rocket',
                                    'Settings' : {
                                        'TopKLearners' : 2,
                                        'SecondRoundTrialsPerLearner' : 5
                                    },
                                },
                                'TerminatorArgs': {
                                    'Name': 'IterationLimited',
                                    'Settings': {
                                        'FinalHistoryLength': 35
                                    }
                                },
                                'TrainerKind': 'SignatureBinaryClassifierTrainer'
                            }
                        },
                        'BatchSize': 5
                      },
                      'Outputs': {
                        'State': '$StateOut',
                        'Results': '$ResultsOut'
                      }
                    },
                  ]
                }";

            JObject graph = JObject.Parse(inputGraph);
            var catalog = ModuleCatalog.CreateInstance(Env);

            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);
            runner.SetInput("TrainingData", datasetTrain);
            runner.SetInput("TestingData", datasetTest);
            runner.RunAll();

            var autoMlState = runner.GetOutput<AutoInference.AutoMlMlState>("StateOut");
            Assert.NotNull(autoMlState);
            var allPipelines = autoMlState.GetAllEvaluatedPipelines();
            var bestPipeline = autoMlState.GetBestPipeline();
            Assert.Equal(allPipelines.Length, numIterations);
            Assert.True(bestPipeline.PerformanceSummary.MetricValue > 0.1);

            var results = runner.GetOutput<IDataView>("ResultsOut");
            Assert.NotNull(results);
            var rows = PipelinePattern.ExtractResults(Env, results,
                "Graph", "MetricValue", "PipelineId", "TrainingMetricValue", "FirstInput", "PredictorModel");
            Assert.True(rows.Length == numIterations);
        }

        [Fact(Skip = "Need CoreTLC specific baseline update")]
        public void TestTextDatasetLearn()
        {
            using (var env = new TlcEnvironment())
            {
                string pathData = GetDataPath(@"../UnitTest/tweets_labeled_10k_test_validation.tsv");
                int batchSize = 5;
                int numIterations = 35;
                int numTransformLevels = 1;
                int numSampleRows = 100;
                AutoInference.SupportedMetric metric = AutoInference.SupportedMetric.AccuracyMicro;

                // Using the simple, uniform random sampling (with replacement) engine
                PipelineOptimizerBase autoMlEngine = new UniformRandomEngine(env);

                // Test initial learning
                var amls = AutoInference.InferPipelines(env, autoMlEngine, pathData, "", out var _, numTransformLevels, batchSize,
                metric, out var _, numSampleRows, new IterationTerminator(numIterations),
                MacroUtils.TrainerKinds.SignatureMultiClassClassifierTrainer);
                env.Check(amls.GetAllEvaluatedPipelines().Length == numIterations);
            }
            Done();
        }

        [Fact]
        public void TestPipelineNodeCloning()
        {
            using (var env = new TlcEnvironment())
            {
                var lr1 = RecipeInference
                    .AllowedLearners(env, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer)
                    .First(learner => learner.PipelineNode != null && learner.LearnerName.Contains("LogisticRegression"));

                var sdca1 = RecipeInference
                    .AllowedLearners(env, MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer)
                    .First(learner => learner.PipelineNode != null && learner.LearnerName.Contains("StochasticDualCoordinateAscent"));

                // Clone and change hyperparam values
                var lr2 = lr1.Clone();
                lr1.PipelineNode.SweepParams[0].RawValue = 1.2f;
                lr2.PipelineNode.SweepParams[0].RawValue = 3.5f;
                var sdca2 = sdca1.Clone();
                sdca1.PipelineNode.SweepParams[0].RawValue = 3;
                sdca2.PipelineNode.SweepParams[0].RawValue = 0;

                // Make sure the changes are propagated to entry point objects
                env.Check(lr1.PipelineNode.UpdateProperties());
                env.Check(lr2.PipelineNode.UpdateProperties());
                env.Check(sdca1.PipelineNode.UpdateProperties());
                env.Check(sdca2.PipelineNode.UpdateProperties());
                env.Check(lr1.PipelineNode.CheckEntryPointStateMatchesParamValues());
                env.Check(lr2.PipelineNode.CheckEntryPointStateMatchesParamValues());
                env.Check(sdca1.PipelineNode.CheckEntryPointStateMatchesParamValues());
                env.Check(sdca2.PipelineNode.CheckEntryPointStateMatchesParamValues());

                // Make sure second object's set of changes didn't overwrite first object's
                env.Check(!lr1.PipelineNode.SweepParams[0].RawValue.Equals(lr2.PipelineNode.SweepParams[0].RawValue));
                env.Check(!sdca2.PipelineNode.SweepParams[0].RawValue.Equals(sdca1.PipelineNode.SweepParams[0].RawValue));
            }
        }

        [Fact]
        public void TestSupportedMetricsByName()
        {
            var names = new List<string>()
            {
                AutoInference.SupportedMetric.AccuracyMacro.Name,
                AutoInference.SupportedMetric.AccuracyMicro.Name,
                AutoInference.SupportedMetric.Auc.Name,
                AutoInference.SupportedMetric.AuPrc.Name,
                AutoInference.SupportedMetric.Dbi.Name,
                AutoInference.SupportedMetric.F1.Name,
                AutoInference.SupportedMetric.LogLossReduction.Name
            };

            foreach (var name in names)
            {
                var metric = AutoInference.SupportedMetric.ByName(name);
                Assert.Equal(metric.Name, name);
            }
        }

        [Fact]
        public void TestHyperparameterFreezing()
        {
            string pathData = GetDataPath("adult.train");
            int numOfSampleRows = 1000;
            int batchSize = 1;
            int numIterations = 10;
            int numTransformLevels = 3;
            AutoInference.SupportedMetric metric = AutoInference.SupportedMetric.Auc;

            // Using the simple, uniform random sampling (with replacement) brain
            PipelineOptimizerBase autoMlBrain = new UniformRandomEngine(Env);

            // Run initial experiments
            var amls = AutoInference.InferPipelines(Env, autoMlBrain, pathData, "", out var _, numTransformLevels, batchSize,
                metric, out var bestPipeline, numOfSampleRows, new IterationTerminator(numIterations),
                MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer);

            // Clear results
            amls.ClearEvaluatedPipelines();

            // Get space, remove transforms and all but one learner, freeze hyperparameters on learner.
            var space = amls.GetSearchSpace();
            var transforms = space.Item1.Where(t =>
                t.ExpertType != typeof(TransformInference.Experts.Categorical)).ToArray();
            var learners = new[] { space.Item2.First() };
            var hyperParam = learners[0].PipelineNode.SweepParams.First();
            var frozenParamValue = hyperParam.RawValue;
            hyperParam.Frozen = true;
            amls.UpdateSearchSpace(learners, transforms);

            // Allow for one more iteration
            amls.UpdateTerminator(new IterationTerminator(numIterations + 1));

            // Do learning. Only retained learner should be left in all pipelines.
            bestPipeline = amls.InferPipelines(numTransformLevels, batchSize, numOfSampleRows);

            // Make sure all pipelines have retained learner
            Assert.True(amls.GetAllEvaluatedPipelines().All(p => p.Learner.LearnerName == learners[0].LearnerName));

            // Make sure hyperparameter value did not change
            Assert.NotNull(bestPipeline);
            Assert.Equal(bestPipeline.Learner.PipelineNode.SweepParams.First().RawValue, frozenParamValue);
        }

        [Fact(Skip = "Dataset not available.")]
        public void TestRegressionPipelineWithMinimizingMetric()
        {
            string pathData = GetDataPath("../Housing (regression)/housing.txt");
            int numOfSampleRows = 100;
            int batchSize = 5;
            int numIterations = 10;
            int numTransformLevels = 1;
            AutoInference.SupportedMetric metric = AutoInference.SupportedMetric.L1;

            // Using the simple, uniform random sampling (with replacement) brain
            PipelineOptimizerBase autoMlBrain = new UniformRandomEngine(Env);

            // Run initial experiments
            var amls = AutoInference.InferPipelines(Env, autoMlBrain, pathData, "", out var _, numTransformLevels, batchSize,
            metric, out var bestPipeline, numOfSampleRows, new IterationTerminator(numIterations),
            MacroUtils.TrainerKinds.SignatureRegressorTrainer);

            // Allow for one more iteration
            amls.UpdateTerminator(new IterationTerminator(numIterations + 1));

            // Do learning. Only retained learner should be left in all pipelines.
            bestPipeline = amls.InferPipelines(numTransformLevels, batchSize, numOfSampleRows);

            // Make sure hyperparameter value did not change
            Assert.NotNull(bestPipeline);
            Assert.True(amls.GetAllEvaluatedPipelines().All(
            p => p.PerformanceSummary.MetricValue >= bestPipeline.PerformanceSummary.MetricValue));
        }

        [Fact]
        public void TestLearnerConstrainingByName()
        {
            string pathData = GetDataPath("adult.train");
            int numOfSampleRows = 1000;
            int batchSize = 1;
            int numIterations = 1;
            int numTransformLevels = 2;
            var retainedLearnerNames = new[] { $"LogisticRegressionBinaryClassifier", $"FastTreeBinaryClassifier" };
            AutoInference.SupportedMetric metric = AutoInference.SupportedMetric.Auc;

            // Using the simple, uniform random sampling (with replacement) brain.
            PipelineOptimizerBase autoMlBrain = new UniformRandomEngine(Env);

            // Run initial experiment.
            var amls = AutoInference.InferPipelines(Env, autoMlBrain, pathData, "", out var _,
            numTransformLevels, batchSize, metric, out var _, numOfSampleRows,
            new IterationTerminator(numIterations), MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer);

            // Keep only logistic regression and FastTree.
            amls.KeepSelectedLearners(retainedLearnerNames);
            var space = amls.GetSearchSpace();

            // Make sure only learners left are those retained.
            Assert.Equal(retainedLearnerNames.Length, space.Item2.Length);
            Assert.True(space.Item2.All(l => retainedLearnerNames.Any(r => r == l.LearnerName)));
        }

        [Fact]
        public void TestRequestedLearners()
        {
            // Get datasets
            var pathData = GetDataPath("adult.train");
            var pathDataTest = GetDataPath("adult.test");
            const int numOfSampleRows = 100;
            const string schema =
                "sep=, col=Features:R4:0,2,4,10-12 col=workclass:TX:1 col=education:TX:3 col=marital_status:TX:5 col=occupation:TX:6 " +
                "col=relationship:TX:7 col=race:TX:8 col=sex:TX:9 col=native_country:TX:13 col=label_IsOver50K_:R4:14 header=+";
            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows);
            var requestedLearners = new[] { $"LogisticRegressionBinaryClassifier", $"FastTreeBinaryClassifier" };
#pragma warning restore 0618
            // Define entrypoint graph
            string inputGraph = @"
                {
                  'Nodes': [                                
                    {
                      'Name': 'Models.PipelineSweeper',
                      'Inputs': {
                        'TrainingData': '$TrainingData',
                        'TestingData': '$TestingData',
                        'StateArguments': {
                            'Name': 'AutoMlState',
                            'Settings': {
                                'Metric': 'Auc',
                                'Engine': {
                                    'Name': 'Rocket',
                                    'Settings' : {
                                        'TopKLearners' : 2,
                                        'SecondRoundTrialsPerLearner' : 0
                                    },
                                },
                                'TerminatorArgs': {
                                    'Name': 'IterationLimited',
                                    'Settings': {
                                        'FinalHistoryLength': 35
                                    }
                                },
                                'TrainerKind': 'SignatureBinaryClassifierTrainer',
                                'RequestedLearners' : [
                                    'LogisticRegressionBinaryClassifier',
                                    'FastTreeBinaryClassifier'
                                ]
                            }
                        },
                        'BatchSize': 5
                      },
                      'Outputs': {
                        'State': '$StateOut',
                        'Results': '$ResultsOut'
                      }
                    },
                  ]
                }";

            JObject graph = JObject.Parse(inputGraph);
            var catalog = ModuleCatalog.CreateInstance(Env);

            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);
            runner.SetInput("TrainingData", datasetTrain);
            runner.SetInput("TestingData", datasetTest);
            runner.RunAll();

            var autoMlState = runner.GetOutput<AutoInference.AutoMlMlState>("StateOut");
            Assert.NotNull(autoMlState);
            var space = autoMlState.GetSearchSpace();

            // Make sure only learners left are those retained.
            Assert.Equal(requestedLearners.Length, space.Item2.Length);
            Assert.True(space.Item2.All(l => requestedLearners.Any(r => r == l.LearnerName)));
        }

        [Fact]
        public void TestMinimizingMetricTransformations()
        {
            var values = new[] { 100d, 10d, -2d, -1d, 5.8d, -3.1d };
            var maxWeight = values.Max();
            var processed = values.Select(v => AutoMlUtils.ProcessWeight(v, maxWeight, false));
            var expectedResult = new[] { 0d, 90d, 102d, 101d, 94.2d, 103.1d };

            Assert.True(processed.Select((x, idx) => System.Math.Abs(x - expectedResult[idx]) < 0.001).All(r => r));
        }
    }
}
