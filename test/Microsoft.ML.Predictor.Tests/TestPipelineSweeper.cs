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
    [Collection("TestPipelineSweeper and TestAutoInference should not be run at the same time")]
    public sealed class TestPipelineSweeper : BaseTestBaseline
    {
        public TestPipelineSweeper(ITestOutputHelper helper)
            : base(helper)
        {
        }

        protected override void Initialize()
        {
            base.Initialize();
            Env.ComponentCatalog.RegisterAssembly(typeof(AutoInference).Assembly);
        }

        [Fact]
        public void PipelineSweeperBasic()
        {
            // Get datasets
            var pathData = GetDataPath("adult.tiny.with-schema.txt");
            var pathDataTest = GetDataPath("adult.tiny.with-schema.txt");
            const int numOfSampleRows = 1000;
            int numIterations = 4;
            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTrain }).Data.Take(numOfSampleRows, Env);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest }).Data.Take(numOfSampleRows, Env);
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
            var catalog = Env.ComponentCatalog;

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
        public void PipelineSweeperNoTransforms()
        {
            // Set up inputs for experiment
            string pathData = GetDataPath("adult.train");
            string pathDataTest = GetDataPath("adult.test");
            const int numOfSampleRows = 1000;
            const string schema = "sep=, col=Features:R4:0,2,4,10-12 col=Label:R4:14 header=+";

            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
#pragma warning restore 0618
            const int batchSize = 5;
            const int numIterations = 20;
            const int numTransformLevels = 2;
            using (var env = new ConsoleEnvironment())
            {
                SupportedMetric metric = PipelineSweeperSupportedMetrics.GetSupportedMetric(PipelineSweeperSupportedMetrics.Metrics.Auc);

                // Using the simple, uniform random sampling (with replacement) engine
                PipelineOptimizerBase autoMlEngine = new UniformRandomEngine(Env);

                // Create search object
                var amls = new AutoInference.AutoMlMlState(Env, metric, autoMlEngine, new IterationTerminator(numIterations),
                    MacroUtils.TrainerKinds.SignatureBinaryClassifierTrainer, datasetTrain, datasetTest);

                // Infer search space
                amls.InferSearchSpace(numTransformLevels);

                // Create macro object
                var pipelineSweepInput = new Microsoft.ML.Legacy.Models.PipelineSweeper()
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
                Assert.True(amlsOut.GetBestPipeline().PerformanceSummary.MetricValue > 0.8);
            }
        }

        [Fact]
        [TestCategory("EntryPoints")]
        public void PipelineSweeperSerialization()
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
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
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
            var catalog = Env.ComponentCatalog;
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
        [TestCategory("EntryPoints")]
        public void PipelineSweeperRoles()
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
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
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
            var catalog = Env.ComponentCatalog;
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
            Assert.True((0.815 < testAuc) && (testAuc < 0.825));

            var results = runner.GetOutput<IDataView>("ResultsOut");
            Assert.NotNull(results);
            var rows = PipelinePattern.ExtractResults(Env, results,
                "Graph", "MetricValue", "PipelineId", "TrainingMetricValue", "FirstInput", "PredictorModel");
            Assert.True(rows.Length == numIterations);
            Assert.True(rows.All(r => r.TrainingMetricValue > 0.1));
        }

        [Fact]
        [TestCategory("EntryPoints")]
        public void PipelineSweeperMultiClassClassification()
        {
            // Get datasets
            // TODO (agoswami) : For now we use the same dataset for train and test since the repo does not have a separate test file for the iris dataset.
            // In the future the PipelineSweeper Macro will have an option to take just one dataset as input, and do the train-test split internally.
            var pathData = GetDataPath(@"iris.txt");
            var pathDataTest = GetDataPath(@"iris.txt");
            int numIterations = 2;
            const string schema = "col=Species:R4:0 col=SepalLength:R4:1 col=SepalWidth:R4:2 col=PetalLength:R4:3 col=PetalWidth:R4:4";
            var inputFileTrain = new SimpleFileHandle(Env, pathData, false, false);
#pragma warning disable 0618
            var datasetTrain = ImportTextData.ImportText(Env, new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data;
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env, new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data;
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
                        'LabelColumns': ['Species'],
                        'StateArguments': {
                            'Name': 'AutoMlState',
                            'Settings': {
                                'Metric': 'AccuracyMicro',
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
                                'RequestedLearners' : [
                                    'LogisticRegressionClassifier',
                                    'StochasticDualCoordinateAscentClassifier'
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
            var catalog = Env.ComponentCatalog;
            var runner = new GraphRunner(Env, catalog, graphJson[FieldNames.Nodes] as JArray);
            runner.SetInput("TrainingData", datasetTrain);
            runner.SetInput("TestingData", datasetTest);
            runner.RunAll();

            var autoMlState = runner.GetOutput<AutoInference.AutoMlMlState>("StateOut");
            Assert.NotNull(autoMlState);
            var allPipelines = autoMlState.GetAllEvaluatedPipelines();
            var bestPipeline = autoMlState.GetBestPipeline();
            Assert.Equal(allPipelines.Length, numIterations);

            var bestMicroAccuracyTrain = bestPipeline.PerformanceSummary.TrainingMetricValue;
            var bestMicroAccuracyTest = bestPipeline.PerformanceSummary.MetricValue;
            Assert.True((0.97 < bestMicroAccuracyTrain) && (bestMicroAccuracyTrain < 0.99));
            Assert.True((0.97 < bestMicroAccuracyTest) && (bestMicroAccuracyTest < 0.99));

            var results = runner.GetOutput<IDataView>("ResultsOut");
            Assert.NotNull(results);
            var rows = PipelinePattern.ExtractResults(Env, results,
                "Graph", "MetricValue", "PipelineId", "TrainingMetricValue", "FirstInput", "PredictorModel");
            Assert.True(rows.Length == numIterations);
            Assert.True(rows.All(r => r.MetricValue > 0.9));
        }

        [Fact]
        public void PipelineSweeperRocketEngine()
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
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
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
            var catalog = Env.ComponentCatalog;

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

        [Fact]
        public void PipelineSweeperRequestedLearners()
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
                new ImportTextData.Input { InputFile = inputFileTrain, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
            var inputFileTest = new SimpleFileHandle(Env, pathDataTest, false, false);
            var datasetTest = ImportTextData.ImportText(Env,
                new ImportTextData.Input { InputFile = inputFileTest, CustomSchema = schema }).Data.Take(numOfSampleRows, Env);
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
            var catalog = Env.ComponentCatalog;

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
    }
}
