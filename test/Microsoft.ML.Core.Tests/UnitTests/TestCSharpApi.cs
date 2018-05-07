// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using ML = Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;
/*using Categorical = Microsoft.ML.Transforms;
using Commands = Microsoft.ML.Transforms;
using Evaluate = Microsoft.ML;
using ImportTextData = Microsoft.ML.Data;
using LogisticRegression = Microsoft.ML.Trainers;
using ModelOperations = Microsoft.ML.Transforms;
using Normalize = Microsoft.ML.Transforms;
using SchemaManipulation = Microsoft.ML.Transforms;
using ScoreModel = Microsoft.ML.Transforms;
using Sdca = Microsoft.ML.Trainers;*/

namespace Microsoft.ML.Runtime.RunTests
{
    public class TestCSharpApi : BaseTestClass
    {
        public TestCSharpApi(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestSimpleExperiment()
        {
            var dataPath = GetDataPath(@"adult.tiny.with-schema.txt");
            using (var env = new TlcEnvironment())
            {
                var experiment = env.CreateExperiment();

                var importInput = new ML.Data.CustomTextLoader();
                var importOutput = experiment.Add(importInput);

                var normalizeInput = new ML.Transforms.MinMaxNormalizer
                {
                    Data = importOutput.Data
                };
                normalizeInput.AddColumn("NumericFeatures");
                var normalizeOutput = experiment.Add(normalizeInput);

                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();
                var data = experiment.GetOutput(normalizeOutput.OutputData);

                var schema = data.Schema;
                Assert.Equal(5, schema.ColumnCount);
                var expected = new[] { "Label", "Workclass", "Categories", "NumericFeatures", "NumericFeatures" };
                for (int i = 0; i < schema.ColumnCount; i++)
                    Assert.Equal(expected[i], schema.GetColumnName(i));
            }
        }

        [Fact]
        public void TestSimpleTrainExperiment()
        {
            var dataPath = GetDataPath(@"adult.tiny.with-schema.txt");
            using (var env = new TlcEnvironment())
            {
                var experiment = env.CreateExperiment();

                var importInput = new ML.Data.CustomTextLoader();
                var importOutput = experiment.Add(importInput);

                var catInput = new ML.Transforms.CategoricalOneHotVectorizer
                {
                    Data = importOutput.Data
                };
                catInput.AddColumn("Categories");
                var catOutput = experiment.Add(catInput);

                var concatInput = new ML.Transforms.ColumnConcatenator
                {
                    Data = catOutput.OutputData
                };
                concatInput.AddColumn("Features", "Categories", "NumericFeatures");
                var concatOutput = experiment.Add(concatInput);

                var sdcaInput = new ML.Trainers.StochasticDualCoordinateAscentBinaryClassifier
                {
                    TrainingData = concatOutput.OutputData,
                    LossFunction = new HingeLossSDCAClassificationLossFunction() { Margin = 1.1f },
                    NumThreads = 1,
                    Shuffle = false
                };
                var sdcaOutput = experiment.Add(sdcaInput);

                var scoreInput = new ML.Transforms.DatasetScorer
                {
                    Data = concatOutput.OutputData,
                    PredictorModel = sdcaOutput.PredictorModel
                };
                var scoreOutput = experiment.Add(scoreInput);

                var evalInput = new ML.Models.BinaryClassificationEvaluator
                {
                    Data = scoreOutput.ScoredData
                };
                var evalOutput = experiment.Add(evalInput);

                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();
                var data = experiment.GetOutput(evalOutput.OverallMetrics);

                var schema = data.Schema;
                var b = schema.TryGetColumnIndex("AUC", out int aucCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == aucCol))
                {
                    var getter = cursor.GetGetter<double>(aucCol);
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double auc = 0;
                    getter(ref auc);
                    Assert.Equal(0.93, auc, 2);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }

        [Fact]
        public void TestTrainTestMacro()
        {
            var dataPath = GetDataPath(@"adult.tiny.with-schema.txt");
            using (var env = new TlcEnvironment())
            {
                var subGraph = env.CreateExperiment();

                var catInput = new ML.Transforms.CategoricalOneHotVectorizer();
                catInput.AddColumn("Categories");
                var catOutput = subGraph.Add(catInput);

                var concatInput = new ML.Transforms.ColumnConcatenator
                {
                    Data = catOutput.OutputData
                };
                concatInput.AddColumn("Features", "Categories", "NumericFeatures");
                var concatOutput = subGraph.Add(concatInput);

                var sdcaInput = new ML.Trainers.StochasticDualCoordinateAscentBinaryClassifier
                {
                    TrainingData = concatOutput.OutputData,
                    LossFunction = new HingeLossSDCAClassificationLossFunction() { Margin = 1.1f },
                    NumThreads = 1,
                    Shuffle = false
                };
                var sdcaOutput = subGraph.Add(sdcaInput);

                var modelCombine = new ML.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(catOutput.Model, concatOutput.Model),
                    PredictorModel = sdcaOutput.PredictorModel
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                var experiment = env.CreateExperiment();

                var importInput = new ML.Data.CustomTextLoader();
                var importOutput = experiment.Add(importInput);

                var trainTestInput = new ML.Models.TrainTestBinaryEvaluator
                {
                    TrainingData = importOutput.Data,
                    TestingData = importOutput.Data,
                    Nodes = subGraph
                };
                trainTestInput.Inputs.Data = catInput.Data;
                trainTestInput.Outputs.Model = modelCombineOutput.PredictorModel;
                var trainTestOutput = experiment.Add(trainTestInput);

                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();
                var data = experiment.GetOutput(trainTestOutput.OverallMetrics);

                var schema = data.Schema;
                var b = schema.TryGetColumnIndex("AUC", out int aucCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == aucCol))
                {
                    var getter = cursor.GetGetter<double>(aucCol);
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double auc = 0;
                    getter(ref auc);
                    Assert.Equal(0.93, auc, 2);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }

        [Fact]
        public void TestCrossValidationBinaryMacro()
        {
            var dataPath = GetDataPath(@"adult.tiny.with-schema.txt");
            using (var env = new TlcEnvironment())
            {
                var subGraph = env.CreateExperiment();

                var catInput = new ML.Transforms.CategoricalOneHotVectorizer();
                catInput.AddColumn("Categories");
                var catOutput = subGraph.Add(catInput);

                var concatInput = new ML.Transforms.ColumnConcatenator
                {
                    Data = catOutput.OutputData
                };
                concatInput.AddColumn("Features", "Categories", "NumericFeatures");
                var concatOutput = subGraph.Add(concatInput);

                var lrInput = new ML.Trainers.BinaryLogisticRegressor
                {
                    TrainingData = concatOutput.OutputData,
                    NumThreads = 1
                };
                var lrOutput = subGraph.Add(lrInput);

                var modelCombine = new ML.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(catOutput.Model, concatOutput.Model),
                    PredictorModel = lrOutput.PredictorModel
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                var experiment = env.CreateExperiment();

                var importInput = new ML.Data.CustomTextLoader();
                var importOutput = experiment.Add(importInput);

                var crossValidateBinary = new ML.Models.BinaryCrossValidator
                {
                    Data = importOutput.Data,
                    Nodes = subGraph
                };
                crossValidateBinary.Inputs.Data = catInput.Data;
                crossValidateBinary.Outputs.Model = modelCombineOutput.PredictorModel;
                var crossValidateOutput = experiment.Add(crossValidateBinary);

                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();
                var data = experiment.GetOutput(crossValidateOutput.OverallMetrics[0]);

                var schema = data.Schema;
                var b = schema.TryGetColumnIndex("AUC", out int aucCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == aucCol))
                {
                    var getter = cursor.GetGetter<double>(aucCol);
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double auc = 0;
                    getter(ref auc);
                    Assert.Equal(0.87, auc, 1);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }

        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/3")]
        public void TestCrossValidationMacro()
        {
            var dataPath = GetDataPath(@"housing.txt");
            using (var env = new TlcEnvironment())
            {
                var subGraph = env.CreateExperiment();

                var nop = new ML.Transforms.NoOperation();
                var nopOutput = subGraph.Add(nop);

                var learnerInput = new ML.Trainers.StochasticDualCoordinateAscentRegressor
                {
                    TrainingData = nopOutput.OutputData,
                    NumThreads = 1
                };
                var learnerOutput = subGraph.Add(learnerInput);

                var modelCombine = new ML.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(nopOutput.Model),
                    PredictorModel = learnerOutput.PredictorModel
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                var experiment = env.CreateExperiment();
                var importInput = new ML.Data.CustomTextLoader();
                var importOutput = experiment.Add(importInput);

                var crossValidate = new ML.Models.CrossValidator
                {
                    Data = importOutput.Data,
                    Nodes = subGraph,
                    Kind = ML.Models.MacroUtilsTrainerKinds.SignatureRegressorTrainer,
                    TransformModel = null
                };
                crossValidate.Inputs.Data = nop.Data;
                crossValidate.Outputs.Model = modelCombineOutput.PredictorModel;
                var crossValidateOutput = experiment.Add(crossValidate);

                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();
                var data = experiment.GetOutput(crossValidateOutput.OverallMetrics[0]);

                var schema = data.Schema;
                var b = schema.TryGetColumnIndex("L1(avg)", out int metricCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == metricCol))
                {
                    var getter = cursor.GetGetter<double>(metricCol);
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double val = 0;
                    getter(ref val);
                    Assert.Equal(3.32, val, 1);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }
    }
}
