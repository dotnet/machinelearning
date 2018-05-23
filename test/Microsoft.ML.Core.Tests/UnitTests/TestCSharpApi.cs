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
using System.Linq;
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

                var importInput = new ML.Data.TextLoader(dataPath);
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

                var importInput = new ML.Data.TextLoader(dataPath);
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

                var importInput = new ML.Data.TextLoader(dataPath);
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

                var lrInput = new ML.Trainers.LogisticRegressionBinaryClassifier
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

                var importInput = new ML.Data.TextLoader(dataPath);
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
                importInput.SetInput(env, experiment);
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

        [Fact]
        public void TestCrossValidationMacro()
        {
            var dataPath = GetDataPath(@"Train-Tiny-28x28.txt");
            using (var env = new TlcEnvironment(42))
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
                var importInput = new ML.Data.TextLoader(dataPath);
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
                importInput.SetInput(env, experiment);
                experiment.Run();
                var data = experiment.GetOutput(crossValidateOutput.OverallMetrics);

                var schema = data.Schema;
                var b = schema.TryGetColumnIndex("L1(avg)", out int metricCol);
                Assert.True(b);
                b = schema.TryGetColumnIndex("Fold Index", out int foldCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == metricCol || col == foldCol))
                {
                    var getter = cursor.GetGetter<double>(metricCol);
                    var foldGetter = cursor.GetGetter<DvText>(foldCol);
                    DvText fold = default;

                    // Get the verage.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double avg = 0;
                    getter(ref avg);
                    foldGetter(ref fold);
                    Assert.True(fold.EqualsStr("Average"));

                    // Get the standard deviation.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double stdev = 0;
                    getter(ref stdev);
                    foldGetter(ref fold);
                    Assert.True(fold.EqualsStr("Standard Deviation"));
                    Assert.Equal(0.096, stdev, 3);

                    double sum = 0;
                    double val = 0;
                    for (int f = 0; f < 2; f++)
                    {
                        b = cursor.MoveNext();
                        Assert.True(b);
                        getter(ref val);
                        foldGetter(ref fold);
                        sum += val;
                        Assert.True(fold.EqualsStr("Fold " + f));
                    }
                    Assert.Equal(avg, sum / 2);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }

        [Fact]
        public void TestCrossValidationMacroWithMultiClass()
        {
            var dataPath = GetDataPath(@"Train-Tiny-28x28.txt");
            using (var env = new TlcEnvironment(42))
            {
                var subGraph = env.CreateExperiment();

                var nop = new ML.Transforms.NoOperation();
                var nopOutput = subGraph.Add(nop);

                var learnerInput = new ML.Trainers.StochasticDualCoordinateAscentClassifier
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
                var importInput = new ML.Data.TextLoader(dataPath);
                var importOutput = experiment.Add(importInput);

                var crossValidate = new ML.Models.CrossValidator
                {
                    Data = importOutput.Data,
                    Nodes = subGraph,
                    Kind = ML.Models.MacroUtilsTrainerKinds.SignatureMultiClassClassifierTrainer,
                    TransformModel = null
                };
                crossValidate.Inputs.Data = nop.Data;
                crossValidate.Outputs.Model = modelCombineOutput.PredictorModel;
                var crossValidateOutput = experiment.Add(crossValidate);

                experiment.Compile();
                importInput.SetInput(env, experiment);
                experiment.Run();
                var data = experiment.GetOutput(crossValidateOutput.OverallMetrics);

                var schema = data.Schema;
                var b = schema.TryGetColumnIndex("Accuracy(micro-avg)", out int metricCol);
                Assert.True(b);
                b = schema.TryGetColumnIndex("Fold Index", out int foldCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == metricCol || col == foldCol))
                {
                    var getter = cursor.GetGetter<double>(metricCol);
                    var foldGetter = cursor.GetGetter<DvText>(foldCol);
                    DvText fold = default;

                    // Get the verage.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double avg = 0;
                    getter(ref avg);
                    foldGetter(ref fold);
                    Assert.True(fold.EqualsStr("Average"));

                    // Get the standard deviation.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double stdev = 0;
                    getter(ref stdev);
                    foldGetter(ref fold);
                    Assert.True(fold.EqualsStr("Standard Deviation"));
                    Assert.Equal(0.025, stdev, 3);

                    double sum = 0;
                    double val = 0;
                    for (int f = 0; f < 2; f++)
                    {
                        b = cursor.MoveNext();
                        Assert.True(b);
                        getter(ref val);
                        foldGetter(ref fold);
                        sum += val;
                        Assert.True(fold.EqualsStr("Fold " + f));
                    }
                    Assert.Equal(avg, sum / 2);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }

                var confusion = experiment.GetOutput(crossValidateOutput.ConfusionMatrix);
                schema = confusion.Schema;
                b = schema.TryGetColumnIndex("Count", out int countCol);
                Assert.True(b);
                b = schema.TryGetColumnIndex("Fold Index", out foldCol);
                Assert.True(b);
                var type = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, countCol);
                Assert.True(type != null && type.ItemType.IsText && type.VectorSize == 10);
                var slotNames = default(VBuffer<DvText>);
                schema.GetMetadata(MetadataUtils.Kinds.SlotNames, countCol, ref slotNames);
                Assert.True(slotNames.Values.Select((s, i) => s.EqualsStr(i.ToString())).All(x => x));
                using (var curs = confusion.GetRowCursor(col => true))
                {
                    var countGetter = curs.GetGetter<VBuffer<double>>(countCol);
                    var foldGetter = curs.GetGetter<DvText>(foldCol);
                    var confCount = default(VBuffer<double>);
                    var foldIndex = default(DvText);
                    int rowCount = 0;
                    var foldCur = "Fold 0";
                    while (curs.MoveNext())
                    {
                        countGetter(ref confCount);
                        foldGetter(ref foldIndex);
                        rowCount++;
                        Assert.True(foldIndex.EqualsStr(foldCur));
                        if (rowCount == 10)
                        {
                            rowCount = 0;
                            foldCur = "Fold 1";
                        }
                    }
                    Assert.Equal(0, rowCount);
                }
            }
        }

        [Fact]
        public void TestCrossValidationMacroWithStratification()
        {
            var dataPath = GetDataPath(@"breast-cancer.txt");
            using (var env = new TlcEnvironment(42))
            {
                var subGraph = env.CreateExperiment();

                var nop = new ML.Transforms.NoOperation();
                var nopOutput = subGraph.Add(nop);

                var learnerInput = new ML.Trainers.StochasticDualCoordinateAscentBinaryClassifier
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
                var importInput = new ML.Data.TextLoader(dataPath);
                importInput.Arguments.Column = new ML.Data.TextLoaderColumn[]
                {
                    new ML.Data.TextLoaderColumn { Name = "Label", Source = new[] { new ML.Data.TextLoaderRange(0) } },
                    new ML.Data.TextLoaderColumn { Name = "Strat", Source = new[] { new ML.Data.TextLoaderRange(1) } },
                    new ML.Data.TextLoaderColumn { Name = "Features", Source = new[] { new ML.Data.TextLoaderRange(2, 9) } }
                };
                var importOutput = experiment.Add(importInput);

                var crossValidate = new ML.Models.CrossValidator
                {
                    Data = importOutput.Data,
                    Nodes = subGraph,
                    TransformModel = null,
                    StratificationColumn = "Strat"
                };
                crossValidate.Inputs.Data = nop.Data;
                crossValidate.Outputs.Model = modelCombineOutput.PredictorModel;
                var crossValidateOutput = experiment.Add(crossValidate);

                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();
                var data = experiment.GetOutput(crossValidateOutput.OverallMetrics);

                var schema = data.Schema;
                var b = schema.TryGetColumnIndex("AUC", out int metricCol);
                Assert.True(b);
                b = schema.TryGetColumnIndex("Fold Index", out int foldCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == metricCol || col == foldCol))
                {
                    var getter = cursor.GetGetter<double>(metricCol);
                    var foldGetter = cursor.GetGetter<DvText>(foldCol);
                    DvText fold = default;

                    // Get the verage.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double avg = 0;
                    getter(ref avg);
                    foldGetter(ref fold);
                    Assert.True(fold.EqualsStr("Average"));

                    // Get the standard deviation.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double stdev = 0;
                    getter(ref stdev);
                    foldGetter(ref fold);
                    Assert.True(fold.EqualsStr("Standard Deviation"));
                    Assert.Equal(0.00485, stdev, 5);

                    double sum = 0;
                    double val = 0;
                    for (int f = 0; f < 2; f++)
                    {
                        b = cursor.MoveNext();
                        Assert.True(b);
                        getter(ref val);
                        foldGetter(ref fold);
                        sum += val;
                        Assert.True(fold.EqualsStr("Fold " + f));
                    }
                    Assert.Equal(avg, sum / 2);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }
    }
}
