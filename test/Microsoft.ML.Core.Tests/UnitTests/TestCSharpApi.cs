// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

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
            var dataPath = GetDataPath("adult.tiny.with-schema.txt");
            using (var env = new ConsoleEnvironment())
            {
                var experiment = env.CreateExperiment();

                var importInput = new Legacy.Data.TextLoader(dataPath);
                var importOutput = experiment.Add(importInput);

                var normalizeInput = new Legacy.Transforms.MinMaxNormalizer
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
            var dataPath = GetDataPath("adult.tiny.with-schema.txt");
            using (var env = new ConsoleEnvironment())
            {
                var experiment = env.CreateExperiment();

                var importInput = new Legacy.Data.TextLoader(dataPath);
                var importOutput = experiment.Add(importInput);

                var catInput = new Legacy.Transforms.CategoricalOneHotVectorizer
                {
                    Data = importOutput.Data
                };
                catInput.AddColumn("Categories");
                var catOutput = experiment.Add(catInput);

                var concatInput = new Legacy.Transforms.ColumnConcatenator
                {
                    Data = catOutput.OutputData
                };
                concatInput.AddColumn("Features", "Categories", "NumericFeatures");
                var concatOutput = experiment.Add(concatInput);

                var sdcaInput = new Legacy.Trainers.StochasticDualCoordinateAscentBinaryClassifier
                {
                    TrainingData = concatOutput.OutputData,
                    LossFunction = new HingeLossSDCAClassificationLossFunction() { Margin = 1.1f },
                    NumThreads = 1,
                    Shuffle = false
                };
                var sdcaOutput = experiment.Add(sdcaInput);

                var scoreInput = new Legacy.Transforms.DatasetScorer
                {
                    Data = concatOutput.OutputData,
                    PredictorModel = sdcaOutput.PredictorModel
                };
                var scoreOutput = experiment.Add(scoreInput);

                var evalInput = new Legacy.Models.BinaryClassificationEvaluator
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
            var dataPath = GetDataPath("adult.tiny.with-schema.txt");
            using (var env = new ConsoleEnvironment())
            {
                var subGraph = env.CreateExperiment();

                var catInput = new Legacy.Transforms.CategoricalOneHotVectorizer();
                catInput.AddColumn("Categories");
                var catOutput = subGraph.Add(catInput);

                var concatInput = new Legacy.Transforms.ColumnConcatenator
                {
                    Data = catOutput.OutputData
                };
                concatInput.AddColumn("Features", "Categories", "NumericFeatures");
                var concatOutput = subGraph.Add(concatInput);

                var sdcaInput = new Legacy.Trainers.StochasticDualCoordinateAscentBinaryClassifier
                {
                    TrainingData = concatOutput.OutputData,
                    LossFunction = new HingeLossSDCAClassificationLossFunction() { Margin = 1.1f },
                    NumThreads = 1,
                    Shuffle = false
                };
                var sdcaOutput = subGraph.Add(sdcaInput);

                var modelCombine = new Legacy.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(catOutput.Model, concatOutput.Model),
                    PredictorModel = sdcaOutput.PredictorModel
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                var experiment = env.CreateExperiment();

                var importInput = new Legacy.Data.TextLoader(dataPath);
                var importOutput = experiment.Add(importInput);

                var trainTestInput = new Legacy.Models.TrainTestBinaryEvaluator
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
            var dataPath = GetDataPath("adult.tiny.with-schema.txt");
            using (var env = new ConsoleEnvironment())
            {
                var subGraph = env.CreateExperiment();

                var catInput = new Legacy.Transforms.CategoricalOneHotVectorizer();
                catInput.AddColumn("Categories");
                var catOutput = subGraph.Add(catInput);

                var concatInput = new Legacy.Transforms.ColumnConcatenator
                {
                    Data = catOutput.OutputData
                };
                concatInput.AddColumn("Features", "Categories", "NumericFeatures");
                var concatOutput = subGraph.Add(concatInput);

                var lrInput = new Legacy.Trainers.LogisticRegressionBinaryClassifier
                {
                    TrainingData = concatOutput.OutputData,
                    NumThreads = 1
                };
                var lrOutput = subGraph.Add(lrInput);

                var modelCombine = new Legacy.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(catOutput.Model, concatOutput.Model),
                    PredictorModel = lrOutput.PredictorModel
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                var experiment = env.CreateExperiment();

                var importInput = new Legacy.Data.TextLoader(dataPath);
                var importOutput = experiment.Add(importInput);

                var crossValidateBinary = new Legacy.Models.BinaryCrossValidator
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
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDatasetmacro.trainFilename);
            using (var env = new ConsoleEnvironment(42))
            {
                var subGraph = env.CreateExperiment();

                var nop = new Legacy.Transforms.NoOperation();
                var nopOutput = subGraph.Add(nop);

                var generate = new Legacy.Transforms.RandomNumberGenerator();
                generate.Column = new[] { new Legacy.Transforms.GenerateNumberTransformColumn() { Name = "Weight1" } };
                generate.Data = nopOutput.OutputData;
                var generateOutput = subGraph.Add(generate);

                var learnerInput = new Legacy.Trainers.PoissonRegressor
                {
                    TrainingData = generateOutput.OutputData,
                    NumThreads = 1,
                    WeightColumn = "Weight1"
                };
                var learnerOutput = subGraph.Add(learnerInput);

                var modelCombine = new Legacy.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(nopOutput.Model, generateOutput.Model),
                    PredictorModel = learnerOutput.PredictorModel
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                var experiment = env.CreateExperiment();
                var importInput = new Legacy.Data.TextLoader(dataPath)
                {
                    Arguments = new Legacy.Data.TextLoaderArguments
                    {
                        Separator = new[] { ';' },
                        HasHeader = true,
                        Column = new[]
                    {
                        new TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoaderRange(11) },
                            Type = Legacy.Data.DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoaderRange(0,10) },
                            Type = Legacy.Data.DataKind.Num
                        }
                    }
                    }
                };
                var importOutput = experiment.Add(importInput);

                var crossValidate = new Legacy.Models.CrossValidator
                {
                    Data = importOutput.Data,
                    Nodes = subGraph,
                    Kind = Legacy.Models.MacroUtilsTrainerKinds.SignatureRegressorTrainer,
                    TransformModel = null,
                    WeightColumn = "Weight1"
                };
                crossValidate.Inputs.Data = nop.Data;
                crossValidate.Outputs.PredictorModel = modelCombineOutput.PredictorModel;
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
                b = schema.TryGetColumnIndex("IsWeighted", out int isWeightedCol);
                using (var cursor = data.GetRowCursor(col => col == metricCol || col == foldCol || col == isWeightedCol))
                {
                    var getter = cursor.GetGetter<double>(metricCol);
                    var foldGetter = cursor.GetGetter<ReadOnlyMemory<char>>(foldCol);
                    ReadOnlyMemory<char> fold = default;
                    var isWeightedGetter = cursor.GetGetter<bool>(isWeightedCol);
                    bool isWeighted = default;
                    double avg = 0;
                    double weightedAvg = 0;
                    for (int w = 0; w < 2; w++)
                    {
                        // Get the average.
                        b = cursor.MoveNext();
                        Assert.True(b);
                        if (w == 1)
                            getter(ref weightedAvg);
                        else
                            getter(ref avg);
                        foldGetter(ref fold);
                        Assert.True(ReadOnlyMemoryUtils.EqualsStr("Average", fold));
                        isWeightedGetter(ref isWeighted);
                        Assert.True(isWeighted == (w == 1));

                        // Get the standard deviation.
                        b = cursor.MoveNext();
                        Assert.True(b);
                        double stdev = 0;
                        getter(ref stdev);
                        foldGetter(ref fold);
                        Assert.True(ReadOnlyMemoryUtils.EqualsStr("Standard Deviation", fold));
                        if (w == 1)
                            Assert.Equal(1.585, stdev, 3);
                        else
                            Assert.Equal(1.39, stdev, 2);
                        isWeightedGetter(ref isWeighted);
                        Assert.True(isWeighted == (w == 1));
                    }
                    double sum = 0;
                    double weightedSum = 0;
                    for (int f = 0; f < 2; f++)
                    {
                        for (int w = 0; w < 2; w++)
                        {
                            b = cursor.MoveNext();
                            Assert.True(b);
                            double val = 0;
                            getter(ref val);
                            foldGetter(ref fold);
                            if (w == 1)
                                weightedSum += val;
                            else
                                sum += val;
                            Assert.True(ReadOnlyMemoryUtils.EqualsStr("Fold " + f, fold));
                            isWeightedGetter(ref isWeighted);
                            Assert.True(isWeighted == (w == 1));
                        }
                    }
                    Assert.Equal(weightedAvg, weightedSum / 2);
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
            using (var env = new ConsoleEnvironment(42))
            {
                var subGraph = env.CreateExperiment();

                var nop = new Legacy.Transforms.NoOperation();
                var nopOutput = subGraph.Add(nop);

                var learnerInput = new Legacy.Trainers.StochasticDualCoordinateAscentClassifier
                {
                    TrainingData = nopOutput.OutputData,
                    NumThreads = 1
                };
                var learnerOutput = subGraph.Add(learnerInput);

                var modelCombine = new Legacy.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(nopOutput.Model),
                    PredictorModel = learnerOutput.PredictorModel
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                var experiment = env.CreateExperiment();
                var importInput = new Legacy.Data.TextLoader(dataPath);
                var importOutput = experiment.Add(importInput);

                var crossValidate = new Legacy.Models.CrossValidator
                {
                    Data = importOutput.Data,
                    Nodes = subGraph,
                    Kind = Legacy.Models.MacroUtilsTrainerKinds.SignatureMultiClassClassifierTrainer,
                    TransformModel = null
                };
                crossValidate.Inputs.Data = nop.Data;
                crossValidate.Outputs.PredictorModel = modelCombineOutput.PredictorModel;
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
                    var foldGetter = cursor.GetGetter<ReadOnlyMemory<char>>(foldCol);
                    ReadOnlyMemory<char> fold = default;

                    // Get the average.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double avg = 0;
                    getter(ref avg);
                    foldGetter(ref fold);
                    Assert.True(ReadOnlyMemoryUtils.EqualsStr("Average", fold));

                    // Get the standard deviation.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double stdev = 0;
                    getter(ref stdev);
                    foldGetter(ref fold);
                    Assert.True(ReadOnlyMemoryUtils.EqualsStr("Standard Deviation", fold));
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
                        Assert.True(ReadOnlyMemoryUtils.EqualsStr("Fold " + f, fold));
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
                Assert.True(type is VectorType vecType && vecType.ItemType is TextType && vecType.Size == 10);
                var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
                schema.GetMetadata(MetadataUtils.Kinds.SlotNames, countCol, ref slotNames);
                Assert.True(slotNames.Values.Select((s, i) => ReadOnlyMemoryUtils.EqualsStr(i.ToString(), s)).All(x => x));
                using (var curs = confusion.GetRowCursor(col => true))
                {
                    var countGetter = curs.GetGetter<VBuffer<double>>(countCol);
                    var foldGetter = curs.GetGetter<ReadOnlyMemory<char>>(foldCol);
                    var confCount = default(VBuffer<double>);
                    var foldIndex = default(ReadOnlyMemory<char>);
                    int rowCount = 0;
                    var foldCur = "Fold 0";
                    while (curs.MoveNext())
                    {
                        countGetter(ref confCount);
                        foldGetter(ref foldIndex);
                        rowCount++;
                        Assert.True(ReadOnlyMemoryUtils.EqualsStr(foldCur, foldIndex));
                        if (rowCount == 10)
                        {
                            rowCount = 0;
                            foldCur = "Fold 1";
                        }
                    }
                    Assert.Equal(0, rowCount);
                }

                var warnings = experiment.GetOutput(crossValidateOutput.Warnings);
                using (var cursor = warnings.GetRowCursor(col => true))
                    Assert.False(cursor.MoveNext());
            }
        }

        [Fact]
        public void TestCrossValidationMacroMultiClassWithWarnings()
        {
            var dataPath = GetDataPath(@"Train-Tiny-28x28.txt");
            using (var env = new ConsoleEnvironment(42))
            {
                var subGraph = env.CreateExperiment();

                var nop = new Legacy.Transforms.NoOperation();
                var nopOutput = subGraph.Add(nop);

                var learnerInput = new Legacy.Trainers.LogisticRegressionClassifier
                {
                    TrainingData = nopOutput.OutputData,
                    NumThreads = 1
                };
                var learnerOutput = subGraph.Add(learnerInput);

                var experiment = env.CreateExperiment();
                var importInput = new Legacy.Data.TextLoader(dataPath);
                var importOutput = experiment.Add(importInput);

                var filter = new Legacy.Transforms.RowRangeFilter();
                filter.Data = importOutput.Data;
                filter.Column = "Label";
                filter.Min = 0;
                filter.Max = 5;
                var filterOutput = experiment.Add(filter);

                var term = new Legacy.Transforms.TextToKeyConverter();
                term.Column = new[]
                {
                    new Legacy.Transforms.TermTransformColumn()
                    {
                        Source = "Label", Name = "Strat", Sort = Legacy.Transforms.TermTransformSortOrder.Value
                    }
                };
                term.Data = filterOutput.OutputData;
                var termOutput = experiment.Add(term);

                var crossValidate = new Legacy.Models.CrossValidator
                {
                    Data = termOutput.OutputData,
                    Nodes = subGraph,
                    Kind = Legacy.Models.MacroUtilsTrainerKinds.SignatureMultiClassClassifierTrainer,
                    TransformModel = null,
                    StratificationColumn = "Strat"
                };
                crossValidate.Inputs.Data = nop.Data;
                crossValidate.Outputs.PredictorModel = learnerOutput.PredictorModel;
                var crossValidateOutput = experiment.Add(crossValidate);

                experiment.Compile();
                importInput.SetInput(env, experiment);
                experiment.Run();
                var warnings = experiment.GetOutput(crossValidateOutput.Warnings);

                var schema = warnings.Schema;
                var b = schema.TryGetColumnIndex("WarningText", out int warningCol);
                Assert.True(b);
                using (var cursor = warnings.GetRowCursor(col => col == warningCol))
                {
                    var getter = cursor.GetGetter<ReadOnlyMemory<char>>(warningCol);

                    b = cursor.MoveNext();
                    Assert.True(b);
                    var warning = default(ReadOnlyMemory<char>);
                    getter(ref warning);
                    Assert.Contains("test instances with class values not seen in the training set.", warning.ToString());
                    b = cursor.MoveNext();
                    Assert.True(b);
                    getter(ref warning);
                    Assert.Contains("Detected columns of variable length: SortedScores, SortedClasses", warning.ToString());
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }

        [Fact]
        public void TestCrossValidationMacroWithStratification()
        {
            var dataPath = GetDataPath(@"breast-cancer.txt");
            using (var env = new ConsoleEnvironment(42))
            {
                var subGraph = env.CreateExperiment();

                var nop = new Legacy.Transforms.NoOperation();
                var nopOutput = subGraph.Add(nop);

                var learnerInput = new Legacy.Trainers.StochasticDualCoordinateAscentBinaryClassifier
                {
                    TrainingData = nopOutput.OutputData,
                    NumThreads = 1
                };
                var learnerOutput = subGraph.Add(learnerInput);

                var modelCombine = new Legacy.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(nopOutput.Model),
                    PredictorModel = learnerOutput.PredictorModel
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                var experiment = env.CreateExperiment();
                var importInput = new Legacy.Data.TextLoader(dataPath);
                importInput.Arguments.Column = new Legacy.Data.TextLoaderColumn[]
                {
                    new Legacy.Data.TextLoaderColumn { Name = "Label", Source = new[] { new Legacy.Data.TextLoaderRange(0) } },
                    new Legacy.Data.TextLoaderColumn { Name = "Strat", Source = new[] { new Legacy.Data.TextLoaderRange(1) } },
                    new Legacy.Data.TextLoaderColumn { Name = "Features", Source = new[] { new Legacy.Data.TextLoaderRange(2, 9) } }
                };
                var importOutput = experiment.Add(importInput);

                var crossValidate = new Legacy.Models.CrossValidator
                {
                    Data = importOutput.Data,
                    Nodes = subGraph,
                    TransformModel = null,
                    StratificationColumn = "Strat"
                };
                crossValidate.Inputs.Data = nop.Data;
                crossValidate.Outputs.PredictorModel = modelCombineOutput.PredictorModel;
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
                    var foldGetter = cursor.GetGetter<ReadOnlyMemory<char>>(foldCol);
                    ReadOnlyMemory<char> fold = default;

                    // Get the verage.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double avg = 0;
                    getter(ref avg);
                    foldGetter(ref fold);
                    Assert.True(ReadOnlyMemoryUtils.EqualsStr("Average", fold));

                    // Get the standard deviation.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double stdev = 0;
                    getter(ref stdev);
                    foldGetter(ref fold);
                    Assert.True(ReadOnlyMemoryUtils.EqualsStr("Standard Deviation", fold));
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
                        Assert.True(ReadOnlyMemoryUtils.EqualsStr("Fold " + f, fold));
                    }
                    Assert.Equal(avg, sum / 2);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }

        [Fact]
        public void TestCrossValidationMacroWithNonDefaultNames()
        {
            string dataPath = GetDataPath(@"adult.tiny.with-schema.txt");
            using (var env = new ConsoleEnvironment(42))
            {
                var subGraph = env.CreateExperiment();

                var textToKey = new Legacy.Transforms.TextToKeyConverter();
                textToKey.Column = new[] { new Legacy.Transforms.TermTransformColumn() { Name = "Label1", Source = "Label" } };
                var textToKeyOutput = subGraph.Add(textToKey);

                var hash = new Legacy.Transforms.HashConverter();
                hash.Column = new[] { new Legacy.Transforms.HashJoinTransformColumn() { Name = "GroupId1", Source = "Workclass" } };
                hash.Data = textToKeyOutput.OutputData;
                var hashOutput = subGraph.Add(hash);

                var learnerInput = new Legacy.Trainers.FastTreeRanker
                {
                    TrainingData = hashOutput.OutputData,
                    NumThreads = 1,
                    LabelColumn = "Label1",
                    GroupIdColumn = "GroupId1"
                };
                var learnerOutput = subGraph.Add(learnerInput);

                var modelCombine = new Legacy.Transforms.ManyHeterogeneousModelCombiner
                {
                    TransformModels = new ArrayVar<ITransformModel>(textToKeyOutput.Model, hashOutput.Model),
                    PredictorModel = learnerOutput.PredictorModel
                };
                var modelCombineOutput = subGraph.Add(modelCombine);

                var experiment = env.CreateExperiment();
                var importInput = new Legacy.Data.TextLoader(dataPath);
                importInput.Arguments.HasHeader = true;
                importInput.Arguments.Column = new TextLoaderColumn[]
                {
                    new TextLoaderColumn { Name = "Label", Source = new[] { new TextLoaderRange(0) } },
                    new TextLoaderColumn { Name = "Workclass", Source = new[] { new TextLoaderRange(1) }, Type = Legacy.Data.DataKind.Text },
                    new TextLoaderColumn { Name = "Features", Source = new[] { new TextLoaderRange(9, 14) } }
                };
                var importOutput = experiment.Add(importInput);

                var crossValidate = new Legacy.Models.CrossValidator
                {
                    Data = importOutput.Data,
                    Nodes = subGraph,
                    TransformModel = null,
                    LabelColumn = "Label1",
                    GroupColumn = "GroupId1",
                    NameColumn = "Workclass",
                    Kind = Legacy.Models.MacroUtilsTrainerKinds.SignatureRankerTrainer
                };
                crossValidate.Inputs.Data = textToKey.Data;
                crossValidate.Outputs.PredictorModel = modelCombineOutput.PredictorModel;
                var crossValidateOutput = experiment.Add(crossValidate);
                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();
                var data = experiment.GetOutput(crossValidateOutput.OverallMetrics);

                var schema = data.Schema;
                var b = schema.TryGetColumnIndex("NDCG", out int metricCol);
                Assert.True(b);
                b = schema.TryGetColumnIndex("Fold Index", out int foldCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == metricCol || col == foldCol))
                {
                    var getter = cursor.GetGetter<VBuffer<double>>(metricCol);
                    var foldGetter = cursor.GetGetter<ReadOnlyMemory<char>>(foldCol);
                    ReadOnlyMemory<char> fold = default;

                    // Get the verage.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    var avg = default(VBuffer<double>);
                    getter(ref avg);
                    foldGetter(ref fold);
                    Assert.True(ReadOnlyMemoryUtils.EqualsStr("Average", fold));

                    // Get the standard deviation.
                    b = cursor.MoveNext();
                    Assert.True(b);
                    var stdev = default(VBuffer<double>);
                    getter(ref stdev);
                    foldGetter(ref fold);
                    Assert.True(ReadOnlyMemoryUtils.EqualsStr("Standard Deviation", fold));
                    Assert.Equal(2.462, stdev.Values[0], 3);
                    Assert.Equal(2.763, stdev.Values[1], 3);
                    Assert.Equal(3.273, stdev.Values[2], 3);

                    var sumBldr = new BufferBuilder<double>(R8Adder.Instance);
                    sumBldr.Reset(avg.Length, true);
                    var val = default(VBuffer<double>);
                    for (int f = 0; f < 2; f++)
                    {
                        b = cursor.MoveNext();
                        Assert.True(b);
                        getter(ref val);
                        foldGetter(ref fold);
                        sumBldr.AddFeatures(0, in val);
                        Assert.True(ReadOnlyMemoryUtils.EqualsStr("Fold " + f, fold));
                    }
                    var sum = default(VBuffer<double>);
                    sumBldr.GetResult(ref sum);
                    for (int i = 0; i < avg.Length; i++)
                        Assert.Equal(avg.Values[i], sum.Values[i] / 2);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }

                data = experiment.GetOutput(crossValidateOutput.PerInstanceMetrics);
                Assert.True(data.Schema.TryGetColumnIndex("Instance", out int nameCol));
                using (var cursor = data.GetRowCursor(col => col == nameCol))
                {
                    var getter = cursor.GetGetter<ReadOnlyMemory<char>>(nameCol);
                    while (cursor.MoveNext())
                    {
                        ReadOnlyMemory<char> name = default;
                        getter(ref name);
                        Assert.Subset(new HashSet<string>() { "Private", "?", "Federal-gov" }, new HashSet<string>() { name.ToString() });
                        if (cursor.Position > 4)
                            break;
                    }
                }
            }
        }

        [Fact]
        public void TestOvaMacro()
        {
            var dataPath = GetDataPath(@"iris.txt");
            using (var env = new ConsoleEnvironment(42))
            {
                // Specify subgraph for OVA
                var subGraph = env.CreateExperiment();
                var learnerInput = new Legacy.Trainers.StochasticDualCoordinateAscentBinaryClassifier { NumThreads = 1 };
                var learnerOutput = subGraph.Add(learnerInput);
                // Create pipeline with OVA and multiclass scoring.
                var experiment = env.CreateExperiment();
                var importInput = new Legacy.Data.TextLoader(dataPath);
                importInput.Arguments.Column = new TextLoaderColumn[]
                {
                    new TextLoaderColumn { Name = "Label", Source = new[] { new TextLoaderRange(0) } },
                    new TextLoaderColumn { Name = "Features", Source = new[] { new TextLoaderRange(1,4) } }
                };
                var importOutput = experiment.Add(importInput);
                var oneVersusAll = new Legacy.Models.OneVersusAll
                {
                    TrainingData = importOutput.Data,
                    Nodes = subGraph,
                    UseProbabilities = true,
                };
                var ovaOutput = experiment.Add(oneVersusAll);
                var scoreInput = new Legacy.Transforms.DatasetScorer
                {
                    Data = importOutput.Data,
                    PredictorModel = ovaOutput.PredictorModel
                };
                var scoreOutput = experiment.Add(scoreInput);
                var evalInput = new Legacy.Models.ClassificationEvaluator
                {
                    Data = scoreOutput.ScoredData
                };
                var evalOutput = experiment.Add(evalInput);
                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();

                var data = experiment.GetOutput(evalOutput.OverallMetrics);
                var schema = data.Schema;
                var b = schema.TryGetColumnIndex(MultiClassClassifierEvaluator.AccuracyMacro, out int accCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == accCol))
                {
                    var getter = cursor.GetGetter<double>(accCol);
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double acc = 0;
                    getter(ref acc);
                    Assert.Equal(0.96, acc, 2);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }

        [Fact]
        public void TestOvaMacroWithUncalibratedLearner()
        {
            var dataPath = GetDataPath(@"iris.txt");
            using (var env = new ConsoleEnvironment(42))
            {
                // Specify subgraph for OVA
                var subGraph = env.CreateExperiment();
                var learnerInput = new Legacy.Trainers.AveragedPerceptronBinaryClassifier { Shuffle = false };
                var learnerOutput = subGraph.Add(learnerInput);
                // Create pipeline with OVA and multiclass scoring.
                var experiment = env.CreateExperiment();
                var importInput = new Legacy.Data.TextLoader(dataPath);
                importInput.Arguments.Column = new TextLoaderColumn[]
                {
                    new TextLoaderColumn { Name = "Label", Source = new[] { new TextLoaderRange(0) } },
                    new TextLoaderColumn { Name = "Features", Source = new[] { new TextLoaderRange(1,4) } }
                };
                var importOutput = experiment.Add(importInput);
                var oneVersusAll = new Legacy.Models.OneVersusAll
                {
                    TrainingData = importOutput.Data,
                    Nodes = subGraph,
                    UseProbabilities = true,
                };
                var ovaOutput = experiment.Add(oneVersusAll);
                var scoreInput = new Legacy.Transforms.DatasetScorer
                {
                    Data = importOutput.Data,
                    PredictorModel = ovaOutput.PredictorModel
                };
                var scoreOutput = experiment.Add(scoreInput);
                var evalInput = new Legacy.Models.ClassificationEvaluator
                {
                    Data = scoreOutput.ScoredData
                };
                var evalOutput = experiment.Add(evalInput);
                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();

                var data = experiment.GetOutput(evalOutput.OverallMetrics);
                var schema = data.Schema;
                var b = schema.TryGetColumnIndex(MultiClassClassifierEvaluator.AccuracyMacro, out int accCol);
                Assert.True(b);
                using (var cursor = data.GetRowCursor(col => col == accCol))
                {
                    var getter = cursor.GetGetter<double>(accCol);
                    b = cursor.MoveNext();
                    Assert.True(b);
                    double acc = 0;
                    getter(ref acc);
                    Assert.Equal(0.71, acc, 2);
                    b = cursor.MoveNext();
                    Assert.False(b);
                }
            }
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TestTensorFlowEntryPoint()
        {
            var dataPath = GetDataPath("Train-Tiny-28x28.txt");
            using (var env = new ConsoleEnvironment(42))
            {
                var experiment = env.CreateExperiment();

                var importInput = new Legacy.Data.TextLoader(dataPath);
                importInput.Arguments.Column = new TextLoaderColumn[]
                {
                    new TextLoaderColumn { Name = "Label", Source = new[] { new TextLoaderRange(0) } },
                    new TextLoaderColumn { Name = "Placeholder", Source = new[] { new TextLoaderRange(1, 784) } }
                };
                var importOutput = experiment.Add(importInput);

                var tfTransformInput = new Legacy.Transforms.TensorFlowScorer
                {
                    Data = importOutput.Data,
                    ModelLocation = "mnist_model/frozen_saved_model.pb",
                    InputColumns = new[] { "Placeholder" },
                    OutputColumns = new[] { "Softmax" },
                };
                var tfTransformOutput = experiment.Add(tfTransformInput);

                experiment.Compile();
                experiment.SetInput(importInput.InputFile, new SimpleFileHandle(env, dataPath, false, false));
                experiment.Run();
                var data = experiment.GetOutput(tfTransformOutput.OutputData);

                var schema = data.Schema;
                Assert.Equal(3, schema.ColumnCount);
                Assert.Equal("Softmax", schema.GetColumnName(2));
                Assert.Equal(10, (schema.GetColumnType(2) as VectorType)?.Size);
            }
        }
    }
}
