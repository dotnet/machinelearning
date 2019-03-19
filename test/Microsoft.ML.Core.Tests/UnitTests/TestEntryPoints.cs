// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Core.Tests.UnitTests;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Ensemble;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Transforms.TimeSeries;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public partial class TestEntryPoints : CoreBaseTestClass
    {
        public TestEntryPoints(ITestOutputHelper output) : base(output)
        {
            Env.ComponentCatalog.RegisterAssembly(typeof(ExponentialAverageTransform).Assembly);
        }

        private IDataView GetBreastCancerDataView()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            return EntryPoints.ImportTextData.TextLoader(Env, new EntryPoints.ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    Columns = new[]
                    {
                        new TextLoader.Column("Label", DataKind.Single, 0),
                        new TextLoader.Column("Features", DataKind.Single,
                            new [] { new TextLoader.Range(1, 9) })
                    }
                },

                InputFile = inputFile
            }).Data;
        }

        private IDataView GetBreastCancerDataviewWithTextColumns()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            return EntryPoints.ImportTextData.TextLoader(Env, new EntryPoints.ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    HasHeader = true,
                    Columns = new[]
                    {
                        new TextLoader.Column("Label", DataKind.Single, 0),
                        new TextLoader.Column("F1", DataKind.String, 1),
                        new TextLoader.Column("F2", DataKind.Int32, 2),
                        new TextLoader.Column("Rest", DataKind.Single, new [] { new TextLoader.Range(3, 9) })
                    }
                },

                InputFile = inputFile
            }).Data;
        }


        [Fact]
        public void EntryPointTrainTestSplit()
        {
            var dataView = GetBreastCancerDataView();
            var splitOutput = TrainTestSplit.Split(Env, new TrainTestSplit.Input { Data = dataView, Fraction = 0.9f });

            int totalRows = CountRows(dataView);
            int trainRows = CountRows(splitOutput.TrainData);
            int testRows = CountRows(splitOutput.TestData);

            Assert.Equal(totalRows, trainRows + testRows);
            Assert.Equal(0.9, (double)trainRows / totalRows, 1);
        }

        private static int CountRows(IDataView dataView)
        {
            int totalRows = 0;
            using (var cursor = dataView.GetRowCursor())
            {
                while (cursor.MoveNext())
                    totalRows++;
            }

            return totalRows;
        }

        [Fact()]
        public void EntryPointFeatureCombiner()
        {
            var dataView = GetBreastCancerDataviewWithTextColumns();

            dataView = Env.CreateTransform("Term{col=F1}", dataView);
            var result = FeatureCombiner.PrepareFeatures(Env, new FeatureCombiner.FeatureCombinerInput() { Data = dataView, Features = new[] { "F1", "F2", "Rest" } }).OutputData;
            var expected = Env.CreateTransform("Convert{col=F2 type=R4}", dataView);
            expected = Env.CreateTransform("KeyToValue{col=F1}", expected);
            expected = Env.CreateTransform("Term{col=F1}", expected);
            expected = Env.CreateTransform("KeyToVector{col=F1}", expected);
            expected = Env.CreateTransform("Concat{col=Features:F1,F2,Rest}", expected);

            expected = Env.CreateTransform("SelectColumns{keepcol=Features hidden=-}", expected);
            result = Env.CreateTransform("SelectColumns{keepcol=Features hidden=-}", result);
            CheckSameValues(result, expected);
            Done();
        }

        [Fact]
        public void EntryPointScoring()
        {
            var dataView = GetBreastCancerDataviewWithTextColumns();
            dataView = Env.CreateTransform("Term{col=F1}", dataView);
            var trainData = FeatureCombiner.PrepareFeatures(Env, new FeatureCombiner.FeatureCombinerInput() { Data = dataView, Features = new[] { "F1", "F2", "Rest" } });
            var lrModel = LogisticRegressionBinaryTrainer.TrainBinary(Env, new LogisticRegressionBinaryTrainer.Options { TrainingData = trainData.OutputData }).PredictorModel;
            var model = ModelOperations.CombineTwoModels(Env, new ModelOperations.SimplePredictorModelInput() { TransformModel = trainData.Model, PredictorModel = lrModel }).PredictorModel;

            var scored1 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = dataView, PredictorModel = model }).ScoredData;
            scored1 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored1, ExtraColumns = new[] { "Label" } }).OutputData;

            var scored2 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = dataView, PredictorModel = lrModel.Apply(Env, trainData.Model) }).ScoredData;
            scored2 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored2, ExtraColumns = new[] { "Label" } }).OutputData;

            Assert.Equal(4, scored1.Schema.Count);
            CheckSameValues(scored1, scored2);
            Done();
        }

        [Fact]
        public void EntryPointApplyModel()
        {
            var dataView = GetBreastCancerDataviewWithTextColumns();

            dataView = Env.CreateTransform("Term{col=F1}", dataView);

            var data1 = FeatureCombiner.PrepareFeatures(Env, new FeatureCombiner.FeatureCombinerInput() { Data = dataView, Features = new[] { "F1", "F2", "Rest" } });
            var data2 = ModelOperations.Apply(Env, new ModelOperations.ApplyTransformModelInput() { Data = dataView, TransformModel = data1.Model });

            CheckSameValues(data1.OutputData, data2.OutputData);
            Done();
        }

        //[Fact]
        //public void EntryPointSchemaManipulation()
        //{
        //    var dv1_data = new[]
        //    {
        //        new Dv1 { Col1 = 11, Col2 = 21, Col3 = 31, Col4 = 41 },
        //        new Dv1 { Col1 = 12, Col2 = 22, Col3 = 32, Col4 = 42 },
        //        new Dv1 { Col1 = 13, Col2 = 23, Col3 = 33, Col4 = 43 },
        //    };
        //    var dv1 = Env.CreateDataView(dv1_data);

        //    var concatOut = SchemaManipulation.ConcatColumns(Env,
        //        new ConcatTransform.Arguments { Column = new[] { ConcatTransform.Column.Parse("ColA:Col1,Col2") }, Data = dv1 });

        //    var postConcatDv = Env.CreateTransform("Concat{col=ColA:Col1,Col2}", dv1);
        //    CheckSameValues(concatOut.OutputData, postConcatDv);

        //    var copyOut = SchemaManipulation.CopyColumns(Env,
        //        new CopyColumnsTransform.Arguments
        //        {
        //            Column = new[] { CopyColumnsTransform.Column.Parse("ColB:Col3"), CopyColumnsTransform.Column.Parse("ColC:Col4") },
        //            Data = concatOut.OutputData,
        //        });

        //    var postCopyDv = Env.CreateTransform("Copy{col=ColB:Col3 col=ColC:Col4}", postConcatDv);
        //    CheckSameValues(copyOut.OutputData, postCopyDv);

        //    var dropOut = SchemaManipulation.DropColumns(Env,
        //        new DropColumnsTransform.Arguments { Column = new[] { "Col1", "Col2", "Col3", "Col4" }, Data = copyOut.OutputData });

        //    var postDropDv = Env.CreateTransform("Drop{col=Col1 col=Col2 col=Col3 col=Col4}", postCopyDv);
        //    CheckSameValues(dropOut.OutputData, postDropDv);

        //    var selectOut = SchemaManipulation.SelectColumns(Env,
        //        new DropColumnsTransform.KeepArguments { Column = new[] { "ColA", "ColB" }, Data = dropOut.OutputData });

        //    var postSelectDv = Env.CreateTransform("Keep{col=ColA col=ColB}", postDropDv);

        //    CheckSameValues(selectOut.OutputData, postSelectDv);

        //    var combinedModel = ModelOperations.CombineTransformModels(Env,
        //        new ModelOperations.CombineTransformModelsInput
        //        {
        //            Models = new[] { concatOut.Model, copyOut.Model, dropOut.Model, selectOut.Model }
        //        }).OutputModel;
        //    CheckSameValues(selectOut.OutputData, combinedModel.Apply(Env, dv1));
        //    Done();
        //}

        /// <summary>
        /// Helper function to get the type of build being used.
        /// </summary>
        /// <returns>Returns "core", "pub", or "dev" depending on the build type.</returns>
        private string GetBuildPrefix()
        {
#if CORECLR
            return "core";
#elif TLCFULLBUILD
            return "dev";
#else
            return "pub";
#endif
        }

        [Fact(Skip = "Execute this test if you want to regenerate the core_manifest and core_ep_list files")]
        public void RegenerateEntryPointCatalog()
        {
            var (epListContents, jObj) = BuildManifests();

            var buildPrefix = GetBuildPrefix();
            var epListFile = buildPrefix + "_ep-list.tsv";

            var entryPointsSubDir = Path.Combine("..", "Common", "EntryPoints");
            var catalog = Env.ComponentCatalog;
            var epListPath = GetBaselinePath(entryPointsSubDir, epListFile);
            DeleteOutputPath(epListPath);

            File.WriteAllLines(epListPath, epListContents);

            var manifestFile = buildPrefix + "_manifest.json";
            var manifestPath = GetBaselinePath(entryPointsSubDir, manifestFile);
            DeleteOutputPath(manifestPath);

            using (var file = File.OpenWrite(manifestPath))
            using (var writer = new StreamWriter(file))
            using (var jw = new JsonTextWriter(writer))
            {
                jw.Formatting = Formatting.Indented;
                jObj.WriteTo(jw);
            }
        }

        [Fact]
        public void EntryPointCatalog()
        {
            var (epListContents, jObj) = BuildManifests();

            var buildPrefix = GetBuildPrefix();
            var epListFile = buildPrefix + "_ep-list.tsv";

            var entryPointsSubDir = Path.Combine("..", "Common", "EntryPoints");
            var catalog = Env.ComponentCatalog;
            var path = DeleteOutputPath(entryPointsSubDir, epListFile);

            File.WriteAllLines(path, epListContents);

            CheckEquality(entryPointsSubDir, epListFile);

            var manifestFile = buildPrefix + "_manifest.json";
            var jPath = DeleteOutputPath(entryPointsSubDir, manifestFile);
            using (var file = File.OpenWrite(jPath))
            using (var writer = new StreamWriter(file))
            using (var jw = new JsonTextWriter(writer))
            {
                jw.Formatting = Formatting.Indented;
                jObj.WriteTo(jw);
            }

            CheckEquality(entryPointsSubDir, manifestFile);
            Done();
        }

        [Fact]
        public void EntryPointCatalogCheckDuplicateParams()
        {
            // Run this test to prevent introducing duplicate param names in entrypoints
            // TODO: fix entrypoints in excludeSet from having duplicate param names
            var excludeSet = new HashSet<string>();
            excludeSet.Add("Data.DataViewReference");
            excludeSet.Add("Models.CrossValidator");
            excludeSet.Add("Models.CrossValidationResultsCombiner");
            excludeSet.Add("Models.PipelineSweeper");
            excludeSet.Add("Models.PipelineSweeper");
            excludeSet.Add("Models.SweepResultExtractor");
            excludeSet.Add("Models.TrainTestEvaluator");
            excludeSet.Add("Transforms.TwoHeterogeneousModelCombiner");
            excludeSet.Add("Transforms.ManyHeterogeneousModelCombiner");

            var (epListContents, jObj) = BuildManifests();
            foreach (var ep in jObj["EntryPoints"])
            {
                if (excludeSet.Contains(ep["Name"].ToString()))
                    continue;

                var variables = new HashSet<string>();
                foreach (var param in ep["Inputs"])
                {
                    var name = param["Name"].ToString();
                    Check(variables.Add(name), "Duplicate param {0} in entrypoint {1}", name, ep["Name"]);
                }
                foreach (var param in ep["Outputs"])
                {
                    var name = param["Name"].ToString();
                    Check(variables.Add(name), "Duplicate param {0} in entrypoint {1}", name, ep["Name"]);
                }
            }

            Done();
        }

        private (IEnumerable<string> epListContents, JObject manifest) BuildManifests()
        {
            Env.ComponentCatalog.RegisterAssembly(typeof(LightGbmBinaryModelParameters).Assembly);
            Env.ComponentCatalog.RegisterAssembly(typeof(TensorFlowTransformer).Assembly);
            Env.ComponentCatalog.RegisterAssembly(typeof(ImageLoadingTransformer).Assembly);
            Env.ComponentCatalog.RegisterAssembly(typeof(SymbolicSgdTrainer).Assembly);
            Env.ComponentCatalog.RegisterAssembly(typeof(SaveOnnxCommand).Assembly);
            Env.ComponentCatalog.RegisterAssembly(typeof(TimeSeriesProcessingEntryPoints).Assembly);
            Env.ComponentCatalog.RegisterAssembly(typeof(ParquetLoader).Assembly);

            var catalog = Env.ComponentCatalog;

            var regex = new Regex(@"\r\n?|\n", RegexOptions.Compiled);
            var epListContents = catalog.AllEntryPoints()
                .Select(x => string.Join("\t",
                x.Name,
                regex.Replace(x.Description, ""),
                x.Method.DeclaringType,
                x.Method.Name,
                x.InputType,
                x.OutputType)
                .Replace(Environment.NewLine, ""))
                .OrderBy(x => x);

            var manifest = JsonManifestUtils.BuildAllManifests(Env, catalog);

            //clean up the description from the new line characters
            if (manifest[FieldNames.TopEntryPoints] != null && manifest[FieldNames.TopEntryPoints] is JArray)
            {
                foreach (JToken entry in manifest[FieldNames.TopEntryPoints].Children())
                    if (entry[FieldNames.Desc] != null)
                        entry[FieldNames.Desc] = regex.Replace(entry[FieldNames.Desc].ToString(), "");
            }

            return (epListContents, manifest);
        }

        [Fact]
        public void EntryPointInputBuilderOptionals()
        {
            var catalog = Env.ComponentCatalog;

            InputBuilder ib1 = new InputBuilder(Env, typeof(LogisticRegressionBinaryTrainer.Options), catalog);
            // Ensure that InputBuilder unwraps the Optional<string> correctly.
            var weightType = ib1.GetFieldTypeOrNull("ExampleWeightColumnName");
            Assert.True(weightType.Equals(typeof(string)));

            var instance = ib1.GetInstance() as LogisticRegressionBinaryTrainer.Options;
            Assert.True(instance.ExampleWeightColumnName == null);

            ib1.TrySetValue("ExampleWeightColumnName", "OtherWeight");
            Assert.Equal("OtherWeight", instance.ExampleWeightColumnName);

            var tok = (JToken)JValue.CreateString("AnotherWeight");
            ib1.TrySetValueJson("ExampleWeightColumnName", tok);
            Assert.Equal("AnotherWeight", instance.ExampleWeightColumnName);
        }

        [Fact]
        public void EntryPointInputRangeChecks()
        {
            TlcModule.RangeAttribute range = null;

            range = new TlcModule.RangeAttribute() { Min = 5.0 };
            Assert.True(EntryPointUtils.IsValueWithinRange(range, 5.1));
            Assert.True(EntryPointUtils.IsValueWithinRange(range, 5.0));
            Assert.False(EntryPointUtils.IsValueWithinRange(range, 4.9));

            range = new TlcModule.RangeAttribute() { Inf = 5.0 };
            Assert.True(EntryPointUtils.IsValueWithinRange(range, 5.1));
            Assert.False(EntryPointUtils.IsValueWithinRange(range, 5.0));
            Assert.False(EntryPointUtils.IsValueWithinRange(range, 4.9));

            range = new TlcModule.RangeAttribute() { Max = 5.0 };
            Assert.False(EntryPointUtils.IsValueWithinRange(range, 5.1));
            Assert.True(EntryPointUtils.IsValueWithinRange(range, 5.0));
            Assert.True(EntryPointUtils.IsValueWithinRange(range, 4.9));

            range = new TlcModule.RangeAttribute() { Sup = 5.0 };
            Assert.False(EntryPointUtils.IsValueWithinRange(range, 5.1));
            Assert.False(EntryPointUtils.IsValueWithinRange(range, 5.0));
            Assert.True(EntryPointUtils.IsValueWithinRange(range, 4.9));

            range = new TlcModule.RangeAttribute() { Max = 1.0, Min = -1.0 };
            Assert.False(EntryPointUtils.IsValueWithinRange(range, -1.1));
            Assert.False(EntryPointUtils.IsValueWithinRange(range, 1.1));
            Assert.True(EntryPointUtils.IsValueWithinRange(range, 0.0));
        }

        [Fact]
        public void EntryPointCreateEnsemble()
        {
            var dataView = GetBreastCancerDataView();
            const int nModels = 5;
            var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = nModels + 1 });
            var predictorModels = new PredictorModel[nModels];
            var individualScores = new IDataView[nModels];
            for (int i = 0; i < nModels; i++)
            {
                var data = splitOutput.TrainData[i];
                var lrInput = new LogisticRegressionBinaryTrainer.Options
                {
                    TrainingData = data,
                    L1Regularization = (Single)0.1 * i,
                    L2Regularization = (Single)0.01 * (1 + i),
                    NormalizeFeatures = NormalizeOption.No
                };
                predictorModels[i] = LogisticRegressionBinaryTrainer.TrainBinary(Env, lrInput).PredictorModel;
                individualScores[i] =
                    ScoreModel.Score(Env,
                        new ScoreModel.Input { Data = splitOutput.TestData[nModels], PredictorModel = predictorModels[i] })
                        .ScoredData;
                individualScores[i] = new ColumnCopyingTransformer(Env, (
                    (AnnotationUtils.Const.ScoreValueKind.Score + i).ToString(),
                     AnnotationUtils.Const.ScoreValueKind.Score)
                    ).Transform(individualScores[i]);

                individualScores[i] = new ColumnSelectingTransformer(Env, null, new[] { AnnotationUtils.Const.ScoreValueKind.Score }).Transform(individualScores[i]);
            }

            var avgEnsembleInput = new EnsembleCreator.ClassifierInput { Models = predictorModels, ModelCombiner = EnsembleCreator.ClassifierCombiner.Average };
            var avgEnsemble = EnsembleCreator.CreateBinaryEnsemble(Env, avgEnsembleInput).PredictorModel;
            var avgScored =
                ScoreModel.Score(Env,
                    new ScoreModel.Input { Data = splitOutput.TestData[nModels], PredictorModel = avgEnsemble }).ScoredData;

            var medEnsembleInput = new EnsembleCreator.ClassifierInput { Models = predictorModels };
            var medEnsemble = EnsembleCreator.CreateBinaryEnsemble(Env, medEnsembleInput).PredictorModel;
            var medScored =
                ScoreModel.Score(Env,
                new ScoreModel.Input { Data = splitOutput.TestData[nModels], PredictorModel = medEnsemble }).ScoredData;

            var regEnsembleInput = new EnsembleCreator.RegressionInput() { Models = predictorModels };
            var regEnsemble = EnsembleCreator.CreateRegressionEnsemble(Env, regEnsembleInput).PredictorModel;
            var regScored =
                ScoreModel.Score(Env,
                new ScoreModel.Input { Data = splitOutput.TestData[nModels], PredictorModel = regEnsemble }).ScoredData;

            var zippedScores = ZipDataView.Create(Env, individualScores);

            var avgComb = new Average(Env).GetCombiner();
            var medComb = new Median(Env).GetCombiner();
            using (var curs1 = avgScored.GetRowCursorForAllColumns())
            using (var curs2 = medScored.GetRowCursorForAllColumns())
            using (var curs3 = regScored.GetRowCursorForAllColumns())
            using (var curs4 = zippedScores.GetRowCursorForAllColumns())
            {
                var scoreColumn = curs1.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var avgScoreGetter = curs1.GetGetter<Single>(scoreColumn.Value);

                scoreColumn = curs2.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var medScoreGetter = curs2.GetGetter<Single>(scoreColumn.Value);

                scoreColumn = curs3.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var regScoreGetter = curs3.GetGetter<Single>(scoreColumn.Value);

                var individualScoreGetters = new ValueGetter<Single>[nModels];
                for (int i = 0; i < nModels; i++)
                {
                    scoreColumn = curs4.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score + i);
                    individualScoreGetters[i] = curs4.GetGetter<Single>(scoreColumn.Value);
                }

                var scoreBuffer = new Single[nModels];
                while (curs1.MoveNext())
                {
                    var move = curs2.MoveNext();
                    Assert.True(move);
                    move = curs3.MoveNext();
                    Assert.True(move);
                    move = curs4.MoveNext();
                    Assert.True(move);

                    Single score = 0;
                    avgScoreGetter(ref score);
                    for (int i = 0; i < nModels; i++)
                        individualScoreGetters[i](ref scoreBuffer[i]);
                    Single avgScore = 0;
                    avgComb(ref avgScore, scoreBuffer, null);
                    Assert.Equal(score, avgScore);

                    medScoreGetter(ref score);
                    Single medScore = 0;
                    medComb(ref medScore, scoreBuffer, null);
                    Assert.Equal(score, medScore);

                    regScoreGetter(ref score);
                    Assert.Equal(score, medScore);
                }
                var moved = curs2.MoveNext();
                Assert.False(moved);
                moved = curs3.MoveNext();
                Assert.False(moved);
                moved = curs4.MoveNext();
                Assert.False(moved);
            }
        }

        [Fact]
        public void EntryPointOptionalParams()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {
                        'InputFile': '$file1'
                      },
                      'Outputs': {
                        'Data': '$data1'
                      }
                    },
                    {
                      'Name': 'Trainers.LogisticRegressionBinaryClassifier',
                      'Inputs': {
                        'TrainingData': '$data1',
                        'NumThreads': 1
                      },
                      'Outputs': {
                        'PredictorModel': '$model1'
                      }
                    }
                  ]
                }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file1", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<PredictorModel>("model1");
            Assert.NotNull(model);
        }

        protected static string EscapePath(string path)
        {
            return path.Replace("\\", "\\\\");
        }

        [Fact]
        public void EntryPointExecGraphCommand()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var modelPath = DeleteOutputPath("model.zip");

            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file1'
                      }},
                      'Outputs': {{
                        'Data': '$data1'
                      }}
                    }},
                    {{
                      'Name': 'Trainers.LogisticRegressionBinaryClassifier',
                      'Inputs': {{
                        'TrainingData': '$data1',
                        'NumThreads': 1
                      }},
                      'Outputs': {{
                        'PredictorModel': '$model1'
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file1' : '{0}'
                  }},
                  'Outputs' : {{
                    'model1' : '{1}'
                  }}
                }}", EscapePath(dataPath), EscapePath(modelPath));

            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();
        }

        //[Fact]
        //public void EntryPointArrayOfVariables()
        //{
        //    string inputGraph = @"
        //        {
        //          ""Nodes"": [
        //            {
        //              ""Name"": ""SchemaManipulation.ConcatColumns"",
        //              ""Inputs"": {
        //                ""Data"": ""$data1"",
        //                ""Column"": [{""Name"":""ColA"", ""Source"":[""Col1"", ""Col2""]}]
        //              },
        //              ""Outputs"": {
        //                ""Model"": ""$model1"",
        //                ""OutputData"": ""$data2""
        //              }
        //            },
        //            {
        //              ""Name"": ""SchemaManipulation.CopyColumns"",
        //              ""Inputs"": {
        //                ""Data"": ""$data2"",
        //                ""Column"": [{""Name"":""ColB"", ""Source"":""Col3""}, {""Name"":""ColC"", ""Source"":""Col4""}]
        //              },
        //              ""Outputs"": {
        //                ""Model"": ""$model2"",
        //                ""OutputData"": ""$data3""
        //              }
        //            },
        //            {
        //              ""Name"": ""ModelOperations.CombineTransformModels"",
        //              ""Inputs"": {
        //                Models: [""$model1"", ""$model2""]
        //              },
        //              ""Outputs"": {
        //                OutputModel: ""$model3""
        //              }
        //            }
        //          ]
        //        }";

        //    JObject graph = JObject.Parse(inputGraph);
        //    var catalog = Env.ComponentCatalog;
        //    var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);

        //    var dv1_data = new[]
        //    {
        //        new Dv1 { Col1 = 11, Col2 = 21, Col3 = 31, Col4 = 41 },
        //        new Dv1 { Col1 = 12, Col2 = 22, Col3 = 32, Col4 = 42 },
        //        new Dv1 { Col1 = 13, Col2 = 23, Col3 = 33, Col4 = 43 },
        //    };
        //    var dv1 = Env.CreateDataView(dv1_data);
        //    runner.SetInput("data1", dv1);
        //    runner.RunAll();
        //    var model = runner.GetOutput<ITransformModel>("model3");
        //    Assert.NotNull(model);
        //}

        [Fact]
        public void EntryPointCalibrate()
        {
            var dataView = GetBreastCancerDataView();

            var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = 3 });

            var lrModel = LogisticRegressionBinaryTrainer.TrainBinary(Env, new LogisticRegressionBinaryTrainer.Options { TrainingData = splitOutput.TestData[0] }).PredictorModel;
            var calibratedLrModel = Calibrate.FixedPlatt(Env,
                new Calibrate.FixedPlattInput { Data = splitOutput.TestData[1], UncalibratedPredictorModel = lrModel }).PredictorModel;

            var scored1 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = splitOutput.TestData[2], PredictorModel = lrModel }).ScoredData;
            scored1 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored1, ExtraColumns = new[] { "Label" } }).OutputData;

            var scored2 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = splitOutput.TestData[2], PredictorModel = calibratedLrModel }).ScoredData;
            scored2 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored2, ExtraColumns = new[] { "Label" } }).OutputData;

            Assert.Equal(4, scored1.Schema.Count);
            CheckSameValues(scored1, scored2);

            var input = new Calibrate.NoArgumentsInput() { Data = splitOutput.TestData[1], UncalibratedPredictorModel = lrModel };
            calibratedLrModel = Calibrate.Platt(Env, input).PredictorModel;
            calibratedLrModel = Calibrate.Naive(Env, input).PredictorModel;
            calibratedLrModel = Calibrate.Pav(Env, input).PredictorModel;

            // This tests that the SchemaBindableCalibratedPredictor doesn't get confused if its sub-predictor is already calibrated.
            var fastForest = new FastForestBinaryTrainer(Env, "Label", "Features");
            var rmd = new RoleMappedData(splitOutput.TrainData[0], "Label", "Features");
            var ffModel = new PredictorModelImpl(Env, rmd, splitOutput.TrainData[0], fastForest.Train(rmd));
            var calibratedFfModel = Calibrate.Platt(Env,
                new Calibrate.NoArgumentsInput() { Data = splitOutput.TestData[0], UncalibratedPredictorModel = ffModel }).PredictorModel;
            var twiceCalibratedFfModel = Calibrate.Platt(Env,
                new Calibrate.NoArgumentsInput() { Data = splitOutput.TestData[0], UncalibratedPredictorModel = calibratedFfModel }).PredictorModel;
            var scoredFf = ScoreModel.Score(Env, new ScoreModel.Input() { Data = splitOutput.TestData[2], PredictorModel = twiceCalibratedFfModel }).ScoredData;
        }

        [Fact]
        public void EntryPointPipelineEnsemble()
        {
            var dataView = GetBreastCancerDataView();
            const int nModels = 5;
            var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = nModels + 1 });
            var predictorModels = new PredictorModel[nModels];
            var individualScores = new IDataView[nModels];
            for (int i = 0; i < nModels; i++)
            {
                var data = splitOutput.TrainData[i];
                data = new ApproximatedKernelMappingEstimator(Env, new[] {
                    new ApproximatedKernelMappingEstimator.ColumnOptions("Features1", 10, false, "Features"),
                    new ApproximatedKernelMappingEstimator.ColumnOptions("Features2", 10, false, "Features"),
                }).Fit(data).Transform(data);

                data = new ColumnConcatenatingTransformer(Env, "Features", new[] { "Features1", "Features2" }).Transform(data);
                data = new ValueToKeyMappingEstimator(Env, "Label", "Label", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue).Fit(data).Transform(data);

                var lrInput = new LogisticRegressionBinaryTrainer.Options
                {
                    TrainingData = data,
                    L1Regularization = (Single)0.1 * i,
                    L2Regularization = (Single)0.01 * (1 + i),
                    NormalizeFeatures = NormalizeOption.Yes
                };
                predictorModels[i] = LogisticRegressionBinaryTrainer.TrainBinary(Env, lrInput).PredictorModel;
                var transformModel = new TransformModelImpl(Env, data, splitOutput.TrainData[i]);

                predictorModels[i] = ModelOperations.CombineTwoModels(Env,
                    new ModelOperations.SimplePredictorModelInput()
                    { PredictorModel = predictorModels[i], TransformModel = transformModel }).PredictorModel;

                individualScores[i] =
                    ScoreModel.Score(Env,
                        new ScoreModel.Input { Data = splitOutput.TestData[nModels], PredictorModel = predictorModels[i] })
                        .ScoredData;
            }

            var binaryEnsembleModel = EnsembleCreator.CreateBinaryPipelineEnsemble(Env,
                new EnsembleCreator.PipelineClassifierInput()
                {
                    ModelCombiner = EnsembleCreator.ClassifierCombiner.Average,
                    Models = predictorModels
                }).PredictorModel;
            var binaryEnsembleCalibrated = Calibrate.Platt(Env,
                new Calibrate.NoArgumentsInput()
                {
                    Data = splitOutput.TestData[nModels],
                    UncalibratedPredictorModel = binaryEnsembleModel
                }).PredictorModel;
            var binaryScored = ScoreModel.Score(Env,
                new ScoreModel.Input
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = binaryEnsembleModel
                }).ScoredData;
            var binaryScoredCalibrated = ScoreModel.Score(Env,
                new ScoreModel.Input
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = binaryEnsembleCalibrated
                }).ScoredData;

            var regressionEnsembleModel = EnsembleCreator.CreateRegressionPipelineEnsemble(Env,
                new EnsembleCreator.PipelineRegressionInput()
                {
                    ModelCombiner = EnsembleCreator.ScoreCombiner.Average,
                    Models = predictorModels
                }).PredictorModel;
            var regressionScored = ScoreModel.Score(Env,
                new ScoreModel.Input
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = regressionEnsembleModel
                }).ScoredData;

            var anomalyEnsembleModel = EnsembleCreator.CreateAnomalyPipelineEnsemble(Env,
                new EnsembleCreator.PipelineAnomalyInput()
                {
                    ModelCombiner = EnsembleCreator.ScoreCombiner.Average,
                    Models = predictorModels
                }).PredictorModel;
            var anomalyScored = ScoreModel.Score(Env,
                new ScoreModel.Input
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = anomalyEnsembleModel
                }).ScoredData;

            // Make sure the scorers have the correct types.
            var scoreCol = binaryScored.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
            Assert.True(scoreCol.HasValue, "Data scored with binary ensemble does not have a score column");
            var type = binaryScored.Schema[scoreCol.Value.Index].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreColumnKind)?.Type;
            Assert.True(type is TextDataViewType, "Binary ensemble scored data does not have correct type of metadata.");
            var kind = default(ReadOnlyMemory<char>);
            binaryScored.Schema[scoreCol.Value.Index].Annotations.GetValue(AnnotationUtils.Kinds.ScoreColumnKind, ref kind);
            Assert.True(ReadOnlyMemoryUtils.EqualsStr(AnnotationUtils.Const.ScoreColumnKind.BinaryClassification, kind),
                $"Binary ensemble scored data column type should be '{AnnotationUtils.Const.ScoreColumnKind.BinaryClassification}', but is instead '{kind}'");

            scoreCol = regressionScored.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
            Assert.True(scoreCol.HasValue, "Data scored with regression ensemble does not have a score column");
            type = regressionScored.Schema[scoreCol.Value.Index].Annotations.Schema[AnnotationUtils.Kinds.ScoreColumnKind].Type;
            Assert.True(type is TextDataViewType, "Regression ensemble scored data does not have correct type of metadata.");
            regressionScored.Schema[scoreCol.Value.Index].Annotations.GetValue(AnnotationUtils.Kinds.ScoreColumnKind, ref kind);
            Assert.True(ReadOnlyMemoryUtils.EqualsStr(AnnotationUtils.Const.ScoreColumnKind.Regression, kind),
                $"Regression ensemble scored data column type should be '{AnnotationUtils.Const.ScoreColumnKind.Regression}', but is instead '{kind}'");

            scoreCol = anomalyScored.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
            Assert.True(scoreCol.HasValue, "Data scored with anomaly detection ensemble does not have a score column");
            type = anomalyScored.Schema[scoreCol.Value.Index].Annotations.Schema[AnnotationUtils.Kinds.ScoreColumnKind].Type;
            Assert.True(type is TextDataViewType, "Anomaly detection ensemble scored data does not have correct type of metadata.");
            anomalyScored.Schema[scoreCol.Value.Index].Annotations.GetValue(AnnotationUtils.Kinds.ScoreColumnKind, ref kind);
            Assert.True(ReadOnlyMemoryUtils.EqualsStr(AnnotationUtils.Const.ScoreColumnKind.AnomalyDetection, kind),
                $"Anomaly detection ensemble scored data column type should be '{AnnotationUtils.Const.ScoreColumnKind.AnomalyDetection}', but is instead '{kind}'");

            var modelPath = DeleteOutputPath("SavePipe", "PipelineEnsembleModel.zip");
            using (var file = Env.CreateOutputFile(modelPath))
            using (var strm = file.CreateWriteStream())
                regressionEnsembleModel.Save(Env, strm);

            PredictorModel loadedFromSaved;
            using (var file = Env.OpenInputFile(modelPath))
            using (var strm = file.OpenReadStream())
                loadedFromSaved = new PredictorModelImpl(Env, strm);

            var scoredFromSaved = ScoreModel.Score(Env,
                new ScoreModel.Input()
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = loadedFromSaved
                }).ScoredData;

            using (var cursReg = regressionScored.GetRowCursorForAllColumns())
            using (var cursBin = binaryScored.GetRowCursorForAllColumns())
            using (var cursBinCali = binaryScoredCalibrated.GetRowCursorForAllColumns())
            using (var cursAnom = anomalyScored.GetRowCursorForAllColumns())
            using (var curs0 = individualScores[0].GetRowCursorForAllColumns())
            using (var curs1 = individualScores[1].GetRowCursorForAllColumns())
            using (var curs2 = individualScores[2].GetRowCursorForAllColumns())
            using (var curs3 = individualScores[3].GetRowCursorForAllColumns())
            using (var curs4 = individualScores[4].GetRowCursorForAllColumns())
            using (var cursSaved = scoredFromSaved.GetRowCursorForAllColumns())
            {
                var scoreColumn = curs0.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter0 = curs0.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = curs1.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter1 = curs1.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = curs2.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter2 = curs2.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = curs3.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter3 = curs3.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = curs4.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter4 = curs4.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = cursReg.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterReg = cursReg.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = cursBin.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterBin = cursBin.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = cursBinCali.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterBinCali = cursBinCali.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = cursSaved.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterSaved = cursSaved.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = cursAnom.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterAnom = cursAnom.GetGetter<Single>(scoreColumn.Value);

                var c = new Average(Env).GetCombiner();
                while (cursReg.MoveNext())
                {
                    Single score = 0;
                    getterReg(ref score);
                    Assert.True(curs0.MoveNext());
                    Assert.True(curs1.MoveNext());
                    Assert.True(curs2.MoveNext());
                    Assert.True(curs3.MoveNext());
                    Assert.True(curs4.MoveNext());
                    Assert.True(cursBin.MoveNext());
                    Assert.True(cursBinCali.MoveNext());
                    Assert.True(cursSaved.MoveNext());
                    Assert.True(cursAnom.MoveNext());
                    Single[] score0 = new Single[5];
                    getter0(ref score0[0]);
                    getter1(ref score0[1]);
                    getter2(ref score0[2]);
                    getter3(ref score0[3]);
                    getter4(ref score0[4]);
                    Single scoreBin = 0;
                    Single scoreBinCali = 0;
                    Single scoreSaved = 0;
                    Single scoreAnom = 0;
                    getterBin(ref scoreBin);
                    getterBinCali(ref scoreBinCali);
                    getterSaved(ref scoreSaved);
                    getterAnom(ref scoreAnom);
                    Assert.True(Single.IsNaN(scoreBin) && Single.IsNaN(score) || scoreBin == score);
                    Assert.True(Single.IsNaN(scoreBinCali) && Single.IsNaN(score) || scoreBinCali == score);
                    Assert.True(Single.IsNaN(scoreSaved) && Single.IsNaN(score) || CompareNumbersWithTolerance(scoreSaved, score, null, 5));
                    Assert.True(Single.IsNaN(scoreAnom) && Single.IsNaN(score) || scoreAnom == score);

                    Single avg = 0;
                    c(ref avg, score0, null);
                    Assert.True(Single.IsNaN(avg) && Single.IsNaN(score) || avg == score);
                }
                Assert.False(curs0.MoveNext());
                Assert.False(curs1.MoveNext());
                Assert.False(curs2.MoveNext());
                Assert.False(curs3.MoveNext());
                Assert.False(curs4.MoveNext());
                Assert.False(cursBin.MoveNext());
                Assert.False(cursBinCali.MoveNext());
                Assert.False(cursSaved.MoveNext());
                Assert.False(cursAnom.MoveNext());
            }
        }


        [Fact]
        public void EntryPointPipelineEnsembleText()
        {
            var dataPath = GetDataPath("lm.sample.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            var dataView = EntryPoints.ImportTextData.TextLoader(Env, new EntryPoints.ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    HasHeader = true,
                    Columns = new[]
                    {
                        new TextLoader.Column("Label", DataKind.String, 0),
                        new TextLoader.Column("Text", DataKind.String, 3)
                    }
                },

                InputFile = inputFile
            }).Data;

            ValueMapper<ReadOnlyMemory<char>, bool> labelToBinary =
                (in ReadOnlyMemory<char> src, ref bool dst) =>
                {
                    if (ReadOnlyMemoryUtils.EqualsStr("Sport", src))
                        dst = true;
                    else
                        dst = false;
                };
            dataView = LambdaColumnMapper.Create(Env, "TextToBinaryLabel", dataView, "Label", "Label",
                TextDataViewType.Instance, BooleanDataViewType.Instance, labelToBinary);

            const int nModels = 5;
            var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = nModels + 1 });
            var predictorModels = new PredictorModel[nModels];
            var individualScores = new IDataView[nModels];
            for (int i = 0; i < nModels; i++)
            {
                var data = splitOutput.TrainData[i];
                if (i % 2 == 0)
                {
                    data = new TextFeaturizingEstimator(Env, "Features", new List<string> { "Text" }, 
                        new TextFeaturizingEstimator.Options { 
                            StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options(),
                        }).Fit(data).Transform(data);
                }
                else
                {
                    data = WordHashBagProducingTransformer.Create(Env,
                        new WordHashBagProducingTransformer.Options()
                        {
                            Columns =
                                new[] { new WordHashBagProducingTransformer.Column() { Name = "Features", Source = new[] { "Text" } }, }
                        },
                        data);
                }
                var lrInput = new LogisticRegressionBinaryTrainer.Options
                {
                    TrainingData = data,
                    L1Regularization = (Single)0.1 * i,
                    L2Regularization = (Single)0.01 * (1 + i),
                    NormalizeFeatures = NormalizeOption.Yes
                };
                predictorModels[i] = LogisticRegressionBinaryTrainer.TrainBinary(Env, lrInput).PredictorModel;
                var transformModel = new TransformModelImpl(Env, data, splitOutput.TrainData[i]);

                predictorModels[i] = ModelOperations.CombineTwoModels(Env,
                    new ModelOperations.SimplePredictorModelInput()
                    { PredictorModel = predictorModels[i], TransformModel = transformModel }).PredictorModel;

                individualScores[i] =
                    ScoreModel.Score(Env,
                        new ScoreModel.Input { Data = splitOutput.TestData[nModels], PredictorModel = predictorModels[i] })
                        .ScoredData;
            }

            var binaryEnsembleModel = EnsembleCreator.CreateBinaryPipelineEnsemble(Env,
                new EnsembleCreator.PipelineClassifierInput()
                {
                    ModelCombiner = EnsembleCreator.ClassifierCombiner.Average,
                    Models = predictorModels
                }).PredictorModel;
            var binaryEnsembleCalibrated = Calibrate.Platt(Env,
                new Calibrate.NoArgumentsInput()
                {
                    Data = splitOutput.TestData[nModels],
                    UncalibratedPredictorModel = binaryEnsembleModel
                }).PredictorModel;
            var binaryScored = ScoreModel.Score(Env,
                new ScoreModel.Input
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = binaryEnsembleModel
                }).ScoredData;
            var binaryScoredCalibrated = ScoreModel.Score(Env,
                new ScoreModel.Input
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = binaryEnsembleCalibrated
                }).ScoredData;

            var regressionEnsembleModel = EnsembleCreator.CreateRegressionPipelineEnsemble(Env,
                new EnsembleCreator.PipelineRegressionInput()
                {
                    ModelCombiner = EnsembleCreator.ScoreCombiner.Average,
                    Models = predictorModels
                }).PredictorModel;
            var regressionScored = ScoreModel.Score(Env,
                new ScoreModel.Input
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = regressionEnsembleModel
                }).ScoredData;

            var modelPath = DeleteOutputPath("SavePipe", "PipelineEnsembleModel.zip");
            using (var file = Env.CreateOutputFile(modelPath))
            using (var strm = file.CreateWriteStream())
                regressionEnsembleModel.Save(Env, strm);

            PredictorModel loadedFromSaved;
            using (var file = Env.OpenInputFile(modelPath))
            using (var strm = file.OpenReadStream())
                loadedFromSaved = new PredictorModelImpl(Env, strm);

            var scoredFromSaved = ScoreModel.Score(Env,
                new ScoreModel.Input()
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = loadedFromSaved
                }).ScoredData;

            using (var cursReg = regressionScored.GetRowCursorForAllColumns())
            using (var cursBin = binaryScored.GetRowCursorForAllColumns())
            using (var cursBinCali = binaryScoredCalibrated.GetRowCursorForAllColumns())
            using (var curs0 = individualScores[0].GetRowCursorForAllColumns())
            using (var curs1 = individualScores[1].GetRowCursorForAllColumns())
            using (var curs2 = individualScores[2].GetRowCursorForAllColumns())
            using (var curs3 = individualScores[3].GetRowCursorForAllColumns())
            using (var curs4 = individualScores[4].GetRowCursorForAllColumns())
            using (var cursSaved = scoredFromSaved.GetRowCursorForAllColumns())
            {
                var scoreColumn = curs0.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter0 = curs0.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = curs1.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter1 = curs1.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = curs2.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter2 = curs2.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = curs3.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter3 = curs3.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = curs4.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter4 = curs4.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = cursReg.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterReg = cursReg.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = cursBin.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterBin = cursBin.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = cursBinCali.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterBinCali = cursBinCali.GetGetter<Single>(scoreColumn.Value);
                scoreColumn = cursSaved.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterSaved = cursSaved.GetGetter<Single>(scoreColumn.Value);

                var c = new Average(Env).GetCombiner();
                while (cursReg.MoveNext())
                {
                    Single score = 0;
                    getterReg(ref score);
                    Assert.True(curs0.MoveNext());
                    Assert.True(curs1.MoveNext());
                    Assert.True(curs2.MoveNext());
                    Assert.True(curs3.MoveNext());
                    Assert.True(curs4.MoveNext());
                    Assert.True(cursBin.MoveNext());
                    Assert.True(cursBinCali.MoveNext());
                    Assert.True(cursSaved.MoveNext());
                    Single[] score0 = new Single[5];
                    getter0(ref score0[0]);
                    getter1(ref score0[1]);
                    getter2(ref score0[2]);
                    getter3(ref score0[3]);
                    getter4(ref score0[4]);
                    Single scoreBin = 0;
                    Single scoreBinCali = 0;
                    Single scoreSaved = 0;
                    getterBin(ref scoreBin);
                    getterBinCali(ref scoreBinCali);
                    getterSaved(ref scoreSaved);
                    Assert.True(Single.IsNaN(scoreBin) && Single.IsNaN(score) || scoreBin == score);
                    Assert.True(Single.IsNaN(scoreBinCali) && Single.IsNaN(score) || scoreBinCali == score);
                    Assert.True(Single.IsNaN(scoreSaved) && Single.IsNaN(score) || scoreSaved == score);

                    Single avg = 0;
                    c(ref avg, score0, null);
                    Assert.True(Single.IsNaN(avg) && Single.IsNaN(score) || avg == score);
                }
                Assert.False(curs0.MoveNext());
                Assert.False(curs1.MoveNext());
                Assert.False(curs2.MoveNext());
                Assert.False(curs3.MoveNext());
                Assert.False(curs4.MoveNext());
                Assert.False(cursBin.MoveNext());
                Assert.False(cursBinCali.MoveNext());
                Assert.False(cursSaved.MoveNext());
            }
        }

        [Fact]
        public void EntryPointMulticlassPipelineEnsemble()
        {
            var dataPath = GetDataPath("iris.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            var dataView = EntryPoints.ImportTextData.TextLoader(Env, new EntryPoints.ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    Columns = new[]
                    {
                        new TextLoader.Column("Label", DataKind.Single, 0),
                        new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 4) })
                    }
                },

                InputFile = inputFile
            }).Data;

            const int nModels = 5;
            var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = nModels + 1 });
            var predictorModels = new PredictorModel[nModels];
            var individualScores = new IDataView[nModels];
            for (int i = 0; i < nModels; i++)
            {
                var data = splitOutput.TrainData[i];
                data = new ApproximatedKernelMappingEstimator(Env, new[] {
                    new ApproximatedKernelMappingEstimator.ColumnOptions("Features1", 10, false, "Features"),
                    new ApproximatedKernelMappingEstimator.ColumnOptions("Features2", 10, false, "Features"),
                }).Fit(data).Transform(data);
                data = new ColumnConcatenatingTransformer(Env, "Features", new[] { "Features1", "Features2" }).Transform(data);

                var mlr = ML.MulticlassClassification.Trainers.LbfgsMaximumEntropy();
                var rmd = new RoleMappedData(data, "Label", "Features");

                predictorModels[i] = new PredictorModelImpl(Env, rmd, data, mlr.Train(rmd));
                var transformModel = new TransformModelImpl(Env, data, splitOutput.TrainData[i]);

                predictorModels[i] = ModelOperations.CombineTwoModels(Env,
                    new ModelOperations.SimplePredictorModelInput()
                    { PredictorModel = predictorModels[i], TransformModel = transformModel }).PredictorModel;

                individualScores[i] =
                    ScoreModel.Score(Env,
                        new ScoreModel.Input { Data = splitOutput.TestData[nModels], PredictorModel = predictorModels[i] })
                        .ScoredData;
            }

            var mcEnsembleModel = EnsembleCreator.CreateMulticlassPipelineEnsemble(Env,
                new EnsembleCreator.PipelineClassifierInput()
                {
                    ModelCombiner = EnsembleCreator.ClassifierCombiner.Average,
                    Models = predictorModels
                }).PredictorModel;
            var mcScored = ScoreModel.Score(Env,
                new ScoreModel.Input
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = mcEnsembleModel
                }).ScoredData;

            var modelPath = DeleteOutputPath("SavePipe", "PipelineEnsembleModel.zip");
            using (var file = Env.CreateOutputFile(modelPath))
            using (var strm = file.CreateWriteStream())
                mcEnsembleModel.Save(Env, strm);

            PredictorModel loadedFromSaved;
            using (var file = Env.OpenInputFile(modelPath))
            using (var strm = file.OpenReadStream())
                loadedFromSaved = new PredictorModelImpl(Env, strm);

            var scoredFromSaved = ScoreModel.Score(Env,
                new ScoreModel.Input()
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = loadedFromSaved
                }).ScoredData;

            using (var curs = mcScored.GetRowCursorForAllColumns())
            using (var cursSaved = scoredFromSaved.GetRowCursorForAllColumns())
            using (var curs0 = individualScores[0].GetRowCursorForAllColumns())
            using (var curs1 = individualScores[1].GetRowCursorForAllColumns())
            using (var curs2 = individualScores[2].GetRowCursorForAllColumns())
            using (var curs3 = individualScores[3].GetRowCursorForAllColumns())
            using (var curs4 = individualScores[4].GetRowCursorForAllColumns())
            {
                var scoreColumn = curs0.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter0 = curs0.GetGetter<VBuffer<Single>>(scoreColumn.Value);
                scoreColumn = curs1.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter1 = curs1.GetGetter<VBuffer<Single>>(scoreColumn.Value);
                scoreColumn = curs2.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter2 = curs2.GetGetter<VBuffer<Single>>(scoreColumn.Value);
                scoreColumn = curs3.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter3 = curs3.GetGetter<VBuffer<Single>>(scoreColumn.Value);
                scoreColumn = curs4.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter4 = curs4.GetGetter<VBuffer<Single>>(scoreColumn.Value);
                scoreColumn = curs.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getter = curs.GetGetter<VBuffer<Single>>(scoreColumn.Value);
                scoreColumn = cursSaved.Schema.GetColumnOrNull(AnnotationUtils.Const.ScoreValueKind.Score);
                Assert.True(scoreColumn.HasValue);
                var getterSaved = cursSaved.GetGetter<VBuffer<Single>>(scoreColumn.Value);

                var c = new MultiAverage(Env, new MultiAverage.Options()).GetCombiner();
                VBuffer<Single> score = default(VBuffer<Single>);
                VBuffer<Single>[] score0 = new VBuffer<Single>[5];
                VBuffer<Single> scoreSaved = default(VBuffer<Single>);
                VBuffer<Single> avg = default(VBuffer<Single>);
                VBuffer<Single> dense1 = default(VBuffer<Single>);
                VBuffer<Single> dense2 = default(VBuffer<Single>);
                while (curs.MoveNext())
                {
                    getter(ref score);
                    Assert.True(curs0.MoveNext());
                    Assert.True(curs1.MoveNext());
                    Assert.True(curs2.MoveNext());
                    Assert.True(curs3.MoveNext());
                    Assert.True(curs4.MoveNext());
                    Assert.True(cursSaved.MoveNext());
                    getter0(ref score0[0]);
                    getter1(ref score0[1]);
                    getter2(ref score0[2]);
                    getter3(ref score0[3]);
                    getter4(ref score0[4]);
                    getterSaved(ref scoreSaved);
                    Assert.True(CompareVBuffers(in scoreSaved, in score, ref dense1, ref dense2));
                    c(ref avg, score0, null);
                    Assert.True(CompareVBuffers(in avg, in score, ref dense1, ref dense2));
                }
                Assert.False(curs0.MoveNext());
                Assert.False(curs1.MoveNext());
                Assert.False(curs2.MoveNext());
                Assert.False(curs3.MoveNext());
                Assert.False(curs4.MoveNext());
                Assert.False(cursSaved.MoveNext());
            }
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void EntryPointPipelineEnsembleGetSummary()
        {
            var dataPath = GetDataPath("breast-cancer-withheader.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            var dataView =
                EntryPoints.ImportTextData.TextLoader(Env,
                    new EntryPoints.ImportTextData.LoaderInput
                    {
                        InputFile = inputFile,
                        Arguments =
                        {
                            Columns = new[]
                            {
                                new TextLoader.Column("Label", DataKind.Single, 0),
                                new TextLoader.Column("Features", DataKind.Single, new[] { new TextLoader.Range(1, 8) }),
                                new TextLoader.Column("Cat", DataKind.String, 9)
                            },
                            HasHeader = true,
                        }
                    })
                    .Data;

            const int nModels = 4;
            var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = nModels });
            var predictorModels = new PredictorModel[nModels];
            for (int i = 0; i < nModels; i++)
            {
                var data = splitOutput.TrainData[i];
                data = new OneHotEncodingEstimator(Env, "Cat").Fit(data).Transform(data);
                data = new ColumnConcatenatingTransformer(Env, new ColumnConcatenatingTransformer.ColumnOptions("Features", i % 2 == 0 ? new[] { "Features", "Cat" } : new[] { "Cat", "Features" })).Transform(data);
                if (i % 2 == 0)
                {
                    var lrInput = new LogisticRegressionBinaryTrainer.Options
                    {
                        TrainingData = data,
                        NormalizeFeatures = NormalizeOption.Yes,
                        NumberOfThreads = 1,
                        ShowTrainingStatistics = true,
                        ComputeStandardDeviation = new ComputeLRTrainingStdThroughMkl()
                    };
                    predictorModels[i] = LogisticRegressionBinaryTrainer.TrainBinary(Env, lrInput).PredictorModel;
                    var transformModel = new TransformModelImpl(Env, data, splitOutput.TrainData[i]);

                    predictorModels[i] = ModelOperations.CombineTwoModels(Env,
                        new ModelOperations.SimplePredictorModelInput()
                        { PredictorModel = predictorModels[i], TransformModel = transformModel }).PredictorModel;
                }
                else if (i % 2 == 1)
                {
                    var trainer = new FastTreeBinaryTrainer(Env, "Label", "Features");
                    var rmd = new RoleMappedData(data, false,
                        RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Feature, "Features"),
                        RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Label, "Label"));
                    var predictor = trainer.Train(rmd);
                    predictorModels[i] = new PredictorModelImpl(Env, rmd, splitOutput.TrainData[i], predictor);
                }
            }

            var binaryEnsembleModel = EnsembleCreator.CreateBinaryPipelineEnsemble(Env,
                new EnsembleCreator.PipelineClassifierInput()
                {
                    ModelCombiner = EnsembleCreator.ClassifierCombiner.Average,
                    Models = predictorModels
                }).PredictorModel;
            var binaryEnsembleCalibrated = Calibrate.Platt(Env,
                new Calibrate.NoArgumentsInput()
                {
                    Data = splitOutput.TestData[0],
                    UncalibratedPredictorModel = binaryEnsembleModel
                }).PredictorModel;

            var summaryDataViews = PipelineEnsemble.Summarize(Env,
                new SummarizePredictor.Input() { PredictorModel = binaryEnsembleCalibrated });

            var summarizable = binaryEnsembleCalibrated.Predictor as ICanSaveSummary;
            Assert.NotNull(summarizable);

            using (var ch = Env.Register("LinearPredictorSummary").Start("Save Data Views"))
            {
                for (int i = 0; i < summaryDataViews.Summaries.Length; i++)
                {
                    var summary = DeleteOutputPath(@"../Common/EntryPoints", $"ensemble-model{i}-summary.txt");
                    var saver = Env.CreateSaver("Text");
                    using (var file = Env.CreateOutputFile(summary))
                        DataSaverUtils.SaveDataView(ch, saver, summaryDataViews.Summaries[i], file);
                    CheckEquality(@"../Common/EntryPoints", $"ensemble-model{i}-summary.txt", digitsOfPrecision: 4);

                    if (summaryDataViews.Stats[i] != null)
                    {
                        var stats = DeleteOutputPath(@"../Common/EntryPoints", $"ensemble-model{i}-stats.txt");
                        using (var file = Env.CreateOutputFile(stats))
                            DataSaverUtils.SaveDataView(ch, saver, summaryDataViews.Stats[i], file);
                        CheckEquality(@"../Common/EntryPoints", $"ensemble-model{i}-stats.txt", digitsOfPrecision: 4);
                    }
                }
            }

            var summaryPath = DeleteOutputPath(@"../Common/EntryPoints", "ensemble-summary.txt");
            using (var file = File.OpenWrite(summaryPath))
            using (var writer = Utils.OpenWriter(file))
                summarizable.SaveSummary(writer, null);

            CheckEquality(@"../Common/EntryPoints", "ensemble-summary.txt", digitsOfPrecision: 4);

            var summaryKvps = binaryEnsembleCalibrated.Predictor as ICanGetSummaryInKeyValuePairs;
            Assert.NotNull(summaryKvps);

            var summaryKvpPath = DeleteOutputPath(@"../Common/EntryPoints", "ensemble-summary-key-value-pairs.txt");
            using (var file = File.OpenWrite(summaryKvpPath))
            using (var writer = Utils.OpenWriter(file))
            {
                var kvps = summaryKvps.GetSummaryInKeyValuePairs(null);
                for (int i = 0; i < kvps.Count; i++)
                {
                    var kvp = kvps[i];
                    var list = kvp.Value as IList<KeyValuePair<string, object>>;
                    Assert.NotNull(list);

                    writer.WriteLine(kvp.Key);
                    for (int j = 0; j < list.Count; j++)
                    {
                        kvp = list[j];
                        writer.WriteLine("{0}: {1}", kvp.Key, kvp.Value);
                    }
                }
            }
            CheckEquality(@"../Common/EntryPoints", "ensemble-summary-key-value-pairs.txt", digitsOfPrecision: 4);

            Done();
        }

        private static bool CompareVBuffers(in VBuffer<Single> v1, in VBuffer<Single> v2, ref VBuffer<Single> dense1, ref VBuffer<Single> dense2)
        {
            if (v1.Length != v2.Length)
                return false;
            v1.CopyToDense(ref dense1);
            v2.CopyToDense(ref dense2);
            var dense1Values = dense1.GetValues();
            var dense2Values = dense2.GetValues();
            for (int i = 0; i < dense1.Length; i++)
            {
                if (!Single.IsNaN(dense1Values[i]) && !Single.IsNaN(dense2Values[i]) && dense1Values[i] != dense2Values[i])
                    return false;
            }
            return true;
        }

        [Fact]
        public void EntryPointParseColumns()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var outputPath = DeleteOutputPath("data.idv");

            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file1'
                      }},
                      'Outputs': {{
                        'Data': '$data1'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.MinMaxNormalizer',
                      'Inputs': {{
                        'Data': '$data1',
                        'Column': [
                          {{
                            'Name': 'Features',
                            'Source': 'Features',
                            'FixZero': true
                          }}
                        ]
                      }},
                      'Outputs': {{
                        'OutputData': '$data2'
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file1' : '{0}'
                  }},
                  'Outputs' : {{
                    'data2' : '{1}'
                  }}
                }}", EscapePath(dataPath), EscapePath(outputPath));

            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();
        }

        [Fact]
        public void EntryPointCountFeatures()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var outputPath = DeleteOutputPath("data.idv");
            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file1'
                      }},
                      'Outputs': {{
                        'Data': '$data1'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.FeatureSelectorByCount',
                      'Inputs': {{
                        'Data': '$data1',
                        'Column': ['Features'],
                        'Count':'2'
                      }},
                      'Outputs': {{
                        'OutputData': '$data2'
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file1' : '{0}'
                  }},
                  'Outputs' : {{
                    'data2' : '{1}'
                  }}
                }}", EscapePath(dataPath), EscapePath(outputPath));

            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();
        }

        [Fact]
        public void EntryPointMutualSelectFeatures()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var outputPath = DeleteOutputPath("data.idv");
            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file1'
                      }},
                      'Outputs': {{
                        'Data': '$data1'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.FeatureSelectorByMutualInformation',
                      'Inputs': {{
                         'Data': '$data1',
                         'Column': ['Features'],
                         'TopK':'2',
                         'Bins': '6'
                      }},
                      'Outputs': {{
                        'OutputData': '$data2'
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file1' : '{0}'
                  }},
                  'Outputs' : {{
                    'data2' : '{1}'
                  }}
                }}", EscapePath(dataPath), EscapePath(outputPath));

            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();
        }

        [Fact]
        public void EntryPointTextToKeyToText()
        {
            var dataPath = GetDataPath("iris.data");
            var outputPath = DeleteOutputPath("data.idv");
            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file1',
                        'CustomSchema': 'sep=comma col=Cat:TX:4'
                      }},
                      'Outputs': {{
                        'Data': '$data1'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.TextToKeyConverter',
                      'Inputs': {{
                        'Data': '$data1',
                        'Column': [{{ 'Name': 'Key', 'Source': 'Cat' }}]
                      }},
                      'Outputs': {{
                        'OutputData': '$data2'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.KeyToTextConverter',
                      'Inputs': {{
                        'Data': '$data2',
                        'Column': [{{ 'Name': 'CatValue', 'Source': 'Key' }}]
                      }},
                      'Outputs': {{
                        'OutputData': '$data3'
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file1' : '{0}'
                  }},
                  'Outputs' : {{
                    'data3' : '{1}'
                  }}
                }}", EscapePath(dataPath), EscapePath(outputPath));

            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), outputPath))
            {
                using (var cursor = loader.GetRowCursorForAllColumns())
                {
                    ReadOnlyMemory<char> cat = default;
                    ReadOnlyMemory<char> catValue = default;
                    uint catKey = 0;

                    var catColumn = loader.Schema.GetColumnOrNull("Cat");
                    Assert.True(catColumn.HasValue);
                    var catGetter = cursor.GetGetter<ReadOnlyMemory<char>>(catColumn.Value);
                    var catValueCol = loader.Schema.GetColumnOrNull("CatValue");
                    Assert.True(catValueCol.HasValue);
                    var catValueGetter = cursor.GetGetter<ReadOnlyMemory<char>>(catValueCol.Value);
                    var keyColumn = loader.Schema.GetColumnOrNull("Key");
                    Assert.True(keyColumn.HasValue);
                    var keyGetter = cursor.GetGetter<uint>(keyColumn.Value);

                    while (cursor.MoveNext())
                    {
                        catGetter(ref cat);
                        catValueGetter(ref catValue);
                        Assert.Equal(cat.ToString(), catValue.ToString());
                        keyGetter(ref catKey);
                        Assert.True(1 <= catKey && catKey <= 3);
                    }
                }
            }
        }

        private void RunTrainScoreEvaluate(string learner, string evaluator, string dataPath, string warningsPath, string overallMetricsPath,
                    string instanceMetricsPath, string confusionMatrixPath = null, string loader = null, string transforms = null,
                    string splitterInput = "AllData")
        {
            if (string.IsNullOrEmpty(transforms))
                transforms = "";
            loader = string.IsNullOrWhiteSpace(loader) ? "" : string.Format(",'CustomSchema': '{0}'", loader);
            var confusionMatrixVar = confusionMatrixPath != null ? ", 'ConfusionMatrix': '$ConfusionMatrix'" : "";
            confusionMatrixPath = confusionMatrixPath != null ? string.Format(", 'ConfusionMatrix' : '{0}'", EscapePath(confusionMatrixPath)) : "";
            var scorerModel = string.IsNullOrEmpty(transforms) ? "Model" : "CombinedModel";
            string inputGraph = $@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file'
                        {loader}
                      }},
                      'Outputs': {{
                        'Data': '$AllData'
                      }}
                    }},
                    {transforms}
                    {{
                      'Name': 'Transforms.TrainTestDatasetSplitter',
                      'Inputs': {{
                        'Data': '${splitterInput}',
                        'Fraction': 0.8
                      }},
                      'Outputs': {{
                        'TrainData': '$TrainData',
                        'TestData': '$TestData'
                      }}
                    }},
                    {{
                      'Name': '{learner}',
                      'Inputs': {{
                        'TrainingData': '$TrainData'
                      }},
                      'Outputs': {{
                        'PredictorModel': '$Model'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.DatasetScorer',
                      'Inputs': {{
                        'Data': '$TestData',
                        'PredictorModel': '$Model'
                      }},
                      'Outputs': {{
                        'ScoredData': '$ScoredData'
                      }}
                    }},
                    {{
                      'Name': '{evaluator}',
                      'Inputs': {{
                        'Data': '$ScoredData'
                      }},
                      'Outputs': {{
                        'Warnings': '$Warnings',
                        'OverallMetrics': '$OverallMetrics',
                        'PerInstanceMetrics': '$PerInstanceMetrics'
                        {confusionMatrixVar}
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file' : '{EscapePath(dataPath)}'
                  }},
                  'Outputs' : {{
                    'Warnings' : '{EscapePath(warningsPath)}',
                    'OverallMetrics' : '{EscapePath(overallMetricsPath)}',
                    'PerInstanceMetrics' : '{EscapePath(instanceMetricsPath)}'
                    {confusionMatrixPath}
                  }}
                }}";

            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();
        }

        [Fact]
        public void EntryPointEvaluateBinary()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var warningsPath = DeleteOutputPath("warnings.idv");
            var overallMetricsPath = DeleteOutputPath("overall.idv");
            var instanceMetricsPath = DeleteOutputPath("instance.idv");
            var confusionMatrixPath = DeleteOutputPath("confusion.idv");

            RunTrainScoreEvaluate("Trainers.LogisticRegressionBinaryClassifier", "Models.BinaryClassificationEvaluator", dataPath, warningsPath, overallMetricsPath, instanceMetricsPath, confusionMatrixPath);

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), warningsPath))
                Assert.Equal(1, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), overallMetricsPath))
                Assert.Equal(1, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), instanceMetricsPath))
                Assert.Equal(138, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), confusionMatrixPath))
                Assert.Equal(2, CountRows(loader));
        }

        [Fact]
        public void EntryPointEvaluateMulticlass()
        {
            var dataPath = GetDataPath("iris.txt");
            var warningsPath = DeleteOutputPath("warnings.idv");
            var overallMetricsPath = DeleteOutputPath("overall.idv");
            var instanceMetricsPath = DeleteOutputPath("instance.idv");
            var confusionMatrixPath = DeleteOutputPath("confusion.idv");

            RunTrainScoreEvaluate("Trainers.LogisticRegressionClassifier", "Models.ClassificationEvaluator", dataPath, warningsPath, overallMetricsPath, instanceMetricsPath, confusionMatrixPath);

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), warningsPath))
                Assert.Equal(0, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), overallMetricsPath))
                Assert.Equal(1, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), instanceMetricsPath))
                Assert.Equal(34, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), confusionMatrixPath))
                Assert.Equal(3, CountRows(loader));
        }

        [Fact]
        public void EntryPointEvaluateRegression()
        {
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDatasetmacro.trainFilename);
            var warningsPath = DeleteOutputPath("warnings.idv");
            var overallMetricsPath = DeleteOutputPath("overall.idv");
            var instanceMetricsPath = DeleteOutputPath("instance.idv");

            RunTrainScoreEvaluate("Trainers.StochasticDualCoordinateAscentRegressor", "Models.RegressionEvaluator",
                dataPath, warningsPath, overallMetricsPath, instanceMetricsPath, loader: TestDatasets.generatedRegressionDatasetmacro.loaderSettings);

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), warningsPath))
                Assert.Equal(0, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), overallMetricsPath))
                Assert.Equal(1, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), instanceMetricsPath))
                Assert.Equal(103, CountRows(loader));
        }

        [Fact]
        public void EntryPointEvaluateRanking()
        {
            var dataPath = GetDataPath(@"adult.tiny.with-schema.txt");
            var warningsPath = DeleteOutputPath("warnings.idv");
            var overallMetricsPath = DeleteOutputPath("overall.idv");
            var instanceMetricsPath = DeleteOutputPath("instance.idv");

            var transforms = @"
                      {
                        'Inputs': {
                            'Column': [
                                {
                                    'Name': 'GroupId',
                                    'Source': 'Workclass'
                                }
                            ],
                            'Data': '$AllData',
                            'MaxNumTerms': 1000000,
                            'Sort': 'ByOccurrence',
                            'TextKeyValues': false
                        },
                        'Name': 'Transforms.TextToKeyConverter',
                        'Outputs': {
                            'Model': '$output_model1',
                            'OutputData': '$output_data1'
                        }
                      },
                      {
                        'Name': 'Transforms.LabelColumnKeyBooleanConverter',
                        'Inputs': {
                            'Data': '$output_data1',
                            'LabelColumn': 'Label',
                            'TextKeyValues': false
                        },
                        'Outputs': {
                            'Model': '$output_model2',
                            'OutputData': '$output_data2'
                        }
                      },
                      {
                        'Name': 'Transforms.ColumnCopier',
                        'Inputs': {
                            'Column': [
                              {
                                'Name': 'Features',
                                'Source': 'NumericFeatures'
                              }
                            ],
                            'Data': '$output_data2'
                        },
                        'Outputs': {
                            'Model': '$output_model3',
                            'OutputData': '$output_data3'
                        }
                      },";

            RunTrainScoreEvaluate("Trainers.FastTreeRanker", "Models.RankingEvaluator",
                dataPath, warningsPath, overallMetricsPath, instanceMetricsPath,
                splitterInput: "output_data3", transforms: transforms);

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), warningsPath))
                Assert.Equal(0, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), overallMetricsPath))
                Assert.Equal(1, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), instanceMetricsPath))
            {
                Assert.Equal(103, CountRows(loader));
                Assert.NotNull(loader.Schema.GetColumnOrNull("GroupId"));
                Assert.NotNull(loader.Schema.GetColumnOrNull("Label"));
            }
        }

        [LightGBMFact]
        public void EntryPointLightGbmBinary()
        {
            Env.ComponentCatalog.RegisterAssembly(typeof(LightGbmBinaryModelParameters).Assembly);
            TestEntryPointRoutine("breast-cancer.txt", "Trainers.LightGbmBinaryClassifier");
        }

        [LightGBMFact]
        public void EntryPointLightGbmMulticlass()
        {
            Env.ComponentCatalog.RegisterAssembly(typeof(LightGbmBinaryModelParameters).Assembly);
            TestEntryPointRoutine(GetDataPath(@"iris.txt"), "Trainers.LightGbmClassifier");
        }

        [Fact]
        public void EntryPointSdcaBinary()
        {
            TestEntryPointRoutine("breast-cancer.txt", "Trainers.StochasticDualCoordinateAscentBinaryClassifier");
        }

        [Fact]
        public void EntryPointSDCAMulticlass()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.StochasticDualCoordinateAscentClassifier");
        }

        [Fact()]
        public void EntryPointSDCARegression()
        {
            TestEntryPointRoutine(TestDatasets.generatedRegressionDatasetmacro.trainFilename, "Trainers.StochasticDualCoordinateAscentRegressor", loader: TestDatasets.generatedRegressionDatasetmacro.loaderSettings);
        }

        [Fact]
        public void EntryPointLogisticRegressionMulticlass()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.LogisticRegressionClassifier");
        }

        [Fact]
        public void EntryPointPcaAnomaly()
        {
            TestEntryPointRoutine("MNIST.Train.0-class.tiny.txt", "Trainers.PcaAnomalyDetector", "col=Features:R4:1-784");
        }

        [Fact]
        public void EntryPointPcaTransform()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Label:0 col=Features:1-9",
                new[]
                {
                    "Transforms.PcaCalculator",
                    "Transforms.PcaCalculator"
                },
                new[]
                {
                   @"'Column': [{
                    'Name': 'Pca1',
                    'Source': 'Features'
                     }],
                    'Rank' : 4,
                    'Oversampling': 2",
                @"'Column': [{
                    'Name': 'Pca2',
                    'Source': 'Features'
                    }],
                    'Rank' : 6,
                    'Center': 'False'"
                });
        }

        [Fact]
        public void EntryPointLightLdaTransformer()
        {
            string dataFile = DeleteOutputPath("SavePipe", "SavePipeTextLightLda-SampleText.txt");
            File.WriteAllLines(dataFile, new[] {
                "The quick brown fox jumps over the lazy dog.",
                "The five boxing wizards jump quickly."
            });

            TestEntryPointPipelineRoutine(dataFile, "sep={ } col=T:TX:0-**",
                new[]
                {
                    "Transforms.TextFeaturizer",
                    "Transforms.LightLda"
                },
                new[]
                {
                   @"'Column': {
                    'Name': 'T',
                    'Source': [
                        'T'
                    ]

                },
                'VectorNormalizer': 'None'",
                    @"'Column': [
                      {
                        'Name': 'T',
                        'Source': 'T'
                      }]"
                });
        }

        [Fact]
        public void EntryPointAveragePerceptron()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.AveragedPerceptronBinaryClassifier");
        }

        [Fact]
        public void EntryPointOnlineGradientDescent()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.OnlineGradientDescentRegressor");
        }

        [Fact]
        public void EntryPointLinearSVM()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.LinearSvmBinaryClassifier");
        }

        [Fact]
        public void EntryPointBinaryEnsemble()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.EnsembleBinaryClassifier", xfNames:
                new[] { "Transforms.ColumnTypeConverter" },
                xfArgs:
                new[] {
                    @"'Column': [{'Name': 'Label','Source': 'Label','Type': 'BL'}]"
                });
        }

        [Fact]
        public void EntryPointClassificationEnsemble()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.EnsembleClassification", xfNames:
                new[] { "Transforms.TextToKeyConverter" },
                xfArgs:
                new[] {
                    @"'Column': [{'Name': 'Label','Source': 'Label'}]"
                });
        }

        [Fact]
        public void EntryPointRegressionEnsemble()
        {
            TestEntryPointRoutine(TestDatasets.generatedRegressionDatasetmacro.trainFilename, "Trainers.EnsembleRegression", loader: TestDatasets.generatedRegressionDatasetmacro.loaderSettings);
        }

        [Fact]
        public void EntryPointNaiveBayesMulticlass()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.NaiveBayesClassifier");
        }

        [Fact]
        public void EntryPointHogwildSGD()
        {
            TestEntryPointRoutine("breast-cancer.txt", "Trainers.StochasticGradientDescentBinaryClassifier");
        }

        [Fact]
        public void EntryPointPoissonRegression()
        {
            TestEntryPointRoutine(TestDatasets.generatedRegressionDatasetmacro.trainFilename, "Trainers.PoissonRegressor", loader: TestDatasets.generatedRegressionDatasetmacro.loaderSettings);
        }

        [Fact]
        public void EntryPointBootstrap()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Label:R4:0 col=Features:R4:1-9",
                new[]
                {
                    "Transforms.ApproximateBootstrapSampler"
                },
                new[]
                {
                    @"'Seed': '1'"
                });
        }

        [Fact]
        public void EntryPointConvert()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=LT:TX:0 col=LB:BL:0 col=FT:TX:1-9 col=LN:0 col=FN:1-9 col=Key:U2[0-9]:2",
                new[]
                {
                    "Transforms.ColumnTypeConverter",
                    "Transforms.ColumnTypeConverter",
                },
                new[]
                {
                    @"'Column': [
                      {
                        'Name': 'Label',
                        'Source': 'LT'
                      },
                      {
                        'Name': 'Label2',
                        'Source': 'LB'
                      },
                      {
                        'Name': 'Feat',
                        'Source': 'FT',
                        'Type': 'R4'
                      },
                      {
                        'Name': 'Key1',
                        'Source': 'Key',
                        'Range': '0-9'
                      }
                      ]",
                    @"'Column': [
                      {
                        'Name': 'Doubles',
                        'Source': 'Feat'
                      }
                      ],
                      'Type': 'R8'",
                });
        }

        [Fact]
        public void EntryPointGroupingOperations()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=G1:TX:0 col=G2:R4:0 col=G3:U4[0-100]:1 col=V1:R4:2 col=V2:TX:3 col=V3:U2[0-10]:4 col=V4:I4:5",
                new[]
                {
                    "Transforms.CombinerByContiguousGroupId",
                    "Transforms.Segregator"
                },
                new[]
                {
                    @"'G': [ 'G1', 'G2', 'G3' ],
                      'Col': [ 'V1', 'V2', 'V3', 'V4' ]",
                    @"'Col': [ 'V1', 'V2', 'V3', 'V4' ]"
                });
        }

        [Fact]
        public void EntryPointNAFilter()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Features:R4:1-9 header+",
                new[]
                {
                    "Transforms.MissingValuesRowDropper"
                },
                new[]
                {
                    @"'Column': [ 'Features' ]",
                });

            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Features:R4:1-9 header+",
                new[]
                {
                    "Transforms.MissingValuesRowDropper"
                },
                new[]
                {
                    @"'Column': [ 'Features' ],
                      'Complement': 'true'",
                });
        }

        [Fact]
        public void EntryPointGcnTransform()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=FV1:2-3 col=FV2:3-4 col=FV3:4-5 col=FV4:7-9 col=Label:0",
                new[]
                {
                    "Transforms.LpNormalizer",
                    "Transforms.LpNormalizer",
                    "Transforms.GlobalContrastNormalizer",
                    "Transforms.GlobalContrastNormalizer",
                },
                new[]
                {
                    @"'Column': [
                      {
                        'Name': 'FV1N',
                        'Source': 'FV1'
                      }
                      ]",
                    @"'Column': [
                      {
                        'Name': 'FV2N',
                        'Source': 'FV2'
                      }
                      ],
                      'SubMean': 'false'",
                    @"'Column': [
                      {
                        'Name': 'FV3N',
                        'Source': 'FV3'
                      }
                      ],
                      'UseStd': 'true'",
                    @"'Column': [
                      {
                        'Name': 'FV4N',
                        'Source': 'FV4'
                      },
                      {
                        'Name': 'FV5N',
                        'Source': 'FV4',
                        'Scale': '0.2'
                      }
                      ],
                      'Scale': '0.1'",
                });
        }

        [Fact]
        public void EntryPointGenerateNumber()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Label:0",
                new[]
                {
                    "Transforms.RandomNumberGenerator",
                    "Transforms.RandomNumberGenerator",
                },
                new[]
                {
                    @"'Column': [
                      {
                        'Name': 'Random'
                      },
                      {
                        'Name': 'Counter1',
                        'Cnt': 'true'
                      }
                      ],
                      'Seed': '45'",
                    @"'Column': [
                      {
                        'Name': 'Counter2'
                      }
                      ],
                      'Cnt': 'true'",
                });
        }

        [Fact]
        public void EntryPointRangeFilter()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Filter:R4:3",
                new[]
                {
                    "Transforms.RowRangeFilter",
                    "Transforms.RowRangeFilter",
                },
                new[]
                {
                    @"'Column': 'Filter',
                      'Min': '3',
                      'Max': '4',
                      'IncludeMax': 'true'",
                    @"'Column': 'Filter',
                      'Min': '2',
                      'Max': '10',
                      'Complement': 'true'",
                });
        }

        [Fact]
        public void EntryPointSkipTakeFilter()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Filter:R4:3",
                new[]
                {
                    "Transforms.RowSkipAndTakeFilter",
                    "Transforms.RowTakeFilter",
                    "Transforms.RowSkipFilter",
                },
                new[]
                {
                    @"'Skip': '1',
                      'Take': '20'",
                    @"'Count': '10'",
                    @"'Count': '5'",
                });
        }

        internal void TestEntryPointPipelineRoutine(string dataFile, string schema, string[] xfNames, string[] xfArgs)
        {
            Env.Assert(Utils.Size(xfNames) == Utils.Size(xfArgs));

            var dataPath = GetDataPath(dataFile);
            var outputPath = DeleteOutputPath("model.zip");
            var outputDataPath = DeleteOutputPath("output-data.idv");
            string xfTemplate =
                @"'Name': '{0}',
                    'Inputs': {{
                      'Data': '$data{1}',
                      {2},
                    }},
                    'Outputs': {{
                      'OutputData': '$data{3}',
                      'Model': '$model{1}'
                    }},";

            string inputGraph =
                $@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file1',
                        'CustomSchema': '{schema}'
                      }},
                      'Outputs': {{
                        'Data': '$data1'
                      }}
                    }},";

            string models = "";
            string sep = "";
            for (int i = 0; i < Utils.Size(xfNames); i++)
            {
                inputGraph =
                    $@"{inputGraph}
                       {{
                         {string.Format(xfTemplate, xfNames[i], i + 1, xfArgs[i], i + 2)}
                       }},";
                models = $@"{models}{sep}'$model{i + 1}'";

                sep = @",";
            }

            inputGraph =
                $@"{inputGraph}
                    {{
                      'Name': 'Transforms.ModelCombiner',
                      'Inputs': {{
                        'Models': [
                          {models}
                        ]
                      }},
                      'Outputs': {{
                        'OutputModel': '$model'
                      }}
                    }},
                    {{
                      'Name': 'Models.DatasetTransformer',
                      'Inputs': {{
                         'Data': '$data1',
                         'TransformModel': '$model'
                      }},
                      'Outputs': {{
                        'OutputData': '$output'
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file1' : '{EscapePath(dataPath)}'
                  }},
                  'Outputs' : {{
                    'model' : '{EscapePath(outputPath)}',
                    'output': '{EscapePath(outputDataPath)}'
                  }}
                }}";

            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();
        }

        internal void TestEntryPointRoutine(string dataFile, string trainerName, string loader = null, string trainerArgs = null, string[] xfNames = null, string[] xfArgs = null)
        {
            var dataPath = GetDataPath(dataFile);
            var outputPath = DeleteOutputPath("model.zip");
            string xfTemplate =
               @"'Name': '{0}',
                    'Inputs': {{
                      'Data': '$data{1}',
                      {2},
                    }},
                    'Outputs': {{
                      'OutputData': '$data{3}'
                    }}";
            var transforms = "";

            for (int i = 0; i < Utils.Size(xfNames); i++)
            {
                transforms =
                    $@"{transforms}
                       {{
                         {string.Format(xfTemplate, xfNames[i], i + 1, xfArgs[i], i + 2)}
                       }},";
            }
            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file1'
                        {3}
                      }},
                      'Outputs': {{
                        'Data': '$data1'
                      }}
                    }},
                    {5}
                    {{
                      'Name': '{2}',
                      'Inputs': {{
                        'TrainingData': '$data{6}'
                         {4}
                      }},
                      'Outputs': {{
                        'PredictorModel': '$model'
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file1' : '{0}'
                  }},
                  'Outputs' : {{
                    'model' : '{1}'
                  }}
                }}", EscapePath(dataPath), EscapePath(outputPath), trainerName,
            string.IsNullOrWhiteSpace(loader) ? "" : string.Format(",'CustomSchema': 'sparse+ {0}'", loader),
            string.IsNullOrWhiteSpace(trainerArgs) ? "" : trainerArgs,
            transforms,
            xfNames != null ? xfNames.Length + 1 : 1
            );

            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();
        }

        [Fact]
        public void TestInputBuilderBasicArgs()
        {
            var catalog = Env.ComponentCatalog;
            bool success = catalog.TryFindEntryPoint("Transforms.MinMaxNormalizer", out ComponentCatalog.EntryPointInfo info);
            Assert.True(success);
            var inputBuilder = new InputBuilder(Env, info.InputType, catalog);

            var args = new NormalizeTransform.MinMaxArguments()
            {
                Columns = new[]
                {
                    NormalizeTransform.AffineColumn.Parse("A"),
                    new NormalizeTransform.AffineColumn() { Name = "B", Source = "B", EnsureZeroUntouched = false },
                },
                EnsureZeroUntouched = true, // Same as default, should not appear in the generated JSON.
                MaximumExampleCount = 1000
            };

            var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            var inputMap = new Dictionary<ParameterBinding, VariableBinding>();

            var parameterBinding = new SimpleParameterBinding("Data");
            inputBindingMap.Add("Data", new List<ParameterBinding>() { parameterBinding });
            inputMap.Add(parameterBinding, new SimpleVariableBinding("data"));

            var result = inputBuilder.GetJsonObject(args, inputBindingMap, inputMap);
            var json = FixWhitespace(result.ToString(Formatting.Indented));

            var expected =
                @"{
  ""Column"": [
    {
      ""Name"": ""A"",
      ""Source"": ""A""
    },
    {
      ""FixZero"": false,
      ""Name"": ""B"",
      ""Source"": ""B""
    }
  ],
  ""MaxTrainingExamples"": 1000,
  ""Data"": ""$data""
}";
            expected = FixWhitespace(expected);
            Assert.Equal(expected, json);
        }

        private static string FixWhitespace(string json)
        {
            return json
                .Trim()
                .Replace("\r", "\n")
                .Replace("\n\n", "\n");
        }

        [Fact]
        public void TestInputBuilderComponentFactories()
        {
            var catalog = Env.ComponentCatalog;
            bool success = catalog.TryFindEntryPoint("Trainers.StochasticDualCoordinateAscentBinaryClassifier", out ComponentCatalog.EntryPointInfo info);
            Assert.True(success);
            var inputBuilder = new InputBuilder(Env, info.InputType, catalog);

            var options = new LegacySdcaBinaryTrainer.Options()
            {
                NormalizeFeatures = NormalizeOption.Yes,
                ConvergenceCheckFrequency = 42
            };

            var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            var inputMap = new Dictionary<ParameterBinding, VariableBinding>();

            var parameterBinding = new SimpleParameterBinding("TrainingData");
            inputBindingMap.Add("TrainingData", new List<ParameterBinding>() { parameterBinding });
            inputMap.Add(parameterBinding, new SimpleVariableBinding("data"));

            var result = inputBuilder.GetJsonObject(options, inputBindingMap, inputMap);
            var json = FixWhitespace(result.ToString(Formatting.Indented));

            var expected =
                @"{
  ""ConvergenceCheckFrequency"": 42,
  ""TrainingData"": ""$data"",
  ""NormalizeFeatures"": ""Yes""
}";
            expected = FixWhitespace(expected);
            Assert.Equal(expected, json);

            options.LossFunction = new HingeLoss.Options();
            result = inputBuilder.GetJsonObject(options, inputBindingMap, inputMap);
            json = FixWhitespace(result.ToString(Formatting.Indented));

            expected =
                @"{
  ""LossFunction"": {
    ""Name"": ""HingeLoss""
  },
  ""ConvergenceCheckFrequency"": 42,
  ""TrainingData"": ""$data"",
  ""NormalizeFeatures"": ""Yes""
}";
            expected = FixWhitespace(expected);
            Assert.Equal(expected, json);

            options.LossFunction = new HingeLoss.Options() { Margin = 2 };
            result = inputBuilder.GetJsonObject(options, inputBindingMap, inputMap);
            json = FixWhitespace(result.ToString(Formatting.Indented));

            expected =
                @"{
  ""LossFunction"": {
    ""Name"": ""HingeLoss"",
    ""Settings"": {
      ""Margin"": 2.0
    }
  },
  ""ConvergenceCheckFrequency"": 42,
  ""TrainingData"": ""$data"",
  ""NormalizeFeatures"": ""Yes""
}";
            expected = FixWhitespace(expected);
            Assert.Equal(expected, json);
        }

        [Fact]
        public void EntryPointNormalizeIfNeeded()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {
                        'InputFile': '$file'
                      },
                      'Outputs': {
                        'Data': '$data1'
                      }
                    },
                    {
                      'Name': 'Transforms.ConditionalNormalizer',
                      'Inputs': {
                        'Data': '$data1',
                        'Column': [{ 'Name': 'Features', 'Source': 'Features' }]
                      },
                      'Outputs': {
                        'OutputData': '$data2',
                        'Model': '$transform'
                      }
                    },
                    {
                      'Name': 'Trainers.LogisticRegressionBinaryClassifier',
                      'Inputs': {
                        'TrainingData': '$data2',
                        'NumThreads': 1
                      },
                      'Outputs': {
                        'PredictorModel': '$predictor'
                      }
                    },
                    {
                      'Name': 'Transforms.ManyHeterogeneousModelCombiner',
                      'Inputs': {
                        'TransformModels': ['$transform'],
                        'PredictorModel': '$predictor'
                      },
                      'Outputs': {
                        'PredictorModel': '$model'
                      }
                    }
                  ]
                }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<PredictorModel>("model");
            Assert.NotNull(model);
        }

        [Fact]
        public void EntryPointTrainTestMacroNoTransformInput()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {
                        'InputFile': '$file'
                      },
                      'Outputs': {
                        'Data': '$data'
                      }
                    },
                    {
                      'Name': 'Transforms.TrainTestDatasetSplitter',
                      'Inputs': {
                        'Data': '$data',
                        'Fraction': 0.8
                      },
                      'Outputs': {
                        'TrainData': '$TrainData',
                        'TestData': '$TestData'
                      }
                    },
                    {
                      'Name': 'Models.TrainTestEvaluator',
                      'Inputs': {
                        'TrainingData': '$TrainData',
                        'TestingData': '$TestData',
                        'Nodes': [
                          {
                            'Name': 'Transforms.ConditionalNormalizer',
                            'Inputs': {
                              'Data': '$data1',
                              'Column': [{ 'Name': 'Features', 'Source': 'Features' }]
                            },
                            'Outputs': {
                              'OutputData': '$data2',
                              'Model': '$transform'
                            }
                          },
                          {
                            'Name': 'Trainers.LogisticRegressionBinaryClassifier',
                            'Inputs': {
                              'TrainingData': '$data2',
                              'NumThreads': 1
                            },
                            'Outputs': {
                              'PredictorModel': '$predictor'
                            }
                          },
                          {
                            'Name': 'Transforms.ManyHeterogeneousModelCombiner',
                            'Inputs': {
                              'TransformModels': ['$transform'],
                              'PredictorModel': '$predictor'
                            },
                            'Outputs': {
                              'PredictorModel': '$model'
                            }
                          }
                        ],
                        'Inputs': {
                          'Data': '$data1'
                        },
                        'Outputs': {
                          'PredictorModel': '$model'
                        }
                      },
                      'Outputs': {
                        'PredictorModel': '$model',
                        'Warnings': '$Warnings',
                        'OverallMetrics': '$OverallMetrics',
                        'PerInstanceMetrics': '$PerInstanceMetrics',
                        'ConfusionMatrix': '$ConfusionMatrix'
                      }
                    },
                  ]
                }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<PredictorModel>("model");
            Assert.NotNull(model);

            var metrics = runner.GetOutput<IDataView>("OverallMetrics");
            Assert.NotNull(metrics);
            using (var cursor = metrics.GetRowCursorForAllColumns())
            {
                var aucCol = cursor.Schema.GetColumnOrNull("AUC");
                Assert.True(aucCol.HasValue);
                var aucGetter = cursor.GetGetter<double>(aucCol.Value);
                Assert.True(cursor.MoveNext());
                double auc = 0;
                aucGetter(ref auc);
                Assert.True(auc > 0.99);
            }
        }

        [Fact]
        public void EntryPointKMeans()
        {
            TestEntryPointRoutine("Train-Tiny-28x28.txt", "Trainers.KMeansPlusPlusClusterer", "col=Weight:R4:0 col=Features:R4:1-784", ",'InitializationAlgorithm':'KMeansPlusPlus'");
        }

        [Fact]
        public void EntryPointTrainTestMacro()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {
                        'InputFile': '$file'
                      },
                      'Outputs': {
                        'Data': '$data'
                      }
                    },
                    {
                      'Name': 'Transforms.TrainTestDatasetSplitter',
                      'Inputs': {
                        'Data': '$data',
                        'Fraction': 0.8
                      },
                      'Outputs': {
                        'TrainData': '$TrainData',
                        'TestData': '$TestData'
                      }
                    },
                    {
                        'Name': 'Transforms.ConditionalNormalizer',
                        'Inputs': {
                            'Data': '$TrainData',
                            'Column': [{ 'Name': 'Features', 'Source': 'Features' }]
                        },
                        'Outputs': {
                            'OutputData': '$data2',
                            'Model': '$transform'
                        }
                    },
                    {
                      'Name': 'Models.TrainTestEvaluator',
                      'Inputs': {
                        'TrainingData': '$data2',
                        'TestingData': '$TestData',
                        'TransformModel': '$transform',
                        'Nodes': [                          
                          {
                            'Name': 'Trainers.LogisticRegressionBinaryClassifier',
                            'Inputs': {
                              'TrainingData': '$data1',
                              'NumThreads': 1
                            },
                            'Outputs': {
                              'PredictorModel': '$model'
                            }
                          }
                        ],
                        'Inputs': {
                          'Data': '$data1'
                        },
                        'Outputs': {
                          'PredictorModel': '$model'
                        }
                      },
                      'Outputs': {
                        'PredictorModel': '$model',
                        'Warnings': '$Warnings',
                        'OverallMetrics': '$OverallMetrics',
                        'PerInstanceMetrics': '$PerInstanceMetrics',
                        'ConfusionMatrix': '$ConfusionMatrix'
                      }
                    },
                  ]
                }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<PredictorModel>("model");
            Assert.NotNull(model);

            var metrics = runner.GetOutput<IDataView>("OverallMetrics");
            Assert.NotNull(metrics);
            using (var cursor = metrics.GetRowCursorForAllColumns())
            {
                var aucCol = cursor.Schema.GetColumnOrNull("AUC");
                Assert.True(aucCol.HasValue);
                var aucGetter = cursor.GetGetter<double>(aucCol.Value);
                Assert.True(cursor.MoveNext());
                double auc = 0;
                aucGetter(ref auc);
                Assert.True(auc > 0.99);
            }
        }

        [Fact]
        public void EntryPointChainedTrainTestMacros()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {
                        'InputFile': '$file'
                      },
                      'Outputs': {
                        'Data': '$data'
                      }
                    },
                    {
                      'Name': 'Transforms.TrainTestDatasetSplitter',
                      'Inputs': {
                        'Data': '$data',
                        'Fraction': 0.8
                      },
                      'Outputs': {
                        'TrainData': '$TrainData',
                        'TestData': '$TestData'
                      }
                    },
                    {
                        'Name': 'Transforms.ConditionalNormalizer',
                        'Inputs': {
                            'Data': '$TrainData',
                            'Column': [{ 'Name': 'Features', 'Source': 'Features' }]
                        },
                        'Outputs': {
                            'OutputData': '$data2',
                            'Model': '$transform1'
                        }
                    },
                    {
                    'Name': 'Transforms.ColumnCopier',
                    'Inputs': {
                        'Column': [
                            {
                                'Name': 'Features2',
                                'Source': 'Features'

                            }
                        ],
                        'Data': '$data2'
                    },
                    'Outputs': {
                        'OutputData': '$data3',
                        'Model': '$transform2'
                        }
                    },
                    {
                        'Name': 'Transforms.ModelCombiner',
                        'Inputs': {
                            'Models': [
                                '$transform1',
                                '$transform2'
                            ]
                        },
                        'Outputs': {
                            'OutputModel': '$CombinedModel'
                        }
                    },
                    {
                      'Name': 'Models.TrainTestEvaluator',
                      'Inputs': {
                        'TrainingData': '$data3',
                        'TestingData': '$TestData',
                        'TransformModel': '$CombinedModel',
                        'Nodes': [                          
                          {
                            'Name': 'Trainers.LogisticRegressionBinaryClassifier',
                            'Inputs': {
                              'TrainingData': '$data1',
                              'NumThreads': 1
                            },
                            'Outputs': {
                              'PredictorModel': '$model'
                            }
                          }
                        ],
                        'Inputs': {
                          'Data': '$data1'
                        },
                        'Outputs': {
                          'PredictorModel': '$model'
                        }
                      },
                      'Outputs': {
                        'PredictorModel': '$model',
                        'Warnings': '$Warnings',
                        'OverallMetrics': '$OverallMetrics',
                        'PerInstanceMetrics': '$PerInstanceMetrics',
                        'ConfusionMatrix': '$ConfusionMatrix'
                      }
                    },
                    {
                      'Name': 'Models.TrainTestEvaluator',
                      'Inputs': {
                        'TrainingData': '$data3',
                        'TestingData': '$TestData',
                        'TransformModel': '$CombinedModel',
                        'Nodes': [                          
                          {
                            'Name': 'Trainers.StochasticGradientDescentBinaryClassifier',
                            'Inputs': {
                              'TrainingData': '$data4',
                              'NumThreads': 1
                            },
                            'Outputs': {
                              'PredictorModel': '$model2'
                            }
                          }
                        ],
                        'Inputs': {
                          'Data': '$data4'
                        },
                        'Outputs': {
                          'PredictorModel': '$model2'
                        }
                      },
                      'Outputs': {
                        'PredictorModel': '$model2',
                        'Warnings': '$Warnings2',
                        'OverallMetrics': '$OverallMetrics2',
                        'PerInstanceMetrics': '$PerInstanceMetrics2',
                        'ConfusionMatrix': '$ConfusionMatrix2'
                      }
                    },
                  ]
                }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<PredictorModel>("model");
            Assert.NotNull(model);

            model = runner.GetOutput<PredictorModel>("model2");
            Assert.NotNull(model);

            var metrics = runner.GetOutput<IDataView>("OverallMetrics");

            Action<IDataView> validateAuc = (metricsIdv) =>
            {
                Assert.NotNull(metricsIdv);
                using (var cursor = metricsIdv.GetRowCursorForAllColumns())
                {
                    var aucCol = cursor.Schema.GetColumnOrNull("AUC");
                    var aucGetter = cursor.GetGetter<double>(aucCol.Value);
                    Assert.True(cursor.MoveNext());
                    double auc = 0;
                    aucGetter(ref auc);
                    Assert.True(auc > 0.99);
                }
            };

            validateAuc(metrics);

            metrics = runner.GetOutput<IDataView>("OverallMetrics2");
            validateAuc(metrics);
        }

        [Fact]
        public void EntryPointChainedCrossValMacros()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {
                        'InputFile': '$file'
                      },
                      'Outputs': {
                        'Data': '$data'
                      }
                    },
                    {
                        'Name': 'Transforms.ConditionalNormalizer',
                        'Inputs': {
                            'Data': '$data',
                            'Column': [{ 'Name': 'Features', 'Source': 'Features' }]
                        },
                        'Outputs': {
                            'OutputData': '$data2',
                            'Model': '$transform1'
                        }
                    },
                    {
                    'Name': 'Transforms.ColumnCopier',
                    'Inputs': {
                        'Column': [
                            {
                                'Name': 'Features2',
                                'Source': 'Features'

                            }
                        ],
                        'Data': '$data2'
                    },
                    'Outputs': {
                        'OutputData': '$data3',
                        'Model': '$transform2'
                        }
                    },
                    {
                        'Name': 'Transforms.ModelCombiner',
                        'Inputs': {
                            'Models': [
                                '$transform1',
                                '$transform2'
                            ]
                        },
                        'Outputs': {
                            'OutputModel': '$CombinedModel'
                        }
                    },
                    {
                      'Name': 'Models.CrossValidator',
                      'Inputs': {
                        'Data': '$data3',
                        'TransformModel': '$CombinedModel',
                        'Kind': 'SignatureBinaryClassifierTrainer',
                        'NumFolds': 3,
                        'Nodes': [                          
                          {
                            'Name': 'Trainers.LogisticRegressionBinaryClassifier',
                            'Inputs': {
                              'TrainingData': '$data6',
                              'NumThreads': 1
                            },
                            'Outputs': {
                              'PredictorModel': '$model'
                            }
                          }
                        ],
                        'Inputs': {
                          'Data': '$data6'
                        },
                        'Outputs': {
                          'PredictorModel': '$model'
                        }
                      },
                      'Outputs': {
                        'PredictorModel': '$model',
                        'Warnings': '$Warnings',
                        'OverallMetrics': '$OverallMetrics',
                        'PerInstanceMetrics': '$PerInstanceMetrics',
                        'ConfusionMatrix': '$ConfusionMatrix'
                      }
                    },
                    {
                      'Name': 'Models.CrossValidator',
                      'Inputs': {
                        'Data': '$data3',
                        'TransformModel': '$CombinedModel',
                        'Kind': 'SignatureBinaryClassifierTrainer',
                        'Nodes': [
                            {
                                'Name': 'Transforms.ColumnCopier',
                                'Inputs': {
                                    'Column': [
                                        {
                                            'Name': 'Features3',
                                            'Source': 'Features'

                                        }
                                    ],
                                    'Data': '$data4'
                                },
                                'Outputs': {
                                    'OutputData': '$data5',
                                    'Model': '$transform3'
                                    }
                               },
                              {
                                'Name': 'Trainers.StochasticDualCoordinateAscentBinaryClassifier',
                                'Inputs': {
                                  'TrainingData': '$data5',
                                  'NumThreads': 1
                                },
                                'Outputs': {
                                  'PredictorModel': '$predictor'
                                }
                              },
                            {
                                'Name': 'Transforms.ManyHeterogeneousModelCombiner',
                                'Inputs': {
                                    'TransformModels': [
                                        '$transform3'
                                    ],
                                    'PredictorModel': '$predictor'
                                },
                                'Outputs': {
                                    'PredictorModel': '$model2'
                                }
                            }
                        ],
                        'Inputs': {
                          'Data': '$data4'
                        },
                        'Outputs': {
                          'PredictorModel': '$model2'
                        }
                      },
                      'Outputs': {
                        'PredictorModel': '$model2',
                        'Warnings': '$Warnings2',
                        'OverallMetrics': '$OverallMetrics2',
                        'PerInstanceMetrics': '$PerInstanceMetrics2',
                        'ConfusionMatrix': '$ConfusionMatrix2'
                      }
                    },   
                  ]
                }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<PredictorModel[]>("model");
            Assert.NotNull(model[0]);

            model = runner.GetOutput<PredictorModel[]>("model2");
            Assert.NotNull(model[0]);

            var metrics = runner.GetOutput<IDataView>("OverallMetrics");

            Action<IDataView> aucValidate = (metricsIdv) =>
            {
                Assert.NotNull(metricsIdv);
                using (var cursor = metrics.GetRowCursorForAllColumns())
                {
                    var aucColumn = cursor.Schema.GetColumnOrNull("AUC");
                    Assert.True(aucColumn.HasValue);
                    var aucGetter = cursor.GetGetter<double>(aucColumn.Value);
                    Assert.True(cursor.MoveNext());
                    double auc = 0;
                    aucGetter(ref auc);
                    Assert.True(auc > 0.99);
                }
            };

            aucValidate(metrics);

            metrics = runner.GetOutput<IDataView>("OverallMetrics2");
            aucValidate(metrics);
        }

        [Fact]
        public void EntryPointSerialization()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {
                        'InputFile': '$file'
                      },
                      'Outputs': {
                        'Data': '$data1'
                      }
                    },
                    {
                      'Name': 'Transforms.ConditionalNormalizer',
                      'Inputs': {
                        'Data': '$data1',
                        'Column': [{ 'Name': 'Features', 'Source': 'Features' }]
                      },
                      'Outputs': {
                        'OutputData': '$data2',
                        'Model': '$transform'
                      }
                    },
                    {
                      'Name': 'Trainers.LogisticRegressionBinaryClassifier',
                      'Inputs': {
                        'TrainingData': '$data2',
                        'NumThreads': 1
                      },
                      'Outputs': {
                        'PredictorModel': '$predictor'
                      }
                    },
                    {
                      'Name': 'Transforms.ManyHeterogeneousModelCombiner',
                      'Inputs': {
                        'TransformModels': ['$transform'],
                        'PredictorModel': '$predictor'
                      },
                      'Outputs': {
                        'PredictorModel': '$model'
                      }
                    }
                  ]
                }";

            JObject graphJson = JObject.Parse(inputGraph);
            var graph = new EntryPointGraph(Env, graphJson[FieldNames.Nodes] as JArray);
            // Serialize the nodes with ToJson() and then executing them to ensure serialization working correctly.
            var nodes = new JArray(graph.AllNodes.Select(node => node.ToJson()));
            var runner = new GraphRunner(Env, nodes);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<PredictorModel>("model");
            Assert.NotNull(model);
        }

        [Fact]
        public void EntryPointNodeSchedulingFields()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.CustomTextLoader',
                      'StageId': '5063dee8f19c4dd89a1fc3a9da5351a7',
                      'Inputs': {
                        'InputFile': '$file'
                      },
                      'Outputs': {
                        'Data': '$data1'
                      }
                    },
                    {
                      'Name': 'Transforms.ConditionalNormalizer',
                      'StageId': '2',
                      'Cost': 2.12,
                      'Inputs': {
                        'Data': '$data1',
                        'Column': [{ 'Name': 'Features', 'Source': 'Features' }]
                      },
                      'Outputs': {
                        'OutputData': '$data2',
                        'Model': '$transform'
                      }
                    },
                    {
                      'Name': 'Trainers.LogisticRegressionBinaryClassifier',
                      'Checkpoint': true,
                      'Cost': 3.14159,
                      'Inputs': {
                        'TrainingData': '$data2',
                        'NumThreads': 1
                      },
                      'Outputs': {
                        'PredictorModel': '$predictor'
                      }
                    }
                  ]
                }";
            JObject graphJson = JObject.Parse(inputGraph);
            var graph = new EntryPointGraph(Env, graphJson[FieldNames.Nodes] as JArray);
            for (int i = 0; i < 2; i++)
            {
                var nodes = graph.AllNodes.ToArray();
                Assert.False(nodes[0].Checkpoint);
                Assert.True(float.IsNaN(nodes[0].Cost));
                Assert.True(nodes[0].StageId == "5063dee8f19c4dd89a1fc3a9da5351a7");

                Assert.False(nodes[1].Checkpoint);
                Assert.True(nodes[1].Cost > 2);
                Assert.True(nodes[1].StageId == "2");

                Assert.True(nodes[2].Checkpoint);
                Assert.True(nodes[2].Cost > 3);
                Assert.True(nodes[2].StageId == "");

                // Serialize the graph and verify again.
                var serNodes = new JArray(graph.AllNodes.Select(node => node.ToJson()));
                graph = new EntryPointGraph(Env, serNodes);
            }
        }

        [Fact]
        public void EntryPointLinearPredictorSummary()
        {
            var dataPath = GetDataPath("breast-cancer-withheader.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);

            var dataView = EntryPoints.ImportTextData.TextLoader(Env, new EntryPoints.ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    Separators = new []{'\t' },
                    HasHeader = true,
                    Columns = new[]
                    {
                        new TextLoader.Column("Label", DataKind.Single, 0),
                        new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 9) })
                    }
                },

                InputFile = inputFile,
            }).Data;

            var lrInput = new LogisticRegressionBinaryTrainer.Options
            {
                TrainingData = dataView,
                NormalizeFeatures = NormalizeOption.Yes,
                NumberOfThreads = 1,
                ShowTrainingStatistics = true,
                ComputeStandardDeviation = new ComputeLRTrainingStdThroughMkl()
            };
            var model = LogisticRegressionBinaryTrainer.TrainBinary(Env, lrInput).PredictorModel;

            var mcLrInput = new LbfgsMaximumEntropyTrainer.Options
            {
                TrainingData = dataView,
                NormalizeFeatures = NormalizeOption.Yes,
                NumberOfThreads = 1,
                ShowTrainingStatistics = true
            };
            var mcModel = LbfgsMaximumEntropyTrainer.TrainMulticlass(Env, mcLrInput).PredictorModel;

            var output = SummarizePredictor.Summarize(Env,
                new SummarizePredictor.Input() { PredictorModel = model });

            var mcOutput = SummarizePredictor.Summarize(Env,
                new SummarizePredictor.Input() { PredictorModel = mcModel });

            using (var ch = Env.Register("LinearPredictorSummary").Start("Save Data Views"))
            {
                var weights = DeleteOutputPath(@"../Common/EntryPoints", "lr-weights.txt");
                var saver = Env.CreateSaver("Text");
                using (var file = Env.CreateOutputFile(weights))
                    DataSaverUtils.SaveDataView(ch, saver, output.Summary, file);

                var stats = DeleteOutputPath(@"../Common/EntryPoints", "lr-stats.txt");
                using (var file = Env.CreateOutputFile(stats))
                    DataSaverUtils.SaveDataView(ch, saver, output.Stats, file);

                weights = DeleteOutputPath(@"../Common/EntryPoints", "mc-lr-weights.txt");
                using (var file = Env.CreateOutputFile(weights))
                    DataSaverUtils.SaveDataView(ch, saver, mcOutput.Summary, file);

                stats = DeleteOutputPath(@"../Common/EntryPoints", "mc-lr-stats.txt");
                using (var file = Env.CreateOutputFile(stats))
                    DataSaverUtils.SaveDataView(ch, saver, mcOutput.Stats, file);
            }

            CheckEquality(@"../Common/EntryPoints", "lr-weights.txt", digitsOfPrecision: 4);
            CheckEquality(@"../Common/EntryPoints", "lr-stats.txt", digitsOfPrecision: 3);
            CheckEquality(@"../Common/EntryPoints", "mc-lr-weights.txt", digitsOfPrecision: 3);
            CheckEquality(@"../Common/EntryPoints", "mc-lr-stats.txt", digitsOfPrecision: 5);
            Done();
        }

        [Fact]
        public void EntryPointPcaPredictorSummary()
        {
            var dataPath = GetDataPath("MNIST.Train.0-class.tiny.txt");
            using (var inputFile = new SimpleFileHandle(Env, dataPath, false, false))
            {
                var dataView = EntryPoints.ImportTextData.TextLoader(Env, new EntryPoints.ImportTextData.LoaderInput()
                {
                    Arguments =
                {
                    AllowSparse = true,
                    Separators = new []{'\t' },
                    HasHeader = false,
                    Columns = new[]
                    {
                        new TextLoader.Column("Features", DataKind.Single, new [] { new TextLoader.Range(1, 784) })
                    }
                },

                    InputFile = inputFile,
                }).Data;

                var pcaInput = new RandomizedPcaTrainer.Options
                {
                    TrainingData = dataView,
                };
                var model = RandomizedPcaTrainer.TrainPcaAnomaly(Env, pcaInput).PredictorModel;

                var output = SummarizePredictor.Summarize(Env,
                    new SummarizePredictor.Input() { PredictorModel = model });

                using (var ch = Env.Register("PcaPredictorSummary").Start("Save Data Views"))
                {
                    var weights = DeleteOutputPath(@"../Common/EntryPoints", "pca-weights.txt");
                    var saver = Env.CreateSaver("Text");
                    using (var file = Env.CreateOutputFile(weights))
                        DataSaverUtils.SaveDataView(ch, saver, output.Summary, file);
                }

                CheckEquality(@"../Common/EntryPoints", "pca-weights.txt", digitsOfPrecision: 4);
                Done();
            }
        }

        [Fact]
        public void EntryPointPrepareLabelConvertPredictedLabel()
        {
            var dataPath = GetDataPath("iris.data");
            var outputPath = DeleteOutputPath("prepare-label-data.idv");
            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.CustomTextLoader',
                      'Inputs': {{
                        'InputFile': '$file1',
                        'CustomSchema': 'sep=comma col=Label:TX:4 col=Features:Num:0-3'
                      }},
                      'Outputs': {{
                        'Data': '$data1'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.LabelColumnKeyBooleanConverter',
                      'Inputs': {{
                        'Data': '$data1',
                        'LabelColumn': 'Label'
                      }},
                      'Outputs': {{
                        'OutputData': '$data2'
                      }}
                    }},
                    {{
                      'Name': 'Trainers.LogisticRegressionClassifier',
                      'Inputs': {{
                        'Data': '$data2'
                      }},
                      'Outputs': {{
                        'PredictorModel': '$model'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.DatasetScorer',
                      'Inputs': {{
                        'Data': '$data2',
                        'PredictorModel': '$model'
                      }},
                      'Outputs': {{
                        'ScoredData': '$data3'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.PredictedLabelColumnOriginalValueConverter',
                      'Inputs': {{
                        'Data': '$data3',
                        'PredictedLabelColumn': 'PredictedLabel'
                      }},
                      'Outputs': {{
                        'OutputData': '$data4'
                      }}
                    }},
                  ],
                  'Inputs' : {{
                    'file1' : '{0}'
                  }},
                  'Outputs' : {{
                    'data4' : '{1}'
                  }}
                }}", EscapePath(dataPath), EscapePath(outputPath));

            var jsonPath = DeleteOutputPath("graph.json");
            File.WriteAllLines(jsonPath, new[] { inputGraph });

            var args = new ExecuteGraphCommand.Arguments() { GraphPath = jsonPath };
            var cmd = new ExecuteGraphCommand(Env, args);
            cmd.Run();

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), outputPath))
            {
                using (var cursor = loader.GetRowCursorForAllColumns())
                {
                    ReadOnlyMemory<char> predictedLabel = default;

                    var predictedLabelCol = loader.Schema.GetColumnOrNull("PredictedLabel");
                    Assert.True(predictedLabelCol.HasValue);
                    var predictedLabelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(predictedLabelCol.Value);

                    while (cursor.MoveNext())
                    {
                        predictedLabelGetter(ref predictedLabel);
                        Assert.True(ReadOnlyMemoryUtils.EqualsStr("Iris-setosa", predictedLabel)
                            || ReadOnlyMemoryUtils.EqualsStr("Iris-versicolor", predictedLabel)
                            || ReadOnlyMemoryUtils.EqualsStr("Iris-virginica", predictedLabel));
                    }
                }
            }
        }

        [Fact]
        public void EntryPointTreeLeafFeaturizer()
        {
            var dataPath = GetDataPath("adult.tiny.with-schema.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
#pragma warning disable 0618
            var dataView = EntryPoints.ImportTextData.ImportText(Env, new EntryPoints.ImportTextData.Input { InputFile = inputFile }).Data;
#pragma warning restore 0618
            var cat = Categorical.CatTransformDict(Env, new OneHotEncodingTransformer.Options()
            {
                Data = dataView,
                Columns = new[] { new OneHotEncodingTransformer.Column { Name = "Categories", Source = "Categories" } }
            });
            var concat = SchemaManipulation.ConcatColumns(Env, new ColumnConcatenatingTransformer.Options()
            {
                Data = cat.OutputData,
                Columns = new[] { new ColumnConcatenatingTransformer.Column { Name = "Features", Source = new[] { "Categories", "NumericFeatures" } } }
            });

            var fastTree = Trainers.FastTree.FastTree.TrainBinary(Env, new FastTreeBinaryTrainer.Options
            {
                FeatureColumnName = "Features",
                NumberOfTrees = 5,
                NumberOfLeaves = 4,
                LabelColumnName = DefaultColumnNames.Label,
                TrainingData = concat.OutputData
            });

            var combine = ModelOperations.CombineModels(Env, new ModelOperations.PredictorModelInput()
            {
                PredictorModel = fastTree.PredictorModel,
                TransformModels = new[] { cat.Model, concat.Model }
            });

            var treeLeaf = TreeFeaturize.Featurizer(Env, new TreeEnsembleFeaturizerTransform.ArgumentsForEntryPoint
            {
                Data = dataView,
                PredictorModel = combine.PredictorModel
            });

            var view = treeLeaf.OutputData;
            var treesCol = view.Schema.GetColumnOrNull("Trees");
            Assert.True(treesCol.HasValue);

            var leavesCol = view.Schema.GetColumnOrNull("Leaves");
            Assert.True(leavesCol.HasValue);

            var pathsCol = view.Schema.GetColumnOrNull("Paths");
            Assert.True(pathsCol.HasValue);


            VBuffer<float> treeValues = default(VBuffer<float>);
            VBuffer<float> leafIndicators = default(VBuffer<float>);
            VBuffer<float> pathIndicators = default(VBuffer<float>);
            using (var curs = view.GetRowCursor(treesCol.Value, leavesCol.Value, pathsCol.Value))
            {
                var treesGetter = curs.GetGetter<VBuffer<float>>(treesCol.Value);
                var leavesGetter = curs.GetGetter<VBuffer<float>>(leavesCol.Value);
                var pathsGetter = curs.GetGetter<VBuffer<float>>(pathsCol.Value);
                while (curs.MoveNext())
                {
                    treesGetter(ref treeValues);
                    leavesGetter(ref leafIndicators);
                    pathsGetter(ref pathIndicators);

                    Assert.Equal(5, treeValues.Length);
                    Assert.Equal(5, treeValues.GetValues().Length);
                    Assert.Equal(20, leafIndicators.Length);
                    Assert.Equal(5, leafIndicators.GetValues().Length);
                    Assert.Equal(15, pathIndicators.Length);
                }
            }
        }

        [Fact]
        public void EntryPointWordEmbeddings()
        {
            string dataFile = DeleteOutputPath("SavePipe", "SavePipeTextWordEmbeddings-SampleText.txt");
            File.WriteAllLines(dataFile, new[] {
                "The quick brown fox jumps over the lazy dog.",
                "The five boxing wizards jump quickly."
            });
            var inputFile = new SimpleFileHandle(Env, dataFile, false, false);
            var dataView = EntryPoints.ImportTextData.TextLoader(Env, new EntryPoints.ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    Separators = new []{' '},
                    Columns = new[]
                    {
                        new TextLoader.Column("Text", DataKind.String,
                            new [] { new TextLoader.Range() { Min = 0, VariableEnd=true, ForceVector=true} })
                    }
                },
                InputFile = inputFile,
            }).Data;
            var embedding = Transforms.Text.TextAnalytics.WordEmbeddings(Env, new WordEmbeddingTransformer.Options()
            {
                Data = dataView,
                Columns = new[] { new WordEmbeddingTransformer.Column { Name = "Features", Source = "Text" } },
                ModelKind = WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding
            });
            var result = embedding.OutputData;
            using (var cursor = result.GetRowCursorForAllColumns())
            {
                var featColumn = result.Schema.GetColumnOrNull("Features");
                Assert.True(featColumn.HasValue);
                var featGetter = cursor.GetGetter<VBuffer<float>>(featColumn.Value);
                VBuffer<float> feat = default;
                while (cursor.MoveNext())
                {
                    featGetter(ref feat);
                    Assert.Equal(150, feat.GetValues().Length);
                    Assert.NotEqual(0, feat.GetValues()[0]);
                }
            }
        }

        [TensorFlowFact]
        public void EntryPointTensorFlowTransform()
        {
            Env.ComponentCatalog.RegisterAssembly(typeof(TensorFlowTransformer).Assembly);

            TestEntryPointPipelineRoutine(GetDataPath("Train-Tiny-28x28.txt"), "col=Label:R4:0 col=Placeholder:R4:1-784",
                new[] { "Transforms.TensorFlowScorer" },
                new[]
                {
                    @"'InputColumns': [ 'Placeholder' ],
                      'ModelLocation': 'mnist_model/frozen_saved_model.pb',
                      'OutputColumns': [ 'Softmax' ]"
                });
        }

        [Fact(Skip = "Needs real time series dataset. https://github.com/dotnet/machinelearning/issues/1120")]
        public void EntryPointSsaChangePoint()
        {
            TestEntryPointPipelineRoutine(GetDataPath(Path.Combine("Timeseries", "A4Benchmark-TS1.csv")), "sep=, col=Features:R4:1 header=+",
                new[]
                {
                    "TimeSeriesProcessingEntryPoints.SsaChangePointDetector",
                    "TimeSeriesProcessingEntryPoints.SsaChangePointDetector",
                },
                new[]
                {
                    @"'Src': 'Features',
                      'Name': 'Anomaly',
                      'Twnd': '500',
                      'Swnd': '50',
                      'Cnf': '93',
                      'Wnd': '20',
                      'Mart': 'Power',
                      'Eps': '0.1'",
                    @"'Src': 'Features',
                      'Name': 'Anomaly2',
                      'Twnd': '500',
                      'Swnd': '50',
                      'Cnf': '93',
                      'Wnd': '20',
                      'Mart': 'Mixture'"
                });
        }

        [Fact]
        public void EntryPointIidSpikeDetector()
        {
            TestEntryPointPipelineRoutine(GetDataPath(Path.Combine("Timeseries", "real_1.csv")), "sep=, col=Features:R4:1 header=+",
                new[]
                {
                    "TimeSeriesProcessingEntryPoints.IidSpikeDetector",
                    "TimeSeriesProcessingEntryPoints.IidSpikeDetector",
                },
                new[]
                {
                    @"'Src': 'Features',
                      'Name': 'Anomaly',
                      'Cnf': '99.5',
                      'Wnd': '200',
                      'Side': 'Positive'",
                    @"'Src': 'Features',
                      'Name': 'Anomaly2',
                      'Cnf': '99.5',
                      'Wnd': '200',
                      'Side': 'Negative'",
                });
        }

        [Fact(Skip = "Needs real time series dataset. https://github.com/dotnet/machinelearning/issues/1120")]
        public void EntryPointSsaSpikeDetector()
        {
            TestEntryPointPipelineRoutine(GetDataPath(Path.Combine("Timeseries", "A4Benchmark-TS2.csv")), "sep=, col=Features:R4:1 header=+",
                new[]
                {
                    "TimeSeriesProcessingEntryPoints.SsaSpikeDetector",
                    "TimeSeriesProcessingEntryPoints.SsaSpikeDetector",
                    "TimeSeriesProcessingEntryPoints.SsaSpikeDetector",
                },
                new[]
                {
                    @"'Src': 'Features',
                      'Name': 'Anomaly',
                      'Twnd': '500',
                      'Swnd': '50',
                      'Err': 'SignedDifference',
                      'Cnf': '99.5',
                      'Wnd': '100',
                      'Side': 'Negative'",
                    @"'Src': 'Features',
                      'Name': 'Anomaly2',
                      'Twnd': '500',
                      'Swnd': '50',
                      'Err': 'SignedDifference',
                      'Cnf': '99.5',
                      'Wnd': '100',
                      'Side': 'Positive'",
                    @"'Src': 'Features',
                      'Name': 'Anomaly3',
                      'Twnd': '500',
                      'Swnd': '50',
                      'Err': 'SignedDifference',
                      'Cnf': '99.5',
                      'Wnd': '100'",
                });
        }

        [Fact]
        public void EntryPointPercentileThreshold()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Input:R4:1",
                new[]
                {
                    "TimeSeriesProcessingEntryPoints.PercentileThresholdTransform"
                },
                new[]
                {
                    @"'Src': 'Input',
                      'Name': 'Output',
                      'Wnd': '10',
                      'Pcnt': '10'"
                });
        }

        [Fact]
        public void EntryPointPValue()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Input:R4:1",
                new[]
                {
                    "TimeSeriesProcessingEntryPoints.PValueTransform"
                },
                new[]
                {
                    @"'Src': 'Input',
                      'Name': 'Output',
                      'Wnd': '10'"
                });
        }

        [Fact]
        public void EntryPointSlidingWindow()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Input:R4:1",
                new[]
                {
                    "TimeSeriesProcessingEntryPoints.SlidingWindowTransform",
                    "TimeSeriesProcessingEntryPoints.SlidingWindowTransform",
                    "TimeSeriesProcessingEntryPoints.SlidingWindowTransform",
                    "TimeSeriesProcessingEntryPoints.SlidingWindowTransform",
                },
                new[]
                {
                    @"'Src': 'Input',
                      'Name': 'Output',
                      'Wnd': '3',
                      'L': '0'",
                    @"'Src': 'Input',
                      'Name': 'Output1',
                      'Wnd': '1',
                      'L': '1'",
                    @"'Src': 'Input',
                      'Name': 'Output2',
                      'Wnd': '1',
                      'L': '2'",
                    @"'Src': 'Input',
                      'Name': 'Output3',
                      'Wnd': '2',
                      'L': '1'"
                });
        }

        [Fact]
        public void EntryPointHashJoinCountTable()
        {
            TestEntryPointPipelineRoutine(GetDataPath("breast-cancer.txt"), "col=Text:Text:1-9 col=Label:0",
                new[]
                {
                    "Transforms.HashConverter",
                },
                new[]
                {
                    @"'Column': [
                      {
                        'Name': 'Temp',
                        'Src': 'Text'
                      },
                      {
                        'Name': 'Temp2',
                        'Src': 'Text',
                        'CustomSlotMap': '0,1;2,3,4,5'
                      }

                      ]"
                });
        }

        [Fact]
        public void TestSimpleExperiment()
        {
            var dataPath = GetDataPath("adult.tiny.with-schema.txt");
            string inputGraph = @"{
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': false,
                                'AllowSparse': false,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': null,
                                'TrimWhitespace': false,
                                'HasHeader': false
                            }
                        },
                        'Outputs': {
                            'Data': '$Var_642faec2bf064255bc9a2b1044e9d116'
                        }
                    }, {
                        'Name': 'Transforms.MinMaxNormalizer',
                        'Inputs': {
                            'Column': [{
                                    'FixZero': null,
                                    'MaxTrainingExamples': null,
                                    'Name': 'NumericFeatures',
                                    'Source': 'NumericFeatures'
                                }
                            ],
                            'FixZero': true,
                            'MaxTrainingExamples': 1000000000,
                            'Data': '$Var_642faec2bf064255bc9a2b1044e9d116'
                        },
                        'Outputs': {
                            'OutputData': '$outputData',
                            'Model': '$Var_9673b095f98f4ebcb19e8eb75a7a12e9'
                        }
                    }
                ]
            }";
            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();
            var data = runner.GetOutput<IDataView>("outputData");
            var schema = data.Schema;
            Assert.Equal(5, schema.Count);
            var expected = new[] { "Label", "Workclass", "Categories", "NumericFeatures", "NumericFeatures" };
            for (int i = 0; i < schema.Count; i++)
                Assert.Equal(expected[i], schema[i].Name);
        }

        [Fact]
        public void TestSimpleTrainExperiment()
        {
            var dataPath = GetDataPath("adult.tiny.with-schema.txt");
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': false,
                                'AllowSparse': false,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': null,
                                'TrimWhitespace': false,
                                'HasHeader': false
                            }
                        },
                        'Outputs': {
                            'Data': '$Var_99eb21288359485f936577da8f2e1061'
                        }
                    }, {
                        'Name': 'Transforms.CategoricalOneHotVectorizer',
                        'Inputs': {
                            'Column': [{
                                    'OutputKind': null,
                                    'MaxNumTerms': null,
                                    'Term': null,
                                    'Sort': null,
                                    'TextKeyValues': null,
                                    'Name': 'Categories',
                                    'Source': 'Categories'
                                }
                            ],
                            'OutputKind': 'Indicator',
                            'MaxNumTerms': 1000000,
                            'Term': null,
                            'Sort': 'ByOccurrence',
                            'TextKeyValues': true,
                            'Data': '$Var_99eb21288359485f936577da8f2e1061'
                        },
                        'Outputs': {
                            'OutputData': '$Var_c9e14b64d1a44114853331e80f1bde57',
                            'Model': '$Var_85534ab1fc57480899180be5bbf20b38'
                        }
                    }, {
                        'Name': 'Transforms.ColumnConcatenator',
                        'Inputs': {
                            'Column': [{
                                    'Name': 'Features',
                                    'Source': [
                                        'Categories',
                                        'NumericFeatures'
                                    ]
                                }
                            ],
                            'Data': '$Var_c9e14b64d1a44114853331e80f1bde57'
                        },
                        'Outputs': {
                            'OutputData': '$Var_51d3ddc9792d4c6eb975e600e87b8cbc',
                            'Model': '$Var_e3888e65f822424ca92959e442827d48'
                        }
                    }, {
                        'Name': 'Trainers.StochasticDualCoordinateAscentBinaryClassifier',
                        'Inputs': {
                            'LossFunction': {
                                'Name': 'HingeLoss',
                                'Settings': {
                                    'Margin': 1.1
                                }
                            },
                            'PositiveInstanceWeight': 1.0,
                            'Calibrator': {
                                'Name': 'PlattCalibrator',
                                'Settings': {}
                            },
                            'MaxCalibrationExamples': 1000000,
                            'L2Const': null,
                            'L1Threshold': null,
                            'NumThreads': 1,
                            'ConvergenceTolerance': 0.1,
                            'MaxIterations': null,
                            'Shuffle': false,
                            'CheckFrequency': null,
                            'BiasLearningRate': 0.0,
                            'LabelColumnName': 'Label',
                            'TrainingData': '$Var_51d3ddc9792d4c6eb975e600e87b8cbc',
                            'FeatureColumnName': 'Features',
                            'NormalizeFeatures': 'Auto',
                            'Caching': 'Auto'
                        },
                        'Outputs': {
                            'PredictorModel': '$Var_e7e860bdbf1c4a628a2a0912673afd36'
                        }
                    }, {
                        'Name': 'Transforms.DatasetScorer',
                        'Inputs': {
                            'Data': '$Var_51d3ddc9792d4c6eb975e600e87b8cbc',
                            'PredictorModel': '$Var_e7e860bdbf1c4a628a2a0912673afd36',
                            'Suffix': null
                        },
                        'Outputs': {
                            'ScoredData': '$Var_be77f9c4e45c43b7a67984304c291bf5',
                            'ScoringTransform': '$Var_826e5697e56a467a81331c5ef3eff37f'
                        }
                    }, {
                        'Name': 'Models.BinaryClassificationEvaluator',
                        'Inputs': {
                            'ProbabilityColumn': null,
                            'Threshold': 0.0,
                            'UseRawScoreThreshold': true,
                            'NumRocExamples': 100000,
                            'MaxAucExamples': -1,
                            'NumAuPrcExamples': 100000,
                            'LabelColumn': null,
                            'WeightColumn': null,
                            'ScoreColumn': null,
                            'StratColumn': null,
                            'Data': '$Var_be77f9c4e45c43b7a67984304c291bf5',
                            'NameColumn': 'Name'
                        },
                        'Outputs': {
                            'ConfusionMatrix': '$Var_cd6d3485a95d4405b469ce65c124e04a',
                            'Warnings': '$Var_94528ba8fca14eb48b7e3f712aced38a',
                            'OverallMetrics': '$Var_2130b277d4e0485f9cc5162c176767fa',
                            'PerInstanceMetrics': '$Var_991a2f8bed28442bb9bd0a0b9ff14e45'
                        }
                    }
                ]
            }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();
            var data = runner.GetOutput<IDataView>("Var_2130b277d4e0485f9cc5162c176767fa");

            var schema = data.Schema;
            var aucCol = schema.GetColumnOrNull("AUC");
            Assert.True(aucCol.HasValue);
            using (var cursor = data.GetRowCursor(aucCol.Value))
            {
                var getter = cursor.GetGetter<double>(aucCol.Value);
                var b = cursor.MoveNext();
                Assert.True(b);
                double auc = 0;
                getter(ref auc);
                Assert.Equal(0.93, auc, 2);
                b = cursor.MoveNext();
                Assert.False(b);
            }
        }

        [LessThanNetCore30OrNotNetCoreFact("netcoreapp3.0 output differs from Baseline")]
        public void TestCrossValidationMacro()
        {
            var dataPath = GetDataPath(TestDatasets.generatedRegressionDatasetmacro.trainFilename);
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': true,
                                'AllowSparse': true,
                                'InputSize': null,
                                'Separator': [
                                    ';'
                                ],
                                'Column': [{
                                        'Name': 'Label',
                                        'Type': 'R4',
                                        'Source': [{
                                                'Min': 11,
                                                'Max': 11,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'Features',
                                        'Type': 'R4',
                                        'Source': [{
                                                'Min': 0,
                                                'Max': 10,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }
                                ],
                                'TrimWhitespace': false,
                                'HasHeader': true
                            }
                        },
                        'Outputs': {
                            'Data': '$Var_6b6d6b5b5f894374a98904481d876a6e'
                        }
                    }, {
                        'Name': 'Models.CrossValidator',
                        'Inputs': {
                            'Data': '$Var_6b6d6b5b5f894374a98904481d876a6e',
                            'TransformModel': null,
                            'Nodes': [{
                                    'Name': 'Transforms.NoOperation',
                                    'Inputs': {
                                        'Data': '$Var_abda1d0923f64b56bd01dc42fb57db33'
                                    },
                                    'Outputs': {
                                        'OutputData': '$Var_65ecee1d96a84b9d9645f616b278e77e',
                                        'Model': '$Var_6807c0e8cb42452c8fc687545aabc43b'
                                    }
                                }, {
                                    'Name': 'Transforms.RandomNumberGenerator',
                                    'Inputs': {
                                        'Column': [{
                                                'Name': 'Weight1',
                                                'UseCounter': null,
                                                'Seed': null
                                            }
                                        ],
                                        'UseCounter': false,
                                        'Seed': 42,
                                        'Data': '$Var_65ecee1d96a84b9d9645f616b278e77e'
                                    },
                                    'Outputs': {
                                        'OutputData': '$Var_8b36a1e70c9f4504973140ad15eac72f',
                                        'Model': '$Var_94bac81ee2e448ba82e3a21e116c0f9c'
                                    }
                                }, {
                                    'Name': 'Trainers.PoissonRegressor',
                                    'Inputs': {
                                        'L2Weight': 1.0,
                                        'L1Weight': 1.0,
                                        'OptTol': 1E-07,
                                        'MemorySize': 20,
                                        'MaxIterations': 2147483647,
                                        'SgdInitializationTolerance': 0.0,
                                        'Quiet': false,
                                        'InitWtsDiameter': 0.0,
                                        'UseThreads': true,
                                        'NumThreads': 1,
                                        'DenseOptimizer': false,
                                        'EnforceNonNegativity': false,
                                        'ExampleWeightColumnName': 'Weight1',
                                        'LabelColumnName': 'Label',
                                        'TrainingData': '$Var_8b36a1e70c9f4504973140ad15eac72f',
                                        'FeatureColumnName': 'Features',
                                        'NormalizeFeatures': 'Auto',
                                        'Caching': 'Auto'
                                    },
                                    'Outputs': {
                                        'PredictorModel': '$Var_5763097adbdb40e3b161540cb0c88b91'
                                    }
                                }, {
                                    'Name': 'Transforms.ManyHeterogeneousModelCombiner',
                                    'Inputs': {
                                        'TransformModels': [
                                            '$Var_6807c0e8cb42452c8fc687545aabc43b',
                                            '$Var_94bac81ee2e448ba82e3a21e116c0f9c'
                                        ],
                                        'PredictorModel': '$Var_5763097adbdb40e3b161540cb0c88b91'
                                    },
                                    'Outputs': {
                                        'PredictorModel': '$Var_77f3e99700ae453586513565171faf55'
                                    }
                                }
                            ],
                            'Inputs': {
                                'Data': '$Var_abda1d0923f64b56bd01dc42fb57db33'
                            },
                            'Outputs': {
                                'PredictorModel': '$Var_77f3e99700ae453586513565171faf55'
                            },
                            'StratificationColumn': null,
                            'NumFolds': 2,
                            'Kind': 'SignatureRegressorTrainer',
                            'LabelColumn': 'Label',
                            'WeightColumn': 'Weight1',
                            'GroupColumn': null,
                            'NameColumn': null
                        },
                        'Outputs': {
                            'PredictorModel': '$Var_5aaddf0cdc6d4d92b05a2804fcc3a2ee',
                            'Warnings': '$Var_33701e91260c4a7184fd595ce392cb08',
                            'OverallMetrics': '$overallMetrics',
                            'PerInstanceMetrics': '$Var_89681d817cf543ecabbe6421bf37acb2',
                            'ConfusionMatrix': '$Var_43fa86567b5f4e129e58bd12a575c06b'
                        }
                    }
                ]
            }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();
            var data = runner.GetOutput<IDataView>("overallMetrics");

            var schema = data.Schema;
            var metricCol = schema.GetColumnOrNull("L1(avg)");
            Assert.True(metricCol.HasValue);
            var foldCol = schema.GetColumnOrNull("Fold Index");
            Assert.True(foldCol.HasValue);
            var isWeightedCol = schema.GetColumnOrNull("IsWeighted");
            Assert.True(isWeightedCol.HasValue);
            using (var cursor = data.GetRowCursor(metricCol.Value, foldCol.Value, isWeightedCol.Value))
            {
                var getter = cursor.GetGetter<double>(metricCol.Value);
                var foldGetter = cursor.GetGetter<ReadOnlyMemory<char>>(foldCol.Value);
                ReadOnlyMemory<char> fold = default;
                var isWeightedGetter = cursor.GetGetter<bool>(isWeightedCol.Value);
                bool isWeighted = default;
                double avg = 0;
                double weightedAvg = 0;
                bool b;
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

        [Fact]
        public void TestCrossValidationMacroWithMulticlass()
        {
            var dataPath = GetDataPath(@"Train-Tiny-28x28.txt");
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': true,
                                'AllowSparse': true,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': null,
                                'TrimWhitespace': false,
                                'HasHeader': false
                            }
                        },
                        'Outputs': {
                            'Data': '$Var_48530e4c7d0c4d0889ba9e6e80e6eb73'
                        }
                    }, {
                        'Name': 'Models.CrossValidator',
                        'Inputs': {
                            'Data': '$Var_48530e4c7d0c4d0889ba9e6e80e6eb73',
                            'TransformModel': null,
                            'Nodes': [{
                                    'Name': 'Transforms.NoOperation',
                                    'Inputs': {
                                        'Data': '$Var_1a2f44ae0aec4af4b4d1337d2cc733da'
                                    },
                                    'Outputs': {
                                        'OutputData': '$Var_a060169d8a924964b71447904c0d2ee9',
                                        'Model': '$Var_dbd0f197ee7145ce91ac26ef62936206'
                                    }
                                }, {
                                    'Name': 'Trainers.StochasticDualCoordinateAscentClassifier',
                                    'Inputs': {
                                        'LossFunction': {
                                            'Name': 'LogLoss',
                                            'Settings': {}
                                        },
                                        'L2Const': null,
                                        'L1Threshold': null,
                                        'NumThreads': 1,
                                        'ConvergenceTolerance': 0.1,
                                        'MaxIterations': null,
                                        'Shuffle': true,
                                        'CheckFrequency': null,
                                        'BiasLearningRate': 0.0,
                                        'LabelColumnName': 'Label',
                                        'TrainingData': '$Var_a060169d8a924964b71447904c0d2ee9',
                                        'FeatureColumnName': 'Features',
                                        'NormalizeFeatures': 'Auto',
                                        'Caching': 'Auto'
                                    },
                                    'Outputs': {
                                        'PredictorModel': '$Var_0bb334380a514e0ab6b2215b0c049846'
                                    }
                                }, {
                                    'Name': 'Transforms.ManyHeterogeneousModelCombiner',
                                    'Inputs': {
                                        'TransformModels': [
                                            '$Var_dbd0f197ee7145ce91ac26ef62936206'
                                        ],
                                        'PredictorModel': '$Var_0bb334380a514e0ab6b2215b0c049846'
                                    },
                                    'Outputs': {
                                        'PredictorModel': '$Var_0b4526e0c7534eada0264802128c32c5'
                                    }
                                }
                            ],
                            'Inputs': {
                                'Data': '$Var_1a2f44ae0aec4af4b4d1337d2cc733da'
                            },
                            'Outputs': {
                                'PredictorModel': '$Var_0b4526e0c7534eada0264802128c32c5'
                            },
                            'StratificationColumn': null,
                            'NumFolds': 2,
                            'Kind': 'SignatureMulticlassClassificationTrainer',
                            'LabelColumn': 'Label',
                            'WeightColumn': null,
                            'GroupColumn': null,
                            'NameColumn': null
                        },
                        'Outputs': {
                            'PredictorModel': '$Var_76decfcf71f5447d92869a4dd9200ea6',
                            'Warnings': '$warnings',
                            'OverallMetrics': '$overallMetrics',
                            'PerInstanceMetrics': '$Var_74d0215056034d6c9a99f90485530b89',
                            'ConfusionMatrix': '$confusionMatrix'
                        }
                    }
                ]
            }
            ";
            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();
            var data = runner.GetOutput<IDataView>("overallMetrics");

            var schema = data.Schema;
            var metricCol = schema.GetColumnOrNull("Accuracy(micro-avg)");
            Assert.True(metricCol.HasValue);
            var foldCol = schema.GetColumnOrNull("Fold Index");
            Assert.True(foldCol.HasValue);
            using (var cursor = data.GetRowCursor(metricCol.Value, foldCol.Value))
            {
                var getter = cursor.GetGetter<double>(metricCol.Value);
                var foldGetter = cursor.GetGetter<ReadOnlyMemory<char>>(foldCol.Value);
                ReadOnlyMemory<char> fold = default;

                // Get the average.
                var b = cursor.MoveNext();
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
                Assert.Equal(0.024809923969586353, stdev, 3);

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

            var confusion = runner.GetOutput<IDataView>("confusionMatrix");
            schema = confusion.Schema;
            var countCol = schema.GetColumnOrNull("Count");
            Assert.True(countCol.HasValue);
            foldCol = schema.GetColumnOrNull("Fold Index");
            Assert.True(foldCol.HasValue);
            var type = schema["Count"].Annotations.Schema[AnnotationUtils.Kinds.SlotNames].Type;
            Assert.True(type is VectorType vecType && vecType.ItemType is TextDataViewType && vecType.Size == 10);
            var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
            schema["Count"].GetSlotNames(ref slotNames);
            var slotNameValues = slotNames.GetValues();
            for (int i = 0; i < slotNameValues.Length; i++)
            {
                Assert.True(ReadOnlyMemoryUtils.EqualsStr(i.ToString(), slotNameValues[i]));
            }
            using (var curs = confusion.GetRowCursorForAllColumns())
            {
                var countGetter = curs.GetGetter<VBuffer<double>>(countCol.Value);
                var foldGetter = curs.GetGetter<ReadOnlyMemory<char>>(foldCol.Value);
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

            var warnings = runner.GetOutput<IDataView>("warnings");
            using (var cursor = warnings.GetRowCursorForAllColumns())
                Assert.False(cursor.MoveNext());
        }

        [Fact]
        public void TestCrossValidationMacroMulticlassWithWarnings()
        {
            var dataPath = GetDataPath(@"Train-Tiny-28x28.txt");
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': true,
                                'AllowSparse': true,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': null,
                                'TrimWhitespace': false,
                                'HasHeader': false
                            }
                        },
                        'Outputs': {
                            'Data': '$Var_48dc3daef3924a22af794e67896272b0'
                        }
                    }, {
                        'Name': 'Transforms.RowRangeFilter',
                        'Inputs': {
                            'Column': 'Label',
                            'Min': 0.0,
                            'Max': 5.0,
                            'Complement': false,
                            'IncludeMin': true,
                            'IncludeMax': null,
                            'Data': '$Var_48dc3daef3924a22af794e67896272b0'
                        },
                        'Outputs': {
                            'OutputData': '$Var_64f1865a99b84b9d9e0c72292c14c3af',
                            'Model': '$Var_4a34fa76d6d04c14b57b8f146010b9ad'
                        }
                    }, {
                        'Name': 'Transforms.TextToKeyConverter',
                        'Inputs': {
                            'Column': [{
                                    'MaxNumTerms': null,
                                    'Term': null,
                                    'Sort': 'ByValue',
                                    'TextKeyValues': null,
                                    'Name': 'Strat',
                                    'Source': 'Label'
                                }
                            ],
                            'MaxNumTerms': 1000000,
                            'Term': null,
                            'Sort': 'ByOccurrence',
                            'TextKeyValues': false,
                            'Data': '$Var_64f1865a99b84b9d9e0c72292c14c3af'
                        },
                        'Outputs': {
                            'OutputData': '$Var_41d7ff9c3dcd45fc869f2691dd628797',
                            'Model': '$Var_ee952a378c624306a7b6b8b65dbb8583'
                        }
                    }, {
                        'Name': 'Models.CrossValidator',
                        'Inputs': {
                            'Data': '$Var_41d7ff9c3dcd45fc869f2691dd628797',
                            'TransformModel': null,
                            'Nodes': [{
                                    'Name': 'Transforms.NoOperation',
                                    'Inputs': {
                                        'Data': '$Var_28af8fabe6dd446a9f20aa97c53c4d4e'
                                    },
                                    'Outputs': {
                                        'OutputData': '$Var_fb8137cb48ac49a7b1b56aa3ed5e0b23',
                                        'Model': '$Var_2cfc22486a4f475f8dc814feccb08f71'
                                    }
                                }, {
                                    'Name': 'Trainers.LogisticRegressionClassifier',
                                    'Inputs': {
                                        'ShowTrainingStats': false,
                                        'L2Weight': 1.0,
                                        'L1Weight': 1.0,
                                        'OptTol': 1E-07,
                                        'MemorySize': 20,
                                        'MaxIterations': 2147483647,
                                        'SgdInitializationTolerance': 0.0,
                                        'Quiet': false,
                                        'InitWtsDiameter': 0.0,
                                        'UseThreads': true,
                                        'NumThreads': 1,
                                        'DenseOptimizer': false,
                                        'EnforceNonNegativity': false,
                                        'ExampleWeightColumnName': null,
                                        'LabelColumnName': 'Label',
                                        'TrainingData': '$Var_fb8137cb48ac49a7b1b56aa3ed5e0b23',
                                        'FeatureColumnName': 'Features',
                                        'NormalizeFeatures': 'Auto',
                                        'Caching': 'Auto'
                                    },
                                    'Outputs': {
                                        'PredictorModel': '$Var_05e29f93f3bb4c31a93d71e051dfbb2a'
                                    }
                                }
                            ],
                            'Inputs': {
                                'Data': '$Var_28af8fabe6dd446a9f20aa97c53c4d4e'
                            },
                            'Outputs': {
                                'PredictorModel': '$Var_05e29f93f3bb4c31a93d71e051dfbb2a'
                            },
                            'StratificationColumn': 'Strat',
                            'NumFolds': 2,
                            'Kind': 'SignatureMulticlassClassificationTrainer',
                            'LabelColumn': 'Label',
                            'WeightColumn': null,
                            'GroupColumn': null,
                            'NameColumn': null
                        },
                        'Outputs': {
                            'PredictorModel': '$Var_2df88bffdbca48d5972decf058c26e3b',
                            'Warnings': '$warning',
                            'OverallMetrics': '$Var_94ec7af856fa4c2aa16f354cf51cee78',
                            'PerInstanceMetrics': '$Var_637187e4984f4eed93cd37ab20685867',
                            'ConfusionMatrix': '$Var_c5fe1a4fbded49898173662f6be2f6cc'
                        }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();
            var warnings = runner.GetOutput<IDataView>("warning");

            var schema = warnings.Schema;
            var warningCol = schema.GetColumnOrNull("WarningText");
            Assert.True(warningCol.HasValue);
            using (var cursor = warnings.GetRowCursor(warningCol.Value))
            {
                var getter = cursor.GetGetter<ReadOnlyMemory<char>>(warningCol.Value);

                var b = cursor.MoveNext();
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

        [Fact]
        public void TestCrossValidationMacroWithStratification()
        {
            var dataPath = GetDataPath(@"breast-cancer.txt");
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': true,
                                'AllowSparse': true,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': [{
                                        'Name': 'Label',
                                        'Type': null,
                                        'Source': [{
                                                'Min': 0,
                                                'Max': 0,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'Strat',
                                        'Type': null,
                                        'Source': [{
                                                'Min': 1,
                                                'Max': 1,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'Features',
                                        'Type': null,
                                        'Source': [{
                                                'Min': 2,
                                                'Max': 9,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }
                                ],
                                'TrimWhitespace': false,
                                'HasHeader': false
                            }
                        },
                        'Outputs': {
                            'Data': '$Var_95d56835dc384629bd288ea0a8879277'
                        }
                    }, {
                        'Name': 'Models.CrossValidator',
                        'Inputs': {
                            'Data': '$Var_95d56835dc384629bd288ea0a8879277',
                            'TransformModel': null,
                            'Nodes': [{
                                    'Name': 'Transforms.NoOperation',
                                    'Inputs': {
                                        'Data': '$Var_e02622de697b478e9b7d84a5220fee8c'
                                    },
                                    'Outputs': {
                                        'OutputData': '$Var_44f5c60e439b49fe9e5bf372be4613ee',
                                        'Model': '$Var_14976738a67940a58cfeffdf795a74c1'
                                    }
                                }, {
                                    'Name': 'Trainers.StochasticDualCoordinateAscentBinaryClassifier',
                                    'Inputs': {
                                        'LossFunction': {
                                            'Name': 'LogLoss',
                                            'Settings': {}
                                        },
                                        'PositiveInstanceWeight': 1.0,
                                        'Calibrator': {
                                            'Name': 'PlattCalibrator',
                                            'Settings': {}
                                        },
                                        'MaxCalibrationExamples': 1000000,
                                        'L2Const': null,
                                        'L1Threshold': null,
                                        'NumThreads': 1,
                                        'ConvergenceTolerance': 0.1,
                                        'MaxIterations': null,
                                        'Shuffle': true,
                                        'CheckFrequency': null,
                                        'BiasLearningRate': 0.0,
                                        'LabelColumnName': 'Label',
                                        'TrainingData': '$Var_44f5c60e439b49fe9e5bf372be4613ee',
                                        'FeatureColumnName': 'Features',
                                        'NormalizeFeatures': 'Auto',
                                        'Caching': 'Auto'
                                    },
                                    'Outputs': {
                                        'PredictorModel': '$Var_d0c2303905c146b6873693e58ed6e2aa'
                                    }
                                }, {
                                    'Name': 'Transforms.ManyHeterogeneousModelCombiner',
                                    'Inputs': {
                                        'TransformModels': [
                                            '$Var_14976738a67940a58cfeffdf795a74c1'
                                        ],
                                        'PredictorModel': '$Var_d0c2303905c146b6873693e58ed6e2aa'
                                    },
                                    'Outputs': {
                                        'PredictorModel': '$Var_250e906783ab442e85af77298c531199'
                                    }
                                }
                            ],
                            'Inputs': {
                                'Data': '$Var_e02622de697b478e9b7d84a5220fee8c'
                            },
                            'Outputs': {
                                'PredictorModel': '$Var_250e906783ab442e85af77298c531199'
                            },
                            'StratificationColumn': 'Strat',
                            'NumFolds': 2,
                            'Kind': 'SignatureBinaryClassifierTrainer',
                            'LabelColumn': 'Label',
                            'WeightColumn': null,
                            'GroupColumn': null,
                            'NameColumn': null
                        },
                        'Outputs': {
                            'PredictorModel': '$Var_c824c370674e4c358012ca07e04ee79e',
                            'Warnings': '$Var_4f7a5c14043247fdb53ea3a264afcb6f',
                            'OverallMetrics': '$overallmetrics',
                            'PerInstanceMetrics': '$Var_1fe20a06e4a14215bc09ba8ff7ae603b',
                            'ConfusionMatrix': '$Var_d159331c1bca445792a37ddd143b3a25'
                        }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();
            var data = runner.GetOutput<IDataView>("overallmetrics");

            var schema = data.Schema;
            var metricCol = schema.GetColumnOrNull("AUC");
            Assert.True(metricCol.HasValue);
            var foldCol = schema.GetColumnOrNull("Fold Index");
            Assert.True(foldCol.HasValue);
            bool b;
            using (var cursor = data.GetRowCursor(metricCol.Value, foldCol.Value))
            {
                var getter = cursor.GetGetter<double>(metricCol.Value);
                var foldGetter = cursor.GetGetter<ReadOnlyMemory<char>>(foldCol.Value);
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
                Assert.Equal(0.00481, stdev, 5);

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

        [Fact]
        public void TestCrossValidationMacroWithNonDefaultNames()
        {
            string dataPath = GetDataPath(@"adult.tiny.with-schema.txt");
            string inputGraph = @"
            {
            'Nodes': [{
                    'Name': 'Data.TextLoader',
                    'Inputs': {
                        'InputFile': '$inputFile',
                        'Arguments': {
                            'UseThreads': true,
                            'HeaderFile': null,
                            'MaxRows': null,
                            'AllowQuoting': true,
                            'AllowSparse': true,
                            'InputSize': null,
                            'Separator': [
                                '\t'
                            ],
                            'Column': [{
                                    'Name': 'Label',
                                    'Type': null,
                                    'Source': [{
                                            'Min': 0,
                                            'Max': 0,
                                            'AutoEnd': false,
                                            'VariableEnd': false,
                                            'AllOther': false,
                                            'ForceVector': false
                                        }
                                    ],
                                    'KeyCount': null
                                }, {
                                    'Name': 'Workclass',
                                    'Type': 'TX',
                                    'Source': [{
                                            'Min': 1,
                                            'Max': 1,
                                            'AutoEnd': false,
                                            'VariableEnd': false,
                                            'AllOther': false,
                                            'ForceVector': false
                                        }
                                    ],
                                    'KeyCount': null
                                }, {
                                    'Name': 'Features',
                                    'Type': null,
                                    'Source': [{
                                            'Min': 9,
                                            'Max': 14,
                                            'AutoEnd': false,
                                            'VariableEnd': false,
                                            'AllOther': false,
                                            'ForceVector': false
                                        }
                                    ],
                                    'KeyCount': null
                                }
                            ],
                            'TrimWhitespace': false,
                            'HasHeader': true
                        }
                    },
                    'Outputs': {
                        'Data': '$Var_bfb5ef5be6f547de88af2409c8c35443'
                    }
                }, {
                    'Name': 'Models.CrossValidator',
                    'Inputs': {
                        'Data': '$Var_bfb5ef5be6f547de88af2409c8c35443',
                        'TransformModel': null,
                        'Nodes': [{
                                'Name': 'Transforms.TextToKeyConverter',
                                'Inputs': {
                                    'Column': [{
                                            'MaxNumTerms': null,
                                            'Term': null,
                                            'Sort': null,
                                            'TextKeyValues': null,
                                            'Name': 'Label1',
                                            'Source': 'Label'
                                        }
                                    ],
                                    'MaxNumTerms': 1000000,
                                    'Term': null,
                                    'Sort': 'ByOccurrence',
                                    'TextKeyValues': false,
                                    'Data': '$Var_48d35aae527f439398805f51e5f0cfab'
                                },
                                'Outputs': {
                                    'OutputData': '$Var_44ac8ba819da483089dacc0f12bae3d6',
                                    'Model': '$Var_2039a1ba743549c1989de460c105b354'
                                }
                            }, {
                                'Name': 'Transforms.HashConverter',
                                'Inputs': {
                                    'Column': [{
                                            'Join': null,
                                            'CustomSlotMap': null,
                                            'NumberOfBits': null,
                                            'Seed': null,
                                            'Ordered': null,
                                            'Name': 'GroupId1',
                                            'Source': 'Workclass'
                                        }
                                    ],
                                    'Join': true,
                                    'NumberOfBits': 31,
                                    'Seed': 314489979,
                                    'Ordered': true,
                                    'Data': '$Var_44ac8ba819da483089dacc0f12bae3d6'
                                },
                                'Outputs': {
                                    'OutputData': '$Var_8f51ed90f5b642b2a80eeb628d67a5b3',
                                    'Model': '$Var_5d04d6405abb40ed9efb0486c2e1688b'
                                }
                            }, {
                                'Name': 'Trainers.FastTreeRanker',
                                'Inputs': {
                                    'CustomGains': [0,3,7,15,31],
                                    'UseDcg': false,
                                    'SortingAlgorithm': 'DescendingStablePessimistic',
                                    'NdcgTruncationLevel': 100,
                                    'ShiftedNdcg': false,
                                    'CostFunctionParam': 'w',
                                    'DistanceWeight2': false,
                                    'NormalizeQueryLambdas': false,
                                    'BestStepRankingRegressionTrees': false,
                                    'UseLineSearch': false,
                                    'MaximumNumberOfLineSearchSteps': 0,
                                    'MinimumStepSize': 0.0,
                                    'OptimizationAlgorithm': 'GradientDescent',
                                    'EarlyStoppingRule': null,
                                    'EarlyStoppingMetrics': 1,
                                    'EnablePruning': false,
                                    'UseTolerantPruning': false,
                                    'PruningThreshold': 0.004,
                                    'PruningWindowSize': 5,
                                    'LearningRate': 0.2,
                                    'Shrinkage': 1.0,
                                    'DropoutRate': 0.0,
                                    'GetDerivativesSampleRate': 1,
                                    'WriteLastEnsemble': false,
                                    'MaximumTreeOutput': 100.0,
                                    'RandomStart': false,
                                    'FilterZeroLambdas': false,
                                    'BaselineScoresFormula': null,
                                    'BaselineAlphaRisk': null,
                                    'PositionDiscountFreeform': null,
                                    'ParallelTrainer': {
                                        'Name': 'Single',
                                        'Settings': {}
                                    },
                                    'NumberOfThreads': 1,
                                    'Seed': 123,
                                    'FeatureSelectionSeed': 123,
                                    'EntropyCoefficient': 0.0,
                                    'HistogramPoolSize': -1,
                                    'DiskTranspose': null,
                                    'FeatureFlocks': true,
                                    'CategoricalSplit': false,
                                    'MaximumCategoricalGroupCountPerNode': 64,
                                    'MaximumCategoricalSplitPointCount': 64,
                                    'MinimumExampleFractionForCategoricalSplit': 0.001,
                                    'MinimumExamplesForCategoricalSplit': 100,
                                    'Bias': 0.0,
                                    'Bundling': 'None',
                                    'MaximumBinCountPerFeature': 255,
                                    'SparsifyThreshold': 0.7,
                                    'FeatureFirstUsePenalty': 0.0,
                                    'FeatureReusePenalty': 0.0,
                                    'GainConfidenceLevel': 0.0,
                                    'SoftmaxTemperature': 0.0,
                                    'ExecutionTime': false,
                                    'NumberOfLeaves': 20,
                                    'MinimumExampleCountPerLeaf': 10,
                                    'NumberOfTrees': 100,
                                    'FeatureFraction': 1.0,
                                    'BaggingSize': 0,
                                    'BaggingExampleFraction': 0.7,
                                    'FeatureFractionPerSplit': 1.0,
                                    'Smoothing': 0.0,
                                    'AllowEmptyTrees': true,
                                    'FeatureCompressionLevel': 1,
                                    'CompressEnsemble': false,
                                    'PrintTestGraph': false,
                                    'PrintTrainValidGraph': false,
                                    'TestFrequency': 2147483647,
                                    'RowGroupColumnName': 'GroupId1',
                                    'ExampleWeightColumnName': null,
                                    'LabelColumnName': 'Label1',
                                    'TrainingData': '$Var_8f51ed90f5b642b2a80eeb628d67a5b3',
                                    'FeatureColumnName': 'Features',
                                    'NormalizeFeatures': 'Auto',
                                    'Caching': 'Auto'
                                },
                                'Outputs': {
                                    'PredictorModel': '$Var_53eb1dedb8234950affa64daaa770427'
                                }
                            }, {
                                'Name': 'Transforms.ManyHeterogeneousModelCombiner',
                                'Inputs': {
                                    'TransformModels': [
                                        '$Var_2039a1ba743549c1989de460c105b354',
                                        '$Var_5d04d6405abb40ed9efb0486c2e1688b'
                                    ],
                                    'PredictorModel': '$Var_53eb1dedb8234950affa64daaa770427'
                                },
                                'Outputs': {
                                    'PredictorModel': '$Var_dd7bc37a393741aea36b46ed609c72a1'
                                }
                            }
                        ],
                        'Inputs': {
                            'Data': '$Var_48d35aae527f439398805f51e5f0cfab'
                        },
                        'Outputs': {
                            'PredictorModel': '$Var_dd7bc37a393741aea36b46ed609c72a1'
                        },
                        'StratificationColumn': null,
                        'NumFolds': 2,
                        'Kind': 'SignatureRankerTrainer',
                        'LabelColumn': 'Label1',
                        'WeightColumn': null,
                        'GroupColumn': {
                            'Value': 'GroupId1',
                            'IsExplicit': true
                        },
                        'NameColumn': {
                            'Value': 'Workclass',
                            'IsExplicit': true
                        }
                    },
                    'Outputs': {
                        'PredictorModel': '$Var_48c4f33d44c1437bb792a8640703e21e',
                        'Warnings': '$Var_8f3381cecb1b48dda606a58d153dc022',
                        'OverallMetrics': '$overallMetrics',
                        'PerInstanceMetrics': '$perInstanceMetric',
                        'ConfusionMatrix': '$Var_a06599dcbf52480a8dbb5f7414ee08fe'
                    }
                }
            ]
            }";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("overallMetrics");

            var schema = data.Schema;
            var metricCol = schema.GetColumnOrNull("NDCG");
            Assert.True(metricCol.HasValue);
            var foldCol = schema.GetColumnOrNull("Fold Index");
            Assert.True(foldCol.HasValue);
            bool b;
            using (var cursor = data.GetRowCursor(metricCol.Value, foldCol.Value))
            {
                var getter = cursor.GetGetter<VBuffer<double>>(metricCol.Value);
                var foldGetter = cursor.GetGetter<ReadOnlyMemory<char>>(foldCol.Value);
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
                var stdevValues = stdev.GetValues();
                Assert.Equal(2.462, stdevValues[0], 3);
                Assert.Equal(2.763, stdevValues[1], 3);
                Assert.Equal(3.273, stdevValues[2], 3);

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

                var avgValues = avg.GetValues();
                var sumValues = sum.GetValues();
                for (int i = 0; i < avgValues.Length; i++)
                    Assert.Equal(avgValues[i], sumValues[i] / 2);
                b = cursor.MoveNext();
                Assert.False(b);
            }

            data = runner.GetOutput<IDataView>("perInstanceMetric");
            var nameCol = data.Schema.GetColumnOrNull("Instance");
            Assert.True(nameCol.HasValue);
            using (var cursor = data.GetRowCursor(nameCol.Value))
            {
                var getter = cursor.GetGetter<ReadOnlyMemory<char>>(nameCol.Value);
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

        [Fact]
        public void TestOvaMacro()
        {
            var dataPath = GetDataPath(@"iris.txt");
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': true,
                                'AllowSparse': true,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': [{
                                        'Name': 'Label',
                                        'Type': null,
                                        'Source': [{
                                                'Min': 0,
                                                'Max': 0,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'Features',
                                        'Type': null,
                                        'Source': [{
                                                'Min': 1,
                                                'Max': 4,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }
                                ],
                                'TrimWhitespace': false,
                                'HasHeader': false
                            }
                        },
                        'Outputs': {
                            'Data': '$Var_672f860e44304ba8bd1c1a6e4b5ba9c5'
                        }
                    }, {
                        'Name': 'Models.OneVersusAll',
                        'Inputs': {
                            'Nodes': [{
                                    'Name': 'Trainers.StochasticDualCoordinateAscentBinaryClassifier',
                                    'Inputs': {
                                        'LossFunction': {
                                            'Name': 'LogLoss',
                                            'Settings': {}
                                        },
                                        'PositiveInstanceWeight': 1.0,
                                        'Calibrator': {
                                            'Name': 'PlattCalibrator',
                                            'Settings': {}
                                        },
                                        'MaxCalibrationExamples': 1000000,
                                        'L2Const': null,
                                        'L1Threshold': null,
                                        'NumThreads': 1,
                                        'ConvergenceTolerance': 0.1,
                                        'MaxIterations': null,
                                        'Shuffle': true,
                                        'CheckFrequency': null,
                                        'BiasLearningRate': 0.0,
                                        'LabelColumnName': 'Label',
                                        'TrainingData': '$Var_9aa1732198964d7f979a0bbec5db66c2',
                                        'FeatureColumnName': 'Features',
                                        'NormalizeFeatures': 'Auto',
                                        'Caching': 'Auto'
                                    },
                                    'Outputs': {
                                        'PredictorModel': '$Var_6219b70478204e599cef4ab3672656ff'
                                    }
                                }
                            ],
                            'OutputForSubGraph': {
                                'Model': '$Var_a229f40df6494a93a794ffd5480d5549'
                            },
                            'UseProbabilities': true,
                            'ExampleWeightColumnName': null,
                            'LabelColumnName': 'Label',
                            'TrainingData': '$Var_672f860e44304ba8bd1c1a6e4b5ba9c5',
                            'FeatureColumnName': 'Features',
                            'NormalizeFeatures': 'Auto',
                            'Caching': 'Auto'
                        },
                        'Outputs': {
                            'PredictorModel': '$Var_a8423859a7994667b7f1075f8b7b0194'
                        }
                    }, {
                        'Name': 'Transforms.DatasetScorer',
                        'Inputs': {
                            'Data': '$Var_672f860e44304ba8bd1c1a6e4b5ba9c5',
                            'PredictorModel': '$Var_a8423859a7994667b7f1075f8b7b0194',
                            'Suffix': null
                        },
                        'Outputs': {
                            'ScoredData': '$Var_5454fd8c353c40288dd8c2d104be788f',
                            'ScoringTransform': '$Var_df35ea0b6e814ed5a1d8f9d673a663b1'
                        }
                    }, {
                        'Name': 'Models.ClassificationEvaluator',
                        'Inputs': {
                            'OutputTopKAcc': null,
                            'NumTopClassesToOutput': 3,
                            'NumClassesConfusionMatrix': 10,
                            'OutputPerClassStatistics': false,
                            'LabelColumn': null,
                            'WeightColumn': null,
                            'ScoreColumn': null,
                            'StratColumn': null,
                            'Data': '$Var_5454fd8c353c40288dd8c2d104be788f',
                            'NameColumn': 'Name'
                        },
                        'Outputs': {
                            'ConfusionMatrix': '$Var_ed441dd1ebcc46f7bf7e096d18b33fd7',
                            'Warnings': '$Var_bec6f9da6bd647808c4a7a05b7e8b1be',
                            'OverallMetrics': '$overallMetrics',
                            'PerInstanceMetrics': '$Var_cfcc191521dd45c58ed6654ced067a28'
                        }
                    }
                ]
            }
            ";
            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("overallMetrics");
            var schema = data.Schema;
            var accCol = schema.GetColumnOrNull(MulticlassClassificationEvaluator.AccuracyMacro);
            Assert.True(accCol.HasValue);
            bool b;
            using (var cursor = data.GetRowCursor(accCol.Value))
            {
                var getter = cursor.GetGetter<double>(accCol.Value);
                b = cursor.MoveNext();
                Assert.True(b);
                double acc = 0;
                getter(ref acc);
                Assert.Equal(0.96, acc, 2);
                b = cursor.MoveNext();
                Assert.False(b);
            }
        }

        [Fact]
        public void TestOvaMacroWithUncalibratedLearner()
        {
            var dataPath = GetDataPath(@"iris.txt");
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': true,
                                'AllowSparse': true,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': [{
                                        'Name': 'Label',
                                        'Type': null,
                                        'Source': [{
                                                'Min': 0,
                                                'Max': 0,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'Features',
                                        'Type': null,
                                        'Source': [{
                                                'Min': 1,
                                                'Max': 4,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }
                                ],
                                'TrimWhitespace': false,
                                'HasHeader': false
                            }
                        },
                        'Outputs': {
                            'Data': '$Var_f38b99289df746319edd57a3ccfb85a2'
                        }
                    }, {
                        'Name': 'Models.OneVersusAll',
                        'Inputs': {
                            'Nodes': [{
                                    'Name': 'Trainers.AveragedPerceptronBinaryClassifier',
                                    'Inputs': {
                                        'LossFunction': {
                                            'Name': 'HingeLoss',
                                            'Settings': {
                                                'Margin': 1.0
                                            }
                                        },
                                        'Calibrator': {
                                            'Name': 'PlattCalibrator',
                                            'Settings': {}
                                        },
                                        'MaxCalibrationExamples': 1000000,
                                        'LearningRate': 1.0,
                                        'DecreaseLearningRate': false,
                                        'ResetWeightsAfterXExamples': null,
                                        'DoLazyUpdates': true,
                                        'L2RegularizerWeight': 0.0,
                                        'RecencyGain': 0.0,
                                        'RecencyGainMulti': false,
                                        'Averaged': true,
                                        'AveragedTolerance': 0.01,
                                        'NumberOfIterations': 1,
                                        'InitialWeights': null,
                                        'InitialWeightsDiameter': 0.0,
                                        'Shuffle': false,
                                        'LabelColumnName': 'Label',
                                        'TrainingData': '$Var_9ccc8bce4f6540eb8a244ab40585602a',
                                        'FeatureColumnName': 'Features',
                                        'NormalizeFeatures': 'Auto',
                                        'Caching': 'Auto'
                                    },
                                    'Outputs': {
                                        'PredictorModel': '$Var_4f1c140c153e4b5fb03fbe3ffb97a68b'
                                    }
                                }
                            ],
                            'OutputForSubGraph': {
                                'Model': '$Var_b47f7facc1c540e39d8b82ab64df6592'
                            },
                            'UseProbabilities': true,
                            'ExampleWeightColumnName': null,
                            'LabelColumnName': 'Label',
                            'TrainingData': '$Var_f38b99289df746319edd57a3ccfb85a2',
                            'FeatureColumnName': 'Features',
                            'NormalizeFeatures': 'Auto',
                            'Caching': 'Auto'
                        },
                        'Outputs': {
                            'PredictorModel': '$Var_d67eb393a0e849c2962961c174eab3da'
                        }
                    }, {
                        'Name': 'Transforms.DatasetScorer',
                        'Inputs': {
                            'Data': '$Var_f38b99289df746319edd57a3ccfb85a2',
                            'PredictorModel': '$Var_d67eb393a0e849c2962961c174eab3da',
                            'Suffix': null
                        },
                        'Outputs': {
                            'ScoredData': '$Var_a20e37dc58d84bf5a1cb13ed13eae5ba',
                            'ScoringTransform': '$Var_49f9a4a57ff043cda5947704678241a0'
                        }
                    }, {
                        'Name': 'Models.ClassificationEvaluator',
                        'Inputs': {
                            'OutputTopKAcc': null,
                            'NumTopClassesToOutput': 3,
                            'NumClassesConfusionMatrix': 10,
                            'OutputPerClassStatistics': false,
                            'LabelColumn': null,
                            'WeightColumn': null,
                            'ScoreColumn': null,
                            'StratColumn': null,
                            'Data': '$Var_a20e37dc58d84bf5a1cb13ed13eae5ba',
                            'NameColumn': 'Name'
                        },
                        'Outputs': {
                            'ConfusionMatrix': '$Var_7db29303b67942e2a6267c20b9c4be77',
                            'Warnings': '$Var_7751126378244c2385940cdf5a0e76e6',
                            'OverallMetrics': '$overallMetrics',
                            'PerInstanceMetrics': '$Var_67109dcdce504a0894a5c2f5616d21f9'
                        }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("overallMetrics");
            var schema = data.Schema;
            var accCol = schema.GetColumnOrNull(MulticlassClassificationEvaluator.AccuracyMacro);
            Assert.True(accCol.HasValue);
            bool b;
            using (var cursor = data.GetRowCursor(accCol.Value))
            {
                var getter = cursor.GetGetter<double>(accCol.Value);
                b = cursor.MoveNext();
                Assert.True(b);
                double acc = 0;
                getter(ref acc);
                Assert.Equal(0.71, acc, 2);
                b = cursor.MoveNext();
                Assert.False(b);
            }
        }

        [TensorFlowFact]
        public void TestTensorFlowEntryPoint()
        {
            var dataPath = GetDataPath("Train-Tiny-28x28.txt");
            Env.ComponentCatalog.RegisterAssembly(typeof(TensorFlowTransformer).Assembly);
            string inputGraph = @"
            {
                'Nodes':
                [{
                        'Name': 'Data.TextLoader',
                        'Inputs': {
                            'InputFile': '$inputFile',
                            'Arguments': {
                                'UseThreads': true,
                                'HeaderFile': null,
                                'MaxRows': null,
                                'AllowQuoting': true,
                                'AllowSparse': true,
                                'InputSize': null,
                                'Separator': [
                                    '\t'
                                ],
                                'Column': [{
                                        'Name': 'Label',
                                        'Type': null,
                                        'Source': [{
                                                'Min': 0,
                                                'Max': 0,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }, {
                                        'Name': 'Placeholder',
                                        'Type': null,
                                        'Source': [{
                                                'Min': 1,
                                                'Max': 784,
                                                'AutoEnd': false,
                                                'VariableEnd': false,
                                                'AllOther': false,
                                                'ForceVector': false
                                            }
                                        ],
                                        'KeyCount': null
                                    }
                                ],
                                'TrimWhitespace': false,
                                'HasHeader': false
                            }
                        },
                        'Outputs': {
                            'Data': '$Var_2802f3e485814063828c2303ec60327c'
                        }
                    }, {
                        'Name': 'Transforms.TensorFlowScorer',
                        'Inputs': {
                            'ModelLocation': 'mnist_model/frozen_saved_model.pb',
                            'InputColumns': [
                                'Placeholder'
                            ],
                            'OutputColumns': [
                                'Softmax'
                            ],
                            'LabelColumn': null,
                            'TensorFlowLabel': null,
                            'OptimizationOperation': null,
                            'LossOperation': null,
                            'MetricOperation': null,
                            'BatchSize': 64,
                            'Epoch': 5,
                            'LearningRateOperation': null,
                            'LearningRate': 0.01,
                            'SaveLocationOperation': 'save/Const',
                            'SaveOperation': 'save/control_dependency',
                            'ReTrain': false,
                            'Data': '$Var_2802f3e485814063828c2303ec60327c'
                        },
                        'Outputs': {
                            'OutputData': '$outputData',
                            'Model': '$Var_c3a191a107c54725acc49e432bfdf104'
                        }
                    }
                ]
            }
            ";
            JObject graph = JObject.Parse(inputGraph);
            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("inputFile", inputFile);
            runner.RunAll();

            var data = runner.GetOutput<IDataView>("outputData");

            var schema = data.Schema;
            Assert.Equal(3, schema.Count);
            Assert.Equal("Softmax", schema[2].Name);
            Assert.Equal(10, (schema[2].Type as VectorType)?.Size);
        }

        [Fact]
        public void LoadEntryPointModel()
        {
            var ml = new MLContext();
            for (int i = 0; i < 5; i++)
            {
                var modelPath = GetDataPath($"backcompat/ep_model{i}.zip");
                ITransformer loadedModel;
                using (var stream = File.OpenRead(modelPath))
                {
                    loadedModel = ml.Model.Load(stream, out var inputSchema);
                }
            }
        }
    }
}
