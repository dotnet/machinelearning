// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Core.Tests.UnitTests;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.PCA;
using Microsoft.ML.Runtime.TextAnalytics;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    public partial class TestEntryPoints : CoreBaseTestClass
    {
        public TestEntryPoints(ITestOutputHelper output) : base(output)
        {
        }

        private IDataView GetBreastCancerDataView()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            return ImportTextData.TextLoader(Env, new ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.R4, 0),
                        new TextLoader.Column("Features", DataKind.R4,
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
            return ImportTextData.TextLoader(Env, new ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", type: null, 0),
                        new TextLoader.Column("F1", DataKind.Text, 1),
                        new TextLoader.Column("F2", DataKind.I4, 2),
                        new TextLoader.Column("Rest", type: null, new [] { new TextLoader.Range(3, 9) })
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
            using (var cursor = dataView.GetRowCursor(col => false))
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

            expected = Env.CreateTransform("ChooseColumns{col=Features}", expected);
            result = Env.CreateTransform("ChooseColumns{col=Features}", result);
            CheckSameValues(result, expected);
            Done();
        }

        [Fact]
        public void EntryPointScoring()
        {
            var dataView = GetBreastCancerDataviewWithTextColumns();
            dataView = Env.CreateTransform("Term{col=F1}", dataView);
            var trainData = FeatureCombiner.PrepareFeatures(Env, new FeatureCombiner.FeatureCombinerInput() { Data = dataView, Features = new[] { "F1", "F2", "Rest" } });
            var lrModel = LogisticRegression.TrainBinary(Env, new LogisticRegression.Arguments { TrainingData = trainData.OutputData }).PredictorModel;
            var model = ModelOperations.CombineTwoModels(Env, new ModelOperations.SimplePredictorModelInput() { TransformModel = trainData.Model, PredictorModel = lrModel }).PredictorModel;

            var scored1 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = dataView, PredictorModel = model }).ScoredData;
            scored1 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored1, ExtraColumns = new[] { "Label" } }).OutputData;

            var scored2 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = dataView, PredictorModel = lrModel.Apply(Env, trainData.Model) }).ScoredData;
            scored2 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored2, ExtraColumns = new[] { "Label" } }).OutputData;

            Assert.Equal(4, scored1.Schema.ColumnCount);
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

        [Fact]
        public void EntryPointCaching()
        {
            var dataView = GetBreastCancerDataviewWithTextColumns();

            dataView = Env.CreateTransform("Term{col=F1}", dataView);

            var cached1 = Cache.CacheData(Env, new Cache.CacheInput() { Data = dataView, Caching = Cache.CachingType.Memory });
            CheckSameValues(dataView, cached1.OutputData);

            var cached2 = Cache.CacheData(Env, new Cache.CacheInput() { Data = dataView, Caching = Cache.CachingType.Disk });
            CheckSameValues(dataView, cached2.OutputData);
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

        [Fact(Skip = "Execute this test if you want to regenerate ep-list and _manifest.json")]
        public void RegenerateEntryPointCatalog()
        {
            var buildPrefix = GetBuildPrefix();
            var epListFile = buildPrefix + "_ep-list.tsv";
            var manifestFile = buildPrefix + "_manifest.json";

            var entryPointsSubDir = Path.Combine("..", "Common", "EntryPoints");
            var catalog = ModuleCatalog.CreateInstance(Env);
            var epListPath = GetBaselinePath(entryPointsSubDir, epListFile);
            DeleteOutputPath(epListPath);

            var regex = new Regex(@"\r\n?|\n", RegexOptions.Compiled);
            File.WriteAllLines(epListPath, catalog.AllEntryPoints()
                .Select(x => string.Join("\t",
                x.Name,
                regex.Replace(x.Description, ""),
                x.Method.DeclaringType,
                x.Method.Name,
                x.InputType,
                x.OutputType)
                .Replace(Environment.NewLine, ""))
                .OrderBy(x => x));


            var jObj = JsonManifestUtils.BuildAllManifests(Env, catalog);

            //clean up the description from the new line characters
            if (jObj[FieldNames.TopEntryPoints] != null && jObj[FieldNames.TopEntryPoints] is JArray)
            {
                foreach (JToken entry in jObj[FieldNames.TopEntryPoints].Children())
                    if (entry[FieldNames.Desc] != null)
                        entry[FieldNames.Desc] = regex.Replace(entry[FieldNames.Desc].ToString(), "");
            }
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
            var buildPrefix = GetBuildPrefix();
            var epListFile = buildPrefix + "_ep-list.tsv";
            var manifestFile = buildPrefix + "_manifest.json";

            var entryPointsSubDir = Path.Combine("..", "Common", "EntryPoints");
            var catalog = ModuleCatalog.CreateInstance(Env);
            var path = DeleteOutputPath(entryPointsSubDir, epListFile);

            var regex = new Regex(@"\r\n?|\n", RegexOptions.Compiled);
            File.WriteAllLines(path, catalog.AllEntryPoints()
                .Select(x => string.Join("\t",
                x.Name,
                regex.Replace(x.Description, ""),
                x.Method.DeclaringType,
                x.Method.Name,
                x.InputType,
                x.OutputType)
                .Replace(Environment.NewLine, ""))
                .OrderBy(x => x));

            CheckEquality(entryPointsSubDir, epListFile);

            var jObj = JsonManifestUtils.BuildAllManifests(Env, catalog);

            //clean up the description from the new line characters
            if (jObj[FieldNames.TopEntryPoints] != null && jObj[FieldNames.TopEntryPoints] is JArray)
            {
                foreach (JToken entry in jObj[FieldNames.TopEntryPoints].Children())
                    if (entry[FieldNames.Desc] != null)
                        entry[FieldNames.Desc] = regex.Replace(entry[FieldNames.Desc].ToString(), "");
            }

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
        public void EntryPointInputBuilderOptionals()
        {
            var catelog = ModuleCatalog.CreateInstance(Env);

            InputBuilder ib1 = new InputBuilder(Env, typeof(LogisticRegression.Arguments), catelog);
            // Ensure that InputBuilder unwraps the Optional<string> correctly.
            var weightType = ib1.GetFieldTypeOrNull("WeightColumn");
            Assert.True(weightType.Equals(typeof(string)));

            var instance = ib1.GetInstance() as LogisticRegression.Arguments;
            Assert.True(!instance.WeightColumn.IsExplicit);
            Assert.True(instance.WeightColumn.Value == DefaultColumnNames.Weight);

            ib1.TrySetValue("WeightColumn", "OtherWeight");
            Assert.True(instance.WeightColumn.IsExplicit);
            Assert.Equal("OtherWeight", instance.WeightColumn.Value);

            var tok = (JToken)JValue.CreateString("AnotherWeight");
            ib1.TrySetValueJson("WeightColumn", tok);
            Assert.True(instance.WeightColumn.IsExplicit);
            Assert.Equal("AnotherWeight", instance.WeightColumn.Value);
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
        public void EntryPointInputArgsChecks()
        {
            var input = new DropColumnsTransform.KeepArguments();
            try
            {
                EntryPointUtils.CheckInputArgs(Env, input);
                Assert.False(true);
            }
            catch
            {
            }

            input.Data = new EmptyDataView(Env, new SimpleSchema(Env, new KeyValuePair<string, ColumnType>("ColA", NumberType.R4)));
            input.Column = new string[0];
            EntryPointUtils.CheckInputArgs(Env, input);
        }

        [Fact]
        public void EntryPointCreateEnsemble()
        {
            var dataView = GetBreastCancerDataView();
            const int nModels = 5;
            var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = nModels + 1 });
            var predictorModels = new IPredictorModel[nModels];
            var individualScores = new IDataView[nModels];
            for (int i = 0; i < nModels; i++)
            {
                var data = splitOutput.TrainData[i];
                var lrInput = new LogisticRegression.Arguments
                {
                    TrainingData = data,
                    L1Weight = (Single)0.1 * i,
                    L2Weight = (Single)0.01 * (1 + i),
                    NormalizeFeatures = NormalizeOption.No
                };
                predictorModels[i] = LogisticRegression.TrainBinary(Env, lrInput).PredictorModel;
                individualScores[i] =
                    ScoreModel.Score(Env,
                        new ScoreModel.Input { Data = splitOutput.TestData[nModels], PredictorModel = predictorModels[i] })
                        .ScoredData;

                individualScores[i] = CopyColumnsTransform.Create(Env,
                    new CopyColumnsTransform.Arguments()
                    {
                        Column = new[]
                        {
                            new CopyColumnsTransform.Column()
                            {
                                Name = MetadataUtils.Const.ScoreValueKind.Score + i,
                                Source = MetadataUtils.Const.ScoreValueKind.Score
                            },
                        }
                    }, individualScores[i]);
                individualScores[i] = new DropColumnsTransform(Env,
                    new DropColumnsTransform.Arguments() { Column = new[] { MetadataUtils.Const.ScoreValueKind.Score } },
                    individualScores[i]);
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
            using (var curs1 = avgScored.GetRowCursor(col => true))
            using (var curs2 = medScored.GetRowCursor(col => true))
            using (var curs3 = regScored.GetRowCursor(col => true))
            using (var curs4 = zippedScores.GetRowCursor(col => true))
            {
                var found = curs1.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out int scoreCol);
                Assert.True(found);
                var avgScoreGetter = curs1.GetGetter<Single>(scoreCol);

                found = curs2.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreCol);
                Assert.True(found);
                var medScoreGetter = curs2.GetGetter<Single>(scoreCol);

                found = curs3.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreCol);
                Assert.True(found);
                var regScoreGetter = curs3.GetGetter<Single>(scoreCol);

                var individualScoreGetters = new ValueGetter<Single>[nModels];
                for (int i = 0; i < nModels; i++)
                {
                    curs4.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score + i, out scoreCol);
                    individualScoreGetters[i] = curs4.GetGetter<Single>(scoreCol);
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file1", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<IPredictorModel>("model1");
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
        //    var catalog = ModuleCatalog.CreateInstance(Env);
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

            var lrModel = LogisticRegression.TrainBinary(Env, new LogisticRegression.Arguments { TrainingData = splitOutput.TestData[0] }).PredictorModel;
            var calibratedLrModel = Calibrate.FixedPlatt(Env,
                new Calibrate.FixedPlattInput { Data = splitOutput.TestData[1], UncalibratedPredictorModel = lrModel }).PredictorModel;

            var scored1 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = splitOutput.TestData[2], PredictorModel = lrModel }).ScoredData;
            scored1 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored1, ExtraColumns = new[] { "Label" } }).OutputData;

            var scored2 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = splitOutput.TestData[2], PredictorModel = calibratedLrModel }).ScoredData;
            scored2 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored2, ExtraColumns = new[] { "Label" } }).OutputData;

            Assert.Equal(4, scored1.Schema.ColumnCount);
            CheckSameValues(scored1, scored2);

            var input = new Calibrate.NoArgumentsInput() { Data = splitOutput.TestData[1], UncalibratedPredictorModel = lrModel };
            calibratedLrModel = Calibrate.Platt(Env, input).PredictorModel;
            calibratedLrModel = Calibrate.Naive(Env, input).PredictorModel;
            calibratedLrModel = Calibrate.Pav(Env, input).PredictorModel;

            // This tests that the SchemaBindableCalibratedPredictor doesn't get confused if its sub-predictor is already calibrated.
            var fastForest = new FastForestClassification(Env, new FastForestClassification.Arguments());
            var rmd = new RoleMappedData(splitOutput.TrainData[0], "Label", "Features");
            var ffModel = new PredictorModel(Env, rmd, splitOutput.TrainData[0], fastForest.Train(rmd));
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
            var predictorModels = new IPredictorModel[nModels];
            var individualScores = new IDataView[nModels];
            for (int i = 0; i < nModels; i++)
            {
                var data = splitOutput.TrainData[i];
                data = new RffTransform(Env, new RffTransform.Arguments()
                {
                    Column = new[]
                    {
                        new RffTransform.Column() {Name = "Features1", Source = "Features"},
                        new RffTransform.Column() {Name = "Features2", Source = "Features"},
                    },
                    NewDim = 10,
                    UseSin = false
                }, data);
                data = new ConcatTransform(Env, new ConcatTransform.Arguments()
                {
                    Column = new[] { new ConcatTransform.Column() { Name = "Features", Source = new[] { "Features1", "Features2" } } }
                }, data);

                data = TermTransform.Create(Env, new TermTransform.Arguments()
                {
                    Column = new[]
                    {
                        new TermTransform.Column()
                        {
                            Name = "Label",
                            Source = "Label",
                            Sort = TermTransform.SortOrder.Value
                        }
                    }
                }, data);

                var lrInput = new LogisticRegression.Arguments
                {
                    TrainingData = data,
                    L1Weight = (Single)0.1 * i,
                    L2Weight = (Single)0.01 * (1 + i),
                    NormalizeFeatures = NormalizeOption.Yes
                };
                predictorModels[i] = LogisticRegression.TrainBinary(Env, lrInput).PredictorModel;
                var transformModel = new TransformModel(Env, data, splitOutput.TrainData[i]);

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
                    ModelCombiner = EntryPoints.EnsembleCreator.ClassifierCombiner.Average,
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

            var regressionEnsembleModel = EntryPoints.EnsembleCreator.CreateRegressionPipelineEnsemble(Env,
                new EntryPoints.EnsembleCreator.PipelineRegressionInput()
                {
                    ModelCombiner = EntryPoints.EnsembleCreator.ScoreCombiner.Average,
                    Models = predictorModels
                }).PredictorModel;
            var regressionScored = ScoreModel.Score(Env,
                new ScoreModel.Input
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = regressionEnsembleModel
                }).ScoredData;

            var anomalyEnsembleModel = EntryPoints.EnsembleCreator.CreateAnomalyPipelineEnsemble(Env,
                new EntryPoints.EnsembleCreator.PipelineAnomalyInput()
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
            var hasScoreCol = binaryScored.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out int scoreIndex);
            Assert.True(hasScoreCol, "Data scored with binary ensemble does not have a score column");
            var type = binaryScored.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.ScoreColumnKind, scoreIndex);
            Assert.True(type != null && type.IsText, "Binary ensemble scored data does not have correct type of metadata.");
            var kind = default(DvText);
            binaryScored.Schema.GetMetadata(MetadataUtils.Kinds.ScoreColumnKind, scoreIndex, ref kind);
            Assert.True(kind.EqualsStr(MetadataUtils.Const.ScoreColumnKind.BinaryClassification),
                $"Binary ensemble scored data column type should be '{MetadataUtils.Const.ScoreColumnKind.BinaryClassification}', but is instead '{kind}'");

            hasScoreCol = regressionScored.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreIndex);
            Assert.True(hasScoreCol, "Data scored with regression ensemble does not have a score column");
            type = regressionScored.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.ScoreColumnKind, scoreIndex);
            Assert.True(type != null && type.IsText, "Regression ensemble scored data does not have correct type of metadata.");
            regressionScored.Schema.GetMetadata(MetadataUtils.Kinds.ScoreColumnKind, scoreIndex, ref kind);
            Assert.True(kind.EqualsStr(MetadataUtils.Const.ScoreColumnKind.Regression),
                $"Regression ensemble scored data column type should be '{MetadataUtils.Const.ScoreColumnKind.Regression}', but is instead '{kind}'");

            hasScoreCol = anomalyScored.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreIndex);
            Assert.True(hasScoreCol, "Data scored with anomaly detection ensemble does not have a score column");
            type = anomalyScored.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.ScoreColumnKind, scoreIndex);
            Assert.True(type != null && type.IsText, "Anomaly detection ensemble scored data does not have correct type of metadata.");
            anomalyScored.Schema.GetMetadata(MetadataUtils.Kinds.ScoreColumnKind, scoreIndex, ref kind);
            Assert.True(kind.EqualsStr(MetadataUtils.Const.ScoreColumnKind.AnomalyDetection),
                $"Anomaly detection ensemble scored data column type should be '{MetadataUtils.Const.ScoreColumnKind.AnomalyDetection}', but is instead '{kind}'");

            var modelPath = DeleteOutputPath("SavePipe", "PipelineEnsembleModel.zip");
            using (var file = Env.CreateOutputFile(modelPath))
            using (var strm = file.CreateWriteStream())
                regressionEnsembleModel.Save(Env, strm);

            IPredictorModel loadedFromSaved;
            using (var file = Env.OpenInputFile(modelPath))
            using (var strm = file.OpenReadStream())
                loadedFromSaved = new PredictorModel(Env, strm);

            var scoredFromSaved = ScoreModel.Score(Env,
                new ScoreModel.Input()
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = loadedFromSaved
                }).ScoredData;

            using (var cursReg = regressionScored.GetRowCursor(col => true))
            using (var cursBin = binaryScored.GetRowCursor(col => true))
            using (var cursBinCali = binaryScoredCalibrated.GetRowCursor(col => true))
            using (var cursAnom = anomalyScored.GetRowCursor(col => true))
            using (var curs0 = individualScores[0].GetRowCursor(col => true))
            using (var curs1 = individualScores[1].GetRowCursor(col => true))
            using (var curs2 = individualScores[2].GetRowCursor(col => true))
            using (var curs3 = individualScores[3].GetRowCursor(col => true))
            using (var curs4 = individualScores[4].GetRowCursor(col => true))
            using (var cursSaved = scoredFromSaved.GetRowCursor(col => true))
            {
                var good = curs0.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out int col);
                Assert.True(good);
                var getter0 = curs0.GetGetter<Single>(col);
                good = curs1.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter1 = curs1.GetGetter<Single>(col);
                good = curs2.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter2 = curs2.GetGetter<Single>(col);
                good = curs3.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter3 = curs3.GetGetter<Single>(col);
                good = curs4.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter4 = curs4.GetGetter<Single>(col);
                good = cursReg.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterReg = cursReg.GetGetter<Single>(col);
                good = cursBin.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterBin = cursBin.GetGetter<Single>(col);
                good = cursBinCali.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterBinCali = cursBinCali.GetGetter<Single>(col);
                good = cursSaved.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterSaved = cursSaved.GetGetter<Single>(col);
                good = cursAnom.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterAnom = cursAnom.GetGetter<Single>(col);

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
                    Assert.True(Single.IsNaN(scoreSaved) && Single.IsNaN(score) || scoreSaved == score);
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
            var dataView = ImportTextData.TextLoader(Env, new ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.TX, 0),
                        new TextLoader.Column("Text", DataKind.TX, 3)
                    }
                },

                InputFile = inputFile
            }).Data;

            ValueMapper<DvText, bool> labelToBinary =
                (ref DvText src, ref bool dst) =>
                {
                    if (src.EqualsStr("Sport"))
                        dst = true;
                    else
                        dst = false;
                };
            dataView = LambdaColumnMapper.Create(Env, "TextToBinaryLabel", dataView, "Label", "Label",
                TextType.Instance, BoolType.Instance, labelToBinary);

            const int nModels = 5;
            var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = nModels + 1 });
            var predictorModels = new IPredictorModel[nModels];
            var individualScores = new IDataView[nModels];
            for (int i = 0; i < nModels; i++)
            {
                var data = splitOutput.TrainData[i];
                if (i % 2 == 0)
                {
                    data = TextTransform.Create(Env,
                        new TextTransform.Arguments()
                        {
                            Column = new TextTransform.Column() { Name = "Features", Source = new[] { "Text" } },
                            StopWordsRemover = new PredefinedStopWordsRemoverFactory()
                        }, data);
                }
                else
                {
                    data = WordHashBagTransform.Create(Env,
                        new WordHashBagTransform.Arguments()
                        {
                            Column =
                                new[] { new WordHashBagTransform.Column() { Name = "Features", Source = new[] { "Text" } }, }
                        },
                        data);
                }
                var lrInput = new LogisticRegression.Arguments
                {
                    TrainingData = data,
                    L1Weight = (Single)0.1 * i,
                    L2Weight = (Single)0.01 * (1 + i),
                    NormalizeFeatures = NormalizeOption.Yes
                };
                predictorModels[i] = LogisticRegression.TrainBinary(Env, lrInput).PredictorModel;
                var transformModel = new TransformModel(Env, data, splitOutput.TrainData[i]);

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

            IPredictorModel loadedFromSaved;
            using (var file = Env.OpenInputFile(modelPath))
            using (var strm = file.OpenReadStream())
                loadedFromSaved = new PredictorModel(Env, strm);

            var scoredFromSaved = ScoreModel.Score(Env,
                new ScoreModel.Input()
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = loadedFromSaved
                }).ScoredData;

            using (var cursReg = regressionScored.GetRowCursor(col => true))
            using (var cursBin = binaryScored.GetRowCursor(col => true))
            using (var cursBinCali = binaryScoredCalibrated.GetRowCursor(col => true))
            using (var curs0 = individualScores[0].GetRowCursor(col => true))
            using (var curs1 = individualScores[1].GetRowCursor(col => true))
            using (var curs2 = individualScores[2].GetRowCursor(col => true))
            using (var curs3 = individualScores[3].GetRowCursor(col => true))
            using (var curs4 = individualScores[4].GetRowCursor(col => true))
            using (var cursSaved = scoredFromSaved.GetRowCursor(col => true))
            {
                var good = curs0.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out int col);
                Assert.True(good);
                var getter0 = curs0.GetGetter<Single>(col);
                good = curs1.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter1 = curs1.GetGetter<Single>(col);
                good = curs2.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter2 = curs2.GetGetter<Single>(col);
                good = curs3.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter3 = curs3.GetGetter<Single>(col);
                good = curs4.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter4 = curs4.GetGetter<Single>(col);
                good = cursReg.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterReg = cursReg.GetGetter<Single>(col);
                good = cursBin.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterBin = cursBin.GetGetter<Single>(col);
                good = cursBinCali.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterBinCali = cursBinCali.GetGetter<Single>(col);
                good = cursSaved.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterSaved = cursSaved.GetGetter<Single>(col);

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
            var dataView = ImportTextData.TextLoader(Env, new ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.R4, 0),
                        new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(1, 4) })
                    }
                },

                InputFile = inputFile
            }).Data;

            const int nModels = 5;
            var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = nModels + 1 });
            var predictorModels = new IPredictorModel[nModels];
            var individualScores = new IDataView[nModels];
            for (int i = 0; i < nModels; i++)
            {
                var data = splitOutput.TrainData[i];
                data = new RffTransform(Env, new RffTransform.Arguments()
                {
                    Column = new[]
                    {
                        new RffTransform.Column() {Name = "Features1", Source = "Features"},
                        new RffTransform.Column() {Name = "Features2", Source = "Features"},
                    },
                    NewDim = 10,
                    UseSin = false
                }, data);
                data = new ConcatTransform(Env, new ConcatTransform.Arguments()
                {
                    Column = new[] { new ConcatTransform.Column() { Name = "Features", Source = new[] { "Features1", "Features2" } } }
                }, data);

                var mlr = new MulticlassLogisticRegression(Env, new MulticlassLogisticRegression.Arguments());
                var rmd = new RoleMappedData(data, "Label", "Features");

                predictorModels[i] = new PredictorModel(Env, rmd, data, mlr.Train(rmd));
                var transformModel = new TransformModel(Env, data, splitOutput.TrainData[i]);

                predictorModels[i] = ModelOperations.CombineTwoModels(Env,
                    new ModelOperations.SimplePredictorModelInput()
                    { PredictorModel = predictorModels[i], TransformModel = transformModel }).PredictorModel;

                individualScores[i] =
                    ScoreModel.Score(Env,
                        new ScoreModel.Input { Data = splitOutput.TestData[nModels], PredictorModel = predictorModels[i] })
                        .ScoredData;
            }

            var mcEnsembleModel = EnsembleCreator.CreateMultiClassPipelineEnsemble(Env,
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

            IPredictorModel loadedFromSaved;
            using (var file = Env.OpenInputFile(modelPath))
            using (var strm = file.OpenReadStream())
                loadedFromSaved = new PredictorModel(Env, strm);

            var scoredFromSaved = ScoreModel.Score(Env,
                new ScoreModel.Input()
                {
                    Data = splitOutput.TestData[nModels],
                    PredictorModel = loadedFromSaved
                }).ScoredData;

            using (var curs = mcScored.GetRowCursor(col => true))
            using (var cursSaved = scoredFromSaved.GetRowCursor(col => true))
            using (var curs0 = individualScores[0].GetRowCursor(col => true))
            using (var curs1 = individualScores[1].GetRowCursor(col => true))
            using (var curs2 = individualScores[2].GetRowCursor(col => true))
            using (var curs3 = individualScores[3].GetRowCursor(col => true))
            using (var curs4 = individualScores[4].GetRowCursor(col => true))
            {
                var good = curs0.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out int col);
                Assert.True(good);
                var getter0 = curs0.GetGetter<VBuffer<Single>>(col);
                good = curs1.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter1 = curs1.GetGetter<VBuffer<Single>>(col);
                good = curs2.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter2 = curs2.GetGetter<VBuffer<Single>>(col);
                good = curs3.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter3 = curs3.GetGetter<VBuffer<Single>>(col);
                good = curs4.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter4 = curs4.GetGetter<VBuffer<Single>>(col);
                good = curs.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getter = curs.GetGetter<VBuffer<Single>>(col);
                good = cursSaved.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out col);
                Assert.True(good);
                var getterSaved = cursSaved.GetGetter<VBuffer<Single>>(col);

                var c = new MultiAverage(Env, new MultiAverage.Arguments()).GetCombiner();
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
                    Assert.True(CompareVBuffers(ref scoreSaved, ref score, ref dense1, ref dense2));
                    c(ref avg, score0, null);
                    Assert.True(CompareVBuffers(ref avg, ref score, ref dense1, ref dense2));
                }
                Assert.False(curs0.MoveNext());
                Assert.False(curs1.MoveNext());
                Assert.False(curs2.MoveNext());
                Assert.False(curs3.MoveNext());
                Assert.False(curs4.MoveNext());
                Assert.False(cursSaved.MoveNext());
            }
        }

        private static bool CompareVBuffers(ref VBuffer<Single> v1, ref VBuffer<Single> v2, ref VBuffer<Single> dense1, ref VBuffer<Single> dense2)
        {
            if (v1.Length != v2.Length)
                return false;
            v1.CopyToDense(ref dense1);
            v2.CopyToDense(ref dense2);
            for (int i = 0; i < dense1.Length; i++)
            {
                if (!Single.IsNaN(dense1.Values[i]) && !Single.IsNaN(dense2.Values[i]) && dense1.Values[i] != dense2.Values[i])
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
                using (var cursor = loader.GetRowCursor(col => true))
                {
                    DvText cat = default(DvText);
                    DvText catValue = default(DvText);
                    uint catKey = 0;

                    bool success = loader.Schema.TryGetColumnIndex("Cat", out int catCol);
                    Assert.True(success);
                    var catGetter = cursor.GetGetter<DvText>(catCol);
                    success = loader.Schema.TryGetColumnIndex("CatValue", out int catValueCol);
                    Assert.True(success);
                    var catValueGetter = cursor.GetGetter<DvText>(catValueCol);
                    success = loader.Schema.TryGetColumnIndex("Key", out int keyCol);
                    Assert.True(success);
                    var keyGetter = cursor.GetGetter<uint>(keyCol);

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
        public void EntryPointEvaluateMultiClass()
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
            var dataPath = GetDataPath(TestDatasets.winequalitymacro.trainFilename);
            var warningsPath = DeleteOutputPath("warnings.idv");
            var overallMetricsPath = DeleteOutputPath("overall.idv");
            var instanceMetricsPath = DeleteOutputPath("instance.idv");

            RunTrainScoreEvaluate("Trainers.StochasticDualCoordinateAscentRegressor", "Models.RegressionEvaluator",
                dataPath, warningsPath, overallMetricsPath, instanceMetricsPath, loader: TestDatasets.winequalitymacro.loaderSettings);

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), warningsPath))
                Assert.Equal(0, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), overallMetricsPath))
                Assert.Equal(1, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), instanceMetricsPath))
                Assert.Equal(975, CountRows(loader));
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
                            'Sort': 'Occurrence',
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

            RunTrainScoreEvaluate("Trainers.FastTreeRanker", "Models.RankerEvaluator",
                dataPath, warningsPath, overallMetricsPath, instanceMetricsPath,
                splitterInput: "output_data3", transforms: transforms);

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), warningsPath))
                Assert.Equal(0, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), overallMetricsPath))
                Assert.Equal(1, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), instanceMetricsPath))
            {
                Assert.Equal(103, CountRows(loader));
                Assert.True(loader.Schema.TryGetColumnIndex("GroupId", out var groupCol));
                Assert.True(loader.Schema.TryGetColumnIndex("Label", out var labelCol));
            }
        }

        [Fact]
        public void EntryPointLightGbmBinary()
        {
            TestEntryPointRoutine("breast-cancer.txt", "Trainers.LightGbmBinaryClassifier");
        }

        [Fact]
        public void EntryPointLightGbmMultiClass()
        {
            TestEntryPointRoutine(GetDataPath(@"iris.txt"), "Trainers.LightGbmClassifier");
        }

        [Fact]
        public void EntryPointSdcaBinary()
        {
            TestEntryPointRoutine("breast-cancer.txt", "Trainers.StochasticDualCoordinateAscentBinaryClassifier");
        }

        [Fact]
        public void EntryPointSDCAMultiClass()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.StochasticDualCoordinateAscentClassifier");
        }

        [Fact()]
        public void EntryPointSDCARegression()
        {
            TestEntryPointRoutine(TestDatasets.winequalitymacro.trainFilename, "Trainers.StochasticDualCoordinateAscentRegressor", loader: TestDatasets.winequalitymacro.loaderSettings);
        }

        [Fact]
        public void EntryPointLogisticRegressionMultiClass()
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
        public void EntryPointLightLdaTransform()
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
            TestEntryPointRoutine("iris.txt", "Trainers.EnsembleBinaryClassifier");
        }

        [Fact]
        public void EntryPointClassificationEnsemble()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.EnsembleClassification");
        }

        [Fact]
        public void EntryPointRegressionEnsemble()
        {
            TestEntryPointRoutine(TestDatasets.winequalitymacro.trainFilename, "Trainers.EnsembleRegression", loader: TestDatasets.winequalitymacro.loaderSettings);
        }

        [Fact]
        public void EntryPointNaiveBayesMultiClass()
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
            TestEntryPointRoutine(TestDatasets.winequalitymacro.trainFilename, "Trainers.PoissonRegressor", loader: TestDatasets.winequalitymacro.loaderSettings);
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
                        'Type': 'I1'
                      },
                      {
                        'Name': 'Key1',
                        'Source': 'Key',
                        'Range': '[5-10,21-24]'
                      }
                      ]",
                    @"'Column': [
                      {
                        'Name': 'Ints',
                        'Source': 'Feat'
                      }
                      ],
                      'Type': 'I4'",
                    @"'Column': [
                      {
                        'Name': 'Floats',
                        'Source': 'Ints'
                      }
                      ],
                      'Type': 'Num'",
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

        internal void TestEntryPointRoutine(string dataFile, string trainerName, string loader = null, string trainerArgs = null)
        {
            var dataPath = GetDataPath(dataFile);
            var outputPath = DeleteOutputPath("model.zip");
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
                    {{
                      'Name': '{2}',
                      'Inputs': {{
                        'TrainingData': '$data1'
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
                string.IsNullOrWhiteSpace(loader) ? "" : string.Format(",'CustomSchema': '{0}'", loader),
                string.IsNullOrWhiteSpace(trainerArgs) ? "" : trainerArgs
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            bool success = catalog.TryFindEntryPoint("Transforms.MinMaxNormalizer", out ModuleCatalog.EntryPointInfo info);
            Assert.True(success);
            var inputBuilder = new InputBuilder(Env, info.InputType, catalog);

            var args = new NormalizeTransform.MinMaxArguments()
            {
                Column = new[]
                {
                    NormalizeTransform.AffineColumn.Parse("A"),
                    new NormalizeTransform.AffineColumn() { Name = "B", Source = "B", FixZero = false },
                },
                FixZero = true, // Same as default, should not appear in the generated JSON.
                MaxTrainingExamples = 1000
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            bool success = catalog.TryFindEntryPoint("Trainers.StochasticDualCoordinateAscentBinaryClassifier", out ModuleCatalog.EntryPointInfo info);
            Assert.True(success);
            var inputBuilder = new InputBuilder(Env, info.InputType, catalog);

            var args = new LinearClassificationTrainer.Arguments()
            {
                NormalizeFeatures = NormalizeOption.Yes,
                CheckFrequency = 42
            };

            var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            var inputMap = new Dictionary<ParameterBinding, VariableBinding>();

            var parameterBinding = new SimpleParameterBinding("TrainingData");
            inputBindingMap.Add("TrainingData", new List<ParameterBinding>() { parameterBinding });
            inputMap.Add(parameterBinding, new SimpleVariableBinding("data"));

            var result = inputBuilder.GetJsonObject(args, inputBindingMap, inputMap);
            var json = FixWhitespace(result.ToString(Formatting.Indented));

            var expected =
                @"{
  ""CheckFrequency"": 42,
  ""TrainingData"": ""$data"",
  ""NormalizeFeatures"": ""Yes""
}";
            expected = FixWhitespace(expected);
            Assert.Equal(expected, json);

            args.LossFunction = new HingeLoss.Arguments();
            result = inputBuilder.GetJsonObject(args, inputBindingMap, inputMap);
            json = FixWhitespace(result.ToString(Formatting.Indented));

            expected =
                @"{
  ""LossFunction"": {
    ""Name"": ""HingeLoss""
  },
  ""CheckFrequency"": 42,
  ""TrainingData"": ""$data"",
  ""NormalizeFeatures"": ""Yes""
}";
            expected = FixWhitespace(expected);
            Assert.Equal(expected, json);

            args.LossFunction = new HingeLoss.Arguments() { Margin = 2 };
            result = inputBuilder.GetJsonObject(args, inputBindingMap, inputMap);
            json = FixWhitespace(result.ToString(Formatting.Indented));

            expected =
                @"{
  ""LossFunction"": {
    ""Name"": ""HingeLoss"",
    ""Settings"": {
      ""Margin"": 2.0
    }
  },
  ""CheckFrequency"": 42,
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<IPredictorModel>("model");
            Assert.NotNull(model);
        }

        [Fact]
        public void EntryPointTrainTestBinaryMacro()
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
                      'Name': 'Models.TrainTestBinaryEvaluator',
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
                          'Model': '$model'
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<IPredictorModel>("model");
            Assert.NotNull(model);

            var metrics = runner.GetOutput<IDataView>("OverallMetrics");
            Assert.NotNull(metrics);
            using (var cursor = metrics.GetRowCursor(col => true))
            {
                Assert.True(cursor.Schema.TryGetColumnIndex("AUC", out int aucCol));
                var aucGetter = cursor.GetGetter<double>(aucCol);
                Assert.True(cursor.MoveNext());
                double auc = 0;
                aucGetter(ref auc);
                Assert.True(auc > 0.99);
            }
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<IPredictorModel>("model");
            Assert.NotNull(model);

            var metrics = runner.GetOutput<IDataView>("OverallMetrics");
            Assert.NotNull(metrics);
            using (var cursor = metrics.GetRowCursor(col => true))
            {
                Assert.True(cursor.Schema.TryGetColumnIndex("AUC", out int aucCol));
                var aucGetter = cursor.GetGetter<double>(aucCol);
                Assert.True(cursor.MoveNext());
                double auc = 0;
                aucGetter(ref auc);
                Assert.True(auc > 0.99);
            }
        }

        [Fact]
        public void EntryPointKMeans()
        {
            TestEntryPointRoutine("Train-Tiny-28x28.txt", "Trainers.KMeansPlusPlusClusterer", "col=Weight:R4:0 col=Features:R4:1-784", ",'InitAlgorithm':'KMeansPlusPlus'");
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<IPredictorModel>("model");
            Assert.NotNull(model);

            var metrics = runner.GetOutput<IDataView>("OverallMetrics");
            Assert.NotNull(metrics);
            using (var cursor = metrics.GetRowCursor(col => true))
            {
                Assert.True(cursor.Schema.TryGetColumnIndex("AUC", out int aucCol));
                var aucGetter = cursor.GetGetter<double>(aucCol);
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<IPredictorModel>("model");
            Assert.NotNull(model);

            model = runner.GetOutput<IPredictorModel>("model2");
            Assert.NotNull(model);

            var metrics = runner.GetOutput<IDataView>("OverallMetrics");
            Assert.NotNull(metrics);
            using (var cursor = metrics.GetRowCursor(col => true))
            {
                Assert.True(cursor.Schema.TryGetColumnIndex("AUC", out int aucCol));
                var aucGetter = cursor.GetGetter<double>(aucCol);
                Assert.True(cursor.MoveNext());
                double auc = 0;
                aucGetter(ref auc);
                Assert.True(auc > 0.99);
            }

            metrics = runner.GetOutput<IDataView>("OverallMetrics2");
            Assert.NotNull(metrics);
            using (var cursor = metrics.GetRowCursor(col => true))
            {
                Assert.True(cursor.Schema.TryGetColumnIndex("AUC", out int aucCol));
                var aucGetter = cursor.GetGetter<double>(aucCol);
                Assert.True(cursor.MoveNext());
                double auc = 0;
                aucGetter(ref auc);
                Assert.True(auc > 0.99);
            }
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            var runner = new GraphRunner(Env, catalog, graph[FieldNames.Nodes] as JArray);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<IPredictorModel[]>("model");
            Assert.NotNull(model[0]);

            model = runner.GetOutput<IPredictorModel[]>("model2");
            Assert.NotNull(model[0]);

            var metrics = runner.GetOutput<IDataView>("OverallMetrics");
            Assert.NotNull(metrics);
            using (var cursor = metrics.GetRowCursor(col => true))
            {
                Assert.True(cursor.Schema.TryGetColumnIndex("AUC", out int aucCol));
                var aucGetter = cursor.GetGetter<double>(aucCol);
                Assert.True(cursor.MoveNext());
                double auc = 0;
                aucGetter(ref auc);
                Assert.True(auc > 0.99);
            }

            metrics = runner.GetOutput<IDataView>("OverallMetrics2");
            Assert.NotNull(metrics);
            using (var cursor = metrics.GetRowCursor(col => true))
            {
                Assert.True(cursor.Schema.TryGetColumnIndex("AUC", out int aucCol));
                var aucGetter = cursor.GetGetter<double>(aucCol);
                Assert.True(cursor.MoveNext());
                double auc = 0;
                aucGetter(ref auc);
                Assert.True(auc > 0.99);
            }
        }

        [Fact]
        public void EntryPointMacroEarlyExpansion()
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
                      'Name': 'Models.TrainTestBinaryEvaluator',
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
                              'Model': '$model'
                            }
                          }
                        ],
                        'Inputs': {
                          'Data': '$data1'
                        },
                        'Outputs': {
                          'Model': '$model'
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

            JObject graphJson = JObject.Parse(inputGraph);
            var catalog = ModuleCatalog.CreateInstance(Env);
            var graph = new EntryPointGraph(Env, catalog, graphJson[FieldNames.Nodes] as JArray);
            Assert.True(graph.Macros.All(x => x.CanStart()));
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            var graph = new EntryPointGraph(Env, catalog, graphJson[FieldNames.Nodes] as JArray);
            // Serialize the nodes with ToJson() and then executing them to ensure serialization working correctly.
            var nodes = new JArray(graph.AllNodes.Select(node => node.ToJson()));
            var runner = new GraphRunner(Env, catalog, nodes);

            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            runner.SetInput("file", inputFile);

            runner.RunAll();

            var model = runner.GetOutput<IPredictorModel>("model");
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
            var catalog = ModuleCatalog.CreateInstance(Env);
            var graph = new EntryPointGraph(Env, catalog, graphJson[FieldNames.Nodes] as JArray);
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
                graph = new EntryPointGraph(Env, catalog, serNodes);
            }
        }

        [Fact]
        public void EntryPointLinearPredictorSummary()
        {
            var dataPath = GetDataPath("breast-cancer-withheader.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);

            var dataView = ImportTextData.TextLoader(Env, new ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    SeparatorChars = new []{'\t' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", type: null, 0),
                        new TextLoader.Column("Features", DataKind.Num, new [] { new TextLoader.Range(1, 9) })
                    }
                },

                InputFile = inputFile,
            }).Data;

            var lrInput = new LogisticRegression.Arguments
            {
                TrainingData = dataView,
                NormalizeFeatures = NormalizeOption.Yes,
                NumThreads = 1,
                // REVIEW: this depends on MKL library which is not available
                ShowTrainingStats = false
            };
            var model = LogisticRegression.TrainBinary(Env, lrInput).PredictorModel;

            var mcLrInput = new MulticlassLogisticRegression.Arguments
            {
                TrainingData = dataView,
                NormalizeFeatures = NormalizeOption.Yes,
                NumThreads = 1,
                ShowTrainingStats = true
            };
            var mcModel = LogisticRegression.TrainMultiClass(Env, mcLrInput).PredictorModel;

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

                // REVIEW: enable this once MKL library is available
                // var stats = DeleteOutputPath(@"../Common/EntryPoints", "lr-stats.txt");
                // using (var file = Env.CreateOutputFile(stats))
                //    DataSaverUtils.SaveDataView(ch, saver, output.Stats, file);

                weights = DeleteOutputPath(@"../Common/EntryPoints", "mc-lr-weights.txt");
                using (var file = Env.CreateOutputFile(weights))
                    DataSaverUtils.SaveDataView(ch, saver, mcOutput.Summary, file);

                var stats = DeleteOutputPath(@"../Common/EntryPoints", "mc-lr-stats.txt");
                using (var file = Env.CreateOutputFile(stats))
                    DataSaverUtils.SaveDataView(ch, saver, mcOutput.Stats, file);

                ch.Done();
            }

            CheckEquality(@"../Common/EntryPoints", "lr-weights.txt");
            // CheckEquality(@"../Common/EntryPoints", "lr-stats.txt");
            CheckEquality(@"../Common/EntryPoints", "mc-lr-weights.txt");
            CheckEquality(@"../Common/EntryPoints", "mc-lr-stats.txt");
            Done();
        }

        [Fact]
        public void EntryPointPcaPredictorSummary()
        {
            var dataPath = GetDataPath("MNIST.Train.0-class.tiny.txt");
            using (var inputFile = new SimpleFileHandle(Env, dataPath, false, false))
            {
                var dataView = ImportTextData.TextLoader(Env, new ImportTextData.LoaderInput()
                {
                    Arguments =
                {
                    SeparatorChars = new []{'\t' },
                    HasHeader = false,
                    Column = new[]
                    {
                        new TextLoader.Column("Features", DataKind.R4, new [] { new TextLoader.Range(1, 784) })
                    }
                },

                    InputFile = inputFile,
                }).Data;

                var pcaInput = new RandomizedPcaTrainer.Arguments
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

                    ch.Done();
                }

                CheckEquality(@"../Common/EntryPoints", "pca-weights.txt");
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
                using (var cursor = loader.GetRowCursor(col => true))
                {
                    DvText predictedLabel = default(DvText);

                    var success = loader.Schema.TryGetColumnIndex("PredictedLabel", out int predictedLabelCol);
                    Assert.True(success);
                    var predictedLabelGetter = cursor.GetGetter<DvText>(predictedLabelCol);

                    while (cursor.MoveNext())
                    {
                        predictedLabelGetter(ref predictedLabel);
                        Assert.True(predictedLabel.EqualsStr("Iris-setosa")
                            || predictedLabel.EqualsStr("Iris-versicolor")
                            || predictedLabel.EqualsStr("Iris-virginica"));
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
            var dataView = ImportTextData.ImportText(Env, new ImportTextData.Input { InputFile = inputFile }).Data;
#pragma warning restore 0618
            var cat = Categorical.CatTransformDict(Env, new CategoricalTransform.Arguments()
            {
                Data = dataView,
                Column = new[] { new CategoricalTransform.Column { Name = "Categories", Source = "Categories" } }
            });
            var concat = SchemaManipulation.ConcatColumns(Env, new ConcatTransform.Arguments()
            {
                Data = cat.OutputData,
                Column = new[] { new ConcatTransform.Column { Name = "Features", Source = new[] { "Categories", "NumericFeatures" } } }
            });

            var fastTree = FastTree.FastTree.TrainBinary(Env, new FastTreeBinaryClassificationTrainer.Arguments
            {
                FeatureColumn = "Features",
                NumTrees = 5,
                NumLeaves = 4,
                LabelColumn = DefaultColumnNames.Label,
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
            Assert.True(view.Schema.TryGetColumnIndex("Trees", out int treesCol));
            Assert.True(view.Schema.TryGetColumnIndex("Leaves", out int leavesCol));
            Assert.True(view.Schema.TryGetColumnIndex("Paths", out int pathsCol));
            VBuffer<float> treeValues = default(VBuffer<float>);
            VBuffer<float> leafIndicators = default(VBuffer<float>);
            VBuffer<float> pathIndicators = default(VBuffer<float>);
            using (var curs = view.GetRowCursor(c => c == treesCol || c == leavesCol || c == pathsCol))
            {
                var treesGetter = curs.GetGetter<VBuffer<float>>(treesCol);
                var leavesGetter = curs.GetGetter<VBuffer<float>>(leavesCol);
                var pathsGetter = curs.GetGetter<VBuffer<float>>(pathsCol);
                while (curs.MoveNext())
                {
                    treesGetter(ref treeValues);
                    leavesGetter(ref leafIndicators);
                    pathsGetter(ref pathIndicators);

                    Assert.Equal(5, treeValues.Length);
                    Assert.Equal(5, treeValues.Count);
                    Assert.Equal(20, leafIndicators.Length);
                    Assert.Equal(5, leafIndicators.Count);
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
            var dataView = ImportTextData.TextLoader(Env, new ImportTextData.LoaderInput()
            {
                Arguments =
                {
                    SeparatorChars = new []{' '},
                    Column = new[]
                    {
                        new TextLoader.Column("Text", DataKind.Text,
                            new [] { new TextLoader.Range() { Min = 0, VariableEnd=true, ForceVector=true} })
                    }
                },
                InputFile = inputFile,
            }).Data;
            var embedding = Transforms.TextAnalytics.WordEmbeddings(Env, new WordEmbeddingsTransform.Arguments()
            {
                Data = dataView,
                Column = new[] { new WordEmbeddingsTransform.Column { Name = "Features", Source = "Text" } },
                ModelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe
            });
            var result = embedding.OutputData;
            using (var cursor = result.GetRowCursor((x => true)))
            {
                Assert.True(result.Schema.TryGetColumnIndex("Features", out int featColumn));
                var featGetter = cursor.GetGetter<VBuffer<float>>(featColumn);
                VBuffer<float> feat = default;
                while (cursor.MoveNext())
                {
                    featGetter(ref feat);
                    Assert.True(feat.Count == 150);
                    Assert.True(feat.Values[0] != 0);
                }
            }
        }

        [Fact]
        public void EntryPointTensorFlowTransform()
        {
            TestEntryPointPipelineRoutine(GetDataPath("Train-Tiny-28x28.txt"), "col=Label:R4:0 col=Placeholder:R4:1-784",
                new[] { "Transforms.TensorFlowScorer" },
                new[]
                {
                    @"'InputColumns': [ 'Placeholder' ],
                      'ModelFile': 'mnist_model/frozen_saved_model.pb',
                      'OutputColumns': [ 'Softmax' ]"
                });
        }
    }
}