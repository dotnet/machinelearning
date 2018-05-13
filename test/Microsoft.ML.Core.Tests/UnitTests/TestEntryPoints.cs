// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Core.Tests.UnitTests;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
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

        [Fact]
        public void EntryPointTrainTestSplit()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            var dataView = ImportTextData.ImportText(Env, new ImportTextData.Input { InputFile = inputFile, CustomSchema = "col=Label:0 col=Features:TX:1-9" }).Data;

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
            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            var dataView = ImportTextData.ImportText(Env, new ImportTextData.Input { InputFile = inputFile, CustomSchema = "col=Label:0 col=F1:TX:1 col=F2:I4:2 col=Rest:3-9" }).Data;
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
            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            var dataView = ImportTextData.ImportText(Env, new ImportTextData.Input { InputFile = inputFile, CustomSchema = "col=Label:0 col=F1:TX:1 col=F2:I4:2 col=Rest:3-9" }).Data;
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
            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            var dataView = ImportTextData.ImportText(Env, new ImportTextData.Input { InputFile = inputFile, CustomSchema = "col=Label:0 col=F1:TX:1 col=F2:I4:2 col=Rest:3-9" }).Data;
            dataView = Env.CreateTransform("Term{col=F1}", dataView);

            var data1 = FeatureCombiner.PrepareFeatures(Env, new FeatureCombiner.FeatureCombinerInput() { Data = dataView, Features = new[] { "F1", "F2", "Rest" } });
            var data2 = ModelOperations.Apply(Env, new ModelOperations.ApplyTransformModelInput() { Data = dataView, TransformModel = data1.Model });

            CheckSameValues(data1.OutputData, data2.OutputData);
            Done();
        }

        [Fact]
        public void EntryPointCaching()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
            var dataView = ImportTextData.ImportText(Env, new ImportTextData.Input { InputFile = inputFile, CustomSchema = "col=Label:0 col=F1:TX:1 col=F2:I4:2 col=Rest:3-9" }).Data;
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

        [Fact]
        public void EntryPointCatalog()
        {
            var buildPrefix = GetBuildPrefix();
            var epListFile = buildPrefix + "_ep-list.tsv";
            var manifestFile = buildPrefix + "_manifest.json";

            var entryPointsSubDir = Path.Combine("..", "Common", "EntryPoints");
            var catalog = ModuleCatalog.CreateInstance(Env);
            var path = DeleteOutputPath(entryPointsSubDir, epListFile);
            File.WriteAllLines(path, catalog.AllEntryPoints()
                .Select(x => string.Join("\t", x.Name, x.Description, x.Method.DeclaringType, x.Method.Name, x.InputType, x.OutputType).Replace(Environment.NewLine, "\\n "))
                .OrderBy(x => x));

            CheckEquality(entryPointsSubDir, epListFile);

            var jObj = JsonManifestUtils.BuildAllManifests(Env, catalog);
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
            Assert.True(string.Compare(instance.WeightColumn.Value, "OtherWeight") == 0);

            var tok = (JToken)JValue.CreateString("AnotherWeight");
            ib1.TrySetValueJson("WeightColumn", tok);
            Assert.True(instance.WeightColumn.IsExplicit);
            Assert.True(string.Compare(instance.WeightColumn.Value, "AnotherWeight") == 0);
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
        public void EntryPointOptionalParams()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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

        //[Fact]
        //public void EntryPointCalibrate()
        //{
        //    var dataPath = GetDataPath("breast-cancer.txt");
        //    var inputFile = new SimpleFileHandle(Env, dataPath, false, false);
        //    var dataView = ImportTextData.ImportText(Env, new ImportTextData.Input { InputFile = inputFile, CustomSchema = "col=Label:0 col=Features:1-9" }).Data;

        //    var splitOutput = CVSplit.Split(Env, new CVSplit.Input { Data = dataView, NumFolds = 3 });

        //    var lrModel = LogisticRegression.TrainBinary(Env, new LogisticRegression.Arguments { TrainingData = splitOutput.TestData[0] }).PredictorModel;
        //    var calibratedLrModel = Calibrate.FixedPlatt(Env,
        //        new Calibrate.FixedPlattInput { Data = splitOutput.TestData[1], UncalibratedPredictorModel = lrModel }).PredictorModel;

        //    var scored1 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = splitOutput.TestData[2], PredictorModel = lrModel }).ScoredData;
        //    scored1 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored1, ExtraColumns = new[] { "Label" } }).OutputData;

        //    var scored2 = ScoreModel.Score(Env, new ScoreModel.Input() { Data = splitOutput.TestData[2], PredictorModel = calibratedLrModel }).ScoredData;
        //    scored2 = ScoreModel.SelectColumns(Env, new ScoreModel.ScoreColumnSelectorInput() { Data = scored2, ExtraColumns = new[] { "Label" } }).OutputData;

        //    Assert.Equal(4, scored1.Schema.ColumnCount);
        //    CheckSameValues(scored1, scored2);

        //    var input = new Calibrate.NoArgumentsInput() { Data = splitOutput.TestData[1], UncalibratedPredictorModel = lrModel };
        //    calibratedLrModel = Calibrate.Platt(Env, input).PredictorModel;
        //    calibratedLrModel = Calibrate.Naive(Env, input).PredictorModel;
        //    calibratedLrModel = Calibrate.Pav(Env, input).PredictorModel;

        //    // This tests that the SchemaBindableCalibratedPredictor doesn't get confused if its sub-predictor is already calibrated.
        //    var fastForest = new FastForestClassification(Env, new FastForestClassification.Arguments());
        //    var rmd = RoleMappedData.Create(splitOutput.TrainData[0],
        //        RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Feature, "Features"),
        //        RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Label, "Label"));
        //    fastForest.Train(rmd);
        //    var ffModel = new PredictorModel(Env, rmd, splitOutput.TrainData[0], fastForest.CreatePredictor());
        //    var calibratedFfModel = Calibrate.Platt(Env,
        //        new Calibrate.NoArgumentsInput() { Data = splitOutput.TestData[0], UncalibratedPredictorModel = ffModel }).PredictorModel;
        //    var twiceCalibratedFfModel = Calibrate.Platt(Env,
        //        new Calibrate.NoArgumentsInput() { Data = splitOutput.TestData[0], UncalibratedPredictorModel = calibratedFfModel }).PredictorModel;
        //    var scoredFf = ScoreModel.Score(Env, new ScoreModel.Input() { Data = splitOutput.TestData[2], PredictorModel = twiceCalibratedFfModel }).ScoredData;
        //}

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
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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
                    string instanceMetricsPath, string confusionMatrixPath = null)
        {
            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.TextLoader',
                      'Inputs': {{
                        'InputFile': '$file'
                      }},
                      'Outputs': {{
                        'Data': '$AllData'
                      }}
                    }},
                    {{
                      'Name': 'Transforms.TrainTestDatasetSplitter',
                      'Inputs': {{
                        'Data': '$AllData',
                        'Fraction': 0.8
                      }},
                      'Outputs': {{
                        'TrainData': '$TrainData',
                        'TestData': '$TestData'
                      }}
                    }},
                    {{
                      'Name': '{0}',
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
                      'Name': '{1}',
                      'Inputs': {{
                        'Data': '$ScoredData'
                      }},
                      'Outputs': {{
                        'Warnings': '$Warnings',
                        'OverallMetrics': '$OverallMetrics',
                        'PerInstanceMetrics': '$PerInstanceMetrics'
                        {6}
                      }}
                    }}
                  ],
                  'Inputs' : {{
                    'file' : '{2}'
                  }},
                  'Outputs' : {{
                    'Warnings' : '{3}',
                    'OverallMetrics' : '{4}',
                    'PerInstanceMetrics' : '{5}'
                    {7}
                  }}
                }}", learner, evaluator, EscapePath(dataPath), EscapePath(warningsPath), EscapePath(overallMetricsPath), EscapePath(instanceMetricsPath),
                confusionMatrixPath != null ? ", 'ConfusionMatrix': '$ConfusionMatrix'" : "",
                confusionMatrixPath != null ? string.Format(", 'ConfusionMatrix' : '{0}'", EscapePath(confusionMatrixPath)) : "");

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

        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/3")]
        public void EntryPointEvaluateRegression()
        {
            var dataPath = GetDataPath("housing.txt");
            var warningsPath = DeleteOutputPath("warnings.idv");
            var overallMetricsPath = DeleteOutputPath("overall.idv");
            var instanceMetricsPath = DeleteOutputPath("instance.idv");

            RunTrainScoreEvaluate("Trainers.StochasticDualCoordinateAscentRegressor", "Models.RegressionEvaluator", dataPath, warningsPath, overallMetricsPath, instanceMetricsPath);

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), warningsPath))
                Assert.Equal(0, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), overallMetricsPath))
                Assert.Equal(1, CountRows(loader));

            using (var loader = new BinaryLoader(Env, new BinaryLoader.Arguments(), instanceMetricsPath))
                Assert.Equal(104, CountRows(loader));
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

        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/3")]
        public void EntryPointSDCARegression()
        {
            TestEntryPointRoutine("housing.txt", "Trainers.StochasticDualCoordinateAscentRegressor");
        }

        [Fact]
        public void EntryPointLogisticRegressionMultiClass()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.LogisticRegressionClassifier");
        }

        [Fact]
        public void EntryPointPcaAnomaly()
        {
            TestEntryPointRoutine("MNIST.Train.0-class.tiny.txt", "Trainers.PcaAnomalyDetector");
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
        public void EntryPointNaiveBayesMultiClass()
        {
            TestEntryPointRoutine("iris.txt", "Trainers.NaiveBayesClassifier");
        }

        [Fact]
        public void EntryPointHogwildSGD()
        {
            TestEntryPointRoutine("breast-cancer.txt", "Trainers.StochasticGradientDescentBinaryClassifier");
        }

        [Fact(Skip = "Missing data set. See https://github.com/dotnet/machinelearning/issues/3")]
        public void EntryPointPoissonRegression()
        {
            TestEntryPointRoutine("housing.txt", "Trainers.PoissonRegressor");
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
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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
        public void EntryPointKMeans()
        {
            TestEntryPointRoutine("Train-Tiny-28x28.txt", "Trainers.KMeansPlusPlusClusterer");
        }

        [Fact]
        public void EntryPointTrainTestMacro()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.TextLoader',
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
        public void EntryPointChainedTrainTestMacros()
        {
            string inputGraph = @"
                {
                  'Nodes': [
                    {
                      'Name': 'Data.TextLoader',
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
                          'Model': '$model2'
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
                      'Name': 'Data.TextLoader',
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
                          'Model': '$model2'
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

            var metrics = runner.GetOutput<IDataView[]>("OverallMetrics");
            Assert.NotNull(metrics[0]);
            using (var cursor = metrics[0].GetRowCursor(col => true))
            {
                Assert.True(cursor.Schema.TryGetColumnIndex("AUC", out int aucCol));
                var aucGetter = cursor.GetGetter<double>(aucCol);
                Assert.True(cursor.MoveNext());
                double auc = 0;
                aucGetter(ref auc);
                Assert.True(auc > 0.99);
            }

            metrics = runner.GetOutput<IDataView[]>("OverallMetrics2");
            Assert.NotNull(metrics[0]);
            using (var cursor = metrics[0].GetRowCursor(col => true))
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
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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
                      'Name': 'Data.TextLoader',
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
        public void EntryPointPrepareLabelConvertPredictedLabel()
        {
            var dataPath = GetDataPath("iris.data");
            var outputPath = DeleteOutputPath("prepare-label-data.idv");
            string inputGraph = string.Format(@"
                {{
                  'Nodes': [
                    {{
                      'Name': 'Data.TextLoader',
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
    }
}