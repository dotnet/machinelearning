// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class DnnImageFeaturizerTests : TestDataPipeBase
    {
        private const int InputSize = 3 * 224 * 224;

        private class TestData
        {
            [VectorType(InputSize)]
            public float[] data_0;
        }
        private class TestDataSize
        {
            [VectorType(2)]
            public float[] data_0;
        }
        private class TestDataXY
        {
            [VectorType(InputSize)]
            public float[] A;
        }
        private class TestDataDifferentType
        {
            [VectorType(InputSize)]
            public string[] data_0;
        }

        private float[] GetSampleArrayData()
        {
            var samplevector = new float[InputSize];
            for (int i = 0; i < InputSize; i++)
                samplevector[i] = (i / ((float)InputSize));
            return samplevector;
        }

        public DnnImageFeaturizerTests(ITestOutputHelper helper) : base(helper)
        {
        }

        [OnnxFact]
        public void TestDnnImageFeaturizer()
        {
            //skip running for x86 as this test using too much memory (over 2GB limit on x86)
            //and very like to hit memory related issue when running on CI
            //TODO: optimized memory usage in related code and enable x86 test run
            if (!Environment.Is64BitProcess)
                return;

            var samplevector = GetSampleArrayData();

            var dataView = DataViewConstructionUtils.CreateFromList(Env,
                new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    },
                });

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[InputSize] } };
            var stringData = new List<TestDataDifferentType> { new TestDataDifferentType() { data_0 = new string[InputSize] } };
            var sizeData = new List<TestDataSize> { new TestDataSize() { data_0 = new float[2] } };
            var pipe = ML.Transforms.DnnFeaturizeImage("output_1", m => m.ModelSelector.ResNet18(m.Environment, m.OutputColumn, m.InputColumn), "data_0");

            var invalidDataWrongNames = ML.Data.LoadFromEnumerable(xyData);
            var invalidDataWrongTypes = ML.Data.LoadFromEnumerable(stringData);
            var invalidDataWrongVectorSize = ML.Data.LoadFromEnumerable(sizeData);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongNames);
            TestEstimatorCore(pipe, dataView, invalidInput: invalidDataWrongTypes);
            pipe.GetOutputSchema(SchemaShape.Create(invalidDataWrongVectorSize.Schema));
            try
            {
                pipe.Fit(invalidDataWrongVectorSize);
                Assert.False(true);
            }
            catch (ArgumentOutOfRangeException) { }
            catch (InvalidOperationException) { }
        }

        [OnnxFact]
        public void OnnxFeaturizerWorkout()
        {
            var env = new MLContext(null);
            var imageHeight = 224;
            var imageWidth = 224;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = ML.Data.LoadFromTextFile(dataFile, new[] {
                new TextLoader.Column("imagePath", DataKind.String, 0),
                new TextLoader.Column("name", DataKind.String, 1)
            });

            var pipe = ML.Transforms.LoadImages("data_0", imageFolder, "imagePath")
                .Append(ML.Transforms.ResizeImages("data_0", imageHeight, imageWidth))
                .Append(ML.Transforms.ExtractPixels("data_0", interleavePixelColors: true))
                .Append(ML.Transforms.DnnFeaturizeImage("output_1", m => m.ModelSelector.ResNet18(m.Environment, m.OutputColumn, m.InputColumn), "data_0"));

            TestEstimatorCore(pipe, data);

            var result = pipe.Fit(data).Transform(data);
            using (var cursor = result.GetRowCursor(result.Schema["output_1"]))
            {
                var buffer = default(VBuffer<float>);
                var getter = cursor.GetGetter<VBuffer<float>>(result.Schema["output_1"]);
                var numRows = 0;
                while (cursor.MoveNext())
                {
                    getter(ref buffer);
                    Assert.Equal(512, buffer.Length);
                    numRows += 1;
                }
                Assert.Equal(4, numRows);
            }
        }

        [OnnxFact]
        public void TestOldSavingAndLoading()
        {
            //skip running for x86 as this test using too much memory (over 2GB limit on x86)
            //and very like to hit memory related issue when running on CI
            //TODO: optimized memory usage in related code and enable x86 run
            if (!Environment.Is64BitProcess)
                return;

            var samplevector = GetSampleArrayData();

            var dataView = ML.Data.LoadFromEnumerable(
                new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    }
                });

            var inputNames = "data_0";
            var outputNames = "output_1";
            var est = ML.Transforms.DnnFeaturizeImage(outputNames, m => m.ModelSelector.ResNet18(m.Environment, m.OutputColumn, m.InputColumn), inputNames);
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);

                using (var cursor = loadedView.GetRowCursor(loadedView.Schema[outputNames]))
                {
                    VBuffer<float> softMaxValue = default;
                    var softMaxGetter = cursor.GetGetter<VBuffer<float>>(loadedView.Schema[outputNames]);
                    float sum = 0f;
                    int i = 0;
                    while (cursor.MoveNext())
                    {
                        softMaxGetter(ref softMaxValue);
                        var values = softMaxValue.DenseValues();
                        foreach (var val in values)
                        {
                            sum += val;
                            if (i == 0)
                                Assert.InRange(val, 0.0, 0.00001);
                            if (i == 7)
                                Assert.InRange(val, 0.62935, 0.62940);
                            if (i == 500)
                                Assert.InRange(val, 0.15521, 0.155225);
                            i++;
                        }
                    }
                    Assert.InRange(sum, 83.50, 84.50);
                }
            }
        }

        internal sealed class ModelInput
        {
            [ColumnName("ImagePath"), LoadColumn(0)]
            public string ImagePath { get; set; }

            [ColumnName("Label"), LoadColumn(1)]
            public string Label { get; set; }
        }

        internal sealed class ModelOutput
        {
            // ColumnName attribute is used to change the column name from
            // its default value, which is the name of the field.
            [ColumnName("PredictedLabel")]
            public String Prediction { get; set; }
            public float[] Score { get; set; }
        }

        [OnnxFact]
        public void TestLoadFromDiskAndPredictionEngine()
        {
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = ML.Data.LoadFromTextFile<ModelInput>(
                                path: dataFile,
                                hasHeader: false,
                                separatorChar: '\t',
                                allowQuoting: true,
                                allowSparse: false);

            var dataProcessPipeline = ML.Transforms.Conversion.MapValueToKey("Label", "Label")
                                      .Append(ML.Transforms.LoadImages("ImagePath_featurized", imageFolder, "ImagePath"))
                                      .Append(ML.Transforms.ResizeImages("ImagePath_featurized", 224, 224, "ImagePath_featurized"))
                                      .Append(ML.Transforms.ExtractPixels("ImagePath_featurized", "ImagePath_featurized"))
                                      .Append(ML.Transforms.DnnFeaturizeImage("ImagePath_featurized", m => m.ModelSelector.ResNet18(m.Environment, m.OutputColumn, m.InputColumn), "ImagePath_featurized"))
                                      .Append(ML.Transforms.Concatenate("Features", new[] { "ImagePath_featurized" }))
                                      .Append(ML.Transforms.NormalizeMinMax("Features", "Features"))
                                      .AppendCacheCheckpoint(ML);

            var trainer = ML.MulticlassClassification.Trainers.OneVersusAll(ML.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", numberOfIterations: 10, featureColumnName: "Features"), labelColumnName: "Label")
                                      .Append(ML.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var model = trainingPipeline.Fit(data);

            string modelPath = GetOutputPath("TestSaveToDiskAndPredictionEngine-model.zip");
            ML.Model.Save(model, data.Schema, modelPath);
            var loadedModel = ML.Model.Load(modelPath, out var inputSchema);

            var predEngine = ML.Model.CreatePredictionEngine<ModelInput, ModelOutput>(loadedModel);
            ModelInput sample = ML.Data.CreateEnumerable<ModelInput>(data, false).First();
            ModelOutput result = predEngine.Predict(sample);
            Assert.Equal("tomato", result.Prediction);
        }
    }
}
