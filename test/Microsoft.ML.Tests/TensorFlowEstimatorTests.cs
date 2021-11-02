// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.TensorFlow;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    [CollectionDefinition("NoParallelization", DisableParallelization = true)]
    public class NoParallelizationCollection { }

    [Collection("NoParallelization")]
    public class TensorFlowEstimatorTests : TestDataPipeBase
    {
        private class TestData
        {
            [VectorType(4)]
            public float[] a;
            [VectorType(4)]
            public float[] b;
        }
        private class TestDataSize
        {
            [VectorType(2)]
            public float[] a;
            [VectorType(2)]
            public float[] b;
        }
        private class TestDataXY
        {
            [VectorType(4)]
            public float[] A;
            [VectorType(4)]
            public float[] B;
        }
        private class TestDataDifferentType
        {
            [VectorType(4)]
            public string[] a;
            [VectorType(4)]
            public string[] b;
        }

        public TensorFlowEstimatorTests(ITestOutputHelper output) : base(output)
        {
        }

        [TensorFlowFact]
        public void TestSimpleCase()
        {
            var modelFile = "model_matmul/frozen_saved_model.pb";

            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestData>(new TestData[] {
                    new TestData()
                    {
                        a = new[] { 1.0f, 2.0f,3.0f, 4.0f },
                        b = new[] { 1.0f, 2.0f,3.0f, 4.0f }
                    },
                     new TestData()
                     {
                         a = new[] { 2.0f, 2.0f,2.0f, 2.0f },
                         b = new[] { 3.0f, 3.0f,3.0f, 3.0f }
                     }
                }));

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[4], B = new float[4] } };
            var stringData = new List<TestDataDifferentType> { new TestDataDifferentType() { a = new string[4], b = new string[4] } };
            var sizeData = new List<TestDataSize> { new TestDataSize() { a = new float[2], b = new float[2] } };
            using var model = ML.Model.LoadTensorFlowModel(modelFile);
            var pipe = model.ScoreTensorFlowModel(new[] { "c" }, new[] { "a", "b" });

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

        [TensorFlowFact]
        public void TestOldSavingAndLoading()
        {
            var modelFile = "model_matmul/frozen_saved_model.pb";

            var dataView = ML.Data.LoadFromEnumerable(
                new List<TestData>(new TestData[] {
                    new TestData()
                    {
                        a = new[] { 1.0f, 2.0f, 3.0f, 4.0f },
                        b = new[] { 1.0f, 2.0f, 3.0f, 4.0f }
                    },
                    new TestData()
                    {
                        a = new[] { 2.0f, 2.0f, 2.0f, 2.0f },
                        b = new[] { 3.0f, 3.0f, 3.0f, 3.0f }
                    },
                    new TestData()
                    {
                        a = new[] { 5.0f, 6.0f, 10.0f, 12.0f },
                        b = new[] { 10.0f, 8.0f, 6.0f, 6.0f }
                    }
                }));
            using var model = ML.Model.LoadTensorFlowModel(modelFile);
            var est = model.ScoreTensorFlowModel(new[] { "c" }, new[] { "a", "b" });
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);
                ValidateTensorFlowTransformer(loadedView);
            }
        }

        [TensorFlowFact]
        public void TestCommandLine()
        {
            // typeof helps to load the TensorFlowTransformer type.
            Type type = typeof(TensorFlowTransformer);
            Assert.Equal(Maml.Main(new[] { @"showschema loader=Text{col=a:R4:0-3 col=b:R4:0-3} xf=TFTransform{inputs=a inputs=b outputs=c modellocation={model_matmul/frozen_saved_model.pb}}" }), (int)0);
        }

        [TensorFlowFact]
        public void TestTensorFlow()
        {
            var modelLocation = "cifar_model/frozen_model.pb";

            var mlContext = new MLContext(seed: 1);
            var imageHeight = 32;
            var imageWidth = 32;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = ML.Data.LoadFromTextFile(dataFile, new[] {
                new TextLoader.Column("imagePath", DataKind.String, 0),
                new TextLoader.Column("name", DataKind.String, 1)
            });

            // Note that CamelCase column names are there to match the TF graph node names.
            var pipe = ML.Transforms.LoadImages("Input", imageFolder, "imagePath")
                .Append(ML.Transforms.ResizeImages("Input", imageHeight, imageWidth))
                .Append(ML.Transforms.ExtractPixels("Input", interleavePixelColors: true))
                .Append(ML.Model.LoadTensorFlowModel(modelLocation).ScoreTensorFlowModel("Output", "Input"));

            TestEstimatorCore(pipe, data);

            using var model = pipe.Fit(data);
            var result = model.Transform(data);
            result.Schema.TryGetColumnIndex("Output", out int output);
            using (var cursor = result.GetRowCursor(result.Schema["Output"]))
            {
                var buffer = default(VBuffer<float>);
                var getter = cursor.GetGetter<VBuffer<float>>(result.Schema["Output"]);
                var numRows = 0;
                while (cursor.MoveNext())
                {
                    getter(ref buffer);
                    Assert.Equal(10, buffer.Length);
                    numRows += 1;
                }
                Assert.Equal(4, numRows);
            }
        }

        [TensorFlowFact]
        public void TreatOutputAsBatched()
        {
            var modelLocation = "cifar_model/frozen_model.pb";

            var mlContext = new MLContext(seed: 1);
            var imageHeight = 32;
            var imageWidth = 32;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = ML.Data.LoadFromTextFile(dataFile, new[] {
                new TextLoader.Column("imagePath", DataKind.String, 0),
                new TextLoader.Column("name", DataKind.String, 1)
            });

            // Note that CamelCase column names are there to match the TF graph node names.
            // Check and make sure save/load work correctly for the new TreatOutputAsBatched value.
            var pipe = ML.Transforms.LoadImages("Input", imageFolder, "imagePath")
                .Append(ML.Transforms.ResizeImages("Input", imageHeight, imageWidth))
                .Append(ML.Transforms.ExtractPixels("Input", interleavePixelColors: true))
                .Append(ML.Model.LoadTensorFlowModel(modelLocation, false).ScoreTensorFlowModel("Output", "Input"));

            TestEstimatorCore(pipe, data);
            var schema = pipe.Fit(data).Transform(data).Schema;

            // The dimensions of the output with treatOutputAsBatched set to false should be * 10
            // as the first dimension of -1 is treated as an unknown dimension.
            Assert.Equal(new VectorDataViewType(NumberDataViewType.Single, 0, 10), schema["Output"].Type);

            // Note that CamelCase column names are there to match the TF graph node names.
            // Test with TreatOutputAsBatched set to default value of true.
            pipe = ML.Transforms.LoadImages("Input", imageFolder, "imagePath")
                .Append(ML.Transforms.ResizeImages("Input", imageHeight, imageWidth))
                .Append(ML.Transforms.ExtractPixels("Input", interleavePixelColors: true))
                .Append(ML.Model.LoadTensorFlowModel(modelLocation).ScoreTensorFlowModel("Output", "Input"));

            TestEstimatorCore(pipe, data);
            schema = pipe.Fit(data).Transform(data).Schema;

            // The dimensions of the output with treatOutputAsBatched set to true should be 10
            // as the first dimension of -1 is treated as the batch dimension.
            Assert.Equal(new VectorDataViewType(NumberDataViewType.Single, 10), schema["Output"].Type);

        }

        [TensorFlowFact]
        public void TestTensorFlowWithSchema()
        {
            const string modelLocation = "cifar_model/frozen_model.pb";

            var mlContext = new MLContext(seed: 1);
            using var tensorFlowModel = TensorFlowUtils.LoadTensorFlowModel(mlContext, modelLocation);
            var schema = tensorFlowModel.GetInputSchema();
            Assert.True(schema.TryGetColumnIndex("Input", out int column));
            var type = (VectorDataViewType)schema[column].Type;
            var imageHeight = type.Dimensions[0];
            var imageWidth = type.Dimensions[1];

            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = ML.Data.LoadFromTextFile(dataFile, new[] {
                new TextLoader.Column("imagePath", DataKind.String, 0),
                new TextLoader.Column("name", DataKind.String, 1)
            });

            // Note that CamelCase column names are there to match the TF graph node names.
            var pipe = ML.Transforms.LoadImages("Input", imageFolder, "imagePath")
                .Append(ML.Transforms.ResizeImages("Input", imageHeight, imageWidth))
                .Append(ML.Transforms.ExtractPixels("Input", interleavePixelColors: true))
                .Append(tensorFlowModel.ScoreTensorFlowModel("Output", "Input"));

            TestEstimatorCore(pipe, data);

            using var model = pipe.Fit(data);
            var result = model.Transform(data);
            result.Schema.TryGetColumnIndex("Output", out int output);
            using (var cursor = result.GetRowCursor(result.Schema["Output"]))
            {
                var buffer = default(VBuffer<float>);
                var getter = cursor.GetGetter<VBuffer<float>>(result.Schema["Output"]);
                var numRows = 0;
                while (cursor.MoveNext())
                {
                    getter(ref buffer);
                    Assert.Equal(10, buffer.Length);
                    numRows += 1;
                }
                Assert.Equal(4, numRows);
            }
        }

        [TensorFlowFact]
        public void TestLoadMultipleModel()
        {
            var modelFile1 = "model_matmul/frozen_saved_model.pb";
            var modelFile2 = "cifar_model/frozen_model.pb";

            MLContext context = new MLContext(seed: 1);

            TensorFlowModel model1 = context.Model.LoadTensorFlowModel(modelFile1);
            TensorFlowModel model2 = context.Model.LoadTensorFlowModel(modelFile2);

            model1.ScoreTensorFlowModel(new[] { "c" }, new[] { "a", "b" });
            model2.ScoreTensorFlowModel("Output", "Input");
        }

        private void ValidateTensorFlowTransformer(IDataView result)
        {
            using (var cursor = result.GetRowCursorForAllColumns())
            {
                VBuffer<float> avalue = default;
                VBuffer<float> bvalue = default;
                VBuffer<float> cvalue = default;

                var aGetter = cursor.GetGetter<VBuffer<float>>(result.Schema["a"]);
                var bGetter = cursor.GetGetter<VBuffer<float>>(result.Schema["b"]);
                var cGetter = cursor.GetGetter<VBuffer<float>>(result.Schema["c"]);
                while (cursor.MoveNext())
                {
                    aGetter(ref avalue);
                    bGetter(ref bvalue);
                    cGetter(ref cvalue);
                    var aValues = avalue.GetValues();
                    var bValues = bvalue.GetValues();
                    var cValues = cvalue.GetValues();
                    Assert.Equal(aValues[0] * bValues[0] + aValues[1] * bValues[2], cValues[0]);
                    Assert.Equal(aValues[0] * bValues[1] + aValues[1] * bValues[3], cValues[1]);
                    Assert.Equal(aValues[2] * bValues[0] + aValues[3] * bValues[2], cValues[2]);
                    Assert.Equal(aValues[2] * bValues[1] + aValues[3] * bValues[3], cValues[3]);
                }
            }
        }
    }
}
