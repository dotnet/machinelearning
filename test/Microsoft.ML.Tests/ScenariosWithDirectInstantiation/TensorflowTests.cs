// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.TensorFlow;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        private class TestData
        {
            [VectorType(4)]
            public float[] a;
            [VectorType(4)]
            public float[] b;
        }

        [TensorFlowFact]
        public void TensorFlowTransformMatrixMultiplicationTest()
        {
            var modelLocation = "model_matmul/frozen_saved_model.pb";
            var mlContext = new MLContext(seed: 1);
            // Pipeline
            var loader = mlContext.Data.LoadFromEnumerable(
                    new List<TestData>(new TestData[] {
                        new TestData() { a = new[] { 1.0f, 2.0f,
                                                     3.0f, 4.0f },
                                         b = new[] { 1.0f, 2.0f,
                                                     3.0f, 4.0f } },
                        new TestData() { a = new[] { 2.0f, 2.0f,
                                                     2.0f, 2.0f },
                                         b = new[] { 3.0f, 3.0f,
                                                     3.0f, 3.0f } } }));
            var trans = mlContext.Model.LoadTensorFlowModel(modelLocation).ScoreTensorFlowModel(new[] { "c" }, new[] { "a", "b" }).Fit(loader).Transform(loader);

            using (var cursor = trans.GetRowCursorForAllColumns())
            {
                var cgetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[2]);
                Assert.True(cursor.MoveNext());
                VBuffer<float> c = default;
                cgetter(ref c);

                var cValues = c.GetValues();
                Assert.Equal(1.0 * 1.0 + 2.0 * 3.0, cValues[0]);
                Assert.Equal(1.0 * 2.0 + 2.0 * 4.0, cValues[1]);
                Assert.Equal(3.0 * 1.0 + 4.0 * 3.0, cValues[2]);
                Assert.Equal(3.0 * 2.0 + 4.0 * 4.0, cValues[3]);

                Assert.True(cursor.MoveNext());
                c = default;
                cgetter(ref c);

                cValues = c.GetValues();
                Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, cValues[0]);
                Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, cValues[1]);
                Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, cValues[2]);
                Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, cValues[3]);

                Assert.False(cursor.MoveNext());
            }
        }

        private class ShapeData
        {
            // Data will be passed as 1-D vector.
            // Intended data shape [5], model shape [None]
            [VectorType(5)]
            public float[] OneDim;

            // Data will be passed as flat vector.
            // Intended data shape [2,2], model shape [2, None]
            [VectorType(4)]
            public float[] TwoDim;

            // Data will be passed as 3-D vector.
            // Intended data shape [1, 2, 2], model shape [1, None, 2]
            [VectorType(1, 2, 2)]
            public float[] ThreeDim;

            // Data will be passed as flat vector.
            // Intended data shape [1, 2, 2, 3], model shape [1, None, None, 3]
            [VectorType(12)]
            public float[] FourDim;

            // Data will be passed as 4-D vector.
            // Intended data shape [2, 2, 2, 2], model shape [2, 2, 2, 2]
            [VectorType(2, 2, 2, 2)]
            public float[] FourDimKnown;
        }

        private List<ShapeData> GetShapeData()
        {
            return new List<ShapeData>(new ShapeData[] {
                        new ShapeData() {   OneDim = new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f },
                                            TwoDim = new[] { 1.0f, 2.0f, 3.0f, 4.0f },
                                            ThreeDim = new[] { 11.0f, 12.0f, 13.0f, 14.0f },
                                            FourDim = new[]{ 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f,
                                                             27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f },
                                            FourDimKnown = new[]{ 41.0f , 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
                                                                  49.0f , 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f}
                                        },
                        new ShapeData() {   OneDim = new[] { 100.1f, 100.2f, 100.3f, 100.4f, 100.5f },
                                            TwoDim = new[] { 101.0f, 102.0f, 103.0f, 104.0f },
                                            ThreeDim = new[] { 111.0f, 112.0f, 113.0f, 114.0f },
                                            FourDim = new[]{ 121.0f, 122.0f, 123.0f, 124.0f, 125.0f, 126.0f,
                                                             127.0f, 128.0f, 129.0f, 130.0f, 131.0f, 132.0f},
                                            FourDimKnown = new[]{ 141.0f , 142.0f, 143.0f, 144.0f, 145.0f, 146.0f, 147.0f, 148.0f,
                                                                  149.0f , 150.0f, 151.0f, 152.0f, 153.0f, 154.0f, 155.0f, 156.0f }
                                        }
                });
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowTransformInputShapeTest()
        {
            var modelLocation = "model_shape_test";
            var mlContext = new MLContext(seed: 1);
            var data = GetShapeData();
            // Pipeline
            var loader = mlContext.Data.LoadFromEnumerable(data);
            var inputs = new string[] { "OneDim", "TwoDim", "ThreeDim", "FourDim", "FourDimKnown" };
            var outputs = new string[] { "o_OneDim", "o_TwoDim", "o_ThreeDim", "o_FourDim", "o_FourDimKnown" };

            var trans = mlContext.Model.LoadTensorFlowModel(modelLocation).ScoreTensorFlowModel(outputs, inputs).Fit(loader).Transform(loader);

            using (var cursor = trans.GetRowCursorForAllColumns())
            {
                int outColIndex = 5;
                var oneDimgetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[outColIndex]);
                var twoDimgetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[outColIndex + 1]);
                var threeDimgetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[outColIndex + 2]);
                var fourDimgetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[outColIndex + 3]);
                var fourDimKnowngetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[outColIndex + 4]);

                VBuffer<float> oneDim = default;
                VBuffer<float> twoDim = default;
                VBuffer<float> threeDim = default;
                VBuffer<float> fourDim = default;
                VBuffer<float> fourDimKnown = default;
                foreach (var sample in data)
                {
                    Assert.True(cursor.MoveNext());

                    oneDimgetter(ref oneDim);
                    twoDimgetter(ref twoDim);
                    threeDimgetter(ref threeDim);
                    fourDimgetter(ref fourDim);
                    fourDimKnowngetter(ref fourDimKnown);

                    var oneDimValues = oneDim.GetValues();
                    Assert.Equal(sample.OneDim.Length, oneDimValues.Length);
                    Assert.True(oneDimValues.SequenceEqual(sample.OneDim));

                    var twoDimValues = twoDim.GetValues();
                    Assert.Equal(sample.TwoDim.Length, twoDimValues.Length);
                    Assert.True(twoDimValues.SequenceEqual(sample.TwoDim));

                    var threeDimValues = threeDim.GetValues();
                    Assert.Equal(sample.ThreeDim.Length, threeDimValues.Length);
                    Assert.True(threeDimValues.SequenceEqual(sample.ThreeDim));

                    var fourDimValues = fourDim.GetValues();
                    Assert.Equal(sample.FourDim.Length, fourDimValues.Length);
                    Assert.True(fourDimValues.SequenceEqual(sample.FourDim));

                    var fourDimKnownValues = fourDimKnown.GetValues();
                    Assert.Equal(sample.FourDimKnown.Length, fourDimKnownValues.Length);
                    Assert.True(fourDimKnownValues.SequenceEqual(sample.FourDimKnown));
                }
                Assert.False(cursor.MoveNext());
            }
        }

        private class TypesData
        {
            [VectorType(2)]
            public double[] f64;
            [VectorType(2)]
            public float[] f32;
            [VectorType(2)]
            public long[] i64;
            [VectorType(2)]
            public int[] i32;
            [VectorType(2)]
            public short[] i16;
            [VectorType(2)]
            public sbyte[] i8;
            [VectorType(2)]
            public ulong[] u64;
            [VectorType(2)]
            public uint[] u32;
            [VectorType(2)]
            public ushort[] u16;
            [VectorType(2)]
            public byte[] u8;
            [VectorType(2)]
            public bool[] b;
        }

        /// <summary>
        /// Test to ensure the supported datatypes can passed to TensorFlow .
        /// </summary>
        [TensorFlowFact]
        public void TensorFlowTransformInputOutputTypesTest()
        {
            // This an identity model which returns the same output as input.
            var model_location = "model_types_test";

            //Data
            var data = new List<TypesData>(
                        new TypesData[] {
                            new TypesData() {   f64 = new[] { -1.0, 2.0 },
                                                f32 = new[] { -1.0f, 2.0f },
                                                i64 = new[] { -1L, 2 },
                                                i32 = new[] { -1, 2 },
                                                i16 = new short[] { -1, 2 },
                                                i8 = new sbyte[] { -1, 2 },
                                                u64 = new ulong[] { 1, 2 },
                                                u32 = new uint[] { 1, 2 },
                                                u16 = new ushort[] { 1, 2 },
                                                u8 = new byte[] { 1, 2 },
                                                b = new bool[] { true, true },
                            },
                           new TypesData() {   f64 = new[] { -3.0, 4.0 },
                                                f32 = new[] { -3.0f, 4.0f },
                                                i64 = new[] { -3L, 4 },
                                                i32 = new[] { -3, 4 },
                                                i16 = new short[] { -3, 4 },
                                                i8 = new sbyte[] { -3, 4 },
                                                u64 = new ulong[] { 3, 4 },
                                                u32 = new uint[] { 3, 4 },
                                                u16 = new ushort[] { 3, 4 },
                                                u8 = new byte[] { 3, 4 },
                                                b = new bool[] { false, false },
                            } });

            var mlContext = new MLContext(seed: 1);
            // Pipeline

            var loader = mlContext.Data.LoadFromEnumerable(data);

            var inputs = new string[] { "f64", "f32", "i64", "i32", "i16", "i8", "u64", "u32", "u16", "u8", "b" };
            var outputs = new string[] { "o_f64", "o_f32", "o_i64", "o_i32", "o_i16", "o_i8", "o_u64", "o_u32", "o_u16", "o_u8", "o_b" };
            var trans = mlContext.Model.LoadTensorFlowModel(model_location).ScoreTensorFlowModel(outputs, inputs).Fit(loader).Transform(loader); ;

            using (var cursor = trans.GetRowCursorForAllColumns())
            {
                var f64getter = cursor.GetGetter<VBuffer<double>>(cursor.Schema[11]);
                var f32getter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[12]);
                var i64getter = cursor.GetGetter<VBuffer<long>>(cursor.Schema[13]);
                var i32getter = cursor.GetGetter<VBuffer<int>>(cursor.Schema[14]);
                var i16getter = cursor.GetGetter<VBuffer<short>>(cursor.Schema[15]);
                var i8getter = cursor.GetGetter<VBuffer<sbyte>>(cursor.Schema[16]);
                var u64getter = cursor.GetGetter<VBuffer<ulong>>(cursor.Schema[17]);
                var u32getter = cursor.GetGetter<VBuffer<uint>>(cursor.Schema[18]);
                var u16getter = cursor.GetGetter<VBuffer<ushort>>(cursor.Schema[19]);
                var u8getter = cursor.GetGetter<VBuffer<byte>>(cursor.Schema[20]);
                var boolgetter = cursor.GetGetter<VBuffer<bool>>(cursor.Schema[21]);


                VBuffer<double> f64 = default;
                VBuffer<float> f32 = default;
                VBuffer<long> i64 = default;
                VBuffer<int> i32 = default;
                VBuffer<short> i16 = default;
                VBuffer<sbyte> i8 = default;
                VBuffer<ulong> u64 = default;
                VBuffer<uint> u32 = default;
                VBuffer<ushort> u16 = default;
                VBuffer<byte> u8 = default;
                VBuffer<bool> b = default;
                foreach (var sample in data)
                {
                    Assert.True(cursor.MoveNext());

                    f64getter(ref f64);
                    f32getter(ref f32);
                    i64getter(ref i64);
                    i32getter(ref i32);
                    i16getter(ref i16);
                    i8getter(ref i8);
                    u64getter(ref u64);
                    u32getter(ref u32);
                    u16getter(ref u16);
                    u8getter(ref u8);
                    u8getter(ref u8);
                    boolgetter(ref b);

                    var f64Values = f64.GetValues();
                    Assert.Equal(2, f64Values.Length);
                    Assert.True(f64Values.SequenceEqual(sample.f64));
                    var f32Values = f32.GetValues();
                    Assert.Equal(2, f32Values.Length);
                    Assert.True(f32Values.SequenceEqual(sample.f32));
                    var i64Values = i64.GetValues();
                    Assert.Equal(2, i64Values.Length);
                    Assert.True(i64Values.SequenceEqual(sample.i64));
                    var i32Values = i32.GetValues();
                    Assert.Equal(2, i32Values.Length);
                    Assert.True(i32Values.SequenceEqual(sample.i32));
                    var i16Values = i16.GetValues();
                    Assert.Equal(2, i16Values.Length);
                    Assert.True(i16Values.SequenceEqual(sample.i16));
                    var i8Values = i8.GetValues();
                    Assert.Equal(2, i8Values.Length);
                    Assert.True(i8Values.SequenceEqual(sample.i8));
                    var u64Values = u64.GetValues();
                    Assert.Equal(2, u64Values.Length);
                    Assert.True(u64Values.SequenceEqual(sample.u64));
                    var u32Values = u32.GetValues();
                    Assert.Equal(2, u32Values.Length);
                    Assert.True(u32Values.SequenceEqual(sample.u32));
                    var u16Values = u16.GetValues();
                    Assert.Equal(2, u16Values.Length);
                    Assert.True(u16Values.SequenceEqual(sample.u16));
                    var u8Values = u8.GetValues();
                    Assert.Equal(2, u8Values.Length);
                    Assert.True(u8Values.SequenceEqual(sample.u8));
                    var bValues = b.GetValues();
                    Assert.Equal(2, bValues.Length);
                    Assert.True(bValues.SequenceEqual(sample.b));
                }
                Assert.False(cursor.MoveNext());
            }
        }

        [Fact(Skip = "Model files are not available yet")]
        public void TensorFlowTransformObjectDetectionTest()
        {
            var modelLocation = @"C:\models\TensorFlow\ssd_mobilenet_v1_coco_2018_01_28\frozen_inference_graph.pb";
            var mlContext = new MLContext(seed: 1);
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = mlContext.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", 32, 32, "ImageReal").Transform(images);

            var pixels = mlContext.Transforms.ExtractPixels("image_tensor", "ImageCropped", outputAsFloatArray: false).Fit(cropped).Transform(cropped);
            var tf = mlContext.Model.LoadTensorFlowModel(modelLocation).ScoreTensorFlowModel(
                new[] { "detection_boxes", "detection_scores", "num_detections", "detection_classes" }, new[] { "image_tensor" }).Fit(pixels).Transform(pixels);

            using (var curs = tf.GetRowCursor(tf.Schema["image_tensor"], tf.Schema["detection_boxes"], tf.Schema["detection_scores"], tf.Schema["detection_classes"], tf.Schema["num_detections"]))
            {
                var getInput = curs.GetGetter<VBuffer<byte>>(tf.Schema["image_tensor"]);
                var getBoxes = curs.GetGetter<VBuffer<float>>(tf.Schema["detection_boxes"]);
                var getScores = curs.GetGetter<VBuffer<float>>(tf.Schema["detection_scores"]);
                var getNum = curs.GetGetter<VBuffer<float>>(tf.Schema["num_detections"]);
                var getClasses = curs.GetGetter<VBuffer<float>>(tf.Schema["detection_classes"]);
                var buffer = default(VBuffer<float>);
                var inputBuffer = default(VBuffer<byte>);
                while (curs.MoveNext())
                {
                    getInput(ref inputBuffer);
                    getBoxes(ref buffer);
                    getScores(ref buffer);
                    getNum(ref buffer);
                    getClasses(ref buffer);
                }
            }
        }

        [Fact(Skip = "Model files are not available yet")]
        public void TensorFlowTransformInceptionTest()
        {
            string inputName = "input";
            string outputName = "softmax2_pre_activation";
            var modelLocation = @"inception5h\tensorflow_inception_graph.pb";
            var mlContext = new MLContext(seed: 1);
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var reader = mlContext.Data.CreateTextLoader(
                   columns: new[]
                   {
                        new TextLoader.Column("ImagePath", DataKind.String , 0),
                        new TextLoader.Column("Name", DataKind.String, 1)

                   },
               hasHeader: false,
               allowSparse: false
               );

            var data = reader.Load(new MultiFileSource(dataFile));
            var images = mlContext.Transforms.LoadImages(imageFolder, ("ImageReal", "ImagePath")).Fit(data).Transform(data);
            var cropped = mlContext.Transforms.ResizeImages("ImageCropped", 224, 224, "ImageReal").Fit(images).Transform(images);
            var pixels = mlContext.Transforms.ExtractPixels(inputName, "ImageCropped", interleavePixelColors: true).Fit(cropped).Transform(cropped);
            var tf = mlContext.Model.LoadTensorFlowModel(modelLocation).ScoreTensorFlowModel(outputName, inputName, true).Fit(pixels).Transform(pixels);

            tf.Schema.TryGetColumnIndex(inputName, out int input);
            tf.Schema.TryGetColumnIndex(outputName, out int b);
            using (var curs = tf.GetRowCursor(tf.Schema[inputName], tf.Schema[outputName]))
            {
                var get = curs.GetGetter<VBuffer<float>>(tf.Schema["softmax2_pre_activation"]);
                var getInput = curs.GetGetter<VBuffer<float>>(tf.Schema["input"]);
                var buffer = default(VBuffer<float>);
                var inputBuffer = default(VBuffer<float>);
                while (curs.MoveNext())
                {
                    getInput(ref inputBuffer);
                    get(ref buffer);
                }
            }
        }

        [TensorFlowFact]
        public void TensorFlowInputsOutputsSchemaTest()
        {
            var mlContext = new MLContext(seed: 1);
            var model_location = "mnist_model/frozen_saved_model.pb";
            var schema = TensorFlowUtils.GetModelSchema(mlContext, model_location);
            Assert.Equal(86, schema.Count);
            Assert.True(schema.TryGetColumnIndex("Placeholder", out int col));
            var type = (VectorType)schema[col].Type;
            Assert.Equal(2, type.Dimensions.Length);
            Assert.Equal(28, type.Dimensions[0]);
            Assert.Equal(28, type.Dimensions[1]);
            var metadataType = schema[col].Annotations.Schema[TensorFlowUtils.TensorflowOperatorTypeKind].Type;
            Assert.NotNull(metadataType);
            Assert.True(metadataType is TextDataViewType);
            ReadOnlyMemory<char> opType = default;
            schema[col].Annotations.GetValue(TensorFlowUtils.TensorflowOperatorTypeKind, ref opType);
            Assert.Equal("Placeholder", opType.ToString());
            metadataType = schema[col].Annotations.Schema.GetColumnOrNull(TensorFlowUtils.TensorflowUpstreamOperatorsKind)?.Type;
            Assert.Null(metadataType);

            Assert.True(schema.TryGetColumnIndex("conv2d/Conv2D/ReadVariableOp", out col));
            type = (VectorType)schema[col].Type;
            Assert.Equal(new[] { 5, 5, 1, 32 }, type.Dimensions);
            metadataType = schema[col].Annotations.Schema[TensorFlowUtils.TensorflowOperatorTypeKind].Type;
            Assert.NotNull(metadataType);
            Assert.True(metadataType is TextDataViewType);
            schema[col].Annotations.GetValue(TensorFlowUtils.TensorflowOperatorTypeKind, ref opType);
            Assert.Equal("Identity", opType.ToString());
            metadataType = schema[col].Annotations.Schema[TensorFlowUtils.TensorflowUpstreamOperatorsKind].Type;
            Assert.NotNull(metadataType);
            VBuffer<ReadOnlyMemory<char>> inputOps = default;
            schema[col].Annotations.GetValue(TensorFlowUtils.TensorflowUpstreamOperatorsKind, ref inputOps);
            Assert.Equal(1, inputOps.Length);
            Assert.Equal("conv2d/kernel", inputOps.GetValues()[0].ToString());

            Assert.True(schema.TryGetColumnIndex("conv2d/Conv2D", out col));
            type = (VectorType)schema[col].Type;
            Assert.Equal(new[] { 28, 28, 32 }, type.Dimensions);
            metadataType = schema[col].Annotations.Schema[TensorFlowUtils.TensorflowOperatorTypeKind].Type;
            Assert.NotNull(metadataType);
            Assert.True(metadataType is TextDataViewType);
            schema[col].Annotations.GetValue(TensorFlowUtils.TensorflowOperatorTypeKind, ref opType);
            Assert.Equal("Conv2D", opType.ToString());
            metadataType = schema[col].Annotations.Schema[TensorFlowUtils.TensorflowUpstreamOperatorsKind].Type;
            Assert.NotNull(metadataType);
            schema[col].Annotations.GetValue(TensorFlowUtils.TensorflowUpstreamOperatorsKind, ref inputOps);
            Assert.Equal(2, inputOps.Length);
            Assert.Equal("reshape/Reshape", inputOps.GetValues()[0].ToString());
            Assert.Equal("conv2d/Conv2D/ReadVariableOp", inputOps.GetValues()[1].ToString());

            Assert.True(schema.TryGetColumnIndex("Softmax", out col));
            type = (VectorType)schema[col].Type;
            Assert.Equal(new[] { 10 }, type.Dimensions);
            metadataType = schema[col].Annotations.Schema[TensorFlowUtils.TensorflowOperatorTypeKind].Type;
            Assert.NotNull(metadataType);
            Assert.True(metadataType is TextDataViewType);
            schema[col].Annotations.GetValue(TensorFlowUtils.TensorflowOperatorTypeKind, ref opType);
            Assert.Equal("Softmax", opType.ToString());
            metadataType = schema[col].Annotations.Schema[TensorFlowUtils.TensorflowUpstreamOperatorsKind].Type;
            Assert.NotNull(metadataType);
            schema[col].Annotations.GetValue(TensorFlowUtils.TensorflowUpstreamOperatorsKind, ref inputOps);
            Assert.Equal(1, inputOps.Length);
            Assert.Equal("sequential/dense_1/BiasAdd", inputOps.GetValues()[0].ToString());

            model_location = "model_matmul/frozen_saved_model.pb";
            schema = TensorFlowUtils.GetModelSchema(mlContext, model_location);
            char name = 'a';
            for (int i = 0; i < schema.Count; i++)
            {
                Assert.Equal(name.ToString(), schema[i].Name);
                type = (VectorType)schema[i].Type;
                Assert.Equal(new[] { 2, 2 }, type.Dimensions);
                name++;
            }
        }

        [TensorFlowFact]
        public void TensorFlowTransformMNISTConvTest()
        {
            var mlContext = new MLContext(seed: 1);
            var reader = mlContext.Data.CreateTextLoader(
                    columns: new[]
                    {
                        new TextLoader.Column("Label", DataKind.UInt32 , new [] { new TextLoader.Range(0) }, new KeyCount(10)),
                        new TextLoader.Column("Placeholder", DataKind.Single, new []{ new TextLoader.Range(1, 784) })

                    },
                hasHeader: true,
                allowSparse: true
                );

            var trainData = reader.Load(GetDataPath(TestDatasets.mnistTiny28.trainFilename));
            var testData = reader.Load(GetDataPath(TestDatasets.mnistOneClass.testFilename));

            var pipe = mlContext.Transforms.CopyColumns(("reshape_input", "Placeholder"))
                .Append(mlContext.Model.LoadTensorFlowModel("mnist_model/frozen_saved_model.pb").ScoreTensorFlowModel(new[] { "Softmax", "dense/Relu" }, new[] { "Placeholder", "reshape_input" }))
                .Append(mlContext.Transforms.Concatenate("Features", "Softmax", "dense/Relu"))
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm("Label", "Features"));

            var trainedModel = pipe.Fit(trainData);
            var predicted = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted);

            Assert.Equal(0.99, metrics.MicroAccuracy, 2);
            Assert.Equal(1.0, metrics.MacroAccuracy, 2);

            var oneSample = GetOneMNISTExample();

            var predictFunction = mlContext.Model.CreatePredictionEngine<MNISTData, MNISTPrediction>(trainedModel);

            var onePrediction = predictFunction.Predict(oneSample);

            Assert.Equal(5, GetMaxIndexForOnePrediction(onePrediction));
        }

        [TensorFlowFact]
        public void TensorFlowTransformMNISTLRTrainingTest()
        {
            const double expectedMicroAccuracy = 0.72173913043478266;
            const double expectedMacroAccruacy = 0.67482993197278918;
            var model_location = "mnist_lr_model";
            try
            {
                var mlContext = new MLContext(seed: 1);
                var reader = mlContext.Data.CreateTextLoader(columns: new[]
                    {
                        new TextLoader.Column("Label", DataKind.Int64, 0),
                        new TextLoader.Column("Placeholder", DataKind.Single, new []{ new TextLoader.Range(1, 784) })
                    },
                    allowSparse: true
                );

                var trainData = reader.Load(GetDataPath(TestDatasets.mnistTiny28.trainFilename));
                var testData = reader.Load(GetDataPath(TestDatasets.mnistOneClass.testFilename));

                var pipe = mlContext.Transforms.Categorical.OneHotEncoding("OneHotLabel", "Label")
                    .Append(mlContext.Transforms.Normalize(new NormalizingEstimator.MinMaxColumnOptions("Features", "Placeholder")))
                    .Append(mlContext.Model.LoadTensorFlowModel(model_location).RetrainTensorFlowModel(
                        inputColumnNames: new[] { "Features" },
                        outputColumnNames: new[] { "Prediction", "b" },
                        labelColumnName: "OneHotLabel",
                        tensorFlowLabel: "Label",
                        optimizationOperation: "SGDOptimizer",
                        lossOperation: "Loss",
                        epoch: 10,
                        learningRateOperation: "SGDOptimizer/learning_rate",
                        learningRate: 0.001f,
                        batchSize: 20))
                    .Append(mlContext.Transforms.Concatenate("Features", "Prediction"))
                    .Append(mlContext.Transforms.Conversion.MapValueToKey("KeyLabel", "Label", maximumNumberOfKeys: 10))
                    .Append(mlContext.MulticlassClassification.Trainers.LightGbm("KeyLabel", "Features"));

                var trainedModel = pipe.Fit(trainData);
                var predicted = trainedModel.Transform(testData);
                var metrics = mlContext.MulticlassClassification.Evaluate(predicted, labelColumnName: "KeyLabel");
                Assert.InRange(metrics.MicroAccuracy, expectedMicroAccuracy, 1);
                Assert.InRange(metrics.MacroAccuracy, expectedMacroAccruacy, 1);
                var predictionFunction = mlContext.Model.CreatePredictionEngine<MNISTData, MNISTPrediction>(trainedModel);

                var oneSample = GetOneMNISTExample();
                var onePrediction = predictionFunction.Predict(oneSample);
                Assert.Equal(0, GetMaxIndexForOnePrediction(onePrediction));


                var trainDataTransformed = trainedModel.Transform(trainData);
                using (var cursor = trainDataTransformed.GetRowCursorForAllColumns())
                {
                    var getter = cursor.GetGetter<VBuffer<float>>(trainDataTransformed.Schema["b"]);
                    if (cursor.MoveNext())
                    {
                        var trainedBias = default(VBuffer<float>);
                        getter(ref trainedBias);
                        Assert.NotEqual(trainedBias.GetValues().ToArray(), new float[] { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f });
                    }
                }
            }
            finally
            {
                // This test changes the state of the model.
                // Cleanup folder so that other test can also use the same model.
                CleanUp(model_location);
            }
        }

        private void CleanUp(string model_location)
        {
            var directories = Directory.GetDirectories(model_location, "variables-*");
            if (directories != null && directories.Length > 0)
            {
                var varDir = Path.Combine(model_location, "variables");
                if (Directory.Exists(varDir))
                    Directory.Delete(varDir, true);
                Directory.Move(directories[0], varDir);
            }
        }

        [TensorFlowFact]
        public void TensorFlowTransformMNISTConvTrainingTest()
        {
            double expectedMicro = 0.73304347826086956;
            double expectedMacro = 0.677551020408163;

            ExecuteTFTransformMNISTConvTrainingTest(false, null, expectedMicro, expectedMacro);
            ExecuteTFTransformMNISTConvTrainingTest(true, 5, expectedMicro, expectedMacro);
        }

        private void ExecuteTFTransformMNISTConvTrainingTest(bool shuffle, int? shuffleSeed, double expectedMicroAccuracy, double expectedMacroAccuracy)
        {
            const string modelLocation = "mnist_conv_model";
            try
            {
                var mlContext = new MLContext(seed: 1);

                var reader = mlContext.Data.CreateTextLoader(new[]
                    {
                        new TextLoader.Column("Label", DataKind.UInt32, new []{ new TextLoader.Range(0) }, new KeyCount(10)),
                        new TextLoader.Column("TfLabel", DataKind.Int64, 0),
                        new TextLoader.Column("Placeholder", DataKind.Single, new []{ new TextLoader.Range(1, 784) })
                    },
                    allowSparse: true
                );

                var trainData = reader.Load(GetDataPath(TestDatasets.mnistTiny28.trainFilename));
                var testData = reader.Load(GetDataPath(TestDatasets.mnistOneClass.testFilename));

                IDataView preprocessedTrainData = null;
                IDataView preprocessedTestData = null;
                if (shuffle)
                {
                    // Shuffle training data set
                    preprocessedTrainData = new RowShufflingTransformer(mlContext, new RowShufflingTransformer.Options()
                    {
                        ForceShuffle = shuffle,
                        ForceShuffleSeed = shuffleSeed
                    }, trainData);

                    // Shuffle test data set
                    preprocessedTestData = new RowShufflingTransformer(mlContext, new RowShufflingTransformer.Options()
                    {
                        ForceShuffle = shuffle,
                        ForceShuffleSeed = shuffleSeed
                    }, testData);
                }
                else
                {
                    preprocessedTrainData = trainData;
                    preprocessedTestData = testData;
                }

                var pipe = mlContext.Transforms.CopyColumns(("Features", "Placeholder"))
                    .Append(mlContext.Model.LoadTensorFlowModel(modelLocation).RetrainTensorFlowModel(
                        inputColumnNames: new[] { "Features" },
                        outputColumnNames: new[] { "Prediction" },
                        labelColumnName: "TfLabel",
                        tensorFlowLabel: "Label",
                        optimizationOperation: "MomentumOp",
                        lossOperation: "Loss",
                        metricOperation: "Accuracy",
                        epoch: 10,
                        learningRateOperation: "learning_rate",
                        learningRate: 0.01f,
                        batchSize: 20))
                    .Append(mlContext.Transforms.Concatenate("Features", "Prediction"))
                    .AppendCacheCheckpoint(mlContext)
                    .Append(mlContext.MulticlassClassification.Trainers.LightGbm(new Trainers.LightGbm.LightGbmMulticlassTrainer.Options()
                    {
                        LabelColumnName = "Label",
                        FeatureColumnName = "Features",
                        Seed = 1,
                        NumberOfThreads = 1,
                        NumberOfIterations = 1
                    }));

                var trainedModel = pipe.Fit(preprocessedTrainData);
                var predicted = trainedModel.Transform(preprocessedTestData);
                var metrics = mlContext.MulticlassClassification.Evaluate(predicted);
                Assert.InRange(metrics.MicroAccuracy, expectedMicroAccuracy - 0.1, expectedMicroAccuracy + 0.1);
                Assert.InRange(metrics.MacroAccuracy, expectedMacroAccuracy - 0.1, expectedMacroAccuracy + 0.1);

                // Create prediction function and test prediction
                var predictFunction = mlContext.Model.CreatePredictionEngine<MNISTData, MNISTPrediction>(trainedModel);

                var oneSample = GetOneMNISTExample();

                var prediction = predictFunction.Predict(oneSample);

                Assert.Equal(2, GetMaxIndexForOnePrediction(prediction));
            }
            finally
            {
                // This test changes the state of the model.
                // Cleanup folder so that other test can also use the same model.
                CleanUp(modelLocation);
            }
        }

        [TensorFlowFact]
        public void TensorFlowTransformMNISTConvSavedModelTest()
        {
            // This test trains a multi-class classifier pipeline where a pre-trained Tenroflow model is used for featurization.
            // Two group of test criteria are checked. One group contains micro and macro accuracies. The other group is the range
            // of predicted label of a single in-memory example.

            var mlContext = new MLContext(seed: 1);
            var reader = mlContext.Data.CreateTextLoader(columns: new[]
                {
                    new TextLoader.Column("Label", DataKind.UInt32 , new [] { new TextLoader.Range(0) }, new KeyCount(10)),
                    new TextLoader.Column("Placeholder", DataKind.Single, new []{ new TextLoader.Range(1, 784) })
                },
                hasHeader: true,
                allowSparse: true
            );

            var trainData = reader.Load(GetDataPath(TestDatasets.mnistTiny28.trainFilename));
            var testData = reader.Load(GetDataPath(TestDatasets.mnistOneClass.testFilename));

            var pipe = mlContext.Transforms.CopyColumns(("reshape_input", "Placeholder"))
                .Append(mlContext.Model.LoadTensorFlowModel("mnist_model").ScoreTensorFlowModel(new[] { "Softmax", "dense/Relu" }, new[] { "Placeholder", "reshape_input" }))
                .Append(mlContext.Transforms.Concatenate("Features", new[] { "Softmax", "dense/Relu" }))
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm("Label", "Features"));

            var trainedModel = pipe.Fit(trainData);
            var predicted = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted);

            // First group of checks
            Assert.Equal(0.99, metrics.MicroAccuracy, 2);
            Assert.Equal(1.0, metrics.MacroAccuracy, 2);

            // An in-memory example. Its label is predicted below.
            var oneSample = GetOneMNISTExample();

            var predictFunction = mlContext.Model.CreatePredictionEngine<MNISTData, MNISTPrediction>(trainedModel);

            var onePrediction = predictFunction.Predict(oneSample);

            // Second group of checks
            Assert.Equal(5, GetMaxIndexForOnePrediction(onePrediction));
        }

        private MNISTData GetOneMNISTExample()
        {
            return new MNISTData()
            {
                Placeholder = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126,
                136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172,
                253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238,
                253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56,
                39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253,
                253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43,
                154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
                190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81,
                240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130,
                183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221,
                253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219,
                253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133,
                11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136,
                253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0 }
            };
        }

        private int GetMaxIndexForOnePrediction(MNISTPrediction onePrediction)
        {
            float maxLabel = -1;
            int maxIndex = -1;
            for (int i = 0; i < onePrediction.PredictedLabels.Length; i++)
            {
                if (onePrediction.PredictedLabels[i] > maxLabel)
                {
                    maxLabel = onePrediction.PredictedLabels[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        public class MNISTData
        {
            public long Label;

            [VectorType(784)]
            public float[] Placeholder;
        }

        public class MNISTPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }

        [TensorFlowFact]
        public void TensorFlowTransformCifar()
        {
            var modelLocation = "cifar_model/frozen_model.pb";

            var mlContext = new MLContext(seed: 1);
            var tensorFlowModel = mlContext.Model.LoadTensorFlowModel(modelLocation);
            var schema = tensorFlowModel.GetInputSchema();
            Assert.True(schema.TryGetColumnIndex("Input", out int column));
            var type = (VectorType)schema[column].Type;
            var imageHeight = type.Dimensions[0];
            var imageWidth = type.Dimensions[1];

            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = mlContext.Data.LoadFromTextFile(dataFile,
                columns: new[]
                    {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                    }
                );

            var pipeEstimator = new ImageLoadingEstimator(mlContext, imageFolder, ("ImageReal", "ImagePath"))
                .Append(new ImageResizingEstimator(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal"))
                .Append(new ImagePixelExtractingEstimator(mlContext, "Input", "ImageCropped", interleavePixelColors: true));

            var pixels = pipeEstimator.Fit(data).Transform(data);
            IDataView trans = tensorFlowModel.ScoreTensorFlowModel("Output", "Input").Fit(pixels).Transform(pixels);

            trans.Schema.TryGetColumnIndex("Output", out int output);
            using (var cursor = trans.GetRowCursor(trans.Schema["Output"]))
            {
                var buffer = default(VBuffer<float>);
                var getter = cursor.GetGetter<VBuffer<float>>(trans.Schema["Output"]);
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
        public void TensorFlowTransformCifarSavedModel()
        {
            var modelLocation = "cifar_saved_model";
            var mlContext = new MLContext(seed: 1);
            var tensorFlowModel = mlContext.Model.LoadTensorFlowModel(modelLocation);
            var schema = tensorFlowModel.GetInputSchema();
            Assert.True(schema.TryGetColumnIndex("Input", out int column));
            var type = (VectorType)schema[column].Type;
            var imageHeight = type.Dimensions[0];
            var imageWidth = type.Dimensions[1];

            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = mlContext.Data.LoadFromTextFile(dataFile, columns: new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                }
            );
            var images = mlContext.Transforms.LoadImages(imageFolder, ("ImageReal", "ImagePath")).Fit(data).Transform(data);
            var cropped = mlContext.Transforms.ResizeImages("ImageCropped", imageWidth, imageHeight, "ImageReal").Fit(images).Transform(images);
            var pixels = mlContext.Transforms.ExtractPixels("Input", "ImageCropped", interleavePixelColors: true).Fit(cropped).Transform(cropped);
            IDataView trans = tensorFlowModel.ScoreTensorFlowModel("Output", "Input").Fit(pixels).Transform(pixels);

            using (var cursor = trans.GetRowCursorForAllColumns())
            {
                var buffer = default(VBuffer<float>);
                var getter = cursor.GetGetter<VBuffer<float>>(trans.Schema["Output"]);
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

        // This test has been created as result of https://github.com/dotnet/machinelearning/issues/2156.
        [TensorFlowFact]
        public void TensorFlowGettingSchemaMultipleTimes()
        {
            var modelLocation = "cifar_saved_model";
            var mlContext = new MLContext(seed: 1);
            for (int i = 0; i < 10; i++)
            {
                var schema = TensorFlowUtils.GetModelSchema(mlContext, modelLocation);
                Assert.NotNull(schema);
            }
        }


        [TensorFlowFact]
        public void TensorFlowTransformCifarInvalidShape()
        {
            var modelLocation = "cifar_model/frozen_model.pb";

            var mlContext = new MLContext(seed: 1);
            var imageHeight = 28;
            var imageWidth = 28;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = mlContext.Data.LoadFromTextFile(dataFile,
                columns: new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.String, 0),
                        new TextLoader.Column("Name", DataKind.String, 1),
                }
            );
            var images = new ImageLoadingTransformer(mlContext, imageFolder, ("ImageReal", "ImagePath")).Transform(data);
            var cropped = new ImageResizingTransformer(mlContext, "ImageCropped", imageWidth, imageHeight, "ImageReal").Transform(images);
            var pixels = new ImagePixelExtractingTransformer(mlContext, "Input", "ImageCropped").Transform(cropped);

            var thrown = false;
            try
            {
                IDataView trans = mlContext.Model.LoadTensorFlowModel(modelLocation).ScoreTensorFlowModel("Output", "Input").Fit(pixels).Transform(pixels);
            }
            catch
            {
                thrown = true;
            }
            Assert.True(thrown);
        }

        /// <summary>
        /// Class to hold features and predictions.
        /// </summary>
        public class TensorFlowSentiment
        {
            public string Sentiment_Text;
            [VectorType(600)]
            public int[] Features;
            [VectorType(2)]
            public float[] Prediction;
        }

        [TensorFlowFact]
        public void TensorFlowSentimentClassificationTest()
        {
            var mlContext = new MLContext(seed: 1);
            var data = new[] { new TensorFlowSentiment() { Sentiment_Text = "this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert  is an amazing actor and now the same being director  father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for  and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also  to the two little boy's that played the  of norman and paul they were just brilliant children are often left out of the  list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all" } };
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            var lookupMap = mlContext.Data.LoadFromTextFile(@"sentiment_model/imdb_word_index.csv",
                columns: new[]
                   {
                        new TextLoader.Column("Words", DataKind.String, 0),
                        new TextLoader.Column("Ids", DataKind.Int32, 1),
                   },
                separatorChar: ','
               );

            // We cannot resize variable length vector to fixed length vector in ML.NET
            // The trick here is to create two pipelines.
            // The first pipeline 'dataPipe' tokenzies the string into words and maps each word to an integer which is an index in the dictionary.
            // Then this integer vector is retrieved from the pipeline and resized to fixed length.
            // The second pipeline 'tfEnginePipe' takes the resized integer vector and passes it to TensoFlow and gets the classification scores.
            var estimator = mlContext.Transforms.Text.TokenizeIntoWords("TokenizedWords", "Sentiment_Text")
                .Append(mlContext.Transforms.Conversion.MapValue(lookupMap, lookupMap.Schema["Words"], lookupMap.Schema["Ids"], 
                    new ColumnOptions[] { ("Features", "TokenizedWords") }));
            var model = estimator.Fit(dataView);
            var dataPipe = mlContext.Model.CreatePredictionEngine<TensorFlowSentiment, TensorFlowSentiment>(model);

            // For explanation on how was the `sentiment_model` created 
            // c.f. https://github.com/dotnet/machinelearning-testdata/blob/master/Microsoft.ML.TensorFlow.TestModels/sentiment_model/README.md
            string modelLocation = @"sentiment_model";
            var pipelineModel = mlContext.Model.LoadTensorFlowModel(modelLocation).ScoreTensorFlowModel(new[] { "Prediction/Softmax" }, new[] { "Features" })
                .Append(mlContext.Transforms.CopyColumns(("Prediction", "Prediction/Softmax")))
                .Fit(dataView);
            var tfEnginePipe = mlContext.Model.CreatePredictionEngine<TensorFlowSentiment, TensorFlowSentiment>(pipelineModel);

            var processedData = dataPipe.Predict(data[0]);
            Array.Resize(ref processedData.Features, 600);
            var prediction = tfEnginePipe.Predict(processedData);

            Assert.Equal(2, prediction.Prediction.Length);
            Assert.InRange(prediction.Prediction[1], 0.650032759 - 0.01, 0.650032759 + 0.01);
        }

        class TextInput
        {
            [LoadColumn(0, 1)]
            [VectorType(2)]
            public string[] A; // Whatever is passed in 'TextInput.A' will be returned as-is in 'TextOutput.AOut'

            [LoadColumn(2, 4)]
            [VectorType(3)]
            public string[] B; // Whatever is passed in 'TextInput.B' will be split on '/' and joined using ' ' and returned in 'TextOutput.BOut'
        }

        class TextOutput
        {
            [VectorType(2)]
            public string[] AOut { get; set; }

            [VectorType(1)]
            public string[] BOut { get; set; }
        }

        [TensorFlowFact]
        public void TensorFlowStringTest()
        {
            var mlContext = new MLContext(seed: 1);
            var tensorFlowModel = mlContext.Model.LoadTensorFlowModel(@"model_string_test");
            var schema = tensorFlowModel.GetModelSchema();
            Assert.True(schema.TryGetColumnIndex("A", out var colIndex));
            Assert.True(schema.TryGetColumnIndex("B", out colIndex));

            var dataview = mlContext.Data.CreateTextLoader<TextInput>().Load(new MultiFileSource(null));

            var pipeline = tensorFlowModel.ScoreTensorFlowModel(new[] { "Original_A", "Joined_Splited_Text" }, new[] { "A", "B" })
                .Append(mlContext.Transforms.CopyColumns(("AOut", "Original_A"), ("BOut", "Joined_Splited_Text")));
            var transformer = mlContext.Model.CreatePredictionEngine<TextInput, TextOutput>(pipeline.Fit(dataview));

            var input = new TextInput
            {
                A = new[] { "This is fine.", "That's ok." },
                B = new[] { "Thank/you/very/much!.", "I/am/grateful/to/you.", "So/nice/of/you." }
            };
            var textOutput = transformer.Predict(input);

            for (int i = 0; i < input.A.Length; i++)
                Assert.Equal(input.A[i], textOutput.AOut[i]);
            Assert.Equal(string.Join(" ", input.B).Replace("/", " "), textOutput.BOut[0]);
        }
    }
}
