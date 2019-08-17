// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.RunTests;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Tools;
using Microsoft.ML.Transforms.Image;
using Xunit;
using Xunit.Abstractions;
using Microsoft.ML.Transforms.Onnx;

namespace Microsoft.ML.Tests
{
    public class OnnxTransformTests : TestDataPipeBase
    {
        private const int inputSize = 150528;

        private class TestData
        {
            [VectorType(inputSize)]
            public float[] data_0;
        }

        private class TestDataMulti
        {
            [VectorType(5)]
            public float[] ina;

            [VectorType(5)]
            public float[] inb;
        }

        private class TestDataMulti2By3
        {
            [VectorType(2, 3)]
            public float[] ina;

            [VectorType(2, 3)]
            public float[] inb;
        }

        private class TestDataSize
        {
            [VectorType(2)]
            public float[] data_0;
        }

        private class TestDataXY
        {
            [VectorType(inputSize)]
            public float[] A;
        }

        private class TestDataDifferntType
        {
            [VectorType(inputSize)]
            public string[] data_0;
        }

        private class TestDataUnknownDimensions
        {
            [VectorType(3)]
            public float[] input;
        }

        class PredictionUnknownDimensions
        {
            [VectorType(1)]
            public long[] argmax { get; set; }
        }

        private class InputWithCustomShape
        {
            [VectorType(3, 3)]
            public float[] input;
        }

        class PredictionWithCustomShape
        {
            [VectorType(3)]
            public long[] argmax { get; set; }
        }

        private float[] GetSampleArrayData()
        {
            var samplevector = new float[inputSize];
            for (int i = 0; i < inputSize; i++)
                samplevector[i] = (i / (inputSize * 1.01f));
            return samplevector;
        }

        public OnnxTransformTests(ITestOutputHelper output) : base(output)
        {
        }

        [OnnxFact]
        void TestSimpleCase()
        {
            var modelFile = "squeezenet/00000001/model.onnx";
            var samplevector = GetSampleArrayData();
            var dataView = ML.Data.LoadFromEnumerable(
                new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    },
                     new TestData()
                     {
                        data_0 = samplevector
                     }
                });

            var xyData = new List<TestDataXY> { new TestDataXY() { A = new float[inputSize] } };
            var stringData = new List<TestDataDifferntType> { new TestDataDifferntType() { data_0 = new string[inputSize] } };
            var sizeData = new List<TestDataSize> { new TestDataSize() { data_0 = new float[2] } };
            var pipe = ML.Transforms.ApplyOnnxModel(new[] { "softmaxout_1" }, new[] { "data_0" }, modelFile);

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

        [OnnxTheory]
        [InlineData(null, false)]
        [InlineData(null, true)]
        void TestOldSavingAndLoading(int? gpuDeviceId, bool fallbackToCpu)
        {
            var modelFile = "squeezenet/00000001/model.onnx";
            var samplevector = GetSampleArrayData();

            var dataView = ML.Data.LoadFromEnumerable(
                new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    }
                });

            var inputNames = new[] { "data_0" };
            var outputNames = new[] { "softmaxout_1" };
            var est = ML.Transforms.ApplyOnnxModel(outputNames, inputNames, modelFile, gpuDeviceId, fallbackToCpu);
            var transformer = est.Fit(dataView);
            var result = transformer.Transform(dataView);
            var resultRoles = new RoleMappedData(result);
            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(Env, Env.Start("saving"), ms, null, resultRoles);
                ms.Position = 0;
                var loadedView = ModelFileUtils.LoadTransforms(Env, dataView, ms);

                var sofMaxOut1Col = loadedView.Schema[outputNames[0]];

                using (var cursor = loadedView.GetRowCursor(sofMaxOut1Col))
                {
                    VBuffer<float> softMaxValue = default;
                    var softMaxGetter = cursor.GetGetter<VBuffer<float>>(sofMaxOut1Col);
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
                                Assert.InRange(val, 0.00004, 0.00005);
                            if (i == 1)
                                Assert.InRange(val, 0.003844, 0.003845);
                            if (i == 999)
                                Assert.InRange(val, 0.0029566, 0.0029567);
                            i++;
                        }
                    }
                    Assert.InRange(sum, 1.0, 1.00001);
                }
            }
        }

        [OnnxFact]
        public void OnnxWorkout()
        {
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet", "00000001", "model.onnx");

            var env = new MLContext();
            var imageHeight = 224;
            var imageWidth = 224;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);

            var data = ML.Data.LoadFromTextFile(dataFile, new[] {
                new TextLoader.Column("imagePath", DataKind.String, 0),
                new TextLoader.Column("name", DataKind.String, 1)
            });
            // Note that CamelCase column names are there to match the TF graph node names.
            var pipe = ML.Transforms.LoadImages("data_0", imageFolder, "imagePath")
                .Append(ML.Transforms.ResizeImages("data_0", imageHeight, imageWidth))
                .Append(ML.Transforms.ExtractPixels("data_0", interleavePixelColors: true))
                .Append(ML.Transforms.ApplyOnnxModel("softmaxout_1", "data_0", modelFile));

            TestEstimatorCore(pipe, data);

            var result = pipe.Fit(data).Transform(data);
            var softmaxOutCol = result.Schema["softmaxout_1"];

            using (var cursor = result.GetRowCursor(softmaxOutCol))
            {
                var buffer = default(VBuffer<float>);
                var getter = cursor.GetGetter<VBuffer<float>>(softmaxOutCol);
                var numRows = 0;
                while (cursor.MoveNext())
                {
                    getter(ref buffer);
                    Assert.Equal(1000, buffer.Length);
                    numRows += 1;
                }
                Assert.Equal(4, numRows);
            }
        }

        [OnnxFact]
        void TestCommandLine()
        {
            var x = Maml.Main(new[] { @"showschema loader=Text{col=data_0:R4:0-150527} xf=Onnx{InputColumns={data_0} OutputColumns={softmaxout_1} model={squeezenet/00000001/model.onnx}}" });
            Assert.Equal(0, x);
        }

        [OnnxFact]
        void TestCommandLineWithCustomShape()
        {
            var x = Maml.Main(new[] { @"showschema loader=Text{col=data_0:R4:0-150527} xf=Onnx{customShapeInfos={Name=data_0 Shape=1 Shape=3 Shape=224 Shape=224} InputColumns={data_0} OutputColumns={softmaxout_1} model={squeezenet/00000001/model.onnx}}" });
            Assert.Equal(0, x);
        }

        [OnnxFact]
        public void OnnxModelScenario()
        {
            var modelFile = "squeezenet/00000001/model.onnx";
            var env = new ConsoleEnvironment(seed: 1);
            var samplevector = GetSampleArrayData();

            var dataView = ML.Data.LoadFromEnumerable(
                new TestData[] {
                    new TestData()
                    {
                        data_0 = samplevector
                    }
                });

            var onnx = ML.Transforms.ApplyOnnxModel("softmaxout_1", "data_0", modelFile).Fit(dataView).Transform(dataView);
            var scoreCol = onnx.Schema["softmaxout_1"];

            using (var curs = onnx.GetRowCursor(scoreCol))
            {
                var getScores = curs.GetGetter<VBuffer<float>>(scoreCol);
                var buffer = default(VBuffer<float>);
                while (curs.MoveNext())
                {
                    getScores(ref buffer);
                    Assert.Equal(1000, buffer.Length);
                }
            }
        }

        [OnnxFact]
        public void OnnxModelMultiInput()
        {
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "twoinput", "twoinput.onnx");
            var env = new ConsoleEnvironment(seed: 1);
            var samplevector = GetSampleArrayData();

            var dataView = ML.Data.LoadFromEnumerable(
                new TestDataMulti[] {
                    new TestDataMulti()
                    {
                        ina = new float[] {1,2,3,4,5},
                        inb = new float[] {1,2,3,4,5}
                    }
                });
            var onnx = ML.Transforms.ApplyOnnxModel(new[] { "outa", "outb" }, new[] { "ina", "inb" }, modelFile).Fit(dataView).Transform(dataView);

            var outaCol = onnx.Schema["outa"];
            var outbCol = onnx.Schema["outb"];
            using (var curs = onnx.GetRowCursor(outaCol, onnx.Schema["outb"]))
            {
                var getScoresa = curs.GetGetter<VBuffer<float>>(outaCol);
                var getScoresb = curs.GetGetter<VBuffer<float>>(outbCol);
                var buffera = default(VBuffer<float>);
                var bufferb = default(VBuffer<float>);

                while (curs.MoveNext())
                {
                    getScoresa(ref buffera);
                    getScoresb(ref bufferb);
                    Assert.Equal(5, buffera.Length);
                    Assert.Equal(5, bufferb.Length);
                    Assert.Equal(0, buffera.GetValues().ToArray().Sum());
                    Assert.Equal(30, bufferb.GetValues().ToArray().Sum());
                }
            }
        }

        [OnnxFact]
        public void TestUnknownDimensions()
        {
            // model contains -1 in input and output shape dimensions
            // model: input dims = [-1, 3], output argmax dims = [-1]
            var modelFile = @"unknowndimensions/test_unknowndimensions_float.onnx";
            var mlContext = new MLContext();
            var data = new TestDataUnknownDimensions[]
                {
                    new TestDataUnknownDimensions(){input = new float[] {1.1f, 1.3f, 1.2f }},
                    new TestDataUnknownDimensions(){input = new float[] {-1.1f, -1.3f, -1.2f }},
                    new TestDataUnknownDimensions(){input = new float[] {-1.1f, -1.3f, 1.2f }},
                };
            var idv = mlContext.Data.LoadFromEnumerable(data);
            var pipeline = ML.Transforms.ApplyOnnxModel(modelFile);
            var transformedValues = pipeline.Fit(idv).Transform(idv);
            var predictions = mlContext.Data.CreateEnumerable<PredictionUnknownDimensions>(transformedValues, reuseRowObject: false).ToArray();

            Assert.Equal(1, predictions[0].argmax[0]);
            Assert.Equal(0, predictions[1].argmax[0]);
            Assert.Equal(2, predictions[2].argmax[0]);
        }

        /// <summary>
        /// This class is used in <see cref="OnnxModelInMemoryImage"/> to describe data points which will be consumed by ML.NET pipeline.
        /// </summary>
        private class ImageDataPoint
        {
            /// <summary>
            /// Height of <see cref="Image"/>.
            /// </summary>
            private const int height = 224;

            /// <summary>
            /// Width of <see cref="Image"/>.
            /// </summary>
            private const int width = 224;

            /// <summary>
            /// Image will be consumed by ONNX image multiclass classification model.
            /// </summary>
            [ImageType(height, width)]
            public Bitmap Image { get; set; }

            /// <summary>
            /// Output of ONNX model. It contains probabilities of all classes.
            /// </summary>
            [ColumnName("softmaxout_1")]
            public float[] Scores { get; set; }

            public ImageDataPoint()
            {
                Image = null;
            }

            public ImageDataPoint(Color color)
            {
                Image = new Bitmap(width, height);
                for (int i = 0; i < width; ++i)
                    for (int j = 0; j < height; ++j)
                        Image.SetPixel(i, j, color);
            }
        }

        /// <summary>
        /// Test applying ONNX transform on in-memory image.
        /// </summary>
        [OnnxFact]
        public void OnnxModelInMemoryImage()
        {
            // Path of ONNX model. It's a multiclass classifier. It consumes an input "data_0" and produces an output "softmaxout_1".
            var modelFile = "squeezenet/00000001/model.onnx";

            // Create in-memory data points. Its Image/Scores field is the input/output of the used ONNX model.
            var dataPoints = new ImageDataPoint[]
            {
                new ImageDataPoint(Color.Red),
                new ImageDataPoint(Color.Green)
            };

            // Convert training data to IDataView, the general data type used in ML.NET.
            var dataView = ML.Data.LoadFromEnumerable(dataPoints);

            // Create a ML.NET pipeline which contains two steps. First, ExtractPixle is used to convert the 224x224 image to a 3x224x224 float tensor.
            // Then the float tensor is fed into a ONNX model with an input called "data_0" and an output called "softmaxout_1". Note that "data_0" and
            // "softmaxout_1" are model input and output names stored in the used ONNX model file. Users may need to inspect their own models to
            // get the right input and output column names.
            var pipeline = ML.Transforms.ExtractPixels("data_0", "Image")                   // Map column "Image" to column "data_0"
                .Append(ML.Transforms.ApplyOnnxModel("softmaxout_1", "data_0", modelFile)); // Map column "data_0" to column "softmaxout_1"
            var model = pipeline.Fit(dataView);
            var onnx = model.Transform(dataView);

            // Convert IDataView back to IEnumerable<ImageDataPoint> so that user can inspect the output, column "softmaxout_1", of the ONNX transform.
            // Note that Column "softmaxout_1" would be stored in ImageDataPont.Scores because the added attributed [ColumnName("softmaxout_1")]
            // tells that ImageDataPont.Scores is equivalent to column "softmaxout_1".
            var transformedDataPoints = ML.Data.CreateEnumerable<ImageDataPoint>(onnx, false).ToList();

            // The scores are probabilities of all possible classes, so they should all be positive.
            foreach (var dataPoint in transformedDataPoints)
                foreach (var score in dataPoint.Scores)
                    Assert.True(score > 0);
        }

        private class ZipMapInput
        {
            [ColumnName("input")]
            [VectorType(3)]
            public float[] Input { get; set; }
        }

        private class ZipMapStringOutput
        {
            [OnnxSequenceType(typeof(IDictionary<string, float>))]
            public IEnumerable<IDictionary<string, float>> output { get; set; }
        }

        private class ZipMapInt64Output
        {
            [OnnxSequenceType(typeof(IDictionary<long, float>))]
            public IEnumerable<IDictionary<long, float>> output { get; set; }
        }

        /// <summary>
        /// A test to check if sequence output works.
        /// </summary>
        [OnnxFact]
        public void TestOnnxZipMapWithInt64Keys()
        {
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "zipmap", "TestZipMapInt64.onnx");

            var dataPoints = new ZipMapInput[] {
                new ZipMapInput() { Input = new float[] {1,2,3}, },
                new ZipMapInput() { Input = new float[] {8,7,6}, },
            };

            var dataView = ML.Data.LoadFromEnumerable(dataPoints);
            var transformedDataView = ML.Transforms.ApplyOnnxModel(new[] { "output" }, new[] { "input" }, modelFile).Fit(dataView).Transform(dataView);

            // Verify output column carried by an IDataView.
            var outputColumn = transformedDataView.Schema["output"];
            using (var curs = transformedDataView.GetRowCursor(outputColumn, transformedDataView.Schema["output"]))
            {
                IEnumerable<IDictionary<long, float>> buffer = null;
                var getMapSequence = curs.GetGetter<IEnumerable<IDictionary<long, float>>>(outputColumn);
                int i = 0;
                while (curs.MoveNext())
                {
                    getMapSequence(ref buffer);
                    Assert.Single(buffer);
                    var dictionary = buffer.First();
                    Assert.Equal(3, dictionary.Count());
                    Assert.Equal(dataPoints[i].Input[0], dictionary[94]);
                    Assert.Equal(dataPoints[i].Input[1], dictionary[17]);
                    Assert.Equal(dataPoints[i].Input[2], dictionary[36]);
                    ++i;
                }
            }

            // Convert IDataView to IEnumerable<ZipMapOutput> and then inspect the values.
            var transformedDataPoints = ML.Data.CreateEnumerable<ZipMapInt64Output>(transformedDataView, false).ToList();

            for (int i = 0; i < transformedDataPoints.Count; ++i)
            {
                Assert.Single(transformedDataPoints[i].output);
                var dictionary = transformedDataPoints[i].output.First();
                Assert.Equal(3, dictionary.Count());
                Assert.Equal(dataPoints[i].Input[0], dictionary[94]);
                Assert.Equal(dataPoints[i].Input[1], dictionary[17]);
                Assert.Equal(dataPoints[i].Input[2], dictionary[36]);
            }
        }

        /// <summary>
        /// A test to check if sequence output works.
        /// </summary>
        [OnnxFact]
        public void TestOnnxZipMapWithStringKeys()
        {
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "zipmap", "TestZipMapString.onnx");

            var dataPoints = new ZipMapInput[] {
                new ZipMapInput() { Input = new float[] {1,2,3}, },
                new ZipMapInput() { Input = new float[] {8,7,6}, },
            };

            var dataView = ML.Data.LoadFromEnumerable(dataPoints);
            var transformedDataView = ML.Transforms.ApplyOnnxModel(new[] { "output" }, new[] { "input" }, modelFile).Fit(dataView).Transform(dataView);

            // Verify output column carried by an IDataView.
            var outputColumn = transformedDataView.Schema["output"];
            using (var curs = transformedDataView.GetRowCursor(outputColumn, transformedDataView.Schema["output"]))
            {
                IEnumerable<IDictionary<string, float>> buffer = null;
                var getMapSequence = curs.GetGetter<IEnumerable<IDictionary<string, float>>>(outputColumn);
                int i = 0;
                while (curs.MoveNext())
                {
                    getMapSequence(ref buffer);
                    Assert.Single(buffer);
                    var dictionary = buffer.First();
                    Assert.Equal(3, dictionary.Count());
                    Assert.Equal(dataPoints[i].Input[0], dictionary["A"]);
                    Assert.Equal(dataPoints[i].Input[1], dictionary["B"]);
                    Assert.Equal(dataPoints[i].Input[2], dictionary["C"]);
                    ++i;
                }
            }

            // Convert IDataView to IEnumerable<ZipMapOutput> and then inspect the values.
            var transformedDataPoints = ML.Data.CreateEnumerable<ZipMapStringOutput>(transformedDataView, false).ToList();

            for (int i = 0; i < transformedDataPoints.Count; ++i)
            {
                Assert.Single(transformedDataPoints[i].output);
                var dictionary = transformedDataPoints[i].output.First();
                Assert.Equal(3, dictionary.Count());
                Assert.Equal(dataPoints[i].Input[0], dictionary["A"]);
                Assert.Equal(dataPoints[i].Input[1], dictionary["B"]);
                Assert.Equal(dataPoints[i].Input[2], dictionary["C"]);
            }
        }

        [OnnxFact]
        public void TestOnnxModelDisposal()
        {
            // Create a ONNX model as a byte[].
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "zipmap", "TestZipMapInt64.onnx");
            var modelInBytes = File.ReadAllBytes(modelFile);

            // Create ONNX model from the byte[].
            var onnxModel = OnnxModel.CreateFromBytes(modelInBytes);

            // Check if a temporal file is crated for storing the byte[].
            Assert.True(File.Exists(onnxModel.ModelFile));

            // Delete the temporal file.
            onnxModel.Dispose();

            // Make sure the temporal file is deleted.
            Assert.False(File.Exists(onnxModel.ModelFile));
        }

        [OnnxFact]
        public void TestOnnxModelNotDisposal()
        {
            // Declare the path the tested ONNX model file.
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "zipmap", "TestZipMapInt64.onnx");

            // Create ONNX model from the model file.
            var onnxModel = new OnnxModel(modelFile);

            // Check if a temporal file is crated for storing the byte[].
            Assert.True(File.Exists(onnxModel.ModelFile));

            // Don't delete the temporal file!
            onnxModel.Dispose();

            // Make sure the temporal file still exists.
            Assert.True(File.Exists(onnxModel.ModelFile));
        }

        private class OnnxMapInput
        {
            [OnnxMapType(typeof(int),typeof(float))]
            public IDictionary<int,float> Input { get; set; }
        }

        private class OnnxMapOutput
        {
            [OnnxMapType(typeof(int),typeof(float))]
            public IDictionary<int,float> Output { get; set; }
        }

        /// <summary>
        /// Use <see cref="CustomMappingCatalog.CustomMapping{TSrc, TDst}(TransformsCatalog, Action{TSrc, TDst}, string, SchemaDefinition, SchemaDefinition)"/>
        /// to test if ML.NET can manipulate <see cref="OnnxMapType"/> properly. ONNXRuntime's C# API doesn't support map yet.
        /// </summary>
        [OnnxFact]
        public void SmokeInMemoryOnnxMapTypeTest()
        {
            var inputDict0 = new Dictionary<int, float> { { 0, 94.17f }, { 1, 17.36f } };
            var inputDict1 = new Dictionary<int, float> { { 0, 12.28f }, { 1, 75.12f } };

            var dataPoints = new[] {
                new OnnxMapInput() { Input = inputDict0 },
                new OnnxMapInput() { Input = inputDict1 }
            };

            Action<OnnxMapInput, OnnxMapOutput> action = (input, output) =>
             {
                 output.Output = new Dictionary<int, float>();
                 foreach (var pair in input.Input)
                 {
                     output.Output.Add(pair.Key + 1, pair.Value);
                 }
             };

            var dataView = ML.Data.LoadFromEnumerable(dataPoints);
            var pipeline = ML.Transforms.CustomMapping(action, contractName: null);
            var model = pipeline.Fit(dataView);
            var transformedDataView = model.Transform(dataView);
            var transformedDataPoints = ML.Data.CreateEnumerable<OnnxMapOutput>(transformedDataView, false).ToList();

            for(int i = 0; i < dataPoints.Count(); ++i)
            {
                Assert.Equal(dataPoints[i].Input.Count(), transformedDataPoints[i].Output.Count());
                foreach(var pair in dataPoints[i].Input)
                    Assert.Equal(pair.Value, transformedDataPoints[i].Output[pair.Key + 1]);
            }
        }

        /// <summary>
        /// A test to check if dynamic shape works.
        /// The source of the test model is <see url="https://github.com/dotnet/machinelearning-testdata/tree/master/Microsoft.ML.Onnx.TestModels/unknowndimensions"/>.
        /// </summary>
        [OnnxFact]
        public void TestOnnxTransformWithCustomShapes()
        {
            // The loaded model has input shape [-1, 3] and output shape [-1].
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "unknowndimensions", "test_unknowndimensions_float.onnx");

            var dataPoints = new InputWithCustomShape[]
                {
                    // It's a flattened 3-by-3 tensor.
                    // [1.1, 1.3, 1.2]
                    // |1.9, 1.3, 1.2|
                    // [1.1, 1.3, 1.8]
                    new InputWithCustomShape(){input = new float[] { 1.1f, 1.3f, 1.2f, 1.9f, 1.3f, 1.2f, 1.1f, 1.3f, 1.8f } },
                    // It's a flattened 3-by-3 tensor.
                    // [0, 0, 1]
                    // |1, 0, 0|
                    // [1, 0, 0]
                    new InputWithCustomShape(){input = new float[] { 0f, 0f, 1f, 1f, 0f, 0f, 1f, 0f, 0f } }
                };

            var shapeDictionary = new Dictionary<string, int[]>() { { nameof(InputWithCustomShape.input), new int[] { 3, 3 } } };

            var dataView = ML.Data.LoadFromEnumerable(dataPoints);

            var transformedDataViews = new IDataView[3];

            // Test three public ONNX APIs with the custom shape.

            // Test 1.
            transformedDataViews[0] = ML.Transforms.ApplyOnnxModel(
                new[] { nameof(PredictionWithCustomShape.argmax) }, new[] { nameof(InputWithCustomShape.input) },
                modelFile, shapeDictionary).Fit(dataView).Transform(dataView);

            // Test 2.
            transformedDataViews[1] = ML.Transforms.ApplyOnnxModel(
                nameof(PredictionWithCustomShape.argmax), nameof(InputWithCustomShape.input),
                modelFile, shapeDictionary).Fit(dataView).Transform(dataView);

            // Test 3.
            transformedDataViews[2] = ML.Transforms.ApplyOnnxModel(
                modelFile, shapeDictionary).Fit(dataView).Transform(dataView);

            // Conduct the same check for all the 3 called public APIs.
            foreach(var transformedDataView in transformedDataViews)
            {
                var transformedDataPoints = ML.Data.CreateEnumerable<PredictionWithCustomShape>(transformedDataView, false).ToList();

                // One data point generates one transformed data point.
                Assert.Equal(dataPoints.Count(), transformedDataPoints.Count);

                // Check result numbers. They are results of applying ONNX argmax along the second axis; for example
                // [1.1, 1.3, 1.2] ---> [1] because 1.3 (indexed by 1) is the largest element.
                // |1.9, 1.3, 1.2| ---> |0|         1.9             0
                // [1.1, 1.3, 1.8] ---> [2]         1.8             2
                var expectedResults = new long[][]
                {
                    new long[] { 1, 0, 2 },
                    new long[] {2, 0, 0 }
                };

                for (int i = 0; i < transformedDataPoints.Count; ++i)
                    Assert.Equal(transformedDataPoints[i].argmax, expectedResults[i]);
            }
        }

        /// <summary>
        /// This function runs a ONNX model with user-specified shapes <paramref name="shapeDictionary"/>.
        /// The source of the test model is <see url="https://github.com/dotnet/machinelearning-testdata/tree/master/Microsoft.ML.Onnx.TestModels/twoinput"/>.
        /// </summary>
        /// <param name="shapeDictionary">Dictionary of tensor shapes. Keys are tensor names
        /// while values the associated shapes.</param>
        private void TryModelWithCustomShapesHelper(IDictionary<string, int[]> shapeDictionary)
        {
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "twoinput", "twoinput.onnx");

            var dataView = ML.Data.LoadFromEnumerable(
                new TestDataMulti2By3[] {
                    new TestDataMulti2By3()
                    {
                        ina = new float[] {1, 2, 3, 4, 5, 6},
                        inb = new float[] {1, 2, 3, 4, 5, 6}
                    }
                });

            // Define a ONNX transform, trains it, and apply it to the input data. 
            var pipeline = ML.Transforms.ApplyOnnxModel(new[] { "outa", "outb" }, new[] { "ina", "inb" },
                modelFile, shapeDictionary);
        }

        /// <summary>
        /// Check if we can throw when shapes are wrong.
        /// </summary>
        [OnnxFact]
        public void SpecifyOnnxShapes()
        {
            // Case 1: This shape conflicts with input shape [1, 1, 1, 5] loaded from the model.
            var shapeDictionary = new Dictionary<string, int[]>() {
                { "ina", new int[] { 2, 3 } },
            };
            bool somethingWrong = false;
            try
            {
                TryModelWithCustomShapesHelper(shapeDictionary);
            }
            catch
            {
                somethingWrong = true;
            }
            Assert.True(somethingWrong);

            // Case 2: This shape works with shape [1, 1, 1, 5] loaded from the model.
            shapeDictionary = new Dictionary<string, int[]>() {
                { "ina", new int[] { 1, 1, -1, 5 } },
            };
            somethingWrong = false;
            TryModelWithCustomShapesHelper(shapeDictionary);
            try
            {
                TryModelWithCustomShapesHelper(shapeDictionary);
            }
            catch
            {
                somethingWrong = true;
            }
            Assert.False(somethingWrong);

            // Case 3: this shape conflicts with output shape [1, 1, 1, 5] loaded from the model.
            shapeDictionary= new Dictionary<string, int[]>() {
                { "outb", new int[] { 5, 6 } },
            };
            somethingWrong= false;
            try
            {
                TryModelWithCustomShapesHelper(shapeDictionary);
            }
            catch
            {
                somethingWrong= true;
            }
            Assert.True(somethingWrong);

            // Case 4: this shape works with output shape [1, 1, 1, 5] loaded from the model.
            shapeDictionary= new Dictionary<string, int[]>() {
                { "outb", new int[] { -1, -1, -1, -1 } },
            };
            somethingWrong= false;
            try
            {
                TryModelWithCustomShapesHelper(shapeDictionary);
            }
            catch
            {
                somethingWrong= true;
            }
            Assert.False(somethingWrong);
        }

        /// <summary>
        /// A test to check if dynamic shape works.
        /// The source of the test model is <see url="https://github.com/dotnet/machinelearning-testdata/tree/master/Microsoft.ML.Onnx.TestModels/unknowndimensions"/>.
        /// </summary>
        [OnnxFact]
        public void TestOnnxTransformSaveAndLoadWithCustomShapes()
        {
            // The loaded model has input shape [-1, 3] and output shape [-1].
            var modelFile = Path.Combine(Directory.GetCurrentDirectory(), "unknowndimensions", "test_unknowndimensions_float.onnx");

            var dataPoints = new InputWithCustomShape[]
                {
                    // It's a flattened 3-by-3 tensor.
                    // [1.1, 1.3, 1.2]
                    // |1.9, 1.3, 1.2|
                    // [1.1, 1.3, 1.8]
                    new InputWithCustomShape(){input = new float[] { 1.1f, 1.3f, 1.2f, 1.9f, 1.3f, 1.2f, 1.1f, 1.3f, 1.8f } },
                    // It's a flattened 3-by-3 tensor.
                    // [0, 0, 1]
                    // |1, 0, 0|
                    // [1, 0, 0]
                    new InputWithCustomShape(){input = new float[] { 0f, 0f, 1f, 1f, 0f, 0f, 1f, 0f, 0f } }
                };

            var shapeDictionary = new Dictionary<string, int[]>() { { nameof(InputWithCustomShape.input), new int[] { 3, 3 } } };

            var dataView = ML.Data.LoadFromEnumerable(dataPoints);

            var pipeline = ML.Transforms.ApplyOnnxModel(nameof(PredictionWithCustomShape.argmax),
                nameof(InputWithCustomShape.input), modelFile, shapeDictionary);

            var model = pipeline.Fit(dataView);

            // Save the trained ONNX transformer into file and then load it back.
            ITransformer loadedModel = null;
            var tempPath = Path.GetTempFileName();
            using (var file = new SimpleFileHandle(Env, tempPath, true, true))
            {
                // Save.
                using (var fs = file.CreateWriteStream())
                    ML.Model.Save(model, null, fs);

                // Load.
                using (var fs = file.OpenReadStream())
                    loadedModel = ML.Model.Load(fs, out var schema);
            }

            var transformedDataView = loadedModel.Transform(dataView);

            // Conduct the same check for all the 3 called public APIs.
            var transformedDataPoints = ML.Data.CreateEnumerable<PredictionWithCustomShape>(transformedDataView, false).ToList();

            // One data point generates one transformed data point.
            Assert.Equal(dataPoints.Count(), transformedDataPoints.Count);

            // Check result numbers. They are results of applying ONNX argmax along the second axis; for example
            // [1.1, 1.3, 1.2] ---> [1] because 1.3 (indexed by 1) is the largest element.
            // |1.9, 1.3, 1.2| ---> |0|         1.9             0
            // [1.1, 1.3, 1.8] ---> [2]         1.8             2
            var expectedResults = new long[][]
            {
                new long[] { 1, 0, 2 },
                new long[] {2, 0, 0 }
            };

            for (int i = 0; i < transformedDataPoints.Count; ++i)
                Assert.Equal(transformedDataPoints[i].argmax, expectedResults[i]);
        }
    }
}
