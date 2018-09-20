// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TensorFlow;
using System.Collections.Generic;
using System.IO;
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

        [Fact]
        public void TensorFlowTransformMatrixMultiplicationTest()
        {
            var model_location = "model_matmul/frozen_saved_model.pb";
            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
            {
                // Pipeline
                var loader = ComponentCreation.CreateDataView(env,
                    new List<TestData>(new TestData[] { new TestData() { a = new[] { 1.0f, 2.0f,
                                                                                     3.0f, 4.0f },
                                                                         b = new[] { 1.0f, 2.0f,
                                                                                     3.0f, 4.0f } },
                        new TestData() { a = new[] { 2.0f, 2.0f,
                                                     2.0f, 2.0f },
                                         b = new[] { 3.0f, 3.0f,
                                                     3.0f, 3.0f } } }));

                var trans = TensorFlowTransform.Create(env, loader, model_location, "c", "a", "b");

                using (var cursor = trans.GetRowCursor(a => true))
                {
                    var cgetter = cursor.GetGetter<VBuffer<float>>(2);
                    Assert.True(cursor.MoveNext());
                    VBuffer<float> c = default;
                    cgetter(ref c);

                    Assert.Equal(1.0 * 1.0 + 2.0 * 3.0, c.Values[0]);
                    Assert.Equal(1.0 * 2.0 + 2.0 * 4.0, c.Values[1]);
                    Assert.Equal(3.0 * 1.0 + 4.0 * 3.0, c.Values[2]);
                    Assert.Equal(3.0 * 2.0 + 4.0 * 4.0, c.Values[3]);

                    Assert.True(cursor.MoveNext());
                    c = default;
                    cgetter(ref c);

                    Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, c.Values[0]);
                    Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, c.Values[1]);
                    Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, c.Values[2]);
                    Assert.Equal(2.0 * 3.0 + 2.0 * 3.0, c.Values[3]);

                    Assert.False(cursor.MoveNext());

                }
            }
        }

        [Fact(Skip = "Model files are not available yet")]
        public void TensorFlowTransformObjectDetectionTest()
        {
            var model_location = @"C:\models\TensorFlow\ssd_mobilenet_v1_coco_2018_01_28\frozen_inference_graph.pb";
            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
            {
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);
                var data = env.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
                var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
                {
                    Column = new ImageLoaderTransform.Column[1]
                    {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                    },
                    ImageFolder = imageFolder
                }, data);
                var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
                {
                    Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =32, ImageWidth = 32, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
                }, images);
                var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
                {
                    Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "image_tensor", UseAlpha=false, InterleaveArgb=true, Convert = false}
                    }
                }, cropped);

                var tf = TensorFlowTransform.Create(env, pixels, model_location,
                    new[] { "detection_boxes", "detection_scores", "num_detections", "detection_classes" },
                    new[] { "image_tensor" });

                tf.Schema.TryGetColumnIndex("image_tensor", out int input);
                tf.Schema.TryGetColumnIndex("detection_boxes", out int boxes);
                tf.Schema.TryGetColumnIndex("detection_scores", out int scores);
                tf.Schema.TryGetColumnIndex("num_detections", out int num);
                tf.Schema.TryGetColumnIndex("detection_classes", out int classes);
                using (var curs = tf.GetRowCursor(col => col == classes || col == num || col == scores || col == boxes || col == input))
                {
                    var getInput = curs.GetGetter<VBuffer<byte>>(input);
                    var getBoxes = curs.GetGetter<VBuffer<float>>(boxes);
                    var getScores = curs.GetGetter<VBuffer<float>>(scores);
                    var getNum = curs.GetGetter<VBuffer<float>>(num);
                    var getClasses = curs.GetGetter<VBuffer<float>>(classes);
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
        }

        [Fact(Skip = "Model files are not available yet")]
        public void TensorFlowTransformInceptionTest()
        {
            var model_location = @"C:\models\TensorFlow\tensorflow_inception_graph.pb";
            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
            {
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);
                var data = env.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
                var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
                {
                    Column = new ImageLoaderTransform.Column[1]
                    {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                    },
                    ImageFolder = imageFolder
                }, data);
                var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
                {
                    Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =224, ImageWidth = 224, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
                }, images);
                var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
                {
                    Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "input", UseAlpha=false, InterleaveArgb=true, Convert = true}
                    }
                }, cropped);

                var tf = TensorFlowTransform.Create(env, pixels, model_location, "softmax2_pre_activation", "input");

                tf.Schema.TryGetColumnIndex("input", out int input);
                tf.Schema.TryGetColumnIndex("softmax2_pre_activation", out int b);
                using (var curs = tf.GetRowCursor(col => col == b || col == input))
                {
                    var get = curs.GetGetter<VBuffer<float>>(b);
                    var getInput = curs.GetGetter<VBuffer<float>>(input);
                    var buffer = default(VBuffer<float>);
                    var inputBuffer = default(VBuffer<float>);
                    while (curs.MoveNext())
                    {
                        getInput(ref inputBuffer);
                        get(ref buffer);
                    }
                }
            }
        }

        [Fact]
        public void TensorFlowInputsOutputsSchemaTest()
        {
            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                var model_location = "mnist_model/frozen_saved_model.pb";
                var schema = TensorFlowUtils.GetModelSchema(env, model_location);
                Assert.Equal(54, schema.ColumnCount);
                Assert.True(schema.TryGetColumnIndex("Placeholder", out int col));
                var type = schema.GetColumnType(col).AsVector;
                Assert.Equal(2, type.DimCount);
                Assert.Equal(28, type.GetDim(0));
                Assert.Equal(28, type.GetDim(1));
                var metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, col);
                Assert.NotNull(metadataType);
                Assert.True(metadataType.IsText);
                DvText opType = default;
                schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
                Assert.True(opType.EqualsStr("Placeholder"));
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
                Assert.Null(metadataType);

                Assert.True(schema.TryGetColumnIndex("conv2d/Conv2D/ReadVariableOp", out col));
                type = schema.GetColumnType(col).AsVector;
                Assert.Equal(4, type.DimCount);
                Assert.Equal(5, type.GetDim(0));
                Assert.Equal(5, type.GetDim(1));
                Assert.Equal(1, type.GetDim(2));
                Assert.Equal(32, type.GetDim(3));
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, col);
                Assert.NotNull(metadataType);
                Assert.True(metadataType.IsText);
                schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
                Assert.True(opType.EqualsStr("Identity"));
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
                Assert.NotNull(metadataType);
                VBuffer<DvText> inputOps = default;
                schema.GetMetadata(TensorFlowUtils.InputOps, col, ref inputOps);
                Assert.Equal(1, inputOps.Length);
                Assert.True(inputOps.Values[0].EqualsStr("conv2d/kernel"));

                Assert.True(schema.TryGetColumnIndex("conv2d/Conv2D", out col));
                type = schema.GetColumnType(col).AsVector;
                Assert.Equal(3, type.DimCount);
                Assert.Equal(28, type.GetDim(0));
                Assert.Equal(28, type.GetDim(1));
                Assert.Equal(32, type.GetDim(2));
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, col);
                Assert.NotNull(metadataType);
                Assert.True(metadataType.IsText);
                schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
                Assert.True(opType.EqualsStr("Conv2D"));
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
                Assert.NotNull(metadataType);
                schema.GetMetadata(TensorFlowUtils.InputOps, col, ref inputOps);
                Assert.Equal(2, inputOps.Length);
                Assert.True(inputOps.Values[0].EqualsStr("reshape/Reshape"));
                Assert.True(inputOps.Values[1].EqualsStr("conv2d/Conv2D/ReadVariableOp"));

                Assert.True(schema.TryGetColumnIndex("Softmax", out col));
                type = schema.GetColumnType(col).AsVector;
                Assert.Equal(1, type.DimCount);
                Assert.Equal(10, type.GetDim(0));
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, col);
                Assert.NotNull(metadataType);
                Assert.True(metadataType.IsText);
                schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
                Assert.True(opType.EqualsStr("Softmax"));
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
                Assert.NotNull(metadataType);
                schema.GetMetadata(TensorFlowUtils.InputOps, col, ref inputOps);
                Assert.Equal(1, inputOps.Length);
                Assert.True(inputOps.Values[0].EqualsStr("sequential/dense_1/BiasAdd"));

                model_location = "model_matmul/frozen_saved_model.pb";
                schema = TensorFlowUtils.GetModelSchema(env, model_location);
                char name = 'a';
                for (int i = 0; i < schema.ColumnCount; i++)
                {
                    Assert.Equal(name.ToString(), schema.GetColumnName(i));
                    type = schema.GetColumnType(i).AsVector;
                    Assert.Equal(2, type.DimCount);
                    Assert.Equal(2, type.GetDim(0));
                    Assert.Equal(2, type.GetDim(1));
                    name++;
                }
            }
        }

        [Fact]
        public void TensorFlowTransformMNISTConvTest()
        {
            var model_location = "mnist_model/frozen_saved_model.pb";
            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
            {
                var dataPath = GetDataPath("Train-Tiny-28x28.txt");
                var testDataPath = GetDataPath("MNIST.Test.tiny.txt");

                // Pipeline
                var loader = TextLoader.ReadFile(env,
                new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.Num,0),
                        new TextLoader.Column("Placeholder", DataKind.Num,new []{new TextLoader.Range(1, 784) })

                    }
                }, new MultiFileSource(dataPath));

                IDataView trans = CopyColumnsTransform.Create(env, new CopyColumnsTransform.Arguments()
                {
                    Column = new[] { new CopyColumnsTransform.Column()
                                        { Name = "reshape_input", Source = "Placeholder" }
                                    }
                }, loader);
                trans = TensorFlowTransform.Create(env, trans, model_location, new[] { "Softmax", "dense/Relu" }, new[] { "Placeholder", "reshape_input" });
                trans = new ConcatTransform(env, "Features", "Softmax", "dense/Relu").Transform(trans);

                var trainer = new LightGbmMulticlassTrainer(env, new LightGbmArguments());

                var cached = new CacheDataView(env, trans, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");
                var pred = trainer.Train(trainRoles);

                // Get scorer and evaluate the predictions from test data
                IDataScorerTransform testDataScorer = GetScorer(env, trans, pred, testDataPath);
                var metrics = Evaluate(env, testDataScorer);

                Assert.Equal(0.99, metrics.AccuracyMicro, 2);
                Assert.Equal(1.0, metrics.AccuracyMacro, 2);

                // Create prediction engine and test predictions
                var model = env.CreatePredictionEngine<MNISTData, MNISTPrediction>(testDataScorer);

                var sample1 = new MNISTData()
                {
                    Placeholder = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26,
                    166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253,
                    225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253,
                    253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198,
                    182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0,
                    43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
                };

                var prediction = model.Predict(sample1);

                float max = -1;
                int maxIndex = -1;
                for (int i = 0; i < prediction.PredictedLabels.Length; i++)
                {
                    if (prediction.PredictedLabels[i] > max)
                    {
                        max = prediction.PredictedLabels[i];
                        maxIndex = i;
                    }
                }

                Assert.Equal(5, maxIndex);
            }
        }

        [Fact]
        public void TensorFlowTransformMNISTConvPipelineTest()
        {
            var model_location = "mnist_model/frozen_saved_model.pb";
            var dataPath = GetDataPath("Train-Tiny-28x28.txt");

            var pipeline = new Legacy.LearningPipeline(seed: 1);
            pipeline.Add(new Microsoft.ML.Legacy.Data.TextLoader(dataPath).CreateFrom<MNISTData>(useHeader: false));
            pipeline.Add(new Legacy.Transforms.ColumnCopier() { Column = new[] { new CopyColumnsTransformColumn() { Name = "reshape_input", Source = "Placeholder" } } });
            pipeline.Add(new TensorFlowScorer()
            {
                ModelFile = model_location,
                OutputColumns = new[] { "Softmax", "dense/Relu" },
                InputColumns = new[] { "Placeholder", "reshape_input" }
            });
            pipeline.Add(new Legacy.Transforms.ColumnConcatenator() { Column = new[] { new ConcatTransformColumn() { Name = "Features", Source = new[] { "Placeholder", "dense/Relu" } } } });
            pipeline.Add(new Legacy.Trainers.LogisticRegressionClassifier());
            TensorFlowUtils.Initialize();
            var model = pipeline.Train<MNISTData, MNISTPrediction>();

            var sample1 = new MNISTData()
            {
                Placeholder = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26,
                    166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253,
                    225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253,
                    253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198,
                    182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0,
                    43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
            };

            MNISTPrediction prediction = model.Predict(sample1);
        }

        public class MNISTData
        {
            [Column("0")]
            public float Label;

            [Column(ordinal: "1-784")]
            [VectorType(784)]
            public float[] Placeholder;
        }

        public class MNISTPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }

        [Fact]
        public void TensorFlowTransformCifar()
        {
            var model_location = "cifar_model/frozen_model.pb";

            using (var env = new ConsoleEnvironment())
            {
                var imageHeight = 32;
                var imageWidth = 32;
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);
                var data = env.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
                var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
                {
                    Column = new ImageLoaderTransform.Column[1]
                    {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                    },
                    ImageFolder = imageFolder
                }, data);
                var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
                {
                    Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
                }, images);

                var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
                {
                    Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "Input", UseAlpha=false, InterleaveArgb=true}
                    }
                }, cropped);


                IDataView trans = TensorFlowTransform.Create(env, pixels, model_location, "Output", "Input");

                trans.Schema.TryGetColumnIndex("Output", out int output);
                using (var cursor = trans.GetRowCursor(col => col == output))
                {
                    var buffer = default(VBuffer<float>);
                    var getter = cursor.GetGetter<VBuffer<float>>(output);
                    var numRows = 0;
                    while (cursor.MoveNext())
                    {
                        getter(ref buffer);
                        Assert.Equal(10, buffer.Length);
                        numRows += 1;
                    }
                    Assert.Equal(3, numRows);
                }
            }
        }

        [Fact]
        public void TensorFlowTransformCifarInvalidShape()
        {
            var model_location = "cifar_model/frozen_model.pb";

            using (var env = new ConsoleEnvironment())
            {
                var imageHeight = 28;
                var imageWidth = 28;
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);
                var data = env.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));

                var images = ImageLoaderTransform.Create(env, new ImageLoaderTransform.Arguments()
                {
                    Column = new ImageLoaderTransform.Column[1]
                    {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                    },
                    ImageFolder = imageFolder
                }, data);
                var cropped = ImageResizerTransform.Create(env, new ImageResizerTransform.Arguments()
                {
                    Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
                }, images);

                var pixels = ImagePixelExtractorTransform.Create(env, new ImagePixelExtractorTransform.Arguments()
                {
                    Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "Input", UseAlpha=false, InterleaveArgb=true}
                    }
                }, cropped);

                var thrown = false;
                try
                {
                    IDataView trans = TensorFlowTransform.Create(env, pixels, model_location, "Output", "Input");
                }
                catch
                {
                    thrown = true;
                }
                Assert.True(thrown);
            }
        }
    }
}
