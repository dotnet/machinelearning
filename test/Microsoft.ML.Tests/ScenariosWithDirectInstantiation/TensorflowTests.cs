﻿// Licensed to the .NET Foundation under one or more agreements.
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
using System;
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

                var trans = TensorFlowTransform.Create(env, loader, model_location, new[] { "c" }, new[] { "a", "b" });

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

                var tf = TensorFlowTransform.Create(env, pixels, model_location, new[] { "softmax2_pre_activation" }, new[] { "input" });

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
            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
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
                ReadOnlyMemory<char> opType = default;
                schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
                Assert.Equal("Placeholder", opType.ToString());
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
                Assert.Equal("Identity", opType.ToString());
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
                Assert.NotNull(metadataType);
                VBuffer<ReadOnlyMemory<char>> inputOps = default;
                schema.GetMetadata(TensorFlowUtils.InputOps, col, ref inputOps);
                Assert.Equal(1, inputOps.Length);
                Assert.Equal("conv2d/kernel", inputOps.Values[0].ToString());

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
                Assert.Equal("Conv2D", opType.ToString());
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
                Assert.NotNull(metadataType);
                schema.GetMetadata(TensorFlowUtils.InputOps, col, ref inputOps);
                Assert.Equal(2, inputOps.Length);
                Assert.Equal("reshape/Reshape", inputOps.Values[0].ToString());
                Assert.Equal("conv2d/Conv2D/ReadVariableOp", inputOps.Values[1].ToString());

                Assert.True(schema.TryGetColumnIndex("Softmax", out col));
                type = schema.GetColumnType(col).AsVector;
                Assert.Equal(1, type.DimCount);
                Assert.Equal(10, type.GetDim(0));
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, col);
                Assert.NotNull(metadataType);
                Assert.True(metadataType.IsText);
                schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
                Assert.Equal("Softmax", opType.ToString());
                metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
                Assert.NotNull(metadataType);
                schema.GetMetadata(TensorFlowUtils.InputOps, col, ref inputOps);
                Assert.Equal(1, inputOps.Length);
                Assert.Equal("sequential/dense_1/BiasAdd", inputOps.Values[0].ToString());

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

                var trainer = new LightGbmMulticlassTrainer(env, "Label", "Features");

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

        /// <summary>
        /// The following python code was used to create tensorflow model.
        /// 
        /// from __future__ import print_function
        /// 
        /// import tensorflow as tf
        /// 
        /// # Import MNIST data
        /// from tensorflow.examples.tutorials.mnist import input_data
        /// mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        /// 
        /// # Parameters
        /// learning_rate = 0.01
        /// training_epochs = 25
        /// batch_size = 100
        /// display_step = 1
        /// 
        /// # tf Graph Input
        /// x = tf.placeholder(tf.float32, [None, 784], name="Features") # mnist data image of shape 28*28=784
        /// y = tf.placeholder(tf.float32, [None, 10], name = "Label") # 0-9 digits recognition => 10 classes
        /// 
        /// # Set model weights
        /// W = tf.Variable(tf.constant(0.1, shape=[784, 10]), name = "W")
        /// b = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float32), name = "b")
        /// 
        /// # Construct model
        /// pred = tf.nn.softmax(tf.matmul(x, W) + b, name = "Prediction") # Softmax
        /// 
        /// # Minimize error using cross entropy
        /// cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1), name="Loss")
        /// # Gradient Descent
        /// optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name = "SGDOptimizer")
        /// 
        /// # Initialize the variables (i.e. assign their default value)
        /// init = tf.global_variables_initializer()
        /// saver_def = tf.train.Saver().as_saver_def()
        /// # Start training
        /// with tf.Session() as sess:
        /// 
        ///     # Run the initializer
        ///    sess.run(init)
        ///    tf.saved_model.simple_save(
        ///     sess,
        ///     export_dir = "./lr",
        ///     inputs = {"Features" : x},
        ///     outputs = {"Prediction": pred})
        /// </summary>
        [Fact]
        public void TensorFlowTransformMNISTLRTemplateTrainingTest()
        {
            var model_location = GetDataPath("lr_model");
            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
            {
                var dataPath = GetDataPath("Train-Tiny-28x28.txt");
                var testDataPath = GetDataPath("MNIST.Test.tiny.txt");

                // Pipeline
                var loader = TextLoader.ReadFile(env,
                new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = false,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.Num,0),
                        new TextLoader.Column("Placeholder", DataKind.Num,new []{new TextLoader.Range(1, 784) })

                    }
                }, new MultiFileSource(dataPath));

                IDataView trans = new CopyColumnsTransform(env, ("Label", "LabelOriginal")).Transform(loader);
                trans = CategoricalTransform.Create(env, trans, "Label");
                trans = NormalizeTransform.CreateMinMaxNormalizer(env, trans, "Features", "Placeholder");

                var args = new TensorFlowTransform.Arguments()
                {
                    Model = model_location,
                    InputColumns = new[] { "Features" },
                    OutputColumns = new[] { "Prediction", "b" },
                    LabeLColumn = "Label",
                    OptimizationOperation = "SGDOptimizer",
                    LossOperation = "Loss",
                    Epoch = 10,
                    LearningRateOperation = "SGDOptimizer/learning_rate",
                    LearningRate = 0.001f,
                    BatchSize = 20,
                    ReTrain = true
                };

                var trainedTfDataView = TensorFlowTransform.Create(env, args, trans);
                trans = new ConcatTransform(env, "Features", "Prediction").Transform(trainedTfDataView);
                trans = new CopyColumnsTransform(env, ("LabelOriginal", "Label")).Transform(trans);

                var trainer = new LightGbmMulticlassTrainer(env, "Label", "Features");

                var cached = new CacheDataView(env, trans, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");

                var pred = trainer.Train(trainRoles);

                // Get scorer and evaluate the predictions from test data
                IDataScorerTransform testDataScorer = GetScorer(env, trans, pred, testDataPath);
                var metrics = Evaluate(env, testDataScorer);

                Assert.Equal(0.72173913043478266, metrics.AccuracyMicro, 2);
                Assert.Equal(0.67482993197278918, metrics.AccuracyMacro, 2);

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

                // Check if the bias actually got changed after the training.
                using (var cursor = trainedTfDataView.GetRowCursor(a => true))
                {
                    trainedTfDataView.Schema.TryGetColumnIndex("b", out int bias);
                    var getter = cursor.GetGetter<VBuffer<float>>(bias);
                    if (cursor.MoveNext())
                    {
                        var trainedBias = default(VBuffer<float>);
                        getter(ref trainedBias);
                        Assert.NotEqual(trainedBias.Values, new float[] { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f });
                    }
                }
            }

            // This test changes the state of the model.
            // Cleanup folder so that other test can also use the same model.
            var varDir = Path.Combine(model_location, "variables");
            if(Directory.Exists(varDir))
                Directory.Delete(varDir, true);
            var directories = Directory.GetDirectories(model_location, "variables-*");
            if (directories != null && directories.Length > 0)
                Directory.Move(directories[0], varDir);
        }

        /// <summary>
        /// The following python code was used to create tensorflow model.
        /// 
        /// def train_model(dataTrain, labCol, config):
        ///     print('Training Data Dimensions: (%d,%d)' % (dataTrain.shape[0],dataTrain.shape[1]))
        ///     colNames = np.array(list(dataTrain))
        ///     features = np.delete(colNames,labCol)
        ///     train_X = dataTrain.ix[:, features].values
        ///     train_X = train_X.reshape(train_X.shape[0], 28,28, 1)
        ///     train_Y = dataTrain.ix[:,labCol].values.ravel()
        ///     
        ///     
        ///     tf.set_random_seed(1)
        ///     lr = tf.placeholder(tf.float32, name = "learning_rate")
        ///     pkeep = tf.placeholder_with_default(1.0, shape=(), name="DropoutProb")
        ///     features = tf.placeholder(tf.float32, [None, train_X.shape[1], train_X.shape[2], train_X.shape[3]], name="Features")
        ///     labels = tf.placeholder(tf.int64, [None], "Label")
        ///     
        ///     K = 6  # first convolutional layer output depth
        ///     L = 12  # second convolutional layer output depth
        ///     M = 24  # third convolutional layer
        ///     N = 200  # fully connected layer (softmax)
        /// 
        ///     W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
        ///     B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        ///     W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
        ///     B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
        ///     W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
        ///     B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
        /// 
        ///     W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
        ///     B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
        /// 
        ///     W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
        ///     B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
        /// 
        ///     # The model
        ///     stride = 1  # output is 28x28
        ///     Y1 = tf.nn.relu(tf.nn.conv2d(features, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
        ///     stride = 2  # output is 14x14
        ///     Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
        ///     stride = 2  # output is 7x7
        ///     Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
        /// 
        ///     # reshape the output from the third convolution for the fully connected layer
        ///     YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
        /// 
        ///     Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
        ///     YY4 = tf.nn.dropout(Y4, pkeep)
        ///     model = tf.matmul(YY4, W5) + B5
        ///     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model), name='Loss')
        /// 
        ///     prediction = tf.nn.softmax(model, name = "Prediction")
        ///     accuracy = tf.reduce_mean( tf.cast(tf.equal( tf.argmax(prediction,1), labels), tf.float32), name = "Accuracy")
        ///     optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, name="MomentumOp")
        /// 
        ///     init = tf.global_variables_initializer()
        ///     # Launch the graph.
        ///     tfconfig = tf.ConfigProto()
        ///     tfconfig.gpu_options.allow_growth = True
        ///     sess = tf.Session(config=tfconfig)
        ///     #sess = tf.Session()
        ///     sess.run(init)
        /// 
        ///     batch_size =config['batch_size']
        ///     train_time_sec = 0
        ///     total_batch = int(train_X.shape[0] / batch_size)
        ///     for epoch in range(config['epochs']):
        ///         avg_loss = 0
        ///         perm = np.arange(train_X.shape[0])
        ///         np.random.shuffle(perm)
        ///         train_X = train_X[perm] 
        ///         train_Y = train_Y[perm] 
        ///         for batch_idx in range(0, train_X.shape[0], batch_size):
        ///             X_batch = train_X[batch_idx:batch_idx+batch_size]
        ///             Y_batch = train_Y[batch_idx:batch_idx+batch_size]
        ///             t0 = time.time()
        ///             _, loss_val, acc = sess.run([optimizer, loss, accuracy], feed_dict={features: X_batch, labels: Y_batch, pkeep:0.9, lr:0.01})
        ///             train_time_sec = train_time_sec + (time.time() - t0)
        ///             avg_loss += loss_val / total_batch
        ///         print('Epoch: ', '%04d' % (epoch+1), 'cost (cross-entropy) = %.4f , acc = %.4f' % (avg_loss, acc))
        /// 
        ///     tf.saved_model.simple_save(
        ///        sess,
        ///        export_dir = "./conv",
        ///        inputs = {"Features" : features},
        ///        outputs = {"Prediction": prediction})
        /// </summary>
        [Fact]
        public void TensorFlowTransformMNISTConvTemplateTrainingTest()
        {
            var model_location = GetDataPath("mnist_conv_model");
            using (var env = new ConsoleEnvironment(seed: 1, conc: 1))
            {
                var dataPath = GetDataPath("Train-Tiny-28x28.txt");
                var testDataPath = GetDataPath("MNIST.Test.tiny.txt");

                // Pipeline
                var loader = TextLoader.ReadFile(env,
                new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = false,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.I8,0),
                        new TextLoader.Column("Placeholder", DataKind.Num,new []{new TextLoader.Range(1, 784) })

                    }
                }, new MultiFileSource(dataPath));

                IDataView trans = new CopyColumnsTransform(env, 
                    ("Placeholder", "Features"), ("Label", "LabelOriginal")).Transform(loader);

                var args = new TensorFlowTransform.Arguments()
                {
                    Model = model_location,
                    InputColumns = new[] { "Features" },
                    OutputColumns = new[] { "Prediction" },
                    LabeLColumn = "Label",
                    OptimizationOperation = "MomentumOp",
                    LossOperation = "Loss",
                    MetricOperation = "Accuracy",
                    Epoch = 10,
                    LearningRateOperation = "learning_rate",
                    LearningRate = 0.01f,
                    FeedDictionary = new Dictionary<string, float>()
                    {
                        { "DropoutProb", 1.0f }
                    },
                    BatchSize = 20,
                    ReTrain = true
                };

                var trainedTfDataView = TensorFlowTransform.Create(env, args, trans);
                trans = new ConcatTransform(env, "Features", "Prediction").Transform(trainedTfDataView);
                trans = new CopyColumnsTransform(env, ("LabelOriginal", "Label")).Transform(trans);
                trans = new ConvertTransform(env, trans, DataKind.R4, "Label");

                var trainer = new LightGbmMulticlassTrainer(env, "Label", "Features");

                var cached = new CacheDataView(env, trans, prefetch: null);
                var trainRoles = new RoleMappedData(cached, label: "Label", feature: "Features");

                var pred = trainer.Train(trainRoles);

                // Get scorer and evaluate the predictions from test data
                IDataScorerTransform testDataScorer = GetScorer(env, trans, pred, testDataPath);
                var metrics = Evaluate(env, testDataScorer);

                Assert.Equal(0.74782608695652175, metrics.AccuracyMicro, 2);
                Assert.Equal(0.608843537414966, metrics.AccuracyMacro, 2);

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

            // This test changes the state of the model.
            // Cleanup folder so that other test can also use the same model.
            var varDir = Path.Combine(model_location, "variables");
            if (Directory.Exists(varDir))
                Directory.Delete(varDir, true);
            var directories = Directory.GetDirectories(model_location, "variables-*");
            if (directories != null && directories.Length > 0)
                Directory.Move(directories[0], varDir);
        }

        [Fact]
        public void TensorFlowTransformMNISTConvSavedModelTest()
        {
            var model_location = "mnist_model";
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

                var trainer = new LightGbmMulticlassTrainer(env, "Label", "Features");

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
                Model = model_location,
                OutputColumns = new[] { "Softmax", "dense/Relu" },
                InputColumns = new[] { "Placeholder", "reshape_input" }
            });
            pipeline.Add(new Legacy.Transforms.ColumnConcatenator() { Column = new[] { new ConcatTransformColumn() { Name = "Features", Source = new[] { "Placeholder", "dense/Relu" } } } });
            pipeline.Add(new Legacy.Trainers.LogisticRegressionClassifier());

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
                var data = TextLoader.Create(env, new TextLoader.Arguments()
                {
                    Column = new[]
                    {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
                }, new MultiFileSource(dataFile));
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


                IDataView trans = TensorFlowTransform.Create(env, pixels, model_location, new[] { "Output" }, new[] { "Input" });

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
        public void TensorFlowTransformCifarSavedModel()
        {
            var model_location = "cifar_saved_model";

            using (var env = new ConsoleEnvironment())
            {
                var imageHeight = 32;
                var imageWidth = 32;
                var dataFile = GetDataPath("images/images.tsv");
                var imageFolder = Path.GetDirectoryName(dataFile);
                var data = TextLoader.Create(env, new TextLoader.Arguments()
                {
                    Column = new[]
                    {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
                }, new MultiFileSource(dataFile));
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


                IDataView trans = TensorFlowTransform.Create(env, pixels, model_location, new[] { "Output" }, new[] { "Input" });

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
                var data = TextLoader.Create(env, new TextLoader.Arguments()
                {
                    Column = new[]
                    {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
                }, new MultiFileSource(dataFile));
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
                    IDataView trans = TensorFlowTransform.Create(env, pixels, model_location, new[] { "Output" }, new[] { "Input" });
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
