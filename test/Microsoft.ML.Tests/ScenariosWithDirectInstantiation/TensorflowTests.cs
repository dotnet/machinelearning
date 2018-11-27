// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.RunTests;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Normalizers;
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

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowTransformMatrixMultiplicationTest()
        {
            var model_location = "model_matmul/frozen_saved_model.pb";
            var mlContext = new MLContext(seed: 1, conc: 1);
            // Pipeline
            var loader = ComponentCreation.CreateDataView(mlContext,
                    new List<TestData>(new TestData[] { new TestData() { a = new[] { 1.0f, 2.0f,
                                                                                     3.0f, 4.0f },
                                                                         b = new[] { 1.0f, 2.0f,
                                                                                     3.0f, 4.0f } },
                        new TestData() { a = new[] { 2.0f, 2.0f,
                                                     2.0f, 2.0f },
                                         b = new[] { 3.0f, 3.0f,
                                                     3.0f, 3.0f } } }));

            var trans = TensorFlowTransform.Create(mlContext, loader, model_location, new[] { "c" }, new[] { "a", "b" });

            using (var cursor = trans.GetRowCursor(a => true))
            {
                var cgetter = cursor.GetGetter<VBuffer<float>>(2);
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

        [Fact(Skip = "Model files are not available yet")]
        public void TensorFlowTransformObjectDetectionTest()
        {
            var model_location = @"C:\models\TensorFlow\ssd_mobilenet_v1_coco_2018_01_28\frozen_inference_graph.pb";
            var mlContext = new MLContext(seed: 1, conc: 1);
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = mlContext.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(mlContext, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(mlContext, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =32, ImageWidth = 32, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);
            var pixels = ImagePixelExtractorTransform.Create(mlContext, new ImagePixelExtractorTransform.Arguments()
            {
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "image_tensor", UseAlpha=false, InterleaveArgb=true, Convert = false}
                    }
            }, cropped);

            var tf = TensorFlowTransform.Create(mlContext, pixels, model_location,
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

        [Fact(Skip = "Model files are not available yet")]
        public void TensorFlowTransformInceptionTest()
        {
            var model_location = @"C:\models\TensorFlow\tensorflow_inception_graph.pb";
            var mlContext = new MLContext(seed: 1, conc: 1);
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = mlContext.CreateLoader("Text{col=ImagePath:TX:0 col=Name:TX:1}", new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(mlContext, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(mlContext, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =224, ImageWidth = 224, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);
            var pixels = ImagePixelExtractorTransform.Create(mlContext, new ImagePixelExtractorTransform.Arguments()
            {
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "input", UseAlpha=false, InterleaveArgb=true, Convert = true}
                    }
            }, cropped);

            var tf = TensorFlowTransform.Create(mlContext, pixels, model_location, new[] { "softmax2_pre_activation" }, new[] { "input" });

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

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowInputsOutputsSchemaTest()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);
            var model_location = "mnist_model/frozen_saved_model.pb";
            var schema = TensorFlowUtils.GetModelSchema(mlContext, model_location);
            Assert.Equal(86, schema.ColumnCount);
            Assert.True(schema.TryGetColumnIndex("Placeholder", out int col));
            var type = (VectorType)schema.GetColumnType(col);
            Assert.Equal(2, type.Dimensions.Length);
            Assert.Equal(28, type.Dimensions[0]);
            Assert.Equal(28, type.Dimensions[1]);
            var metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, col);
            Assert.NotNull(metadataType);
            Assert.True(metadataType is TextType);
            ReadOnlyMemory<char> opType = default;
            schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
            Assert.Equal("Placeholder", opType.ToString());
            metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
            Assert.Null(metadataType);

            Assert.True(schema.TryGetColumnIndex("conv2d/Conv2D/ReadVariableOp", out col));
            type = (VectorType)schema.GetColumnType(col);
            Assert.Equal(new[] { 5, 5, 1, 32 }, type.Dimensions);
            metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, col);
            Assert.NotNull(metadataType);
            Assert.True(metadataType is TextType);
            schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
            Assert.Equal("Identity", opType.ToString());
            metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
            Assert.NotNull(metadataType);
            VBuffer<ReadOnlyMemory<char>> inputOps = default;
            schema.GetMetadata(TensorFlowUtils.InputOps, col, ref inputOps);
            Assert.Equal(1, inputOps.Length);
            Assert.Equal("conv2d/kernel", inputOps.GetValues()[0].ToString());

            Assert.True(schema.TryGetColumnIndex("conv2d/Conv2D", out col));
            type = (VectorType)schema.GetColumnType(col);
            Assert.Equal(new[] { 28, 28, 32 }, type.Dimensions);
            metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, col);
            Assert.NotNull(metadataType);
            Assert.True(metadataType is TextType);
            schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
            Assert.Equal("Conv2D", opType.ToString());
            metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
            Assert.NotNull(metadataType);
            schema.GetMetadata(TensorFlowUtils.InputOps, col, ref inputOps);
            Assert.Equal(2, inputOps.Length);
            Assert.Equal("reshape/Reshape", inputOps.GetValues()[0].ToString());
            Assert.Equal("conv2d/Conv2D/ReadVariableOp", inputOps.GetValues()[1].ToString());

            Assert.True(schema.TryGetColumnIndex("Softmax", out col));
            type = (VectorType)schema.GetColumnType(col);
            Assert.Equal(new[] { 10 }, type.Dimensions);
            metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.OpType, col);
            Assert.NotNull(metadataType);
            Assert.True(metadataType is TextType);
            schema.GetMetadata(TensorFlowUtils.OpType, col, ref opType);
            Assert.Equal("Softmax", opType.ToString());
            metadataType = schema.GetMetadataTypeOrNull(TensorFlowUtils.InputOps, col);
            Assert.NotNull(metadataType);
            schema.GetMetadata(TensorFlowUtils.InputOps, col, ref inputOps);
            Assert.Equal(1, inputOps.Length);
            Assert.Equal("sequential/dense_1/BiasAdd", inputOps.GetValues()[0].ToString());

            model_location = "model_matmul/frozen_saved_model.pb";
            schema = TensorFlowUtils.GetModelSchema(mlContext, model_location);
            char name = 'a';
            for (int i = 0; i < schema.ColumnCount; i++)
            {
                Assert.Equal(name.ToString(), schema.GetColumnName(i));
                type = (VectorType)schema.GetColumnType(i);
                Assert.Equal(new[] { 2, 2 }, type.Dimensions);
                name++;
            }
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowTransformMNISTConvTest()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);
            var reader = mlContext.Data.TextReader(
                new TextLoader.Arguments()
                {
                    Separator = "tab",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.U4 , new [] { new TextLoader.Range(0) }, new KeyRange(0, 9)),
                        new TextLoader.Column("Placeholder", DataKind.R4, new []{ new TextLoader.Range(1, 784) })

                    }
                });

            var trainData = reader.Read(GetDataPath(TestDatasets.mnistTiny28.trainFilename));
            var testData = reader.Read(GetDataPath(TestDatasets.mnistOneClass.testFilename));

            var pipe = mlContext.Transforms.CopyColumns(("Placeholder", "reshape_input"))
                .Append(new TensorFlowEstimator(mlContext, "mnist_model/frozen_saved_model.pb", new[] { "Placeholder", "reshape_input" }, new[] { "Softmax", "dense/Relu" }))
                .Append(mlContext.Transforms.Concatenate("Features", "Softmax", "dense/Relu"))
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm("Label", "Features"));

            var trainedModel = pipe.Fit(trainData);
            var predicted = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted);

            Assert.Equal(0.99, metrics.AccuracyMicro, 2);
            Assert.Equal(1.0, metrics.AccuracyMacro, 2);

            var oneSample = GetOneMNISTExample();

            var predictFunction = trainedModel.MakePredictionFunction<MNISTData, MNISTPrediction>(mlContext);

            var onePrediction = predictFunction.Predict(oneSample);

            Assert.Equal(5, GetMaxIndexForOnePrediction(onePrediction));
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowTransformMNISTLRTrainingTest()
        {
            const double expectedMicroAccuracy = 0.72173913043478266;
            const double expectedMacroAccruacy = 0.67482993197278918;
            var model_location = "mnist_lr_model";
            try
            {
                var mlContext = new MLContext(seed: 1, conc: 1);
                var reader = mlContext.Data.TextReader(
                    new TextLoader.Arguments
                    {
                        Separator = "tab",
                        HasHeader = false,
                        Column = new[]
                        {
                            new TextLoader.Column("Label", DataKind.I8, 0),
                            new TextLoader.Column("Placeholder", DataKind.R4, new []{ new TextLoader.Range(1, 784) })
                        }
                    });

                var trainData = reader.Read(GetDataPath(TestDatasets.mnistTiny28.trainFilename));
                var testData = reader.Read(GetDataPath(TestDatasets.mnistOneClass.testFilename));

                var pipe = mlContext.Transforms.Categorical.OneHotEncoding("Label", "OneHotLabel")
                    .Append(mlContext.Transforms.Normalize(new NormalizingEstimator.MinMaxColumn("Placeholder", "Features")))
                    .Append(new TensorFlowEstimator(mlContext, new TensorFlowTransform.Arguments()
                    {
                        ModelLocation = model_location,
                        InputColumns = new[] { "Features" },
                        OutputColumns = new[] { "Prediction", "b" },
                        LabelColumn = "OneHotLabel",
                        TensorFlowLabel = "Label",
                        OptimizationOperation = "SGDOptimizer",
                        LossOperation = "Loss",
                        Epoch = 10,
                        LearningRateOperation = "SGDOptimizer/learning_rate",
                        LearningRate = 0.001f,
                        BatchSize = 20,
                        ReTrain = true
                    }))
                    .Append(mlContext.Transforms.Concatenate("Features", "Prediction"))
                    .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "KeyLabel", maxNumTerms: 10))
                    .Append(mlContext.MulticlassClassification.Trainers.LightGbm("KeyLabel", "Features"));

                var trainedModel = pipe.Fit(trainData);
                var predicted = trainedModel.Transform(testData);
                var metrics = mlContext.MulticlassClassification.Evaluate(predicted, label: "KeyLabel");
                Assert.InRange(metrics.AccuracyMicro, expectedMicroAccuracy, 1);
                Assert.InRange(metrics.AccuracyMacro, expectedMacroAccruacy, 1);
                var predictionFunction = trainedModel.MakePredictionFunction<MNISTData, MNISTPrediction>(mlContext);

                var oneSample = GetOneMNISTExample();
                var onePrediction = predictionFunction.Predict(oneSample);
                Assert.Equal(0, GetMaxIndexForOnePrediction(onePrediction));


                var trainDataTransformed = trainedModel.Transform(trainData);
                using (var cursor = trainDataTransformed.GetRowCursor(a => true))
                {
                    trainDataTransformed.Schema.TryGetColumnIndex("b", out int bias);
                    var getter = cursor.GetGetter<VBuffer<float>>(bias);
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

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowTransformMNISTConvTrainingTest()
        {
            ExecuteTFTransformMNISTConvTrainingTest(false, null, 0.74782608695652175, 0.608843537414966);
            ExecuteTFTransformMNISTConvTrainingTest(true, 5, 0.75652173913043474, 0.610204081632653);
        }

        private void ExecuteTFTransformMNISTConvTrainingTest(bool shuffle, int? shuffleSeed, double expectedMicroAccuracy, double expectedMacroAccruacy)
        {
            const string modelLocation = "mnist_conv_model";
            try
            {
                var mlContext = new MLContext(seed: 1, conc: 1);

                var reader = mlContext.Data.TextReader(new TextLoader.Arguments
                {
                    Separator = "tab",
                    HasHeader = false,
                    Column = new[]
                    {
                        new TextLoader.Column("Label", DataKind.U4, new []{ new TextLoader.Range(0) }, new KeyRange(0, 9)),
                        new TextLoader.Column("TfLabel", DataKind.I8, 0),
                        new TextLoader.Column("Placeholder", DataKind.R4, new []{ new TextLoader.Range(1, 784) })
                    }
                });

                var trainData = reader.Read(GetDataPath(TestDatasets.mnistTiny28.trainFilename));
                var testData = reader.Read(GetDataPath(TestDatasets.mnistOneClass.testFilename));

                IDataView preprocessedTrainData = null;
                IDataView preprocessedTestData = null;
                if (shuffle)
                {
                    // Shuffle training data set
                    preprocessedTrainData = new RowShufflingTransformer(mlContext, new RowShufflingTransformer.Arguments()
                    {
                        ForceShuffle = shuffle,
                        ForceShuffleSeed = shuffleSeed
                    }, trainData);

                    // Shuffle test data set
                    preprocessedTestData = new RowShufflingTransformer(mlContext, new RowShufflingTransformer.Arguments()
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

                var pipe = mlContext.Transforms.CopyColumns(("Placeholder", "Features"))
                    .Append(new TensorFlowEstimator(mlContext, new TensorFlowTransform.Arguments()
                    {
                        ModelLocation = modelLocation,
                        InputColumns = new[] { "Features" },
                        OutputColumns = new[] { "Prediction" },
                        LabelColumn = "TfLabel",
                        TensorFlowLabel = "Label",
                        OptimizationOperation = "MomentumOp",
                        LossOperation = "Loss",
                        MetricOperation = "Accuracy",
                        Epoch = 10,
                        LearningRateOperation = "learning_rate",
                        LearningRate = 0.01f,
                        BatchSize = 20,
                        ReTrain = true
                    }))
                    .Append(mlContext.Transforms.Concatenate("Features", "Prediction"))
                    .Append(mlContext.MulticlassClassification.Trainers.LightGbm("Label", "Features"));

                var trainedModel = pipe.Fit(preprocessedTrainData);
                var predicted = trainedModel.Transform(preprocessedTestData);
                var metrics = mlContext.MulticlassClassification.Evaluate(predicted);

                // First group of checks. They check if the overall prediction quality is ok using a test set.
                Assert.InRange(metrics.AccuracyMicro, expectedMicroAccuracy-.01, expectedMicroAccuracy+.01);
                Assert.InRange(metrics.AccuracyMacro, expectedMacroAccruacy-.01, expectedMicroAccuracy+.01);

                // Create prediction function and test prediction
                var predictFunction = trainedModel.MakePredictionFunction<MNISTData, MNISTPrediction>(mlContext);

                var oneSample = GetOneMNISTExample();

                var prediction = predictFunction.Predict(oneSample);

                Assert.Equal(5, GetMaxIndexForOnePrediction(prediction));
            }
            finally
            {
                // This test changes the state of the model.
                // Cleanup folder so that other test can also use the same model.
                CleanUp(modelLocation);
            }
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowTransformMNISTConvSavedModelTest()
        {
            // This test trains a multi-class classifier pipeline where a pre-trained Tenroflow model is used for featurization.
            // Two group of test criteria are checked. One group contains micro and macro accuracies. The other group is the range
            // of predicted label of a single in-memory example.

            var mlContext = new MLContext(seed: 1, conc: 1);
            var reader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.U4 , new [] { new TextLoader.Range(0) }, new KeyRange(0, 9)),
                    new TextLoader.Column("Placeholder", DataKind.R4, new []{ new TextLoader.Range(1, 784) })
                }
            });

            var trainData = reader.Read(GetDataPath(TestDatasets.mnistTiny28.trainFilename));
            var testData = reader.Read(GetDataPath(TestDatasets.mnistOneClass.testFilename));

            var pipe = mlContext.Transforms.CopyColumns(("Placeholder", "reshape_input"))
                .Append(new TensorFlowEstimator(mlContext, "mnist_model", new[] { "Placeholder", "reshape_input" }, new[] { "Softmax", "dense/Relu" }))
                .Append(mlContext.Transforms.Concatenate("Features", new[] { "Softmax", "dense/Relu" }))
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm("Label", "Features"));

            var trainedModel = pipe.Fit(trainData);
            var predicted = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted);

            // First group of checks
            Assert.Equal(0.99, metrics.AccuracyMicro, 2);
            Assert.Equal(1.0, metrics.AccuracyMacro, 2);

            // An in-memory example. Its label is predicted below.
            var oneSample = GetOneMNISTExample();

            var predictFunction = trainedModel.MakePredictionFunction<MNISTData, MNISTPrediction>(mlContext);

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
            [Column("0")]
            public long Label;

            [Column(ordinal: "1-784")]
            [VectorType(784)]
            public float[] Placeholder;
        }

        public class MNISTPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowTransformCifar()
        {
            var model_location = "cifar_model/frozen_model.pb";

            var mlContext = new MLContext(seed: 1, conc: 1);
            var tensorFlowModel = TensorFlowUtils.LoadTensorFlowModel(mlContext, model_location);
            var schema = tensorFlowModel.GetInputSchema();
            Assert.True(schema.TryGetColumnIndex("Input", out int column));
            var type = (VectorType)schema.GetColumnType(column);
            var imageHeight = type.Dimensions[0];
            var imageWidth = type.Dimensions[1];

            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));

            var pipeEstimator = new ImageLoadingEstimator(mlContext, imageFolder, ("ImagePath", "ImageReal"))
                .Append(new ImageResizingEstimator(mlContext, "ImageReal", "ImageCropped", imageWidth, imageHeight))
                .Append(new ImagePixelExtractingEstimator(mlContext, "ImageCropped", "Input", interleave: true));

            var pixels = pipeEstimator.Fit(data).Transform(data);

            IDataView trans = TensorFlowTransform.Create(mlContext, pixels, tensorFlowModel, new[] { "Output" }, new[] { "Input" });

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

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))] // TensorFlow is 64-bit only
        public void TensorFlowTransformCifarSavedModel()
        {
            var model_location = "cifar_saved_model";

            var mlContext = new MLContext(seed: 1, conc: 1);
            var tensorFlowModel = TensorFlowUtils.LoadTensorFlowModel(mlContext, model_location);
            var schema = tensorFlowModel.GetInputSchema();
            Assert.True(schema.TryGetColumnIndex("Input", out int column));
            var type = (VectorType)schema.GetColumnType(column);
            var imageHeight = type.Dimensions[0];
            var imageWidth = type.Dimensions[1];

            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(mlContext, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(mlContext, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(mlContext, new ImagePixelExtractorTransform.Arguments()
            {
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "Input", UseAlpha=false, InterleaveArgb=true}
                    }
            }, cropped);


            IDataView trans = TensorFlowTransform.Create(mlContext, pixels, tensorFlowModel, new[] { "Output" }, new[] { "Input" });

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

        [ConditionalFact(typeof(Environment), nameof(Environment.Is64BitProcess))]
        public void TensorFlowTransformCifarInvalidShape()
        {
            var model_location = "cifar_model/frozen_model.pb";

            var mlContext = new MLContext(seed: 1, conc: 1);
            var imageHeight = 28;
            var imageWidth = 28;
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var data = TextLoader.Create(mlContext, new TextLoader.Arguments()
            {
                Column = new[]
                {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Name", DataKind.TX, 1),
                    }
            }, new MultiFileSource(dataFile));
            var images = ImageLoaderTransform.Create(mlContext, new ImageLoaderTransform.Arguments()
            {
                Column = new ImageLoaderTransform.Column[1]
                {
                        new ImageLoaderTransform.Column() { Source=  "ImagePath", Name="ImageReal" }
                },
                ImageFolder = imageFolder
            }, data);
            var cropped = ImageResizerTransform.Create(mlContext, new ImageResizerTransform.Arguments()
            {
                Column = new ImageResizerTransform.Column[1]{
                        new ImageResizerTransform.Column() { Source = "ImageReal", Name= "ImageCropped", ImageHeight =imageHeight, ImageWidth = imageWidth, Resizing = ImageResizerTransform.ResizingKind.IsoCrop}
                    }
            }, images);

            var pixels = ImagePixelExtractorTransform.Create(mlContext, new ImagePixelExtractorTransform.Arguments()
            {
                Column = new ImagePixelExtractorTransform.Column[1]{
                        new ImagePixelExtractorTransform.Column() {  Source= "ImageCropped", Name = "Input", UseAlpha=false, InterleaveArgb=true}
                    }
            }, cropped);

            var thrown = false;
            try
            {
                IDataView trans = TensorFlowTransform.Create(mlContext, pixels, model_location, new[] { "Output" }, new[] { "Input" });
            }
            catch
            {
                thrown = true;
            }
            Assert.True(thrown);
        }
    }
}
