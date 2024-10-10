// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.TensorExtensionMethods;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using Microsoft.ML.TorchSharp.Utils;
using Microsoft.ML;
using System.IO;
using Microsoft.ML.Data.IO;
using Microsoft.ML.TorchSharp.Loss;
using Microsoft.ML.Transforms.Image;
using static Microsoft.ML.TorchSharp.AutoFormerV2.ObjectDetectionTrainer;
using Microsoft.ML.TorchSharp.AutoFormerV2;
using static Microsoft.ML.Data.AnnotationUtils;

[assembly: LoadableClass(typeof(ObjectDetectionTransformer), null, typeof(SignatureLoadModel),
    ObjectDetectionTransformer.UserName, ObjectDetectionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ObjectDetectionTransformer), null, typeof(SignatureLoadRowMapper),
    ObjectDetectionTransformer.UserName, ObjectDetectionTransformer.LoaderSignature)]

namespace Microsoft.ML.TorchSharp.AutoFormerV2
{
    public class ObjectDetectionTrainer : IEstimator<ObjectDetectionTransformer>
    {
        public sealed class Options : TransformInputBase
        {
            /// <summary>
            /// The label column name.
            /// </summary>
            public string LabelColumnName = DefaultColumnNames.Label;

            /// <summary>
            /// The label column name.
            /// </summary>
            public string PredictedLabelColumnName = DefaultColumnNames.PredictedLabel;

            /// <summary>
            /// The Bounding Box column name.
            /// </summary>
            public string BoundingBoxColumnName = "BoundingBoxes";

            /// <summary>
            /// The Predicted Bounding Box column name.
            /// </summary>
            public string PredictedBoundingBoxColumnName = "PredictedBoundingBoxes";

            /// <summary>
            /// The Image column name.
            /// </summary>
            public string ImageColumnName = "Image";

            /// <summary>
            /// The Confidence column name.
            /// </summary>
            public string ScoreColumnName = DefaultColumnNames.Score;

            /// <summary>
            /// Gets or sets the IOU threshold for removing duplicate bounding boxes.
            /// </summary>
            [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "<Pending>")]
            public double IOUThreshold = 0.5;

            /// <summary>
            /// Gets or sets the confidenct threshold for bounding box category.
            /// </summary>
            public double ScoreThreshold = 0.5;

            /// <summary>
            /// Gets or sets the epoch steps in learning rate scheduler to reduce learning rate.
            /// </summary>
            public List<int> Steps = new List<int> { 6 };

            /// <summary>
            /// Stop training when reaching this number of epochs.
            /// </summary>
            public int MaxEpoch = 10;

            /// <summary>
            /// The validation set used while training to improve model quality.
            /// </summary>
            public IDataView ValidationSet = null;

            /// <summary>
            /// Number of classes for the data.
            /// </summary>
            internal int NumberOfClasses;

            /// <summary>
            /// Gets or sets the initial learning rate in optimizer.
            /// </summary>
            public double InitLearningRate = 1.0;

            /// <summary>
            /// Gets or sets the weight decay in optimizer.
            /// </summary>
            public double WeightDecay = 0.0;

            /// <summary>
            /// How often to log the loss.
            /// </summary>
            public int LogEveryNStep = 50;
        }

        private protected readonly IHost Host;
        internal readonly Options Option;
        private const string ModelUrl = "models/autoformer_11m_torchsharp.bin";

        internal ObjectDetectionTrainer(IHostEnvironment env, Options options)
        {
            Host = Contracts.CheckRef(env, nameof(env)).Register(nameof(ObjectDetectionTrainer));
            Contracts.Assert(options.MaxEpoch > 0);
            Contracts.AssertValue(options.BoundingBoxColumnName);
            Contracts.AssertValue(options.LabelColumnName);
            Contracts.AssertValue(options.ImageColumnName);
            Contracts.AssertValue(options.ScoreColumnName);
            Contracts.AssertValue(options.PredictedLabelColumnName);

            Option = options;
        }

        internal ObjectDetectionTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string predictedLabelColumnName = DefaultColumnNames.PredictedLabel,
            string scoreColumnName = DefaultColumnNames.Score,
            string boundingBoxColumnName = "BoundingBoxes",
            string predictedBoundingBoxColumnName = "PredictedBoundingBoxes",
            string imageColumnName = "Image",
            int maxEpoch = 10) :
            this(env, new Options
            {
                LabelColumnName = labelColumnName,
                PredictedLabelColumnName = predictedLabelColumnName,
                ScoreColumnName = scoreColumnName,
                BoundingBoxColumnName = boundingBoxColumnName,
                PredictedBoundingBoxColumnName = predictedBoundingBoxColumnName,
                ImageColumnName = imageColumnName,
                MaxEpoch = maxEpoch
            })
        {
        }

        public ObjectDetectionTransformer Fit(IDataView input)
        {
            CheckInputSchema(SchemaShape.Create(input.Schema));

            ObjectDetectionTransformer transformer = default;

            using (var ch = Host.Start("TrainModel"))
            using (var pch = Host.StartProgressChannel("Training model"))
            {
                var header = new ProgressHeader(new[] { "Loss" }, new[] { "total images" });

                var trainer = new Trainer(this, ch, input);
                pch.SetHeader(header,
                    e =>
                    {
                        e.SetProgress(0, trainer.Updates, trainer.RowCount);
                        e.SetMetric(0, trainer.LossValue);
                    });

                for (int i = 0; i < Option.MaxEpoch; i++)
                {
                    ch.Trace($"Starting epoch {i}");
                    Host.CheckAlive();
                    trainer.Train(Host, input, pch);
                    ch.Trace($"Finished epoch {i}");
                }
                var labelCol = input.Schema.GetColumnOrNull(Option.LabelColumnName);

                transformer = new ObjectDetectionTransformer(Host, Option, trainer.Model, new DataViewSchema.DetachedColumn(labelCol.Value));
                trainer.Optimizer.Dispose();

                transformer.GetOutputSchema(input.Schema);
            }
            return transformer;
        }

        internal class Trainer
        {
            public AutoFormerV2 Model;
            public torch.Device Device;
            public Optimizer Optimizer;
            public LRScheduler LearningRateScheduler;
            protected readonly ObjectDetectionTrainer Parent;
            public FocalLoss Loss;
            public int Updates;
            public float LossValue;
            public readonly int RowCount;
            private readonly IChannel _channel;

            public Trainer(ObjectDetectionTrainer parent, IChannel ch, IDataView input)
            {
                Parent = parent;
                Updates = 0;
                LossValue = 0;
                _channel = ch;

                // Get row count and figure out num of unique labels
                RowCount = GetRowCountAndSetLabelCount(input);
                Device = TorchUtils.InitializeDevice(Parent.Host);

                // Initialize the model and load pre-trained weights
                Model = new AutoFormerV2(
                    Parent.Option.NumberOfClasses,
                    embedChannels: new List<int>() { 64, 128, 256, 448 },
                    depths: new List<int>() { 2, 2, 6, 2 },
                    numHeads: new List<int>() { 2, 4, 8, 14 },
                    device: Device);

                Model.load(GetModelPath(), false);

                // Figure out if we are running on GPU or CPU
                Device = TorchUtils.InitializeDevice(Parent.Host);

                // Move to GPU if we are running there
                if (Device.type == DeviceType.CUDA)
                    Model.cuda();

                // Get the parameters that need optimization and set up the optimizer
                Optimizer = SGD(
                    Model.parameters(),
                    learningRate: Parent.Option.InitLearningRate,
                    weight_decay: Parent.Option.WeightDecay);

                Loss = new FocalLoss();

                LearningRateScheduler = MultiStepLR(Optimizer, Parent.Option.Steps);
            }

            private protected int GetRowCountAndSetLabelCount(IDataView input)
            {
                var labelCol = input.GetColumn<VBuffer<uint>>(Parent.Option.LabelColumnName);
                var rowCount = 0;
                var uniqueLabels = new HashSet<uint>();

                foreach (var label in labelCol)
                {
                    rowCount++;
                    label.DenseValues().ToList().ForEach(x => uniqueLabels.Add(x));
                }

                Parent.Option.NumberOfClasses = uniqueLabels.Count;
                return rowCount;
            }

            private string GetModelPath()
            {
                var destDir = Path.Combine(((IHostEnvironmentInternal)Parent.Host).TempFilePath, "mlnet");
                var destFileName = ModelUrl.Split('/').Last();

                Directory.CreateDirectory(destDir);

                string relativeFilePath = Path.Combine(destDir, destFileName);

                int timeout = 10 * 60 * 1000;
                using (var ch = (Parent.Host as IHostEnvironment).Start("Ensuring model file is present."))
                {
                    var ensureModel = ResourceManagerUtils.Instance.EnsureResourceAsync(Parent.Host, ch, ModelUrl, destFileName, destDir, timeout);
                    ensureModel.Wait();
                    var errorResult = ResourceManagerUtils.GetErrorMessage(out var errorMessage, ensureModel.Result);
                    if (errorResult != null)
                    {
                        var directory = Path.GetDirectoryName(errorResult.FileName);
                        var name = Path.GetFileName(errorResult.FileName);
                        throw ch.Except($"{errorMessage}\nmodel file could not be downloaded!");
                    }
                }

                return relativeFilePath;
            }

            public void Train(IHost host, IDataView input, IProgressChannel pch)
            {
                // Get the cursor and the correct columns based on the inputs
                DataViewRowCursor cursor = input.GetRowCursor(input.Schema[Parent.Option.LabelColumnName], input.Schema[Parent.Option.BoundingBoxColumnName], input.Schema[Parent.Option.ImageColumnName]);

                var boundingBoxGetter = cursor.GetGetter<VBuffer<float>>(input.Schema[Parent.Option.BoundingBoxColumnName]);
                var imageGetter = cursor.GetGetter<MLImage>(input.Schema[Parent.Option.ImageColumnName]);
                var labelGetter = cursor.GetGetter<VBuffer<uint>>(input.Schema[Parent.Option.LabelColumnName]);

                var cursorValid = true;
                Updates = 0;

                Model.train();
                Model.FreezeBN();

                if (host is IHostEnvironmentInternal hostInternal)
                {
                    torch.random.manual_seed(hostInternal.Seed ?? 1);
                    torch.cuda.manual_seed(hostInternal.Seed ?? 1);
                }
                else
                {
                    torch.random.manual_seed(1);
                    torch.cuda.manual_seed(1);
                }

                while (cursorValid)
                {
                    cursorValid = TrainStep(host, cursor, boundingBoxGetter, imageGetter, labelGetter, pch);
                }

                LearningRateScheduler.step();
            }

            private bool TrainStep(IHost host,
                DataViewRowCursor cursor,
                ValueGetter<VBuffer<float>> boundingBoxGetter,
                ValueGetter<MLImage> imageGetter,
                ValueGetter<VBuffer<uint>> labelGetter,
                IProgressChannel pch)
            {
                using var disposeScope = torch.NewDisposeScope();
                var cursorValid = true;
                Tensor imageTensor = default;
                Tensor targetTensor = default;

                host.CheckAlive();
                cursorValid = cursor.MoveNext();
                if (cursorValid)
                {
                    (imageTensor, targetTensor) = PrepareData(labelGetter, imageGetter, boundingBoxGetter);
                }
                else
                {
                    return cursorValid;
                }

                Updates++;
                host.CheckAlive();

                Optimizer.zero_grad();

                var (classification, regression, anchors) = Model.forward(imageTensor);
                var lossValue = Loss.forward(classification, regression, anchors, targetTensor);
                lossValue.backward();
                torch.nn.utils.clip_grad_norm_(Model.parameters(), 0.1);

                Optimizer.step();
                host.CheckAlive();

                if (Updates % Parent.Option.LogEveryNStep == 0)
                {
                    pch.Checkpoint(lossValue.ToDouble(), Updates);
                    _channel.Info($"Row: {Updates}, Loss: {lossValue.ToDouble()}");
                }

                return cursorValid;
            }

            private (Tensor image, Tensor Label) PrepareData(ValueGetter<VBuffer<uint>> labelGetter, ValueGetter<MLImage> imageGetter, ValueGetter<VBuffer<float>> boundingBoxGetter)
            {
                using (var _ = torch.NewDisposeScope())
                {
                    MLImage image = default;
                    imageGetter(ref image);
                    var midTensor0 = torch.tensor(image.GetBGRPixels, device: Device);
                    var midTensor1 = midTensor0.@float();
                    var midTensor2 = midTensor1.reshape(1, image.Height, image.Width, 3);
                    var midTensor3 = midTensor2.transpose(0, 3);
                    var midTensor4 = midTensor3.reshape(3, image.Height, image.Width);
                    var chunks = midTensor4.chunk(3, 0);

                    List<Tensor> part = new List<Tensor>();
                    part.Add(chunks[2]);
                    part.Add(chunks[1]);
                    part.Add(chunks[0]);

                    using var midTensor = torch.cat(part, 0);
                    using var reMidTensor = midTensor.reshape(1, 3, image.Height, image.Width);
                    var padW = 32 - (image.Width % 32);
                    var padH = 32 - (image.Height % 32);
                    using var transMidTensor = torch.zeros(1, 3, image.Height + padH, image.Width + padW, device: Device);
                    transMidTensor[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..image.Height), RangeUtil.ToTensorIndex(..image.Width)] = reMidTensor / 255.0;
                    var imageTensor = Normalize(transMidTensor, Device);

                    VBuffer<uint> labels = default;
                    labelGetter(ref labels);

                    VBuffer<float> boxes = default;
                    boundingBoxGetter(ref boxes);

                    var labelValues = labels.GetValues();
                    var boxValues = boxes.GetValues();
                    Contracts.Assert(boxValues.Length == labelValues.Length * 4, "Must have 4 coordinates for each label");

                    int b = 0;
                    var labelTensor = torch.zeros(1, labels.Length, 5, dtype: ScalarType.Int64, device: Device);
                    for (int i = 0; i < labels.Length; i++)
                    {
                        long x0 = (long)boxValues[b++];
                        long y0 = (long)boxValues[b++];
                        long x1 = (long)boxValues[b++];
                        long y1 = (long)boxValues[b++];
                        // Our labels are 1 based, the TorchSharp model is 0 based so subtract 1 to they align correctly.
                        long cl = labelValues[i] - 1;
                        labelTensor[RangeUtil.ToTensorIndex(..), i, 0] = x0;
                        labelTensor[RangeUtil.ToTensorIndex(..), i, 1] = y0;
                        labelTensor[RangeUtil.ToTensorIndex(..), i, 2] = x1;
                        labelTensor[RangeUtil.ToTensorIndex(..), i, 3] = y1;
                        labelTensor[RangeUtil.ToTensorIndex(..), i, 4] = cl;
                    }
                    return (imageTensor.MoveToOuterDisposeScope(), labelTensor.MoveToOuterDisposeScope());
                }
            }

            [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:Private field name not in: _camelCase format", Justification = "<Pending>")]
            private static readonly double[] MEAN = { 0.406, 0.456, 0.485 };

            [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_PrivateFieldName:Private field name not in: _camelCase format", Justification = "<Pending>")]
            private static readonly double[] STD = { 0.225, 0.224, 0.229 };

            internal static Tensor Normalize(Tensor x, Device device)
            {
                using (var _ = torch.NewDisposeScope())
                {
                    var meanTensor = MEAN.ToTensor(new long[4] { 1L, MEAN.Length, 1L, 1L }).to_type(ScalarType.Float32).to(device);
                    var stdTensor = STD.ToTensor(new long[4] { 1L, STD.Length, 1L, 1L }).to_type(ScalarType.Float32).to(device);
                    x = (x - meanTensor) / stdTensor;
                    return x.MoveToOuterDisposeScope();
                }
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);

            var metadata = new List<SchemaShape.Column>();
            metadata.Add(new SchemaShape.Column(Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                TextDataViewType.Instance, false));

            var scoreMetadata = new List<SchemaShape.Column>();

            scoreMetadata.Add(new SchemaShape.Column(Kinds.ScoreColumnKind, SchemaShape.Column.VectorKind.Scalar,
                TextDataViewType.Instance, false));
            scoreMetadata.Add(new SchemaShape.Column(Kinds.ScoreValueKind, SchemaShape.Column.VectorKind.Scalar,
                TextDataViewType.Instance, false));
            scoreMetadata.Add(new SchemaShape.Column(Kinds.ScoreColumnSetId, SchemaShape.Column.VectorKind.Scalar,
                NumberDataViewType.UInt32, true));
            scoreMetadata.Add(new SchemaShape.Column(Kinds.TrainingLabelValues, SchemaShape.Column.VectorKind.Vector,
                TextDataViewType.Instance, false));

            // Get label column for score column annotations. Already verified it exists.
            inputSchema.TryFindColumn(Option.LabelColumnName, out var labelCol);

            outColumns[Option.PredictedLabelColumnName] = new SchemaShape.Column(Option.PredictedLabelColumnName, SchemaShape.Column.VectorKind.VariableVector,
                    NumberDataViewType.UInt32, true, new SchemaShape(metadata.ToArray()));

            outColumns[Option.PredictedBoundingBoxColumnName] = new SchemaShape.Column(Option.PredictedBoundingBoxColumnName, SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.Single, false);

            outColumns[Option.ScoreColumnName] = new SchemaShape.Column(Option.ScoreColumnName, SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.Single, false, new SchemaShape(scoreMetadata.ToArray()));


            return new SchemaShape(outColumns.Values);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(Option.LabelColumnName, out var labelCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Option.LabelColumnName);
            if (labelCol.Kind != SchemaShape.Column.VectorKind.VariableVector || labelCol.ItemType.RawType != typeof(UInt32))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Option.LabelColumnName,
                    new VectorDataViewType(new KeyDataViewType(typeof(uint), uint.MaxValue)).ToString(), labelCol.GetTypeString());

            if (!inputSchema.TryFindColumn(Option.BoundingBoxColumnName, out var boundingBoxCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "BoundingBox", Option.BoundingBoxColumnName);
            if (boundingBoxCol.Kind != SchemaShape.Column.VectorKind.VariableVector || boundingBoxCol.ItemType.RawType != typeof(Single))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "BoundingBox", Option.BoundingBoxColumnName,
                    new VectorDataViewType(NumberDataViewType.Single).ToString(), boundingBoxCol.GetTypeString());

            if (!inputSchema.TryFindColumn(Option.ImageColumnName, out var imageCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Image", Option.ImageColumnName);
            if (imageCol.ItemType.RawType != typeof(MLImage))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Image", Option.ImageColumnName,
                    new ImageDataViewType().ToString(), imageCol.GetTypeString());
        }
    }

    public class ObjectDetectionTransformer : RowToRowTransformerBase, IDisposable
    {
        private protected readonly Device Device;
        private protected readonly AutoFormerV2 Model;
        internal readonly ObjectDetectionTrainer.Options Options;

        public readonly SchemaShape.Column PredictedLabelColumnName;
        public readonly SchemaShape.Column PredictedBoundingBoxColumn;
        public readonly SchemaShape.Column ConfidenceColumn;
        public readonly DataViewSchema.DetachedColumn LabelColumn;

        internal const string LoadName = "ObjDetTrainer";
        internal const string UserName = "Obj Detection Trainer";
        internal const string ShortName = "OBJDETC";
        internal const string Summary = "Object Detection";
        internal const string LoaderSignature = "OBJDETC";

        private static readonly FuncStaticMethodInfo1<object, Delegate> _decodeInitMethodInfo
            = new FuncStaticMethodInfo1<object, Delegate>(DecodeInit<int>);
        private bool _disposedValue;

        internal ObjectDetectionTransformer(IHostEnvironment env, ObjectDetectionTrainer.Options options, AutoFormerV2 model, DataViewSchema.DetachedColumn labelColumn)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ObjectDetectionTransformer)))
        {
            Device = TorchUtils.InitializeDevice(env);

            Options = options;
            LabelColumn = labelColumn;
            PredictedLabelColumnName = new SchemaShape.Column(Options.PredictedLabelColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.UInt32, false);
            PredictedBoundingBoxColumn = new SchemaShape.Column(Options.PredictedBoundingBoxColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);
            ConfidenceColumn = new SchemaShape.Column(Options.ScoreColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

            Model = model;
            Model.eval();

            if (Device.type == DeviceType.CUDA)
                Model.cuda();
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);

            var labelAnnotationsColumn = new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, LabelColumn.Annotations.Schema[AnnotationUtils.Kinds.SlotNames].Type, false);
            var predLabelMetadata = new SchemaShape(new SchemaShape.Column[] { labelAnnotationsColumn }
                .Concat(AnnotationUtils.GetTrainerOutputAnnotation()));

            var scoreMetadata = new List<SchemaShape.Column>();

            scoreMetadata.Add(new SchemaShape.Column(Kinds.ScoreColumnKind, SchemaShape.Column.VectorKind.Scalar,
                TextDataViewType.Instance, false));
            scoreMetadata.Add(new SchemaShape.Column(Kinds.ScoreValueKind, SchemaShape.Column.VectorKind.Scalar,
                TextDataViewType.Instance, false));
            scoreMetadata.Add(new SchemaShape.Column(Kinds.ScoreColumnSetId, SchemaShape.Column.VectorKind.Scalar,
                NumberDataViewType.UInt32, true));
            scoreMetadata.Add(new SchemaShape.Column(Kinds.TrainingLabelValues, SchemaShape.Column.VectorKind.Vector,
                TextDataViewType.Instance, false));

            outColumns[Options.PredictedLabelColumnName] = new SchemaShape.Column(Options.PredictedLabelColumnName, SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.UInt32, true, predLabelMetadata);

            outColumns[Options.PredictedBoundingBoxColumnName] = new SchemaShape.Column(Options.PredictedBoundingBoxColumnName, SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.Single, false);

            outColumns[Options.ScoreColumnName] = new SchemaShape.Column(Options.ScoreColumnName, SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.Single, false, new SchemaShape(scoreMetadata.ToArray()));

            return new SchemaShape(outColumns.Values);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            if (!inputSchema.TryFindColumn(Options.ImageColumnName, out var imageCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Image", Options.ImageColumnName);
            if (imageCol.ItemType != new ImageDataViewType())
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "Image", Options.ImageColumnName,
                    new ImageDataViewType().ToString(), imageCol.GetTypeString());
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "OBJ-DETC",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ObjectDetectionTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: id of label column name
            // int: id of predicted label column name
            // int: id of the BoundingBoxColumnName name
            // int: id of the PredictedBoundingBoxColumnName name
            // int: id of ImageColumnName name
            // int: id of Score column name
            // int: number of classes
            // double: score threshold
            // double: iou threshold
            // LabelValues
            // BinaryStream: TS Model

            ctx.SaveNonEmptyString(Options.LabelColumnName);
            ctx.SaveNonEmptyString(Options.PredictedLabelColumnName);
            ctx.SaveNonEmptyString(Options.BoundingBoxColumnName);
            ctx.SaveNonEmptyString(Options.PredictedBoundingBoxColumnName);
            ctx.SaveNonEmptyString(Options.ImageColumnName);
            ctx.SaveNonEmptyString(Options.ScoreColumnName);

            ctx.Writer.Write(Options.NumberOfClasses);

            ctx.Writer.Write(Options.ScoreThreshold);
            ctx.Writer.Write(Options.IOUThreshold);

            var labelColType = LabelColumn.Annotations.Schema[AnnotationUtils.Kinds.KeyValues].Type as VectorDataViewType;
            Microsoft.ML.Internal.Utilities.Utils.MarshalActionInvoke(SaveLabelValues<int>, labelColType.ItemType.RawType, ctx);

            ctx.SaveBinaryStream("TSModel", w =>
            {
                Model.save(w);
            });
        }

        private void SaveLabelValues<T>(ModelSaveContext ctx)
        {
            ValueGetter<VBuffer<T>> getter = LabelColumn.Annotations.GetGetter<VBuffer<T>>(LabelColumn.Annotations.Schema[AnnotationUtils.Kinds.KeyValues]);
            var val = default(VBuffer<T>);
            getter(ref val);

            BinarySaver saver = new BinarySaver(Host, new BinarySaver.Arguments());
            int bytesWritten;
            var labelColType = LabelColumn.Annotations.Schema[AnnotationUtils.Kinds.KeyValues].Type as VectorDataViewType;
            if (!saver.TryWriteTypeAndValue<VBuffer<T>>(ctx.Writer.BaseStream, labelColType, ref val, out bytesWritten))
                throw Host.Except("We do not know how to serialize label names of type '{0}'", labelColType.ItemType);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new ObjDetMapper(this, schema);

        //Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureLoadModel.
        private static ObjectDetectionTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: id of label column name
            // int: id of predicted label column name
            // int: id of the BoundingBoxColumnName name
            // int: id of the PredictedBoundingBoxColumnName name
            // int: id of ImageColumnName name
            // int: id of Score column name
            // int: number of classes
            // double: score threshold
            // double: iou threshold
            // LabelValues
            // BinaryStream: TS Model

            var options = new Options()
            {
                LabelColumnName = ctx.LoadString(),
                PredictedLabelColumnName = ctx.LoadString(),
                BoundingBoxColumnName = ctx.LoadString(),
                PredictedBoundingBoxColumnName = ctx.LoadString(),
                ImageColumnName = ctx.LoadString(),
                ScoreColumnName = ctx.LoadString(),
                NumberOfClasses = ctx.Reader.ReadInt32(),
                ScoreThreshold = ctx.Reader.ReadDouble(),
                IOUThreshold = ctx.Reader.ReadDouble(),
            };

            var ch = env.Start("Load Model");

            var model = new AutoFormerV2(options.NumberOfClasses,
                embedChannels: new List<int>() { 64, 128, 256, 448 },
                depths: new List<int>() { 2, 2, 6, 2 },
                numHeads: new List<int>() { 2, 4, 8, 14 },
                device: TorchUtils.InitializeDevice(env));

            BinarySaver saver = new BinarySaver(env, new BinarySaver.Arguments());
            DataViewType type;
            object value;
            env.CheckDecode(saver.TryLoadTypeAndValue(ctx.Reader.BaseStream, out type, out value));
            var vecType = type as VectorDataViewType;
            env.CheckDecode(vecType != null);
            env.CheckDecode(value != null);
            var labelGetter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_decodeInitMethodInfo, vecType.ItemType.RawType, value);

            var meta = new DataViewSchema.Annotations.Builder();
            meta.Add(AnnotationUtils.Kinds.KeyValues, type, labelGetter);

            var labelCol = new DataViewSchema.DetachedColumn(options.LabelColumnName, type, meta.ToAnnotations());

            if (!ctx.TryLoadBinaryStream("TSModel", r => model.load(r)))
                throw env.ExceptDecode();

            return new ObjectDetectionTransformer(env, options, model, labelCol);
        }

        private static Delegate DecodeInit<T>(object value)
        {
            VBuffer<T> buffValue = (VBuffer<T>)value;
            ValueGetter<VBuffer<T>> buffGetter = (ref VBuffer<T> dst) => buffValue.CopyTo(ref dst);
            return buffGetter;
        }

        private class ObjDetMapper : MapperBase
        {
            private readonly ObjectDetectionTransformer _parent;
            private readonly HashSet<int> _inputColIndices;

            private static readonly FuncInstanceMethodInfo1<ObjDetMapper, DataViewSchema.DetachedColumn, Delegate> _makeLabelAnnotationGetter
                = FuncInstanceMethodInfo1<ObjDetMapper, DataViewSchema.DetachedColumn, Delegate>.Create(target => target.GetLabelAnnotations<int>);


            public ObjDetMapper(ObjectDetectionTransformer parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(ObjDetMapper)), inputSchema, parent)
            {
                _parent = parent;
                _inputColIndices = new HashSet<int>();

                if (inputSchema.TryGetColumnIndex(parent.Options.ImageColumnName, out var col))
                    _inputColIndices.Add(col);

                if (Host is IHostEnvironmentInternal hostInternal)
                {
                    torch.random.manual_seed(hostInternal.Seed ?? 1);
                    torch.cuda.manual_seed(hostInternal.Seed ?? 1);
                }
                else
                {
                    torch.random.manual_seed(1);
                    torch.cuda.manual_seed(1);
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {

                var info = new DataViewSchema.DetachedColumn[3];
                var keyType = _parent.LabelColumn.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorDataViewType;
                var getter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_makeLabelAnnotationGetter, this, keyType.ItemType.RawType, _parent.LabelColumn);

                var meta = new DataViewSchema.Annotations.Builder();
                meta.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreColumnKind.MulticlassClassification.AsMemory(); });
                meta.Add(AnnotationUtils.Kinds.ScoreColumnSetId, AnnotationUtils.ScoreColumnSetIdType, GetScoreColumnSetId(InputSchema));
                meta.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreValueKind.Score.AsMemory(); });
                meta.Add(AnnotationUtils.Kinds.TrainingLabelValues, keyType, getter);

                var labelBuilder = new DataViewSchema.Annotations.Builder();
                labelBuilder.Add(AnnotationUtils.Kinds.KeyValues, keyType, getter);

                info[0] = new DataViewSchema.DetachedColumn(_parent.Options.PredictedLabelColumnName, new VectorDataViewType(new KeyDataViewType(typeof(uint), _parent.Options.NumberOfClasses)), labelBuilder.ToAnnotations());

                info[1] = new DataViewSchema.DetachedColumn(_parent.Options.ScoreColumnName, new VectorDataViewType(NumberDataViewType.Single), meta.ToAnnotations());

                info[2] = new DataViewSchema.DetachedColumn(_parent.Options.PredictedBoundingBoxColumnName, new VectorDataViewType(NumberDataViewType.Single));
                return info;

            }

            private Delegate GetLabelAnnotations<T>(DataViewSchema.DetachedColumn labelCol)
            {
                return labelCol.Annotations.GetGetter<VBuffer<T>>(labelCol.Annotations.Schema[AnnotationUtils.Kinds.KeyValues]);
            }

            private ValueGetter<uint> GetScoreColumnSetId(DataViewSchema schema)
            {
                int c;
                var max = schema.GetMaxAnnotationKind(out c, AnnotationUtils.Kinds.ScoreColumnSetId);
                uint id = checked(max + 1);
                return
                    (ref uint dst) => dst = id;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
                => throw new NotImplementedException("This should never be called!");

            private Delegate CreateGetter(DataViewRow input, int iinfo, TensorCacher outputCacher)
            {
                var ch = Host.Start("Make Getter");
                if (iinfo == 0)
                    return MakePredictedLabelGetter(input, ch, outputCacher);
                else if (iinfo == 1)
                    return MakeScoreGetter(input, ch, outputCacher);
                else
                    return MakeBoundingBoxGetter(input, ch, outputCacher);
            }

            private Delegate MakeScoreGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<MLImage> getImage = default;

                getImage = input.GetGetter<MLImage>(input.Schema[_parent.Options.ImageColumnName]);

                MLImage image = default;

                ValueGetter<VBuffer<float>> score = (ref VBuffer<float> dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref image, ref getImage);
                    var editor = VBufferEditor.Create(ref dst, outputCacher.ScoresBuffer.Length);

                    for (var i = 0; i < outputCacher.ScoresBuffer.Length; i++)
                    {
                        editor.Values[i] = outputCacher.ScoresBuffer[i];
                    }
                    dst = editor.Commit();
                };

                return score;
            }

            private Delegate MakePredictedLabelGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<MLImage> getImage = default;

                getImage = input.GetGetter<MLImage>(input.Schema[_parent.Options.ImageColumnName]);

                MLImage image = default;

                ValueGetter<VBuffer<UInt32>> predictedLabel = (ref VBuffer<UInt32> dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref image, ref getImage);
                    var editor = VBufferEditor.Create(ref dst, outputCacher.PredictedLabelsBuffer.Length);

                    for (var i = 0; i < outputCacher.PredictedLabelsBuffer.Length; i++)
                    {
                        editor.Values[i] = outputCacher.PredictedLabelsBuffer[i];
                    }
                    dst = editor.Commit();
                };

                return predictedLabel;
            }

            private Delegate MakeBoundingBoxGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<MLImage> getImage = default;

                getImage = input.GetGetter<MLImage>(input.Schema[_parent.Options.ImageColumnName]);

                MLImage image = default;

                ValueGetter<VBuffer<float>> score = (ref VBuffer<float> dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref image, ref getImage);
                    var editor = VBufferEditor.Create(ref dst, outputCacher.BoxBuffer.Length);

                    for (var i = 0; i < outputCacher.BoxBuffer.Length; i++)
                    {
                        editor.Values[i] = outputCacher.BoxBuffer[i];
                    }
                    dst = editor.Commit();
                };

                return score;
            }

            public override Delegate[] CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Contracts.Assert(input.Schema == base.InputSchema);

                TensorCacher outputCacher = new TensorCacher(_parent.Options.NumberOfClasses);
                var ch = Host.Start("Make Getters");
                _parent.Model.eval();

                int n = OutputColumns.Value.Length;
                var result = new Delegate[n];
                for (int i = 0; i < n; i++)
                {
                    if (!activeOutput(i))
                        continue;
                    result[i] = CreateGetter(input, i, outputCacher);
                }
                disposer = () =>
                {
                    outputCacher.Dispose();
                };
                return result;
            }

            private Tensor PrepInputTensors(ref MLImage image, ValueGetter<MLImage> imageGetter)
            {
                imageGetter(ref image);
                using (var preprocessScope = torch.NewDisposeScope())
                {
                    var midTensor0 = torch.tensor(image.GetBGRPixels, device: _parent.Device);
                    var midTensor1 = midTensor0.@float();
                    var midTensor2 = midTensor1.reshape(1, image.Height, image.Width, 3);
                    var midTensor3 = midTensor2.transpose(0, 3);
                    var midTensor4 = midTensor3.reshape(3, image.Height, image.Width);
                    var chunks = midTensor4.chunk(3, 0);
                    var part = new List<Tensor>();

                    part.Add(chunks[2]);
                    part.Add(chunks[1]);
                    part.Add(chunks[0]);

                    var midTensor = torch.cat(part, 0);
                    var reMidTensor = midTensor.reshape(1, 3, image.Height, image.Width);
                    var padW = 32 - (image.Width % 32);
                    var padH = 32 - (image.Height % 32);
                    var transMidTensor = torch.zeros(1, 3, image.Height + padH, image.Width + padW, device: _parent.Device);
                    transMidTensor[RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..), RangeUtil.ToTensorIndex(..image.Height), RangeUtil.ToTensorIndex(..image.Width)] = reMidTensor / 255.0;
                    var imageTensor = ObjectDetectionTrainer.Trainer.Normalize(transMidTensor, _parent.Device);
                    return imageTensor.MoveToOuterDisposeScope();
                }
            }

            private (Tensor, Tensor, Tensor) PrepAndRunModel(Tensor inputTensor)
            {
                return _parent.Model.forward(inputTensor);
            }

            private protected class TensorCacher : IDisposable
            {
                public long Position;

                public int MaxLength;
                public UInt32[] PredictedLabelsBuffer;
                public Single[] ScoresBuffer;
                public Single[] BoxBuffer;

                public TensorCacher(int maxLength)
                {
                    Position = -1;
                    MaxLength = maxLength;

                    PredictedLabelsBuffer = default;
                    ScoresBuffer = default;
                    BoxBuffer = default;
                }

                private bool _isDisposed;

                public void Dispose()
                {
                    if (_isDisposed)
                        return;

                    _isDisposed = true;
                }
            }

            private protected void UpdateCacheIfNeeded(long position, TensorCacher outputCache, ref MLImage image, ref ValueGetter<MLImage> getImage)
            {
                if (outputCache.Position != position)
                {

                    var imageTensor = PrepInputTensors(ref image, getImage);
                    _parent.Model.eval();

                    (var pred, var score, var box) = PrepAndRunModel(imageTensor);

                    ImageUtils.Postprocess(imageTensor, pred, score, box, out outputCache.PredictedLabelsBuffer, out outputCache.ScoresBuffer, out outputCache.BoxBuffer, _parent.Options.ScoreThreshold, _parent.Options.IOUThreshold);

                    pred.Dispose();
                    score.Dispose();
                    box.Dispose();

                    outputCache.Position = position;
                }
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => (activeOutput(0) || activeOutput(1) || activeOutput(2)) && _inputColIndices.Any(i => i == col);
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                }

                Model.Dispose();
                _disposedValue = true;
            }
        }

        ~ObjectDetectionTransformer()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
