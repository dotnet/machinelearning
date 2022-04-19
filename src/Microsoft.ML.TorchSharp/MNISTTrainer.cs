// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;

namespace Microsoft.ML.TorchSharp
{
#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public sealed class MNISTTrainer :
        TrainerEstimatorBase<MulticlassPredictionTransformer<MNISTModelParameters>,
            MNISTModelParameters>
    {
        internal const string LoadName = "MNISTTrainer";
        internal const string UserName = "MNIST Trainer";
        internal const string ShortName = "MNISTCLSS";
        internal const string Summary = "Runs MNIST on images.";

        private readonly string _labelColumnName;
        private readonly string _predictedLabelColumnName;
        private readonly string _featureColumnName;
        private readonly string _scoreColumnName;

        internal MNISTTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string scoreColumn = DefaultColumnNames.Score,
            string predictedLabelColumn = DefaultColumnNames.PredictedLabel,
            IDataView validationSet = null)
                : base(Contracts.CheckRef(env, nameof(env)).Register(LoadName),
                  new SchemaShape.Column(featureColumn, SchemaShape.Column.VectorKind.Vector,
                      NumberDataViewType.Single, false),
                  new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar,
                      TextDataViewType.Instance, false))
        {
            _labelColumnName = labelColumn;
            _predictedLabelColumnName = predictedLabelColumn;
            _featureColumnName = featureColumn;
            _scoreColumnName = scoreColumn;
        }

        public override TrainerInfo Info => _info;
        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);

        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            //bool success = inputSchema.TryFindColumn(_labelColumnName, out _);
            //Contracts.Assert(success);
            var metadata = new List<SchemaShape.Column>();
            metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                TextDataViewType.Instance, false));

            return new[]
            {
                new SchemaShape.Column(_predictedLabelColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.Byte, true, new SchemaShape(metadata.ToArray()))
            };
        }

        private protected override MulticlassPredictionTransformer<MNISTModelParameters> MakeTransformer(MNISTModelParameters model, DataViewSchema trainSchema)
            => new MulticlassPredictionTransformer<MNISTModelParameters>(Host, model, trainSchema,
                FeatureColumn.Name, LabelColumn.Name, _scoreColumnName, _predictedLabelColumnName);

        private protected override MNISTModelParameters TrainModelCore(TrainContext trainContext)
        {

            return new MNISTModelParameters(Host);
        }
    }

    public sealed class MNISTModelParameters : ModelParametersBase<VBuffer<float>>, IValueMapper, IDisposable
    {
        private bool _isDisposed;
        private readonly Device _device;
        private readonly Module _model;

        internal const string LoaderSignature = "MNISTPred";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MNIST",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MNISTModelParameters).Assembly.FullName);
        }

        private readonly VectorDataViewType _inputType;
        private readonly VectorDataViewType _outputType;

        internal MNISTModelParameters(IHostEnvironment env) : base(env, LoaderSignature)
        {
            _device = ((IHostEnvironmentInternal)env).GpuDeviceId != null && cuda.is_available() ? CUDA : CPU;
            Contracts.Assert(((IHostEnvironmentInternal)env).FallbackToCpu != false || _device != CPU, "Fallback to CPU is false but no GPU detected");
            _model = new MNISTModel("model", _device);
            _model.load(@"C:\Repos\TorchSharpExamples\src\CSharp\CSharpExamples\mnist.model.bin");

            if (((IHostEnvironmentInternal)env).Seed.HasValue)
                torch.random.manual_seed(((IHostEnvironmentInternal)env).Seed.Value);

            _inputType = new VectorDataViewType(NumberDataViewType.Single);
            _outputType = new VectorDataViewType(NumberDataViewType.Single, 10);
        }

        /// <summary> Return the type of prediction task.</summary>
        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        DataViewType IValueMapper.InputType => _inputType;

        DataViewType IValueMapper.OutputType => _outputType;

        private MNISTModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // State Dictionary.
            _device = ((IHostEnvironmentInternal)env).GpuDeviceId != null && cuda.is_available() ? CUDA : CPU;
            _model = new MNISTModel("MNIST", _device);

            if (!ctx.TryLoadBinaryStream("TSModel", r => _model.load(r)))
                throw env.ExceptDecode();

            _inputType = new VectorDataViewType(NumberDataViewType.Single);
            _outputType = new VectorDataViewType(NumberDataViewType.Single);
        }

        internal static MNISTModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MNISTModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // State Dictionary.

            ctx.SaveBinaryStream("TSModel", w =>
            {
                _model.save(w);
            });
        }

        //private class Classifier
        //{
        //    private readonly Runner _runner;
        //    private readonly ImageClassificationTrainer.ImageProcessor _imageProcessor;

        //    public Classifier(ImageClassificationModelParameters model)
        //    {
        //        _runner = new Runner(model._session, new[] { model._graphInputTensor }, new[] { model._graphOutputTensor });
        //        _imageProcessor = new ImageClassificationTrainer.ImageProcessor(model._session,
        //            model._imagePreprocessorTensorInput, model._imagePreprocessorTensorOutput);
        //    }

        //    public void Score(in VBuffer<byte> image, Span<float> classProbabilities)
        //    {
        //        var processedTensor = _imageProcessor.ProcessImage(image);
        //        if (processedTensor != null)
        //        {
        //            var outputTensor = _runner.AddInput(processedTensor, 0).Run();
        //            outputTensor[0].CopyTo(classProbabilities);
        //            outputTensor[0].Dispose();
        //            processedTensor.Dispose();
        //        }
        //    }
        //}

        ValueMapper<TSrc, TDst> IValueMapper.GetMapper<TSrc, TDst>()
        {
            Host.Check(typeof(TSrc) == typeof(VBuffer<float>));
            Host.Check(typeof(TDst) == typeof(VBuffer<float>));
            //Classifier classifier = new Classifier(this);
            ValueMapper<VBuffer<float>, VBuffer<float>> del = (in VBuffer<float> src, ref VBuffer<float> dst) =>
            {
                using (var d = torch.NewDisposeScope())
                {
                    var tensor = torch.tensor(src.GetValues().ToArray(), 1, 1, 28, 28, ScalarType.Float32, _device);
                    //tensor.DecoupleFromNativeHandle();
                    var prediction = _model.forward(tensor);
                    var m = prediction.argmax(1).ToByte();
                    var editor = VBufferEditor.Create(ref dst, 10);
                    unsafe
                    {
                        fixed (byte* predictionPointer = prediction.bytes)
                        fixed (float* dstPointer = editor.Values)
                        {
                            Buffer.MemoryCopy(predictionPointer, dstPointer, 40, 40);
                        }
                    }
                    dst = editor.Commit();
                }
            };

            return (ValueMapper<TSrc, TDst>)(Delegate)del;
        }

        public void Dispose()
        {
            if (_isDisposed)
                return;

            //torch.random.ma();

            _isDisposed = true;
        }
    }
}
#pragma warning restore MSML_GeneralName // This name should be PascalCased
