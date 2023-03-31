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
using static TorchSharp.torch.nn;
using Microsoft.ML.TorchSharp.Utils;
using System.IO;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.TorchSharp
{
    public abstract class TorchSharpBaseTrainer : IEstimator<TorchSharpBaseTransformer>
    {
        public abstract TorchSharpBaseTransformer Fit(IDataView input);

        public abstract SchemaShape GetOutputSchema(SchemaShape inputSchema);

        public abstract class Options : TransformInputBase
        {
            /// <summary>
            /// The label column name.
            /// </summary>
            public string LabelColumnName = DefaultColumnNames.Label;

            /// <summary>
            /// The Score column name.
            /// </summary>
            public string ScoreColumnName = DefaultColumnNames.Score;

            /// <summary>
            /// The Prediction column name.
            /// </summary>
            public string PredictionColumnName = DefaultColumnNames.PredictedLabel;

            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            public int BatchSize = 32;

            /// <summary>
            /// The start learning rate for polynomial decay scheduler.
            /// </summary>
            public double StartLearningRateRatio = .1;

            /// <summary>
            /// The final learning rate for polynomial decay scheduler.
            /// </summary>
            public double FinalLearningRateRatio = .9;

            /// <summary>
            /// Coefficiency of weight decay. Should be within [0, +Inf).
            /// </summary>
            public double WeightDecay = 0;

            /// <summary>
            /// Stop training when reaching this number of epochs.
            /// </summary>
            public int MaxEpoch = 100;

            /// <summary>
            /// The validation set used while training to improve model quality.
            /// </summary>
            public IDataView ValidationSet = null;

            /// <summary>
            /// Number of classes for the data.
            /// </summary>
            internal int NumberOfClasses;
        }
    }

    public abstract class TorchSharpBaseTrainer<TLabelCol, TTargetsCol> : TorchSharpBaseTrainer
    {
        private protected readonly IHost Host;
        internal readonly Options Option;

        internal TorchSharpBaseTrainer(IHostEnvironment env, Options options)
        {
            Host = Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchSharpBaseTrainer));
            Contracts.Assert(options.BatchSize > 0);
            Contracts.Assert(options.MaxEpoch > 0);
            Contracts.AssertValue(options.LabelColumnName);
            Contracts.AssertValue(options.PredictionColumnName);
            Contracts.AssertValue(options.ScoreColumnName);
            Option = options;
        }

        public override TorchSharpBaseTransformer Fit(IDataView input)
        {
            CheckInputSchema(SchemaShape.Create(input.Schema));

            TorchSharpBaseTransformer<TLabelCol, TTargetsCol> transformer = default;

            using (var ch = Host.Start("TrainModel"))
            using (var pch = Host.StartProgressChannel("Training model"))
            {
                var header = new ProgressHeader(new[] { "Accuracy" }, null);
                var trainer = CreateTrainer(this, ch, input);
                pch.SetHeader(header, e => e.SetMetric(0, trainer.Accuracy));
                for (int i = 0; i < Option.MaxEpoch; i++)
                {
                    ch.Trace($"Starting epoch {i}");
                    Host.CheckAlive();
                    trainer.Train(Host, input);
                    ch.Trace($"Finished epoch {i}");
                    if (Option.ValidationSet != null)
                        trainer.Validate(pch, ch, i);
                }
                var labelCol = input.Schema.GetColumnOrNull(Option.LabelColumnName);

                transformer = CreateTransformer(Host, Option, trainer.Model, new DataViewSchema.DetachedColumn(labelCol.Value));

                transformer.GetOutputSchema(input.Schema);
            }
            return transformer;
        }

        private protected abstract void CheckInputSchema(SchemaShape inputSchema);
        private protected abstract TorchSharpBaseTransformer<TLabelCol, TTargetsCol> CreateTransformer(IHost host, TorchSharpBaseTrainer<TLabelCol, TTargetsCol>.Options options, Module model, DataViewSchema.DetachedColumn labelColumn);
        private protected abstract TrainerBase CreateTrainer(TorchSharpBaseTrainer<TLabelCol, TTargetsCol> parent, IChannel ch, IDataView input);

        internal abstract class TrainerBase
        {
            public Module Model;
            public torch.Device Device;
            public optim.Optimizer Optimizer;
            public optim.lr_scheduler.LRScheduler LearningRateScheduler;
            protected readonly TorchSharpBaseTrainer<TLabelCol, TTargetsCol> Parent;
            public int Updates;
            public float Accuracy;
            public readonly int TrainingRowCount;

            public TrainerBase(TorchSharpBaseTrainer<TLabelCol, TTargetsCol> parent, IChannel ch, IDataView input)
            {
                Parent = parent;
                Updates = 0;
                Accuracy = 0;

                // Get row count and figure out num of unique labels
                TrainingRowCount = GetRowCountAndSetLabelCount(input);

                // Initialize the model and load pre-trained weights
                Model = CreateModule(ch, input);

                // Figure out if we are running on GPU or CPU
                Device = TorchUtils.InitializeDevice(Parent.Host);

                // Move to GPU if we are running there
                if (Device == CUDA)
                    Model.cuda();
            }

            private protected abstract int GetRowCountAndSetLabelCount(IDataView input);
            private protected abstract Module CreateModule(IChannel ch, IDataView input);

            public string GetModelPath(string modelUrl)
            {
                var destDir = Path.Combine(((IHostEnvironmentInternal)Parent.Host).TempFilePath, "mlnet");
                var destFileName = modelUrl.Split('/').Last();

                Directory.CreateDirectory(destDir);

                string relativeFilePath = Path.Combine(destDir, destFileName);

                int timeout = 10 * 60 * 1000;
                using (var ch = (Parent.Host as IHostEnvironment).Start("Ensuring model file is present."))
                {
                    var ensureModel = ResourceManagerUtils.Instance.EnsureResourceAsync(Parent.Host, ch, modelUrl, destFileName, destDir, timeout);
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

            public void Validate(IProgressChannel pch, IChannel ch, int epoch)
            {
                var validationSet = Parent.Option.ValidationSet;
                Model.eval();

                DataViewRowCursor cursor = GetRowCursor(validationSet);

                InitializeDataGetters(validationSet, cursor);
                var labelGetter = cursor.GetGetter<TLabelCol>(validationSet.Schema[Parent.Option.LabelColumnName]);

                // Pre-allocate the memory so it's only done once (though this step needs to be optimized)
                List<Tensor> inputTensors = new List<Tensor>(Parent.Option.BatchSize);
                List<TTargetsCol> targets = new List<TTargetsCol>(Parent.Option.BatchSize);
                int numCorrect = 0;
                int numRows = 0;

                var cursorValid = true;
                while (cursorValid)
                {
                    cursorValid = ValidateStep(cursor, labelGetter, ref inputTensors, ref targets, ref numCorrect, ref numRows);
                }
                Accuracy = numCorrect / (float)numRows;
                pch.Checkpoint(Accuracy);
                ch.Info($"Accuracy for epoch {epoch}: {Accuracy}");

                Model.train();
            }

            private protected abstract void InitializeDataGetters(IDataView input, DataViewRowCursor cursor);

            private bool ValidateStep(DataViewRowCursor cursor,
                ValueGetter<TLabelCol> labelGetter,
                ref List<Tensor> inputTensors,
                ref List<TTargetsCol> targets,
                ref int numCorrect,
                ref int numRows)
            {
                // Make sure list is clear before use
                inputTensors.Clear();
                targets.Clear();
                using var disposeScope = torch.NewDisposeScope();
                var cursorValid = true;
                for (int i = 0; i < Parent.Option.BatchSize && cursorValid; i++)
                {
                    cursorValid = cursor.MoveNext();
                    if (cursorValid)
                    {
                        inputTensors.Add(PrepareRowTensor());
                        TLabelCol target = default;
                        labelGetter(ref target);
                        targets.Add(AddToTargets(target));
                    }
                    else
                    {
                        inputTensors.TrimExcess();
                        targets.TrimExcess();
                        if (inputTensors.Count() == 0)
                            return cursorValid;
                    }
                }

                using (torch.no_grad())
                {
                    var inputTensor = PrepareBatchTensor(ref inputTensors, device: Device);
                    var targetsTensor = CreateTargetsTensor(ref targets, device: Device);
                    RunModelAndUpdateValidationStats(ref inputTensor, ref targetsTensor, ref numCorrect);
                    numRows = inputTensors.Count;
                }

                return cursorValid;
            }

            private protected abstract void RunModelAndUpdateValidationStats(ref Tensor inputTensor, ref Tensor targetsTensor, ref int numCorrect);

            public void Train(IHost host, IDataView input)
            {
                // Get the cursor and the correct columns based on the inputs
                DataViewRowCursor cursor = GetRowCursor(input);

                InitializeDataGetters(input, cursor);
                var labelGetter = cursor.GetGetter<TLabelCol>(input.Schema[Parent.Option.LabelColumnName]);

                // Pre-allocate the memory so it's only done once (though this step needs to be optimized)
                List<Tensor> inputTensors = new List<Tensor>(Parent.Option.BatchSize);
                List<TTargetsCol> targets = new List<TTargetsCol>(Parent.Option.BatchSize);

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

                var cursorValid = true;
                while (cursorValid)
                {
                    cursorValid = TrainStep(host, cursor, labelGetter, ref inputTensors, ref targets);
                }
            }

            private bool TrainStep(IHost host,
                DataViewRowCursor cursor,
                ValueGetter<TLabelCol> labelGetter,
            ref List<Tensor> inputTensors,
            ref List<TTargetsCol> targets)
            {
                // Make sure list is clear before use
                inputTensors.Clear();
                targets.Clear();
                using var disposeScope = torch.NewDisposeScope();
                var cursorValid = true;
                for (int i = 0; i < Parent.Option.BatchSize && cursorValid; i++)
                {
                    host.CheckAlive();
                    cursorValid = cursor.MoveNext();
                    if (cursorValid)
                    {
                        inputTensors.Add(PrepareRowTensor());
                        TLabelCol target = default;
                        labelGetter(ref target);
                        targets.Add(AddToTargets(target));
                    }
                    else
                    {
                        inputTensors.TrimExcess();
                        targets.TrimExcess();
                        if (inputTensors.Count() == 0)
                            return cursorValid;
                    }
                }

                Updates++;
                host.CheckAlive();
                Model.train();
                Optimizer.zero_grad();

                var inputTensor = PrepareBatchTensor(ref inputTensors, device: Device);
                var targetsTensor = CreateTargetsTensor(ref targets, device: Device);

                RunModelAndBackPropagate(ref inputTensor, ref targetsTensor);
                host.CheckAlive();

                OptimizeStep();

                return cursorValid;
            }

            private protected abstract void RunModelAndBackPropagate(ref Tensor inputTensorm, ref Tensor targetsTensor);

            private protected abstract torch.Tensor PrepareRowTensor();
            private protected abstract torch.Tensor PrepareBatchTensor(ref List<Tensor> inputTensors, Device device);

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected abstract TTargetsCol AddToTargets(TLabelCol target);

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private protected abstract Tensor CreateTargetsTensor(ref List<TTargetsCol> targets, Device device);
            private protected abstract DataViewRowCursor GetRowCursor(IDataView input);
            private protected abstract torch.Tensor GetPredictions(torch.Tensor logits);
            private protected abstract torch.Tensor GetTargets(torch.Tensor labels);
            private protected abstract int GetNumCorrect(torch.Tensor predictions, torch.Tensor targets);

            private protected virtual void OptimizeStep()
            {
                Optimizer.step();
                LearningRateScheduler.step();
            }
        }
    }


    public abstract class TorchSharpBaseTransformer : RowToRowTransformerBase
    {
        private protected TorchSharpBaseTransformer(IHost host) : base(host)
        {
        }
    }

    public abstract class TorchSharpBaseTransformer<TLabelCol, TTargetsCol> : TorchSharpBaseTransformer
    {
        private protected readonly Device Device;
        private protected readonly Module Model;
        internal readonly TorchSharpBaseTrainer.Options Options;

        private protected readonly string ScoreColumnName;
        public readonly DataViewSchema.DetachedColumn LabelColumn;

        internal TorchSharpBaseTransformer(IHostEnvironment env, TorchSharpBaseTrainer.Options options, Module model, DataViewSchema.DetachedColumn labelColumn)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchSharpBaseTransformer)))
        {
            Device = TorchUtils.InitializeDevice(env);

            Options = options;
            LabelColumn = labelColumn;
            ScoreColumnName = Options.ScoreColumnName;

            Model = model;

            if (Device == CUDA)
                Model.cuda();
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            return GetOutputSchemaCore(inputSchema);
        }

        private protected abstract void CheckInputSchema(SchemaShape inputSchema);
        private protected abstract SchemaShape GetOutputSchemaCore(SchemaShape inputSchema);
        private protected abstract override void SaveModel(ModelSaveContext ctx);

        private protected void SaveBaseModel(ModelSaveContext ctx, VersionInfo versionInfo)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(versionInfo);

            // *** Binary format ***
            // int: id of label column name
            // int: id of the score column name
            // int: id of output column name
            // int: number of classes
            // BinaryStream: TS Model
            ctx.SaveNonEmptyString(Options.LabelColumnName);
            ctx.SaveNonEmptyString(Options.ScoreColumnName);
            ctx.SaveNonEmptyString(Options.PredictionColumnName);
            ctx.Writer.Write(Options.NumberOfClasses);

            ctx.SaveBinaryStream("TSModel", w =>
            {
                Model.save(w);
            });
        }
        private protected abstract IRowMapper GetRowMapper(TorchSharpBaseTransformer<TLabelCol, TTargetsCol> parent, DataViewSchema schema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => GetRowMapper(this, schema);

        private protected abstract class TorchSharpBaseMapper : MapperBase
        {
            private protected readonly TorchSharpBaseTransformer<TLabelCol, TTargetsCol> Parent;
            private protected readonly HashSet<int> InputColIndices;

            private static readonly FuncInstanceMethodInfo1<TorchSharpBaseMapper, DataViewSchema.DetachedColumn, Delegate> _makeLabelAnnotationGetter
                = FuncInstanceMethodInfo1<TorchSharpBaseMapper, DataViewSchema.DetachedColumn, Delegate>.Create(target => target.GetLabelAnnotations<int>);

            private Delegate GetLabelAnnotations<T>(DataViewSchema.DetachedColumn labelCol)
            {
                return labelCol.Annotations.GetGetter<VBuffer<T>>(labelCol.Annotations.Schema[AnnotationUtils.Kinds.KeyValues]);
            }

            public TorchSharpBaseMapper(TorchSharpBaseTransformer<TLabelCol, TTargetsCol> parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(TorchSharpBaseMapper)), inputSchema, parent)
            {
                Parent = parent;
                InputColIndices = new HashSet<int>();
                AddInputColumnIndices(inputSchema);

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

            private protected abstract void AddInputColumnIndices(DataViewSchema inputSchema);

            private protected ValueGetter<uint> GetScoreColumnSetId(DataViewSchema schema)
            {
                int c;
                var max = schema.GetMaxAnnotationKind(out c, AnnotationUtils.Kinds.ScoreColumnSetId);
                uint id = checked(max + 1);
                return
                    (ref uint dst) => dst = id;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
                => throw new NotImplementedException("This should never be called!");

            private protected abstract Delegate CreateGetter(DataViewRow input, int iinfo, TensorCacher outputCacher);

            public override Delegate[] CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Contracts.Assert(input.Schema == base.InputSchema);

                TensorCacher outputCacher = GetTensorCacher();
                var ch = Host.Start("Make Getters");
                Parent.Model.eval();

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

            private protected abstract TensorCacher GetTensorCacher();

            private protected abstract class TensorCacher : IDisposable
            {
                public long Position;

                public TensorCacher()
                {
                    Position = -1;
                }

                public abstract void Dispose();
                public abstract void DisposeCore();

            }

            private protected abstract class TensorCacher<TOut> : TensorCacher
            {
                public TOut Result;

                public TensorCacher() : base()
                {
                    Result = default;
                }

                private bool _isDisposed;

                public override void Dispose()
                {
                    if (_isDisposed)
                        return;

                    DisposeCore();
                    _isDisposed = true;
                }
            }

            private protected override void SaveModel(ModelSaveContext ctx) => Parent.SaveModel(ctx);

        }
    }
}
