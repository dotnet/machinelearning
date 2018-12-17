// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

[assembly: LoadableClass(MultiClassNearestNeighborTrainer.Summary, typeof(MultiClassNearestNeighborTrainer), typeof(MultiClassNearestNeighborTrainer.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    MultiClassNearestNeighborTrainer.UserName,
    MultiClassNearestNeighborTrainer.LoadName,
    MultiClassNearestNeighborTrainer.ShortName)]

[assembly: LoadableClass(typeof(MultiClassNearestNeighborModelParameters), null, typeof(SignatureLoadModel),
    "Multiclass Nearest Neightbor predictor", MultiClassNearestNeighborModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(MultiClassNearestNeighborTrainer), null, typeof(SignatureEntryPointModule), MultiClassNearestNeighborTrainer.LoadName)]

namespace Microsoft.ML.Trainers
{
    public sealed class MultiClassNearestNeighborTrainer : NearestNeighborBase<MultiClassNearestNeighborTrainer.Arguments, MulticlassPredictionTransformer<MultiClassNearestNeighborModelParameters>, MultiClassNearestNeighborModelParameters>
    {
        public const string LoadName = "MCNearestNeighbor";
        internal const string UserName = "Multiclass Nearest Neighbor";
        internal const string ShortName = "MulticlassKNN";
        internal const string Summary = "Trains a multiclass nearest neighbor predictor.";

        public class Arguments : ArgumentsBase
        {
        }

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        /// <summary>
        /// Initializes a new instance of <see cref="MultiClassNearestNeighborTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="neareastNeighbors">Number of neighbors to use by default to determine result.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public MultiClassNearestNeighborTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            int neareastNeighbors = Arguments.Defaults.K,
            Action<Arguments> advancedSettings = null)
            : base(env, featureColumn, TrainerUtils.MakeU4ScalarColumn(labelColumn), neareastNeighbors, advancedSettings)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="MultiClassNearestNeighborTrainer"/>
        /// </summary>
        internal MultiClassNearestNeighborTrainer(IHostEnvironment env, Arguments args)
            : base(env, args, TrainerUtils.MakeU4ScalarColumn(args.LabelColumn))
        {
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var predLabelMetadata = new SchemaShape(labelCol.Metadata.Where(x => x.Name == MetadataUtils.Kinds.KeyValues)
                .Concat(MetadataUtils.GetTrainerOutputMetadata()));

            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(MetadataUtils.MetadataForMulticlassScoreColumn(labelCol))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true, predLabelMetadata)
            };
        }

        protected override MulticlassPredictionTransformer<MultiClassNearestNeighborModelParameters> MakeTransformer(MultiClassNearestNeighborModelParameters model, Schema trainSchema)
            => new MulticlassPredictionTransformer<MultiClassNearestNeighborModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);

        public MulticlassPredictionTransformer<MultiClassNearestNeighborModelParameters> Train(IDataView trainData, IPredictor initialPredictor = null)
            => TrainTransformer(trainData, initPredictor: initialPredictor);

        private protected override MultiClassNearestNeighborModelParameters TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;
            Host.Check(data.Schema.Label != null, "Missing Label column");
            Host.Check(data.Schema.Label.Type == NumberType.Float || data.Schema.Label.Type is KeyType,
                "Invalid type for Label column, only floats and known-size keys are supported");

            Host.Check(data.Schema.Feature != null, "Missing Feature column");
            int featureCount;
            data.CheckFeatureFloatVector(out featureCount);
            int labelCount = 0;
            if (data.Schema.Label.Type.IsKey)
                labelCount = data.Schema.Label.Type.KeyCount;

            var examples = new List<VBuffer<float>>();
            var labels = new List<int>();
            if (context.InitialPredictor is MultiClassNearestNeighborModelParameters oldPredictor)
            {
                examples.AddRange(oldPredictor.Objects);
                labels.AddRange(oldPredictor.Labels);
            }
            using (var ch = Host.Start("Data extraction"))
            using (var cursor = new MultiClassLabelCursor(labelCount, data, CursOpt.Features | CursOpt.Label))
            {
                while (cursor.MoveNext())
                {
                    labels.Add(cursor.Label);
                    int size = cursor.Label + 1;
                    labelCount = labelCount < size ? size : labelCount;
                    var featureValues = cursor.Features.GetValues();
                    VBuffer<float> example = new VBuffer<float>();
                    cursor.Features.CopyTo(ref example, 0, cursor.Features.Length);
                    examples.Add(example);
                }
            }

            return new MultiClassNearestNeighborModelParameters(Host, data.Schema.Feature.Type.VectorSize, labelCount, examples.ToArray(), labels.ToArray(), false, K, UseDistanceAsWeight, UseManhattanDistance);
        }

        [TlcModule.EntryPoint(Name = "Trainers.MCNearestNeighbor",
            Desc = "Train a multiclass Nearest Neighbor Trainer.",
            UserName = UserName,
            ShortName = ShortName)]
        public static CommonOutputs.MulticlassClassificationOutput TrainMultiClassNearestNeighborTrainer(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainMCNearestNeighborTrainer");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.MulticlassClassificationOutput>(host, input,
                () => new MultiClassNearestNeighborTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }

    public sealed class MultiClassNearestNeighborModelParameters : NearestNeighborModelParametersBase<VBuffer<float>, int>, IValueMapper, ICanSaveModel
    {
        internal const string LoaderSignature = "MCKnnPred";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MCKNNPRE",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MultiClassNearestNeighborModelParameters).Assembly.FullName);
        }

        private readonly VectorType _inputType;
        private readonly VectorType _outputType;
        public readonly int NumClasses;

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        ColumnType IValueMapper.InputType => _inputType;

        ColumnType IValueMapper.OutputType => _outputType;

        /// <summary>
        /// Creates Neareast Neighbor model parameters.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="featureCount">Amount of features in each element in <paramref name="objects"/>.</param>
        /// <param name="numClasses">Number of classes total.</param>
        /// <param name="objects">Data on which to make predictions on.</param>
        /// <param name="labels">Labels for each element.</param>
        /// <param name="copyIn">If true then the <paramref name="objects"/> vectors and  will be subject to
        /// a deep copy, if false then this constructor will take ownership of the passed in  <paramref name="objects"/> vectors.
        /// If false then the caller must take care to not use or modify the input vectors once this object
        /// is constructed, and should probably remove all references.</param>
        /// <param name="k">Number of neighbors to look at for prediction.</param>
        /// <param name="useDistanceAsWeight">Use inverse distance as weight during scoring. By default we weights for all neighbors are same (1 / <paramref  name="k"/>).</param>
        /// <param name="useManhattanDistance">Use manhattan distance between points. By default we use eucilidiean distance.</param>
        public MultiClassNearestNeighborModelParameters(IHostEnvironment env, int featureCount, int numClasses, IList<VBuffer<float>> objects, IList<int> labels, bool copyIn,
            int k, bool useDistanceAsWeight, bool useManhattanDistance)
            : base(env, featureCount, objects, labels, copyIn, k, useDistanceAsWeight, useManhattanDistance)
        {
            NumClasses = numClasses;
            _inputType = new VectorType(NumberType.Float, FeatureCount);
            _outputType = new VectorType(NumberType.R4, NumClasses);
        }

        private MultiClassNearestNeighborModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx)
        {

            // ***Binary format***
            // <base>
            // int: number of classes.
            NumClasses = ctx.Reader.ReadInt32();
            _inputType = new VectorType(NumberType.Float, FeatureCount);
            _outputType = new VectorType(NumberType.Float, NumClasses);
        }

        /// <summary>
        /// Save the predictor in binary format.
        /// </summary>
        /// <param name="ctx">The context to save to</param>
        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
            var writer = ctx.Writer;

            // ***Binary format***
            // <base>
            // int: number of classes.
            writer.Write(NumClasses);
        }

        protected override int ReadLabel(BinaryReader reader)
        {
            return reader.ReadInt32();
        }

        protected override void WriteLabel(BinaryWriter writer, int label)
        {
            writer.Write(label);
        }

        private static MultiClassNearestNeighborModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MultiClassNearestNeighborModelParameters(env, ctx);
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<float>));

            ValueMapper<VBuffer<float>, VBuffer<float>> del = Map;
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private void Map(in VBuffer<float> src, ref VBuffer<float> dst)
        {
            Host.Check(src.Length == FeatureCount, "Invalid number of features passed.");

            (var weights, var labels) = GetScores(in src);
            var scores = new float[NumClasses];
            for (int i = 0; i < weights.Count; i++)
            {
                var weight = weights[i];
                // for just distance we already have sqr distance.
                if (UseManhattanDistance)
                    weight = weight * weight;
                scores[labels[i]] += weight;
            }
            var editor = VBufferEditor.Create(ref dst, NumClasses, NumClasses);
            for (int i = 0; i < NumClasses; i++)
                editor.Values[i] = scores[i];

            dst = editor.Commit();
        }
    }
}
