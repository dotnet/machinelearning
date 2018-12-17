using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;

[assembly: LoadableClass(BinaryNearestNeighborTrainer.Summary, typeof(BinaryNearestNeighborTrainer), typeof(BinaryNearestNeighborTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    BinaryNearestNeighborTrainer.UserName,
    BinaryNearestNeighborTrainer.LoadName,
    BinaryNearestNeighborTrainer.ShortName)]

[assembly: LoadableClass(typeof(BinaryNearestNeighborModelParameters), null, typeof(SignatureLoadModel),
    "Binary Nearest Neightbor predictor", BinaryNearestNeighborModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(BinaryNearestNeighborTrainer), null, typeof(SignatureEntryPointModule), BinaryNearestNeighborTrainer.LoadName)]

namespace Microsoft.ML.Trainers
{
    public sealed class BinaryNearestNeighborTrainer : NearestNeighborBase<BinaryNearestNeighborTrainer.Arguments, BinaryPredictionTransformer<BinaryNearestNeighborModelParameters>, BinaryNearestNeighborModelParameters>
    {
        public const string LoadName = "BinaryNearestNeighbor";
        internal const string UserName = "Binary Nearest Neighbor";
        internal const string ShortName = "BinaryKNN";
        internal const string Summary = "Trains a binary nearest neighbor predictor.";

        public class Arguments : ArgumentsBase
        {
        }

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        /// <summary>
        /// Initializes a new instance of <see cref="BinaryNearestNeighborTrainer"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="neareastNeighbors">Number of neighbors to use by default to determine result.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public BinaryNearestNeighborTrainer(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            int neareastNeighbors = ArgumentsBase.Defaults.K,
            Action<Arguments> advancedSettings = null)
            : base(env, featureColumn, TrainerUtils.MakeBoolScalarLabel(labelColumn), neareastNeighbors, advancedSettings)
        {
            Host.CheckNonEmpty(featureColumn, nameof(featureColumn));
            Host.CheckNonEmpty(labelColumn, nameof(labelColumn));
        }

        /// <summary>
        /// Initializes a new instance of <see cref="BinaryNearestNeighborTrainer"/>
        /// </summary>
        internal BinaryNearestNeighborTrainer(IHostEnvironment env, Arguments args)
            : base(env, args, TrainerUtils.MakeBoolScalarLabel(args.LabelColumn))
        {
        }

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        protected override BinaryPredictionTransformer<BinaryNearestNeighborModelParameters> MakeTransformer(BinaryNearestNeighborModelParameters model, Schema trainSchema)
        => new BinaryPredictionTransformer<BinaryNearestNeighborModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        public BinaryPredictionTransformer<BinaryNearestNeighborModelParameters> Train(IDataView trainData, IPredictor initialPredictor = null)
            => TrainTransformer(trainData, initPredictor: initialPredictor);

        private protected override BinaryNearestNeighborModelParameters TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;
            Host.Check(data.Schema.Label != null, "Missing Label column");
            /*Host.Check(data.Schema.Label.Type == NumberType.Float || data.Schema.Label.Type is KeyType,
                "Invalid type for Label column, only floats and known-size keys are supported");
                */
            Host.Check(data.Schema.Feature != null, "Missing Feature column");
            int featureCount;
            data.CheckFeatureFloatVector(out featureCount);
            var examples = new List<VBuffer<float>>();
            var labels = new List<float>();
            if (context.InitialPredictor is RegressionNearestNeighborModelParameters oldPredictor)
            {
                examples.AddRange(oldPredictor.Objects);
                labels.AddRange(oldPredictor.Labels);
            }

            var cursorFactory = new FloatLabelCursor.Factory(data, CursOpt.Features | CursOpt.Label);
            using (var ch = Host.Start("Data extraction"))
            using (var cursor = cursorFactory.Create())
            {
                while (cursor.MoveNext())
                {
                    labels.Add(cursor.Label);
                    var featureValues = cursor.Features.GetValues();
                    VBuffer<float> example = new VBuffer<float>();
                    cursor.Features.CopyTo(ref example, 0, cursor.Features.Length);
                    examples.Add(example);
                }
            }

            return new BinaryNearestNeighborModelParameters(Host, data.Schema.Feature.Type.VectorSize, examples.ToArray(), labels.ToArray(), false, K, UseDistanceAsWeight, UseManhattanDistance);
        }

        [TlcModule.EntryPoint(Name = "Trainers.BinaryNearestNeighbor",
            Desc = "Train a binary Nearest Neighbor Trainer.",
            UserName = UserName,
            ShortName = ShortName)]
        public static CommonOutputs.BinaryClassificationOutput TrainBinarysNearestNeighborTrainer(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainBinaryNearestNeighborTrainer");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new BinaryNearestNeighborTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }
    }

    public sealed class BinaryNearestNeighborModelParameters : NearestNeighborModelParametersBase<float, float>, IValueMapper, ICanSaveModel
    {
        internal const string LoaderSignature = "BinNearestNeighborPred";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BIKNNPRE",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(BinaryNearestNeighborModelParameters).Assembly.FullName);
        }

        private readonly VectorType _inputType;
        private readonly ColumnType _outputType;

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        ColumnType IValueMapper.InputType => _inputType;

        ColumnType IValueMapper.OutputType => _outputType;

        /// <summary>
        /// Creates Neareast Neighbor model parameters.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="featureCount">Amount of features in each element in <paramref name="objects"/>.</param>
        /// <param name="objects">Data on which to make predictions on.</param>
        /// <param name="labels">Labels for each element.</param>
        /// <param name="copyIn">If true then the <paramref name="objects"/> vectors and  will be subject to
        /// a deep copy, if false then this constructor will take ownership of the passed in  <paramref name="objects"/> vectors.
        /// If false then the caller must take care to not use or modify the input vectors once this object
        /// is constructed, and should probably remove all references.</param>
        /// <param name="k">Number of neighbors to look at for prediction.</param>
        /// <param name="useDistanceAsWeight">Use inverse distance as weight during scoring. By default we weights for all neighbors are same (1 / <paramref  name="k"/>).</param>
        /// <param name="useManhattanDistance">Use manhattan distance between points. By default we use eucilidiean distance.</param>
        public BinaryNearestNeighborModelParameters(IHostEnvironment env, int featureCount, IList<VBuffer<float>> objects, IList<float> labels, bool copyIn,
            int k, bool useDistanceAsWeight, bool useManhattanDistance)
            : base(env, featureCount, objects, labels, copyIn, k, useDistanceAsWeight, useManhattanDistance)
        {
            _inputType = new VectorType(NumberType.Float, FeatureCount);
            _outputType = NumberType.Float;
        }

        private BinaryNearestNeighborModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx)
        {
            // ***Binary format***
            // <base>
            _inputType = new VectorType(NumberType.Float, FeatureCount);
            _outputType = NumberType.Float;
        }

        /// <summary>
        /// Save the predictor in binary format.
        /// </summary>
        /// <param name="ctx">The context to save to</param>
        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
            // ***Binary format***
            // <base>
        }

        protected override float ReadLabel(BinaryReader reader)
        {
            return reader.ReadFloat();
        }

        protected override void WriteLabel(BinaryWriter writer, float label)
        {
            writer.Write(label);
        }

        private static BinaryNearestNeighborModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new BinaryNearestNeighborModelParameters(env, ctx);
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Contracts.Check(typeof(TIn) == typeof(VBuffer<float>));
            Contracts.Check(typeof(TOut) == typeof(float));

            ValueMapper<VBuffer<float>, float> del =
                (in VBuffer<float> src, ref float dst) =>
                {
                    if (src.Length != FeatureCount)
                        throw Contracts.Except("Input is of length {0}, but predictor expected length {1}", src.Length, FeatureCount);
                    dst = Score(in src);
                };
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private float Score(in VBuffer<float> src)
        {
            Host.Check(src.Length == FeatureCount, "Invalid number of features passed.");

            (var weights, var labels) = GetScores(in src);
            float result = 0;
            for (int i = 0; i < weights.Count; i++)
            {
                var weight = weights[i];
                // for just distance we already have sqr distance.
                if (UseManhattanDistance)
                    weight = weight * weight;
                result += (labels[i] <= 0 ? -1 : 1) * weight;
            }
            return result;
        }
    }
}
