// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Trainers.Online;
using Microsoft.ML.Training;

namespace Microsoft.ML.Trainers
{
    using TScalarTrainer = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;

    public abstract class MetaMulticlassTrainer<TTransformer, TModel> : ITrainerEstimator<TTransformer, TModel>, ITrainer<IPredictor>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor", ShortName = "p", SortOrder = 4, SignatureType = typeof(SignatureBinaryClassifierTrainer))]
            [TGUI(Label = "Predictor Type", Description = "Type of underlying binary predictor")]
            internal IComponentFactory<TScalarTrainer> PredictorType;

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", SortOrder = 150, NullName = "<None>", SignatureType = typeof(SignatureCalibrator))]
            public IComponentFactory<ICalibratorTrainer> Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", SortOrder = 150, ShortName = "numcali")]
            public int MaxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.Multiple, HelpText = "Whether to treat missing labels as having negative labels, instead of keeping them missing", SortOrder = 150, ShortName = "missNeg")]
            public bool ImputeMissingLabelsAsNegative;
        }

        /// <summary>
        /// The label column that the trainer expects.
        /// </summary>
        public readonly SchemaShape.Column LabelColumn;

        private protected readonly ArgumentsBase Args;
        private protected readonly IHost Host;
        private protected readonly ICalibratorTrainer Calibrator;
        private protected readonly TScalarTrainer Trainer;

        public PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        private protected SchemaShape.Column[] OutputColumns;

        public TrainerInfo Info { get; }

        /// <summary>
        /// Initializes the <see cref="MetaMulticlassTrainer{TTransformer, TModel}"/> from the <see cref="ArgumentsBase"/> class.
        /// </summary>
        /// <param name="env">The private instance of the <see cref="IHostEnvironment"/>.</param>
        /// <param name="args">The legacy arguments <see cref="ArgumentsBase"/>class.</param>
        /// <param name="name">The component name.</param>
        /// <param name="labelColumn">The label column for the metalinear trainer and the binary trainer.</param>
        /// <param name="singleEstimator">The binary estimator.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not explicitly provided, it will default to <see cref="PlattCalibratorTrainer"/></param>
        internal MetaMulticlassTrainer(IHostEnvironment env, ArgumentsBase args, string name, string labelColumn = null,
            TScalarTrainer singleEstimator = null, ICalibratorTrainer calibrator = null)
        {
            Host = Contracts.CheckRef(env, nameof(env)).Register(name);
            Host.CheckValue(args, nameof(args));
            Args = args;

            if (labelColumn != null)
                LabelColumn = new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true);

            Trainer = singleEstimator ?? CreateTrainer();

            Calibrator = calibrator ?? new PlattCalibratorTrainer(env);
            if (args.Calibrator != null)
                Calibrator = args.Calibrator.CreateComponent(Host);

            // Regarding caching, no matter what the internal predictor, we're performing many passes
            // simply by virtue of this being a meta-trainer, so we will still cache.
            Info = new TrainerInfo(normalization: Trainer.Info.NeedNormalization);
        }

        private TScalarTrainer CreateTrainer()
        {
            return Args.PredictorType != null ?
                Args.PredictorType.CreateComponent(Host) :
                new LinearSvmTrainer(Host, new LinearSvmTrainer.Options());
        }

        private protected IDataView MapLabelsCore<T>(ColumnType type, InPredicate<T> equalsTarget, RoleMappedData data)
        {
            Host.AssertValue(type);
            Host.Assert(type.RawType == typeof(T));
            Host.AssertValue(equalsTarget);
            Host.AssertValue(data);
            Host.Assert(data.Schema.Label.HasValue);

            var lab = data.Schema.Label.Value;

            InPredicate<T> isMissing;
            if (!Args.ImputeMissingLabelsAsNegative && Conversions.Instance.TryGetIsNAPredicate(type, out isMissing))
            {
                return LambdaColumnMapper.Create(Host, "Label mapper", data.Data,
                    lab.Name, lab.Name, type, NumberType.Float,
                    (in T src, ref float dst) =>
                        dst = equalsTarget(in src) ? 1 : (isMissing(in src) ? float.NaN : default(float)));
            }
            return LambdaColumnMapper.Create(Host, "Label mapper", data.Data,
                lab.Name, lab.Name, type, NumberType.Float,
                (in T src, ref float dst) =>
                    dst = equalsTarget(in src) ? 1 : default(float));
        }

        private protected abstract TModel TrainCore(IChannel ch, RoleMappedData data, int count);

        /// <summary>
        /// The legacy train method.
        /// </summary>
        /// <param name="context">The trainig context for this learner.</param>
        /// <returns>The trained model.</returns>
        IPredictor ITrainer<IPredictor>.Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;

            data.CheckFeatureFloatVector();

            int count;
            data.CheckMultiClassLabel(out count);
            Host.Assert(count > 0);

            using (var ch = Host.Start("Training"))
            {
                var pred = TrainCore(ch, data, count) as IPredictor;
                ch.Check(pred != null, "Training did not result in a predictor");
                return pred;
            }
        }

        /// <summary>
        ///  Gets the output columns.
        /// </summary>
        /// <param name="inputSchema">The input schema. </param>
        /// <returns>The output <see cref="SchemaShape"/></returns>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (LabelColumn.IsValid)
            {
                if (!inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol))
                    throw Host.ExceptSchemaMismatch(nameof(labelCol), "label", LabelColumn.Name);

                if (!LabelColumn.IsCompatibleWith(labelCol))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", LabelColumn.Name, LabelColumn.GetTypeString(), labelCol.GetTypeString());
            }

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            foreach (var col in GetOutputColumnsCore(inputSchema))
                outColumns[col.Name] = col;

            return new SchemaShape(outColumns.Values);
        }

        private SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            if (LabelColumn.IsValid)
            {
                bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
                Contracts.Assert(success);

                var metadata = new SchemaShape(labelCol.Metadata.Where(x => x.Name == MetadataUtils.Kinds.KeyValues)
                                .Concat(MetadataForScoreColumn()));
                return new[]
                {
                    new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(MetadataUtils.MetadataForMulticlassScoreColumn(labelCol))),
                    new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true, metadata)
                };
            }
            else
                return new[]
                {
                    new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false, new SchemaShape(MetadataForScoreColumn())),
                    new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true, new SchemaShape(MetadataForScoreColumn()))
                };
        }

        /// <summary>
        /// Normal metadata that we produce for score columns.
        /// </summary>
        private static IEnumerable<SchemaShape.Column> MetadataForScoreColumn()
        {
            var cols = new List<SchemaShape.Column>();
            cols.Add(new SchemaShape.Column(MetadataUtils.Kinds.ScoreColumnSetId, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true));
            cols.Add(new SchemaShape.Column(MetadataUtils.Kinds.ScoreColumnKind, SchemaShape.Column.VectorKind.Scalar, TextType.Instance, false));
            cols.Add(new SchemaShape.Column(MetadataUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false));
            cols.Add(new SchemaShape.Column(MetadataUtils.Kinds.ScoreValueKind, SchemaShape.Column.VectorKind.Scalar, TextType.Instance, false));

            return cols;
        }

        IPredictor ITrainer.Train(TrainContext context) => ((ITrainer<IPredictor>)this).Train(context);

        /// <summary>
        /// Fits the data to the trainer.
        /// </summary>
        /// <param name="input">The input data to fit to.</param>
        /// <returns>The transformer.</returns>
        public abstract TTransformer Fit(IDataView input);
    }
}