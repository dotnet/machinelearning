// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
namespace Microsoft.ML.Trainers
{
    using TScalarTrainer = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;

    public abstract class MetaMulticlassClassificationTrainer<TTransformer, TModel> : ITrainerEstimator<TTransformer, TModel>, ITrainer<IPredictor>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
    {
        internal abstract class OptionsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor", ShortName = "p", SortOrder = 4, SignatureType = typeof(SignatureBinaryClassifierTrainer))]
            [TGUI(Label = "Predictor Type", Description = "Type of underlying binary predictor")]
            internal IComponentFactory<TScalarTrainer> PredictorType;

            [Argument(ArgumentType.Multiple, HelpText = "Output calibrator", ShortName = "cali", SortOrder = 150, NullName = "<None>", SignatureType = typeof(SignatureCalibrator))]
            internal IComponentFactory<ICalibratorTrainer> Calibrator = new PlattCalibratorTrainerFactory();

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Number of instances to train the calibrator", SortOrder = 150, ShortName = "numcali")]
            internal int MaxCalibrationExamples = 1000000000;

            [Argument(ArgumentType.Multiple, HelpText = "Whether to treat missing labels as having negative labels, or exclude their rows from dataview.", SortOrder = 150, ShortName = "missNeg")]
            public bool ImputeMissingLabelsAsNegative;
        }

        /// <summary>
        /// The label column that the trainer expects.
        /// </summary>
        private protected readonly SchemaShape.Column LabelColumn;

        private protected readonly OptionsBase Args;
        private protected readonly IHost Host;
        private protected readonly ICalibratorTrainer Calibrator;
        private protected readonly TScalarTrainer Trainer;

        PredictionKind ITrainer.PredictionKind => PredictionKind;
        private protected PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        private protected SchemaShape.Column[] OutputColumns;

        public TrainerInfo Info { get; }

        /// <summary>
        /// Initializes the <see cref="MetaMulticlassClassificationTrainer{TTransformer, TModel}"/> from the <see cref="OptionsBase"/> class.
        /// </summary>
        /// <param name="env">The private instance of the <see cref="IHostEnvironment"/>.</param>
        /// <param name="options">The legacy arguments <see cref="OptionsBase"/>class.</param>
        /// <param name="name">The component name.</param>
        /// <param name="labelColumn">The label column for the metalinear trainer and the binary trainer.</param>
        /// <param name="singleEstimator">The binary estimator.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not explicitly provided, it will default to <see cref="PlattCalibratorTrainer"/></param>
        internal MetaMulticlassClassificationTrainer(IHostEnvironment env, OptionsBase options, string name, string labelColumn = null,
            TScalarTrainer singleEstimator = null, ICalibratorTrainer calibrator = null)
        {
            Host = Contracts.CheckRef(env, nameof(env)).Register(name);
            Host.CheckValue(options, nameof(options));
            Args = options;

            if (labelColumn != null)
                LabelColumn = new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true);

            Trainer = singleEstimator ?? CreateTrainer();

            Calibrator = calibrator ?? new PlattCalibratorTrainer(env);
            if (options.Calibrator != null)
                Calibrator = options.Calibrator.CreateComponent(Host);

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

        private protected IDataView MapLabelsCore<T>(DataViewType type, InPredicate<T> equalsTarget, RoleMappedData data)
        {
            Host.AssertValue(type);
            Host.Assert(type.RawType == typeof(T));
            Host.AssertValue(equalsTarget);
            Host.AssertValue(data);
            Host.Assert(data.Schema.Label.HasValue);

            var label = data.Schema.Label.Value;
            IDataView dataView = data.Data;
            if (!Args.ImputeMissingLabelsAsNegative)
                dataView = new NAFilter(Host, data.Data, false, label.Name);

            return LambdaColumnMapper.Create(Host, "Label mapper", data.Data,
                label.Name, label.Name, type, BooleanDataViewType.Instance,
                (in T src, ref bool dst) =>
                    dst = equalsTarget(in src) ? true : false);
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
            data.CheckMulticlassLabel(out count);
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

                var metadata = new SchemaShape(labelCol.Annotations.Where(x => x.Name == AnnotationUtils.Kinds.KeyValues)
                                .Concat(MetadataForScoreColumn()));
                return new[]
                {
                    new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol))),
                    new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true, metadata)
                };
            }
            else
                return new[]
                {
                    new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(MetadataForScoreColumn())),
                    new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true, new SchemaShape(MetadataForScoreColumn()))
                };
        }

        /// <summary>
        /// Normal metadata that we produce for score columns.
        /// </summary>
        private static IEnumerable<SchemaShape.Column> MetadataForScoreColumn()
        {
            var cols = new List<SchemaShape.Column>();
            cols.Add(new SchemaShape.Column(AnnotationUtils.Kinds.ScoreColumnSetId, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true));
            cols.Add(new SchemaShape.Column(AnnotationUtils.Kinds.ScoreColumnKind, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false));
            cols.Add(new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false));
            cols.Add(new SchemaShape.Column(AnnotationUtils.Kinds.ScoreValueKind, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false));

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