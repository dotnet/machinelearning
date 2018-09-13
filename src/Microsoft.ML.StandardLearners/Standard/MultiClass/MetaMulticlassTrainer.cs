// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Training;
using System.Linq;

namespace Microsoft.ML.Runtime.Learners
{
    using TScalarTrainer = ITrainerEstimator<IPredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;

    public abstract class MetaMulticlassTrainer<TTransformer, TModel> : ITrainerEstimator<TTransformer, TModel>, ITrainer<TModel>
        where TTransformer : IPredictionTransformer<TModel>
        where TModel : IPredictor
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor", ShortName = "p", SortOrder = 4, SignatureType = typeof(SignatureBinaryClassifierTrainer))]
            [TGUI(Label = "Predictor Type", Description = "Type of underlying binary predictor")]
            public IComponentFactory<TScalarTrainer> PredictorType;

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

        protected readonly ArgumentsBase Args;
        protected readonly IHost Host;
        protected readonly ICalibratorTrainer Calibrator;

        private TScalarTrainer _trainer;

        public PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        protected SchemaShape.Column[] OutputColumns { get; }

        public TrainerInfo Info { get; }

        public TScalarTrainer PredictorType;

        /// <summary>
        /// Initializes the <see cref="MetaMulticlassTrainer{TTransformer, TModel}"/> from the Arguments class.
        /// </summary>
        /// <param name="env">The private instance of the <see cref="IHostEnvironment"/>.</param>
        /// <param name="args">The legacy arguments <see cref="ArgumentsBase"/>class.</param>
        /// <param name="name">The component name.</param>
        /// <param name="labelColumn">The label column for the metalinear trainer and the binary trainer.</param>
        /// <param name="singleEstimator">The binary estimator.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not explicitly provided, it will default to <see cref="PlattCalibratorCalibratorTrainer"/></param>
        internal MetaMulticlassTrainer(IHostEnvironment env, ArgumentsBase args, string name, string labelColumn = null,
            TScalarTrainer singleEstimator = null, ICalibratorTrainer calibrator = null)
        {
            Host = Contracts.CheckRef(env, nameof(env)).Register(name);
            Host.CheckValue(args, nameof(args));
            Args = args;

            if (labelColumn != null)
                LabelColumn = new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);

            // Create the first trainer so errors in the args surface early.
            _trainer = singleEstimator ?? CreateTrainer();

            Calibrator = calibrator ?? new PlattCalibratorTrainer(env);

            if (args.Calibrator != null)
                Calibrator = args.Calibrator.CreateComponent(Host);

            // Regarding caching, no matter what the internal predictor, we're performing many passes
            // simply by virtue of this being a meta-trainer, so we will still cache.
            Info = new TrainerInfo(normalization: _trainer.Info.NeedNormalization);

            OutputColumns = new[]

            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false)
            };
        }

        private TScalarTrainer CreateTrainer()
        {
            return Args.PredictorType != null ?
                Args.PredictorType.CreateComponent(Host) :
                new LinearSvm(Host, new LinearSvm.Arguments());
        }

        protected IDataView MapLabelsCore<T>(ColumnType type, RefPredicate<T> equalsTarget, RoleMappedData data)
        {
            Host.AssertValue(type);
            Host.Assert(type.RawType == typeof(T));
            Host.AssertValue(equalsTarget);
            Host.AssertValue(data);
            Host.AssertValue(data.Schema.Label);

            var lab = data.Schema.Label;

            RefPredicate<T> isMissing;
            if (!Args.ImputeMissingLabelsAsNegative && Conversions.Instance.TryGetIsNAPredicate(type, out isMissing))
            {
                return LambdaColumnMapper.Create(Host, "Label mapper", data.Data,
                    lab.Name, lab.Name, type, NumberType.Float,
                    (ref T src, ref float dst) =>
                        dst = equalsTarget(ref src) ? 1 : (isMissing(ref src) ? float.NaN : default(float)));
            }
            return LambdaColumnMapper.Create(Host, "Label mapper", data.Data,
                lab.Name, lab.Name, type, NumberType.Float,
                (ref T src, ref float dst) =>
                    dst = equalsTarget(ref src) ? 1 : default(float));
        }

        protected TScalarTrainer GetTrainer()
        {
            // We may have instantiated the first trainer to use already, from the constructor.
            // If so capture it and set the retained trainer to null; otherwise create a new one.
            var train = _trainer ?? CreateTrainer();
            _trainer = null;
            return train;
        }

        protected abstract TModel TrainCore(IChannel ch, RoleMappedData data, int count);

        /// <summary>
        /// The legacy train method.
        /// </summary>
        /// <param name="context">The trainig context for this learner.</param>
        /// <returns>The trained model.</returns>
        public TModel Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;

            data.CheckFeatureFloatVector();

            int count;
            data.CheckMultiClassLabel(out count);
            Host.Assert(count > 0);

            using (var ch = Host.Start("Training"))
            {
                var pred = TrainCore(ch, data, count);
                ch.Check(pred != null, "Training did not result in a predictor");
                ch.Done();
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

            if (LabelColumn != null)
            {
                if (!inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol))
                    throw Host.Except($"Label column '{LabelColumn.Name}' is not found");

                if (!labelCol.IsKey || labelCol.ItemType != NumberType.R4 || labelCol.ItemType != NumberType.R8)
                    throw Host.ExceptSchemaMismatch(nameof(labelCol), DefaultColumnNames.PredictedLabel, labelCol.Name, "R8, R4 or a Key", labelCol.GetTypeString());
            }

            var outColumns = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var col in OutputColumns)
                outColumns[col.Name] = col;

            return new SchemaShape(outColumns.Values);
        }

        IPredictor ITrainer.Train(TrainContext context) => Train(context);

        /// <summary>
        /// Fits the data to the trainer.
        /// </summary>
        /// <param name="input">The input data to fit to.</param>
        /// <returns>The transformer.</returns>
        public abstract TTransformer Fit(IDataView input);
    }
}