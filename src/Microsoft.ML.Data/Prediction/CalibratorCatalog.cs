// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Calibrator;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Model;
using Microsoft.ML.Training;

[assembly: LoadableClass(typeof(CalibratorTransformer<PlattCalibrator>), typeof(PlattCalibratorTransformer), null,
    typeof(SignatureLoadModel), "", PlattCalibratorTransformer.LoadName)]

[assembly: LoadableClass(typeof(CalibratorTransformer<NaiveCalibrator>), typeof(NaiveCalibratorTransformer), null,
    typeof(SignatureLoadModel), "", NaiveCalibratorTransformer.LoadName)]

[assembly: LoadableClass(typeof(CalibratorTransformer<PavCalibrator>), typeof(PavCalibratorTransformer), null,
    typeof(SignatureLoadModel), "", PavCalibratorTransformer.LoadName)]

namespace Microsoft.ML.Calibrator
{

    /// <summary>
    /// An interface for probability calibrators.
    /// </summary>
    public interface ICalibrator
    {
        /// <summary> Given a classifier output, produce the probability.</summary>
        float PredictProbability(float output);
    }

    /// <summary>
    /// Base class for CalibratorEstimators.
    /// </summary>
    /// <remarks>
    /// CalibratorEstimators take an <see cref="IDataView"/> (the output of a <see cref="BinaryClassifierScorer"/>)
    /// that contains a &quot;Score&quot; column, and converts the scores to probabilities(through binning, interpolation etc.), based on the <typeparamref name="TICalibrator"/> type.
    /// They are used in pipelines where the binary classifier produces non-calibrated scores.
    /// </remarks>
    /// <example>
    /// <format type="text/markdown">
    /// <![CDATA[
    ///  [!code-csharp[Calibrators](~/../docs/samples/docs/samples/Microsoft.ML.Samples/Dynamic/Calibrator.cs)]
    /// ]]></format>
    /// </example>
    public abstract class CalibratorEstimatorBase<TCalibratorTrainer, TICalibrator> : IEstimator<CalibratorTransformer<TICalibrator>>
        where TCalibratorTrainer : ICalibratorTrainer
        where TICalibrator : class, ICalibrator
    {
        protected readonly IHostEnvironment Host;
        protected readonly TCalibratorTrainer CalibratorTrainer;

        protected readonly IPredictor Predictor;
        protected readonly SchemaShape.Column ScoreColumn;
        protected readonly SchemaShape.Column FeatureColumn;
        protected readonly SchemaShape.Column LabelColumn;
        protected readonly SchemaShape.Column WeightColumn;
        protected readonly SchemaShape.Column PredictedLabel;

        protected CalibratorEstimatorBase(IHostEnvironment env,
            TCalibratorTrainer calibratorTrainer,
            IPredictor predictor = null,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null)
        {
            Host = env;
            Predictor = predictor;
            CalibratorTrainer = calibratorTrainer;

            ScoreColumn = TrainerUtils.MakeR4ScalarColumn(DefaultColumnNames.Score); // Do we fantom this being named anything else (renaming column)? Complete metadata?
            LabelColumn = TrainerUtils.MakeBoolScalarLabel(labelColumn);
            FeatureColumn = TrainerUtils.MakeR4VecFeature(featureColumn);
            PredictedLabel = new SchemaShape.Column(DefaultColumnNames.PredictedLabel,
                SchemaShape.Column.VectorKind.Scalar,
                BoolType.Instance,
                false,
                new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()));

            if (weightColumn != null)
                WeightColumn = TrainerUtils.MakeR4ScalarWeightColumn(weightColumn);
        }

        /// <summary>
        /// Gets the output <see cref="SchemaShape"/> of the <see cref="IDataView"/> after fitting the calibrator.
        /// Fitting the calibrator will add a column named "Probability" to the schema. If you already had such a column, a new one will be added.
        /// </summary>
        /// <param name="inputSchema">The input <see cref="SchemaShape"/>.</param>
        SchemaShape IEstimator<CalibratorTransformer<TICalibrator>>.GetOutputSchema(SchemaShape inputSchema)
        {
            Action<SchemaShape.Column, string> checkColumnValid = (SchemaShape.Column column, string columnRole) =>
            {
                if (column.IsValid)
                {
                    if (!inputSchema.TryFindColumn(column.Name, out var outCol))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), columnRole, column.Name);
                    if (!column.IsCompatibleWith(outCol))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), columnRole, column.Name, column.GetTypeString(), outCol.GetTypeString());
                }
            };

            // check the input schema
            checkColumnValid(ScoreColumn, DefaultColumnNames.Score);
            checkColumnValid(WeightColumn, DefaultColumnNames.Weight);
            checkColumnValid(LabelColumn, DefaultColumnNames.Label);
            checkColumnValid(FeatureColumn, DefaultColumnNames.Features);
            checkColumnValid(PredictedLabel, DefaultColumnNames.PredictedLabel);

            //create the new Probability column
            var outColumns = inputSchema.ToDictionary(x => x.Name);
            outColumns[DefaultColumnNames.Probability] = new SchemaShape.Column(DefaultColumnNames.Probability,
                SchemaShape.Column.VectorKind.Scalar,
                NumberType.R4,
                false,
                new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true)));

            return new SchemaShape(outColumns.Values);
        }

        /// <summary>
        /// Fits the scored <see cref="IDataView"/> creating a <see cref="CalibratorTransformer{TICalibrator}"/> that can transform the data by adding a
        /// <see cref="DefaultColumnNames.Probability"/> column containing the calibrated <see cref="DefaultColumnNames.Score"/>.
        /// </summary>
        /// <param name="input"></param>
        /// <returns>A trained <see cref="CalibratorTransformer{TICalibrator}"/> that will transform the data by adding the
        /// <see cref="DefaultColumnNames.Probability"/> column.</returns>
        public CalibratorTransformer<TICalibrator> Fit(IDataView input)
        {
            TICalibrator calibrator = null;

            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            roles.Add(RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, DefaultColumnNames.Score));
            roles.Add(RoleMappedSchema.ColumnRole.Label.Bind(LabelColumn.Name));
            roles.Add(RoleMappedSchema.ColumnRole.Feature.Bind(FeatureColumn.Name));
            if (WeightColumn.IsValid)
                roles.Add(RoleMappedSchema.ColumnRole.Weight.Bind(WeightColumn.Name));

            var roleMappedData = new RoleMappedData(input, opt: false, roles.ToArray());

            using (var ch = Host.Start("Creating calibrator."))
                calibrator = (TICalibrator)CalibratorUtils.TrainCalibrator(Host, ch, CalibratorTrainer, Predictor, roleMappedData);

            return Create(Host, calibrator);
        }

        /// <summary>
        /// Implemented by deriving classes that create a concrete calibrator.
        /// </summary>
        protected abstract CalibratorTransformer<TICalibrator> Create(IHostEnvironment env, TICalibrator calibrator);
    }

    /// <summary>
    /// CalibratorTransfomers, the artifact of calling Fit on a <see cref="CalibratorEstimatorBase{TCalibratorTrainer, TICalibrator}"/>.
    /// If you pass a scored data, to the <see cref="CalibratorTransformer{TICalibrator}"/> Transform method, it will add the Probability column
    /// to the dataset. The Probability column is the value of the Score normalized to be a valid probability.
    /// The CalibratorTransformer is an instance of <see cref="ISingleFeaturePredictionTransformer{TModel}"/> where score can be viewed as a feature
    /// while probability is treated as the label.
    /// </summary>
    /// <typeparam name="TICalibrator">The <see cref="ICalibrator"/> used to transform the data.</typeparam>
    public abstract class CalibratorTransformer<TICalibrator> : RowToRowTransformerBase, ISingleFeaturePredictionTransformer<TICalibrator>
        where TICalibrator : class, ICalibrator
    {
        private TICalibrator _calibrator;
        private readonly string _loaderSignature;

        internal CalibratorTransformer(IHostEnvironment env, TICalibrator calibrator, string loaderSignature)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CalibratorTransformer<TICalibrator>)))
        {
            Host.CheckRef(calibrator, nameof(calibrator));

            _loaderSignature = loaderSignature;
            _calibrator = calibrator;
        }

        // Factory method for SignatureLoadModel.
        internal CalibratorTransformer(IHostEnvironment env, ModelLoadContext ctx, string loaderSignature)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CalibratorTransformer<TICalibrator>)))
        {
            Contracts.AssertValue(ctx);

            _loaderSignature = loaderSignature;
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // model: _calibrator
            ctx.LoadModel<TICalibrator, SignatureLoadModel>(env, out _calibrator, @"Calibrator");
        }

        string ISingleFeaturePredictionTransformer<TICalibrator>.FeatureColumn => DefaultColumnNames.Score;

        ColumnType ISingleFeaturePredictionTransformer<TICalibrator>.FeatureColumnType => NumberType.Float;

        TICalibrator IPredictionTransformer<TICalibrator>.Model => _calibrator;

        bool ITransformer.IsRowToRowMapper => true;

        public override void Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // model: _calibrator
            ctx.SaveModel(_calibrator, @"Calibrator");
        }

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper<TICalibrator>(this, _calibrator, schema);

        protected VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CALTRANS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: _loaderSignature,
                loaderAssemblyName: typeof(CalibratorTransformer<>).Assembly.FullName);
        }

    private sealed class Mapper<TCalibrator> : MapperBase
        where TCalibrator : class, ICalibrator
        {
            private TCalibrator _calibrator;
            private int _scoreColIndex;
            private CalibratorTransformer<TCalibrator> _parent;

            internal Mapper(CalibratorTransformer<TCalibrator> parent, TCalibrator calibrator, Schema inputSchema) :
                base(parent.Host, inputSchema, parent)
            {
                _calibrator = calibrator;
                _parent = parent;

                _scoreColIndex = inputSchema.GetColumnOrNull(DefaultColumnNames.Score)?.Index ?? -1;

                parent.Host.Check(_scoreColIndex > 0, "The data to calibrate contains no 'Score' column");
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
               => col => col == _scoreColIndex;

            public override void Save(ModelSaveContext ctx) => _parent.Save(ctx);

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                return new[]
                {
                    new Schema.DetachedColumn(DefaultColumnNames.Probability, NumberType.Float, null)
                };
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                disposer = null;

                Host.Assert(input.IsColumnActive(_scoreColIndex));
                var getScore = input.GetGetter<float>(_scoreColIndex);

                float score = default;

                ValueGetter<float> probability = (ref float dst) =>
                {
                    getScore(ref score);
                    dst = _calibrator.PredictProbability(score);
                };

                return probability;
            }
        }
    }

    /// <summary>
    /// The PlattCalibratorEstimator.
    /// </summary>
    /// <remarks>
    /// For the usage pattern see the example in <see cref="CalibratorEstimatorBase{TCalibratorTrainer, TICalibrator}"/>.
    /// </remarks>
    public sealed class PlattCalibratorEstimator : CalibratorEstimatorBase<PlattCalibratorTrainer, PlattCalibrator>
    {
        /// <summary>
        /// Initializes a new instance of <see cref="PlattCalibratorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="predictor">The predictor used to train the data.</param>
        /// <param name="labelColumn">The label column name.</param>
        /// <param name="featureColumn">The feature column name.</param>
        /// <param name="weightColumn">The weight column name.</param>
        public PlattCalibratorEstimator(IHostEnvironment env,
            IPredictor predictor,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null) : base(env, new PlattCalibratorTrainer(env), predictor, labelColumn, featureColumn, weightColumn)
        {

        }

        protected override CalibratorTransformer<PlattCalibrator> Create(IHostEnvironment env, PlattCalibrator calibrator)
        => new PlattCalibratorTransformer(env, calibrator);
    }

    /// <summary>
    /// Obtains the probability values by fitting the sigmoid:  f(x) = 1 / (1 + exp(-slope * x + offset).
    /// </summary>
    /// <remarks>
    /// For the usage pattern see the example in <see cref="CalibratorEstimatorBase{TCalibratorTrainer, TICalibrator}"/>.
    /// </remarks>
    public sealed class FixedPlattCalibratorEstimator : CalibratorEstimatorBase<FixedPlattCalibratorTrainer, PlattCalibrator>
    {
        /// <summary>
        /// Initializes a new instance of <see cref="FixedPlattCalibratorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="predictor">The predictor used to train the data.</param>
        /// <param name="slope">The slope in the function of the exponent of the sigmoid.</param>
        /// <param name="offset">The offset in the function of the exponent of the sigmoid.</param>
        /// <param name="labelColumn">The label column name.</param>
        /// <param name="featureColumn">The feature column name.</param>
        /// <param name="weightColumn">The weight column name.</param>
        public FixedPlattCalibratorEstimator(IHostEnvironment env,
            IPredictor predictor,
            double slope = 1,
            double offset = 0,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null) : base(env, new FixedPlattCalibratorTrainer(env, new FixedPlattCalibratorTrainer.Arguments()
            {
                Slope = slope,
                Offset = offset
            }), predictor, labelColumn, featureColumn, weightColumn)
        {

        }

        protected override CalibratorTransformer<PlattCalibrator> Create(IHostEnvironment env, PlattCalibrator calibrator)
            => new PlattCalibratorTransformer(env, calibrator);
    }

    /// <summary>
    /// The <see cref="ITransformer"/> implementation obtained by training a <see cref="FixedPlattCalibratorEstimator"/> or a <see cref="PlattCalibratorEstimator"/>.
    /// </summary>
    public sealed class PlattCalibratorTransformer : CalibratorTransformer<PlattCalibrator>
    {
        internal const string LoadName = "PlattCalibratTransf";

        internal PlattCalibratorTransformer(IHostEnvironment env, PlattCalibrator calibrator)
          : base(env, calibrator, LoadName)
        {

        }

        // Factory method for SignatureLoadModel.
        internal PlattCalibratorTransformer(IHostEnvironment env, ModelLoadContext ctx)
            :base(env, ctx, LoadName)
        {

        }
    }

    /// <summary>
    /// The naive binning-based calibratorEstimator.
    /// </summary>
    /// <remarks>
    /// It divides the range of the outputs into equally sized bins. In each bin,
    /// the probability of belonging to class 1, is the number of class 1 instances in the bin, divided by the total number
    /// of instances in the bin.
    /// For the usage pattern see the example in <see cref="CalibratorEstimatorBase{TCalibratorTrainer, TICalibrator}"/>.
    /// </remarks>
    public sealed class NaiveCalibratorEstimator : CalibratorEstimatorBase<NaiveCalibratorTrainer, NaiveCalibrator>
    {
        /// <summary>
        /// Initializes a new instance of <see cref="NaiveCalibratorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="predictor">The predictor used to train the data.</param>
        /// <param name="labelColumn">The label column name.</param>
        /// <param name="featureColumn">The feature column name.</param>
        /// <param name="weightColumn">The weight column name.</param>
        public NaiveCalibratorEstimator(IHostEnvironment env,
            IPredictor predictor,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null) : base(env, new NaiveCalibratorTrainer(env), predictor, labelColumn, featureColumn, weightColumn)
        {

        }

        protected override CalibratorTransformer<NaiveCalibrator> Create(IHostEnvironment env, NaiveCalibrator calibrator)
        => new NaiveCalibratorTransformer(env, calibrator);
    }

    /// <summary>
    /// The <see cref="ITransformer"/> implementation obtained by training a <see cref="NaiveCalibratorEstimator"/>
    /// </summary>
    public sealed class NaiveCalibratorTransformer : CalibratorTransformer<NaiveCalibrator>
    {
        internal const string LoadName = "NaiveCalibratTransf";

        internal NaiveCalibratorTransformer(IHostEnvironment env, NaiveCalibrator calibrator)
          : base(env, calibrator, LoadName)
        {

        }

        // Factory method for SignatureLoadModel.
        internal NaiveCalibratorTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoadName)
        {

        }
    }

    /// <summary>
    /// The PavCalibratorEstimator.
    /// </summary>
    /// <remarks>
    /// For the usage pattern see the example in <see cref="CalibratorEstimatorBase{TCalibratorTrainer, TICalibrator}"/>.
    /// </remarks>
    public sealed class PavCalibratorEstimator : CalibratorEstimatorBase<PavCalibratorTrainer, PavCalibrator>
    {
        /// <summary>
        /// Initializes a new instance of <see cref="PavCalibratorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="predictor">The predictor used to train the data.</param>
        /// <param name="labelColumn">The label column name.</param>
        /// <param name="featureColumn">The feature column name.</param>
        /// <param name="weightColumn">The weight column name.</param>
        public PavCalibratorEstimator(IHostEnvironment env,
            IPredictor predictor,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weightColumn = null) : base(env, new PavCalibratorTrainer(env), predictor, labelColumn, featureColumn, weightColumn)
        {

        }

        protected override CalibratorTransformer<PavCalibrator> Create(IHostEnvironment env, PavCalibrator calibrator)
            => new PavCalibratorTransformer(env, calibrator);

    }

    /// <summary>
    /// The <see cref="ITransformer"/> implementation obtained by training a <see cref="PavCalibratorEstimator"/>
    /// </summary>
    public sealed class PavCalibratorTransformer : CalibratorTransformer<PavCalibrator>
    {
        internal const string LoadName = "PavCalibratTransf";

        internal PavCalibratorTransformer(IHostEnvironment env, PavCalibrator calibrator)
          : base(env, calibrator, LoadName)
        {

        }

        // Factory method for SignatureLoadModel.
        private PavCalibratorTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoadName)
        {

        }
    }
}
