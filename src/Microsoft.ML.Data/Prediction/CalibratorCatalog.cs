// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(typeof(CalibratorTransformer<PlattCalibrator>), typeof(PlattCalibratorTransformer), null,
    typeof(SignatureLoadModel), "", PlattCalibratorTransformer.LoadName)]

[assembly: LoadableClass(typeof(CalibratorTransformer<NaiveCalibrator>), typeof(NaiveCalibratorTransformer), null,
    typeof(SignatureLoadModel), "", NaiveCalibratorTransformer.LoadName)]

[assembly: LoadableClass(typeof(CalibratorTransformer<IsotonicCalibrator>), typeof(IsotonicCalibratorTransformer), null,
    typeof(SignatureLoadModel), "", IsotonicCalibratorTransformer.LoadName)]

namespace Microsoft.ML.Calibrators
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
    /// Base class for calibrator estimators.
    /// </summary>
    /// <remarks>
    /// CalibratorEstimators take an <see cref="IDataView"/> (the output of a <see cref="BinaryClassifierScorer"/>)
    /// that contains a &quot;Score&quot; column, and converts the scores to probabilities(through binning, interpolation etc.), based on the <typeparamref name="TICalibrator"/> type.
    /// They are used in pipelines where the binary classifier produces non-calibrated scores.
    /// </remarks>
    public abstract class CalibratorEstimatorBase<TICalibrator> : IEstimator<CalibratorTransformer<TICalibrator>>, IHaveCalibratorTrainer
        where TICalibrator : class, ICalibrator
    {
        [BestFriend]
        private protected readonly IHostEnvironment Host;
        private readonly ICalibratorTrainer _calibratorTrainer;
        ICalibratorTrainer IHaveCalibratorTrainer.CalibratorTrainer => _calibratorTrainer;

        [BestFriend]
        private protected readonly SchemaShape.Column ScoreColumn;
        [BestFriend]
        private protected readonly SchemaShape.Column LabelColumn;
        [BestFriend]
        private protected readonly SchemaShape.Column WeightColumn;
        [BestFriend]
        private protected readonly SchemaShape.Column PredictedLabel;

        [BestFriend]
        private protected CalibratorEstimatorBase(IHostEnvironment env,
            ICalibratorTrainer calibratorTrainer, string labelColumn, string scoreColumn, string weightColumn)
        {
            Host = env;
            _calibratorTrainer = calibratorTrainer;

            if (!string.IsNullOrWhiteSpace(labelColumn))
                LabelColumn = TrainerUtils.MakeBoolScalarLabel(labelColumn);
            else
                env.CheckParam(!calibratorTrainer.NeedsTraining, nameof(labelColumn), "For trained calibrators, " + nameof(labelColumn) + " must be specified.");
            ScoreColumn = TrainerUtils.MakeR4ScalarColumn(scoreColumn); // Do we fathom this being named anything else (renaming column)? Complete metadata?

            if (weightColumn != null)
                WeightColumn = TrainerUtils.MakeR4ScalarWeightColumn(weightColumn);
        }

        /// <summary>
        /// Gets the output <see cref="SchemaShape"/> of the <see cref="IDataView"/> after fitting the calibrator.
        /// Fitting the calibrator will add a column named "Probability" to the schema. If you already had such a column, a new one will be added.
        /// The same annotation data that would be produced by <see cref="AnnotationUtils.GetTrainerOutputAnnotation(bool)"/> is marked as
        /// being present on the output, if it is present on the input score column.
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

            // Check the input schema.
            checkColumnValid(ScoreColumn, "score");
            checkColumnValid(WeightColumn, "weight");
            checkColumnValid(LabelColumn, "label");

            bool success = inputSchema.TryFindColumn(ScoreColumn.Name, out var inputScoreCol);
            Host.Assert(success);
            const SchemaShape.Column.VectorKind scalar = SchemaShape.Column.VectorKind.Scalar;

            var annotations = new List<SchemaShape.Column>();
            annotations.Add(new SchemaShape.Column(AnnotationUtils.Kinds.IsNormalized,
                SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
            // We only propagate this training column metadata if it looks like it's all there, and all correct.
            if (inputScoreCol.Annotations.TryFindColumn(AnnotationUtils.Kinds.ScoreColumnSetId, out var setIdCol) &&
                setIdCol.Kind == scalar && setIdCol.IsKey && setIdCol.ItemType == NumberDataViewType.UInt32 &&
                inputScoreCol.Annotations.TryFindColumn(AnnotationUtils.Kinds.ScoreColumnKind, out var kindCol) &&
                kindCol.Kind == scalar && kindCol.ItemType is TextDataViewType &&
                inputScoreCol.Annotations.TryFindColumn(AnnotationUtils.Kinds.ScoreValueKind, out var valueKindCol) &&
                valueKindCol.Kind == scalar && valueKindCol.ItemType is TextDataViewType)
            {
                annotations.Add(setIdCol);
                annotations.Add(kindCol);
                annotations.Add(valueKindCol);
            }

            // Create the new Probability column.
            var outColumns = inputSchema.ToDictionary(x => x.Name);
            outColumns[DefaultColumnNames.Probability] = new SchemaShape.Column(DefaultColumnNames.Probability,
                SchemaShape.Column.VectorKind.Scalar,
                NumberDataViewType.Single,
                false, new SchemaShape(annotations));

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
            using (var ch = Host.Start("Creating calibrator."))
            {
                var calibrator = (TICalibrator)CalibratorUtils.TrainCalibrator(Host, ch,
                    _calibratorTrainer, input, LabelColumn.Name, ScoreColumn.Name, WeightColumn.Name);
                return Create(Host, calibrator, ScoreColumn.Name);
            }
        }

        /// <summary>
        /// Implemented by deriving classes that create a concrete calibrator.
        /// </summary>
        [BestFriend]
        private protected abstract CalibratorTransformer<TICalibrator> Create(IHostEnvironment env, TICalibrator calibrator, string scoreColumnName);
    }

    /// <summary>
    /// An instance of this class is the result of calling <see cref="CalibratorEstimatorBase{TICalibrator}.Fit(IDataView)"/>.
    /// If you pass a scored data, to the <see cref="CalibratorTransformer{TICalibrator}"/> Transform method, it will add the Probability column
    /// to the dataset. The Probability column is the value of the Score normalized to be a valid probability.
    /// The <see cref="CalibratorTransformer{TICalibrator}"/> is an instance of <see cref="ISingleFeaturePredictionTransformer{TModel}"/>
    /// where score can be viewed as a feature while probability is treated as the label.
    /// </summary>
    /// <typeparam name="TICalibrator">The <see cref="ICalibrator"/> used to transform the data.</typeparam>
    public abstract class CalibratorTransformer<TICalibrator> : RowToRowTransformerBase, ISingleFeaturePredictionTransformer<TICalibrator>, ISingleFeaturePredictionTransformer
        where TICalibrator : class, ICalibrator
    {
        private readonly TICalibrator _calibrator;
        private readonly string _loaderSignature;
        private readonly string _scoreColumnName;

        private protected CalibratorTransformer(IHostEnvironment env, TICalibrator calibrator, string loaderSignature, string scoreColumnName)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CalibratorTransformer<TICalibrator>)))
        {
            _loaderSignature = loaderSignature;
            _calibrator = calibrator;
            _scoreColumnName = scoreColumnName;
        }

        // Factory method for SignatureLoadModel.
        private protected CalibratorTransformer(IHostEnvironment env, ModelLoadContext ctx, string loaderSignature)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CalibratorTransformer<TICalibrator>)))
        {
            Contracts.AssertValue(ctx);

            _loaderSignature = loaderSignature;
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // model: _calibrator
            // scoreColumnName: _scoreColumnName
            ctx.LoadModel<TICalibrator, SignatureLoadModel>(env, out _calibrator, "Calibrator");
            if (ctx.Header.ModelVerWritten >= 0x00010002)
            {
                _scoreColumnName = ctx.LoadString();
            }
            else
            {
                _scoreColumnName = DefaultColumnNames.Score;
            }
        }

        string ISingleFeaturePredictionTransformer<TICalibrator>.FeatureColumnName => DefaultColumnNames.Score;
        string ISingleFeaturePredictionTransformer.FeatureColumnName => ((ISingleFeaturePredictionTransformer<TICalibrator>)this).FeatureColumnName;

        DataViewType ISingleFeaturePredictionTransformer<TICalibrator>.FeatureColumnType => NumberDataViewType.Single;

        TICalibrator IPredictionTransformer<TICalibrator>.Model => _calibrator;

        bool ITransformer.IsRowToRowMapper => true;

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // model: _calibrator
            // scoreColumnName: _scoreColumnName
            ctx.SaveModel(_calibrator, "Calibrator");
            ctx.SaveString(_scoreColumnName);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper<TICalibrator>(this, _calibrator, schema, _scoreColumnName);

        private protected VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CALTRANS",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Added _scoreColumnName
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: _loaderSignature,
                loaderAssemblyName: typeof(CalibratorTransformer<>).Assembly.FullName);
        }

        private sealed class Mapper<TCalibrator> : MapperBase, ISaveAsOnnx
            where TCalibrator : class, ICalibrator
        {
            private readonly TCalibrator _calibrator;
            private readonly int _scoreColIndex;
            private readonly CalibratorTransformer<TCalibrator> _parent;
            private readonly string _scoreColumnName;

            bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => _calibrator is ICanSaveOnnx onnxMapper ? onnxMapper.CanSaveOnnx(ctx) : false;

            internal Mapper(CalibratorTransformer<TCalibrator> parent, TCalibrator calibrator, DataViewSchema inputSchema, string scoreColumnName) :
                base(parent.Host, inputSchema, parent)
            {
                _calibrator = calibrator;
                _parent = parent;

                _scoreColumnName = scoreColumnName;
                _scoreColIndex = inputSchema.GetColumnOrNull(_scoreColumnName)?.Index ?? -1;

                parent.Host.Check(_scoreColIndex >= 0, "The data to calibrate contains no \'" + scoreColumnName + "\' column.");
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
               => col => col == _scoreColIndex;

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);

            void ISaveAsOnnx.SaveAsOnnx(OnnxContext ctx)
            {
                var scoreName = InputSchema[_scoreColIndex].Name;
                var probabilityName = GetOutputColumnsCore()[0].Name;
                Host.CheckValue(ctx, nameof(ctx));
                if (_calibrator is ISingleCanSaveOnnx onnx)
                {
                    Host.Check(onnx.CanSaveOnnx(ctx), "Cannot be saved as ONNX.");
                    scoreName = ctx.GetVariableName(scoreName);
                    probabilityName = ctx.AddIntermediateVariable(NumberDataViewType.Single, probabilityName);
                    onnx.SaveAsOnnx(ctx, new[] { scoreName, probabilityName }, ""); // No need for featureColumn
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var builder = new DataViewSchema.Annotations.Builder();
                var annotation = InputSchema[_scoreColIndex].Annotations;
                var schema = annotation.Schema;

                // We only propagate this training column metadata if it looks like it's all there, and all correct.
                if (schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreColumnSetId) is DataViewSchema.Column setIdCol &&
                    setIdCol.Type is KeyDataViewType setIdType && setIdType.RawType == typeof(uint) &&
                    schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreColumnKind) is DataViewSchema.Column kindCol &&
                    kindCol.Type is TextDataViewType &&
                    schema.GetColumnOrNull(AnnotationUtils.Kinds.ScoreValueKind) is DataViewSchema.Column valueKindCol &&
                    valueKindCol.Type is TextDataViewType)
                {
                    builder.Add(setIdCol.Name, setIdType, annotation.GetGetter<uint>(setIdCol));
                    // Now, this next one I'm a little less sure about. It is entirely reasonable for someone to, say,
                    // try to calibrate the result of a regression or ranker training, or something else. But should we
                    // just pass through this class just like that? Having thought through the alternatives I view this
                    // as the least harmful thing we could be doing, but it is something to consider I may be wrong
                    // about if it proves that it ever causes problems to, say, have something identified as a probability
                    // column but be marked as being a regression task, or what have you.
                    builder.Add(kindCol.Name, kindCol.Type, annotation.GetGetter<ReadOnlyMemory<char>>(kindCol));
                    builder.Add(valueKindCol.Name, valueKindCol.Type, annotation.GetGetter<ReadOnlyMemory<char>>(valueKindCol));
                }
                // Probabilities are always considered normalized.
                builder.Add(AnnotationUtils.Kinds.IsNormalized, BooleanDataViewType.Instance, (ref bool value) => value = true);

                return new[]
                {
                    new DataViewSchema.DetachedColumn(DefaultColumnNames.Probability, NumberDataViewType.Single, builder.ToAnnotations())
                };
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                disposer = null;

                Host.Assert(input.IsColumnActive(input.Schema[_scoreColIndex]));
                var getScore = input.GetGetter<float>(input.Schema[_scoreColIndex]);

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
    /// The Platt calibrator estimator.
    /// </summary>
    public sealed class PlattCalibratorEstimator : CalibratorEstimatorBase<PlattCalibrator>
    {
        /// <summary>
        /// Initializes a new instance of <see cref="PlattCalibratorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        ///  /// <param name="labelColumnName">The name of the label column.This is consumed both when this estimator
        /// is fit and when the estimator is consumed.</param>
        /// <param name="scoreColumnName">The name of the score column.This is consumed when this estimator is fit,
        /// but not consumed by the resulting transformer.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional). Note that if specified this is
        /// consumed when this estimator is fit, but not consumed by the resulting transformer.</param>
        internal PlattCalibratorEstimator(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string scoreColumnName = DefaultColumnNames.Score,
            string exampleWeightColumnName = null) : base(env, new PlattCalibratorTrainer(env), labelColumnName, scoreColumnName, exampleWeightColumnName)
        {
        }

        [BestFriend]
        private protected override CalibratorTransformer<PlattCalibrator> Create(IHostEnvironment env, PlattCalibrator calibrator, string scoreColumnName)
            => new PlattCalibratorTransformer(env, calibrator, scoreColumnName);
    }

    /// <summary>
    /// Obtains the probability values by applying the sigmoid:  f(x) = 1 / (1 + exp(-slope * x + offset).
    /// Note that unlike, say, <see cref="PlattCalibratorEstimator"/>, the fit function here is trivial
    /// and just "fits" a calibrator with the provided parameters.
    /// </summary>
    public sealed class FixedPlattCalibratorEstimator : CalibratorEstimatorBase<PlattCalibrator>
    {
        /// <summary>
        /// Initializes a new instance of <see cref="FixedPlattCalibratorEstimator"/>.
        /// </summary>
        /// <remarks>
        /// Note that unlike many other calibrator estimators this one has the parameters pre-specified.
        /// This means that it does not have a label or weight column specified as an input during training.
        /// </remarks>
        /// <param name="env">The environment to use.</param>
        /// <param name="slope">The slope in the function of the exponent of the sigmoid.</param>
        /// <param name="offset">The offset in the function of the exponent of the sigmoid.</param>
        /// <param name="scoreColumn">The score column name. This is consumed both when this estimator
        /// is fit and when the estimator is consumed.</param>
        internal FixedPlattCalibratorEstimator(IHostEnvironment env,
            double slope = 1,
            double offset = 0,
            string scoreColumn = DefaultColumnNames.Score)
            : base(env, new FixedPlattCalibratorTrainer(env, new FixedPlattCalibratorTrainer.Arguments()
            {
                Slope = slope,
                Offset = offset
            }), null, scoreColumn, null)
        {

        }

        [BestFriend]
        private protected override CalibratorTransformer<PlattCalibrator> Create(IHostEnvironment env, PlattCalibrator calibrator, string scoreColumnName)
            => new PlattCalibratorTransformer(env, calibrator, scoreColumnName);
    }

    /// <summary>
    /// The <see cref="ITransformer"/> implementation obtained by training a <see cref="FixedPlattCalibratorEstimator"/> or a <see cref="PlattCalibratorEstimator"/>.
    /// </summary>
    public sealed class PlattCalibratorTransformer : CalibratorTransformer<PlattCalibrator>
    {
        internal const string LoadName = "PlattCalibratTransf";

        internal PlattCalibratorTransformer(IHostEnvironment env, PlattCalibrator calibrator, string scoreColumnName)
          : base(env, calibrator, LoadName, scoreColumnName)
        {
        }

        // Factory method for SignatureLoadModel.
        internal PlattCalibratorTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoadName)
        {
        }
    }

    /// <summary>
    /// The naive binning-based calibrator estimator.
    /// </summary>
    /// <remarks>
    /// It divides the range of the outputs into equally sized bins. In each bin,
    /// the probability of belonging to class 1, is the number of class 1 instances in the bin, divided by the total number
    /// of instances in the bin.
    /// </remarks>
    public sealed class NaiveCalibratorEstimator : CalibratorEstimatorBase<NaiveCalibrator>
    {
        /// <summary>
        /// Initializes a new instance of <see cref="NaiveCalibratorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The label column name. This is consumed when this estimator is fit,
        /// but not consumed by the resulting transformer.</param>
        /// <param name="scoreColumn">The score column name. This is consumed both when this estimator
        /// is fit and when the estimator is consumed.</param>
        /// <param name="weightColumn">The optional weight column name. Note that if specified this is
        /// consumed when this estimator is fit, but not consumed by the resulting transformer.</param>
        internal NaiveCalibratorEstimator(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string scoreColumn = DefaultColumnNames.Score,
            string weightColumn = null) : base(env, new NaiveCalibratorTrainer(env), labelColumn, scoreColumn, weightColumn)
        {
        }

        [BestFriend]
        private protected override CalibratorTransformer<NaiveCalibrator> Create(IHostEnvironment env, NaiveCalibrator calibrator, string scoreColumnName)
            => new NaiveCalibratorTransformer(env, calibrator, scoreColumnName);
    }

    /// <summary>
    /// The <see cref="ITransformer"/> implementation obtained by training a <see cref="NaiveCalibratorEstimator"/>
    /// </summary>
    public sealed class NaiveCalibratorTransformer : CalibratorTransformer<NaiveCalibrator>
    {
        internal const string LoadName = "NaiveCalibratTransf";

        internal NaiveCalibratorTransformer(IHostEnvironment env, NaiveCalibrator calibrator, string scoreColumnName)
          : base(env, calibrator, LoadName, scoreColumnName)
        {
        }

        // Factory method for SignatureLoadModel.
        internal NaiveCalibratorTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoadName)
        {
        }
    }

    /// <summary>
    /// The isotonic calbrated estimator.
    /// </summary>
    /// <remarks>
    /// Calibrator finds a stepwise constant function (using the Pool Adjacent Violators Algorithm aka PAV) that minimizes the squared error.
    /// </remarks>
    public sealed class IsotonicCalibratorEstimator : CalibratorEstimatorBase<IsotonicCalibrator>
    {
        /// <summary>
        /// Initializes a new instance of <see cref="IsotonicCalibratorEstimator"/>
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="labelColumn">The label column name. This is consumed when this estimator is fit,
        /// but not consumed by the resulting transformer.</param>
        /// <param name="scoreColumn">The score column name. This is consumed both when this estimator
        /// is fit and when the estimator is consumed.</param>
        /// <param name="weightColumn">The optional weight column name. Note that if specified this is
        /// consumed when this estimator is fit, but not consumed by the resulting transformer.</param>
        internal IsotonicCalibratorEstimator(IHostEnvironment env,
            string labelColumn = DefaultColumnNames.Label,
            string scoreColumn = DefaultColumnNames.Score,
            string weightColumn = null) : base(env, new PavCalibratorTrainer(env), labelColumn, scoreColumn, weightColumn)
        {
        }

        [BestFriend]
        private protected override CalibratorTransformer<IsotonicCalibrator> Create(IHostEnvironment env, IsotonicCalibrator calibrator, string scoreColumnName)
            => new IsotonicCalibratorTransformer(env, calibrator, scoreColumnName);

    }

    /// <summary>
    /// The <see cref="ITransformer"/> implementation obtained by training a <see cref="IsotonicCalibratorEstimator"/>
    /// </summary>
    public sealed class IsotonicCalibratorTransformer : CalibratorTransformer<IsotonicCalibrator>
    {
        internal const string LoadName = "PavCalibratTransf";

        internal IsotonicCalibratorTransformer(IHostEnvironment env, IsotonicCalibrator calibrator, string scoreColumnName)
          : base(env, calibrator, LoadName, scoreColumnName)
        {
        }

        // Factory method for SignatureLoadModel.
        private IsotonicCalibratorTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoadName)
        {
        }
    }
}
