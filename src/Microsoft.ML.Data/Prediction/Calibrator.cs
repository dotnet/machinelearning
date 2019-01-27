// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Calibrator;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Onnx;
using Microsoft.ML.Model.Pfa;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(PlattCalibratorTrainer.Summary, typeof(PlattCalibratorTrainer), null, typeof(SignatureCalibrator),
    PlattCalibratorTrainer.UserName,
    PlattCalibratorTrainer.LoadName,
    "SigmoidCalibration")]

[assembly: LoadableClass(FixedPlattCalibratorTrainer.Summary, typeof(FixedPlattCalibratorTrainer), typeof(FixedPlattCalibratorTrainer.Arguments), typeof(SignatureCalibrator),
    FixedPlattCalibratorTrainer.UserName,
    FixedPlattCalibratorTrainer.LoadName,
    "FixedSigmoidCalibration")]

[assembly: LoadableClass(PavCalibratorTrainer.Summary, typeof(PavCalibratorTrainer), null, typeof(SignatureCalibrator),
    PavCalibratorTrainer.UserName,
    PavCalibratorTrainer.LoadName,
    "PAV")]

[assembly: LoadableClass(NaiveCalibratorTrainer.Summary, typeof(NaiveCalibratorTrainer), null, typeof(SignatureCalibrator),
    NaiveCalibratorTrainer.UserName,
    NaiveCalibratorTrainer.LoadName,
    "Naive",
    "NaiveCalibration")]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(PlattCalibrator), null, typeof(SignatureLoadModel),
    "Platt Calibration Executor",
    PlattCalibrator.LoaderSignature)]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(PavCalibrator), null, typeof(SignatureLoadModel),
    "PAV Calibration Executor",
    PavCalibrator.LoaderSignature)]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(NaiveCalibrator), null, typeof(SignatureLoadModel),
    "Naive Calibration Executor",
    NaiveCalibrator.LoaderSignature)]

[assembly: LoadableClass(typeof(CalibratedPredictor), null, typeof(SignatureLoadModel),
    "Calibrated Predictor Executor",
    CalibratedPredictor.LoaderSignature, "BulkCaliPredExec")]

[assembly: LoadableClass(typeof(FeatureWeightsCalibratedPredictor), null, typeof(SignatureLoadModel),
    "Feature Weights Calibrated Predictor Executor",
    FeatureWeightsCalibratedPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(ParameterMixingCalibratedPredictor), null, typeof(SignatureLoadModel),
    "Parameter Mixing Calibrated Predictor Executor",
    ParameterMixingCalibratedPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(SchemaBindableCalibratedPredictor), null, typeof(SignatureLoadModel),
    "Schema Bindable Calibrated Predictor", SchemaBindableCalibratedPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(Calibrate), null, typeof(SignatureEntryPointModule), "Calibrate")]

[assembly: EntryPointModule(typeof(FixedPlattCalibratorTrainer.Arguments))]
[assembly: EntryPointModule(typeof(NaiveCalibratorTrainerFactory))]
[assembly: EntryPointModule(typeof(PavCalibratorTrainerFactory))]
[assembly: EntryPointModule(typeof(PlattCalibratorTrainerFactory))]

namespace Microsoft.ML.Internal.Calibration
{
    /// <summary>
    /// Signature for the loaders of calibrators.
    /// </summary>
    public delegate void SignatureCalibrator();

    [TlcModule.ComponentKind("CalibratorTrainer")]
    public interface ICalibratorTrainerFactory : IComponentFactory<ICalibratorTrainer>
    {
    }

    public interface ICalibratorTrainer
    {
        /// <summary>
        /// True if the calibrator needs training, false otherwise.
        /// </summary>
        bool NeedsTraining { get; }

        /// <summary> Training calibrators:  provide the  output and the class label </summary>
        /// <returns> True if it needs more examples, false otherwise</returns>
        bool ProcessTrainingExample(float output, bool labelIs1, float weight);

        /// <summary> Finish up training after seeing all examples </summary>
        ICalibrator FinishTraining(IChannel ch);
    }

    /// <summary>
    /// An interface for predictors that take care of their own calibration given an input data view.
    /// </summary>
    [BestFriend]
    internal interface ISelfCalibratingPredictor
    {
        IPredictor Calibrate(IChannel ch, IDataView data, ICalibratorTrainer caliTrainer, int maxRows);
    }

    [BestFriend]
    public abstract class CalibratedPredictorBase :
        IDistPredictorProducing<float, float>,
        ICanSaveInIniFormat,
        ICanSaveInTextFormat,
        ICanSaveInSourceCode,
        ICanSaveSummary,
        ICanGetSummaryInKeyValuePairs
    {
        protected readonly IHost Host;

        public IPredictorProducing<float> SubPredictor { get; }
        public ICalibrator Calibrator { get; }
        public PredictionKind PredictionKind => SubPredictor.PredictionKind;

        protected CalibratedPredictorBase(IHostEnvironment env, string name, IPredictorProducing<float> predictor, ICalibrator calibrator)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
            Host.CheckValue(predictor, nameof(predictor));
            Host.CheckValue(calibrator, nameof(calibrator));

            SubPredictor = predictor;
            Calibrator = calibrator;
        }

        void ICanSaveInIniFormat.SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator)
        {
            Host.Check(calibrator == null, "Too many calibrators.");
            var saver = SubPredictor as ICanSaveInIniFormat;
            saver?.SaveAsIni(writer, schema, Calibrator);
        }

        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubPredictor as ICanSaveInTextFormat;
            if (saver != null)
                saver.SaveAsText(writer, schema);
        }

        void ICanSaveInSourceCode.SaveAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubPredictor as ICanSaveInSourceCode;
            if (saver != null)
                saver.SaveAsCode(writer, schema);
        }

        void ICanSaveSummary.SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubPredictor as ICanSaveSummary;
            if (saver != null)
                saver.SaveSummary(writer, schema);
        }

        ///<inheritdoc/>
        IList<KeyValuePair<string, object>> ICanGetSummaryInKeyValuePairs.GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubPredictor as ICanGetSummaryInKeyValuePairs;
            if (saver != null)
                return saver.GetSummaryInKeyValuePairs(schema);

            return null;
        }

        protected void SaveCore(ModelSaveContext ctx)
        {
            ctx.SaveModel(SubPredictor, ModelFileUtils.DirPredictor);
            ctx.SaveModel(Calibrator, @"Calibrator");
        }

        protected static IPredictorProducing<float> GetPredictor(IHostEnvironment env, ModelLoadContext ctx)
        {
            IPredictorProducing<float> predictor;
            ctx.LoadModel<IPredictorProducing<float>, SignatureLoadModel>(env, out predictor, ModelFileUtils.DirPredictor);
            return predictor;
        }

        protected static ICalibrator GetCalibrator(IHostEnvironment env, ModelLoadContext ctx)
        {
            ICalibrator calibrator;
            ctx.LoadModel<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            return calibrator;
        }
    }

    public abstract class ValueMapperCalibratedPredictorBase : CalibratedPredictorBase, IValueMapperDist, IFeatureContributionMapper, ICalculateFeatureContribution,
        IDistCanSavePfa, IDistCanSaveOnnx
    {
        private readonly IValueMapper _mapper;
        private readonly IFeatureContributionMapper _featureContribution;

        ColumnType IValueMapper.InputType => _mapper.InputType;
        ColumnType IValueMapper.OutputType => _mapper.OutputType;
        ColumnType IValueMapperDist.DistType => NumberType.Float;
        bool ICanSavePfa.CanSavePfa => (_mapper as ICanSavePfa)?.CanSavePfa == true;

        public FeatureContributionCalculator FeatureContributionCalculator => new FeatureContributionCalculator(this);

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => (_mapper as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true;

        protected ValueMapperCalibratedPredictorBase(IHostEnvironment env, string name, IPredictorProducing<float> predictor, ICalibrator calibrator)
            : base(env, name, predictor, calibrator)
        {
            Contracts.AssertValue(Host);

            _mapper = SubPredictor as IValueMapper;
            Host.Check(_mapper != null, "The predictor does not implement IValueMapper");
            Host.Check(_mapper.OutputType == NumberType.Float, "The output type of the predictor is expected to be float");

            _featureContribution = predictor as IFeatureContributionMapper;
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            return _mapper.GetMapper<TIn, TOut>();
        }

        ValueMapper<TIn, TOut, TDist> IValueMapperDist.GetMapper<TIn, TOut, TDist>()
        {
            Host.Check(typeof(TOut) == typeof(float));
            Host.Check(typeof(TDist) == typeof(float));
            var map = ((IValueMapper)this).GetMapper<TIn, float>();
            ValueMapper<TIn, float, float> del =
                (in TIn src, ref float score, ref float prob) =>
                {
                    map(in src, ref score);
                    prob = Calibrator.PredictProbability(score);
                };
            return (ValueMapper<TIn, TOut, TDist>)(Delegate)del;
        }

        ValueMapper<TSrc, VBuffer<float>> IFeatureContributionMapper.GetFeatureContributionMapper<TSrc, TDst>(int top, int bottom, bool normalize)
        {
            // REVIEW: checking this a bit too late.
            Host.Check(_featureContribution != null, "Predictor does not implement IFeatureContributionMapper");
            return _featureContribution.GetFeatureContributionMapper<TSrc, TDst>(top, bottom, normalize);
        }

        JToken ISingleCanSavePfa.SaveAsPfa(BoundPfaContext ctx, JToken input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(input, nameof(input));

            Host.Assert(_mapper is ISingleCanSavePfa);
            var mapper = (ISingleCanSavePfa)_mapper;
            return mapper.SaveAsPfa(ctx, input);
        }

        void IDistCanSavePfa.SaveAsPfa(BoundPfaContext ctx, JToken input,
            string score, out JToken scoreToken, string prob, out JToken probToken)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(input, nameof(input));
            Host.CheckValueOrNull(score);
            Host.CheckValueOrNull(prob);

            JToken scoreExpression = ((ISingleCanSavePfa)this).SaveAsPfa(ctx, input);
            scoreToken = ctx.DeclareVar(score, scoreExpression);
            var calibrator = Calibrator as ISingleCanSavePfa;
            if (calibrator?.CanSavePfa != true)
            {
                ctx.Hide(prob);
                probToken = null;
                return;
            }
            JToken probExpression = calibrator.SaveAsPfa(ctx, scoreToken);
            probToken = ctx.DeclareVar(prob, probExpression);
        }

        bool IDistCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumnName)
            => ((ISingleCanSaveOnnx)this).SaveAsOnnx(ctx, outputNames, featureColumnName);

        bool ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumnName)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(outputNames, nameof(outputNames));

            Host.Assert(_mapper is ISingleCanSaveOnnx);

            var mapper = (ISingleCanSaveOnnx)_mapper;
            if (!mapper.SaveAsOnnx(ctx, new[] { outputNames[1] }, featureColumnName))
                return false;

            var calibrator = Calibrator as ISingleCanSaveOnnx;
            if (!(calibrator?.CanSaveOnnx(ctx) == true && calibrator.SaveAsOnnx(ctx, new[] { outputNames[1], outputNames[2] }, featureColumnName)))
                ctx.RemoveVariable(outputNames[1], true);

            return true;
        }

    }

    [BestFriend]
    internal sealed class CalibratedPredictor : ValueMapperCalibratedPredictorBase, ICanSaveModel
    {
        internal CalibratedPredictor(IHostEnvironment env, IPredictorProducing<float> predictor, ICalibrator calibrator)
            : base(env, RegistrationName, predictor, calibrator)
        {
        }

        internal const string LoaderSignature = "CaliPredExec";
        internal const string RegistrationName = "CalibratedPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CALIPRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CalibratedPredictor).Assembly.FullName);
        }
        private static VersionInfo GetVersionInfoBulk()
        {
            return new VersionInfo(
                modelSignature: "BCALPRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CalibratedPredictor).Assembly.FullName);
        }

        private CalibratedPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, GetPredictor(env, ctx), GetCalibrator(env, ctx))
        {
        }

        private static CalibratedPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            // Can load either the old "bulk" model or standard "cali". The two formats are identical.
            var ver1 = GetVersionInfo();
            var ver2 = GetVersionInfoBulk();
            var ver = ctx.Header.ModelSignature == ver2.ModelSignature ? ver2 : ver1;
            ctx.CheckAtModel(ver);
            return new CalibratedPredictor(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveCore(ctx);
        }
    }

    [BestFriend]
    internal sealed class FeatureWeightsCalibratedPredictor :
        ValueMapperCalibratedPredictorBase,
        IPredictorWithFeatureWeights<float>,
        ICanSaveModel
    {
        private readonly IPredictorWithFeatureWeights<float> _featureWeights;

        internal FeatureWeightsCalibratedPredictor(IHostEnvironment env, IPredictorWithFeatureWeights<float> predictor,
            ICalibrator calibrator)
            : base(env, RegistrationName, predictor, calibrator)
        {
            _featureWeights = predictor;
        }

        internal const string LoaderSignature = "FeatWCaliPredExec";
        internal const string RegistrationName = "FeatureWeightsCalibratedPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FTWTCALP",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FeatureWeightsCalibratedPredictor).Assembly.FullName);
        }

        private FeatureWeightsCalibratedPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, GetPredictor(env, ctx), GetCalibrator(env, ctx))
        {
            Host.Check(SubPredictor is IPredictorWithFeatureWeights<float>, "Predictor does not implement " + nameof(IPredictorWithFeatureWeights<float>));
            _featureWeights = (IPredictorWithFeatureWeights<float>)SubPredictor;
        }

        private static FeatureWeightsCalibratedPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new FeatureWeightsCalibratedPredictor(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveCore(ctx);
        }

        public void GetFeatureWeights(ref VBuffer<float> weights)
        {
            _featureWeights.GetFeatureWeights(ref weights);
        }
    }

    /// <summary>
    /// Encapsulates a predictor and a calibrator that implement <see cref="IParameterMixer"/>.
    /// Its implementation of <see cref="IParameterMixer.CombineParameters"/> combines both the predictors and the calibrators.
    /// </summary>
    public sealed class ParameterMixingCalibratedPredictor :
        ValueMapperCalibratedPredictorBase,
        IParameterMixer<float>,
        IPredictorWithFeatureWeights<float>,
        ICanSaveModel
    {
        private readonly IPredictorWithFeatureWeights<float> _featureWeights;

        internal ParameterMixingCalibratedPredictor(IHostEnvironment env, IPredictorWithFeatureWeights<float> predictor, ICalibrator calibrator)
            : base(env, RegistrationName, predictor, calibrator)
        {
            Host.Check(predictor is IParameterMixer<float>, "Predictor does not implement " + nameof(IParameterMixer<float>));
            Host.Check(calibrator is IParameterMixer, "Calibrator does not implement " + nameof(IParameterMixer));
            _featureWeights = predictor;
        }

        internal const string LoaderSignature = "PMixCaliPredExec";
        internal const string RegistrationName = "ParameterMixingCalibratedPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PMIXCALP",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ParameterMixingCalibratedPredictor).Assembly.FullName);
        }

        private ParameterMixingCalibratedPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, GetPredictor(env, ctx), GetCalibrator(env, ctx))
        {
            Host.Check(SubPredictor is IParameterMixer<float>, "Predictor does not implement " + nameof(IParameterMixer));
            Host.Check(SubPredictor is IPredictorWithFeatureWeights<float>, "Predictor does not implement " + nameof(IPredictorWithFeatureWeights<float>));
            _featureWeights = (IPredictorWithFeatureWeights<float>)SubPredictor;
        }

        private static ParameterMixingCalibratedPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new ParameterMixingCalibratedPredictor(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveCore(ctx);
        }

        public void GetFeatureWeights(ref VBuffer<float> weights)
        {
            _featureWeights.GetFeatureWeights(ref weights);
        }

        IParameterMixer<float> IParameterMixer<float>.CombineParameters(IList<IParameterMixer<float>> models)
        {
            var predictors = models.Select(
                m =>
                {
                    var model = m as ParameterMixingCalibratedPredictor;
                    Contracts.Assert(model != null);
                    return (IParameterMixer<float>)model.SubPredictor;
                }).ToArray();
            var calibrators = models.Select(
                m =>
                {
                    var model = m as ParameterMixingCalibratedPredictor;
                    Contracts.Assert(model != null);
                    return (IParameterMixer)model.Calibrator;
                }).ToArray();
            var combinedPredictor = predictors[0].CombineParameters(predictors);
            var combinedCalibrator = calibrators[0].CombineParameters(calibrators);
            return new ParameterMixingCalibratedPredictor(Host, (IPredictorWithFeatureWeights<float>)combinedPredictor, (ICalibrator)combinedCalibrator);
        }
    }

    [BestFriend]
    internal sealed class SchemaBindableCalibratedPredictor : CalibratedPredictorBase, ISchemaBindableMapper, ICanSaveModel,
        IBindableCanSavePfa, IBindableCanSaveOnnx, IFeatureContributionMapper
    {
        private sealed class Bound : ISchemaBoundRowMapper
        {
            private readonly SchemaBindableCalibratedPredictor _parent;
            private readonly ISchemaBoundRowMapper _predictor;
            private readonly int _scoreCol;

            public ISchemaBindableMapper Bindable => _parent;
            public RoleMappedSchema InputRoleMappedSchema => _predictor.InputRoleMappedSchema;
            public Schema InputSchema => _predictor.InputSchema;
            public Schema OutputSchema { get; }

            public Bound(IHostEnvironment env, SchemaBindableCalibratedPredictor parent, RoleMappedSchema schema)
            {
                Contracts.AssertValue(env);
                env.AssertValue(parent);
                _parent = parent;
                _predictor = _parent._bindable.Bind(env, schema) as ISchemaBoundRowMapper;
                env.Check(_predictor != null, "Predictor is not a row-to-row mapper");
                if (!_predictor.OutputSchema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out _scoreCol))
                    throw env.Except("Predictor does not output a score");
                var scoreType = _predictor.OutputSchema[_scoreCol].Type;
                env.Check(scoreType is NumberType);
                OutputSchema = ScoreSchemaFactory.CreateBinaryClassificationSchema();
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < OutputSchema.Count; i++)
                {
                    if (predicate(i))
                        return _predictor.GetDependencies(col => true);
                }
                return col => false;
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                return _predictor.GetInputColumnRoles();
            }

            public Row GetRow(Row input, Func<int, bool> predicate)
            {
                Func<int, bool> predictorPredicate = col => false;
                for (int i = 0; i < OutputSchema.Count; i++)
                {
                    if (predicate(i))
                    {
                        predictorPredicate = col => true;
                        break;
                    }
                }
                var predictorRow = _predictor.GetRow(input, predictorPredicate);
                var getters = new Delegate[OutputSchema.Count];
                for (int i = 0; i < OutputSchema.Count - 1; i++)
                {
                    var type = predictorRow.Schema[i].Type;
                    if (!predicate(i))
                        continue;
                    getters[i] = Utils.MarshalInvoke(GetPredictorGetter<int>, type.RawType, predictorRow, i);
                }
                if (predicate(OutputSchema.Count - 1))
                    getters[OutputSchema.Count - 1] = GetProbGetter(predictorRow);
                return new SimpleRow(OutputSchema, predictorRow, getters);
            }

            private Delegate GetPredictorGetter<T>(Row input, int col)
            {
                return input.GetGetter<T>(col);
            }

            private Delegate GetProbGetter(Row input)
            {
                var scoreGetter = RowCursorUtils.GetGetterAs<Single>(NumberType.R4, input, _scoreCol);
                ValueGetter<Single> probGetter =
                    (ref Single dst) =>
                    {
                        Single score = 0;
                        scoreGetter(ref score);
                        dst = _parent.Calibrator.PredictProbability(score);
                    };
                return probGetter;
            }
        }

        private readonly ISchemaBindableMapper _bindable;
        private readonly IFeatureContributionMapper _featureContribution;

        internal const string LoaderSignature = "SchemaBindableCalibrated";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BINDCALI",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SchemaBindableCalibratedPredictor).Assembly.FullName);
        }

        /// <summary>
        /// Whether we can save as PFA. Note that this depends on whether the underlying predictor
        /// can save as PFA, since in the event that this in particular does not get saved,
        /// </summary>
        bool ICanSavePfa.CanSavePfa => (_bindable as ICanSavePfa)?.CanSavePfa == true;

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => (_bindable as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true;

        internal SchemaBindableCalibratedPredictor(IHostEnvironment env, IPredictorProducing<Single> predictor, ICalibrator calibrator)
            : base(env, LoaderSignature, predictor, calibrator)
        {
            _bindable = ScoreUtils.GetSchemaBindableMapper(Host, SubPredictor);
            _featureContribution = SubPredictor as IFeatureContributionMapper;
        }

        private SchemaBindableCalibratedPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, GetPredictor(env, ctx), GetCalibrator(env, ctx))
        {
            _bindable = ScoreUtils.GetSchemaBindableMapper(Host, SubPredictor);
            _featureContribution = SubPredictor as IFeatureContributionMapper;
        }

        private static SchemaBindableCalibratedPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new SchemaBindableCalibratedPredictor(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            SaveCore(ctx);
        }

        void IBindableCanSavePfa.SaveAsPfa(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputs)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(Utils.Size(outputs) == 2, nameof(outputs), "Expected this to have two outputs");
            Host.Check(((ICanSavePfa)this).CanSavePfa, "Called despite not being savable");

            ctx.Hide(outputs);
        }

        bool IBindableCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, RoleMappedSchema schema, string[] outputs)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckParam(Utils.Size(outputs) == 2, nameof(outputs), "Expected this to have two outputs");
            Host.CheckValue(schema, nameof(schema));
            Host.Check(((ICanSaveOnnx)this).CanSaveOnnx(ctx), "Called despite not being savable");
            return false;
        }

        public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Host.CheckValue(env, nameof(env));
            env.CheckValue(schema, nameof(schema));
            return new Bound(Host, this, schema);
        }

        ValueMapper<TSrc, VBuffer<float>> IFeatureContributionMapper.GetFeatureContributionMapper<TSrc, TDst>(int top, int bottom, bool normalize)
        {
            // REVIEW: checking this a bit too late.
            Host.Check(_featureContribution != null, "Predictor does not implement " + nameof(IFeatureContributionMapper));
            return _featureContribution.GetFeatureContributionMapper<TSrc, TDst>(top, bottom, normalize);
        }
    }

    [BestFriend]
    internal static class CalibratorUtils
    {
        // maximum number of rows passed to the calibrator.
        private const int _maxCalibrationExamples = 1000000;

        private static bool NeedCalibration(IHostEnvironment env, IChannel ch, ICalibratorTrainer calibrator,
            ITrainer trainer, IPredictor predictor, RoleMappedSchema schema)
        {
            if (!trainer.Info.NeedCalibration)
            {
                ch.Info("Not training a calibrator because it is not needed.");
                return false;
            }

            if (calibrator == null)
            {
                ch.Info("Not training a calibrator because a valid calibrator trainer was not provided.");
                return false;
            }

            if (schema.Feature == null)
            {
                ch.Info("Not training a calibrator because there is no features column.");
                return false;
            }

            if (schema.Label == null)
            {
                ch.Info("Not training a calibrator because there is no label column.");
                return false;
            }

            if (!(predictor is IPredictorProducing<float>))
            {
                ch.Info("Not training a calibrator because the predictor does not implement IPredictorProducing<float>.");
                return false;
            }

            var bindable = ScoreUtils.GetSchemaBindableMapper(env, predictor);
            var bound = bindable.Bind(env, schema);
            var outputSchema = bound.OutputSchema;
            int scoreCol;
            if (!outputSchema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreCol))
            {
                ch.Info("Not training a calibrator because the predictor does not output a score column.");
                return false;
            }
            var type = outputSchema[scoreCol].Type;
            if (type != NumberType.Float)
            {
                ch.Info("Not training a calibrator because the predictor output is {0}, but expected to be {1}.", type, NumberType.R4);
                return false;
            }
            return true;
        }

        /// <summary>
        /// Trains a calibrator, if needed.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="ch">The channel.</param>
        /// <param name="calibrator">The calibrator trainer.</param>
        /// <param name="maxRows">The maximum rows to use for calibrator training.</param>
        /// <param name="trainer">The trainer used to train the predictor.</param>
        /// <param name="predictor">The predictor that needs calibration.</param>
        /// <param name="data">The examples to used for calibrator training.</param>
        /// <returns>The original predictor, if no calibration is needed,
        /// or a metapredictor that wraps the original predictor and the newly trained calibrator.</returns>
        public static IPredictor TrainCalibratorIfNeeded(IHostEnvironment env, IChannel ch, ICalibratorTrainer calibrator,
            int maxRows, ITrainer trainer, IPredictor predictor, RoleMappedData data)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckValue(trainer, nameof(trainer));
            ch.CheckValue(predictor, nameof(predictor));
            ch.CheckValue(data, nameof(data));

            if (!NeedCalibration(env, ch, calibrator, trainer, predictor, data.Schema))
                return predictor;

            return GetCalibratedPredictor(env, ch, calibrator, predictor, data, maxRows);
        }

        /// <summary>
        /// Trains a calibrator.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="ch">The channel.</param>
        /// <param name="caliTrainer">The calibrator trainer.</param>
        /// <param name="predictor">The predictor that needs calibration.</param>
        /// <param name="data">The examples to used for calibrator training.</param>
        /// <param name="maxRows">The maximum rows to use for calibrator training.</param>
        /// <returns>The original predictor, if no calibration is needed,
        /// or a metapredictor that wraps the original predictor and the newly trained calibrator.</returns>
        public static IPredictor GetCalibratedPredictor(IHostEnvironment env, IChannel ch, ICalibratorTrainer caliTrainer,
            IPredictor predictor, RoleMappedData data, int maxRows = _maxCalibrationExamples)
        {
            var trainedCalibrator = TrainCalibrator(env, ch, caliTrainer, predictor, data, maxRows);
            return CreateCalibratedPredictor(env, (IPredictorProducing<float>)predictor, trainedCalibrator);
        }

        /// <summary>
        /// Trains a calibrator.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="ch">The channel.</param>
        /// <param name="caliTrainer">The calibrator trainer.</param>
        /// <param name="predictor">The predictor that needs calibration.</param>
        /// <param name="data">The examples to used for calibrator training.</param>
        /// <param name="maxRows">The maximum rows to use for calibrator training.</param>
        /// <returns>The original predictor, if no calibration is needed,
        /// or a metapredictor that wraps the original predictor and the newly trained calibrator.</returns>
        public static ICalibrator TrainCalibrator(IHostEnvironment env, IChannel ch, ICalibratorTrainer caliTrainer, IPredictor predictor, RoleMappedData data, int maxRows = _maxCalibrationExamples)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckValue(predictor, nameof(predictor));
            ch.CheckValue(data, nameof(data));
            ch.CheckParam(data.Schema.Label.HasValue, nameof(data), "data must have a Label column");

            var scored = ScoreUtils.GetScorer(predictor, data, env, null);

            if (caliTrainer.NeedsTraining)
            {
                int labelCol;
                if (!scored.Schema.TryGetColumnIndex(data.Schema.Label.Value.Name, out labelCol))
                    throw ch.Except("No label column found");
                int scoreCol;
                if (!scored.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreCol))
                    throw ch.Except("No score column found");
                int weightCol = -1;
                if (data.Schema.Weight?.Name is string weightName && scored.Schema.GetColumnOrNull(weightName)?.Index is int weightIdx)
                    weightCol = weightIdx;
                ch.Info("Training calibrator.");

                var cols = weightCol > -1 ?
                    new Schema.Column[] { scored.Schema[labelCol], scored.Schema[scoreCol], scored.Schema[weightCol] } :
                    new Schema.Column[] { scored.Schema[labelCol], scored.Schema[scoreCol] };

                using (var cursor = scored.GetRowCursor(cols))
                {
                    var labelGetter = RowCursorUtils.GetLabelGetter(cursor, labelCol);
                    var scoreGetter = RowCursorUtils.GetGetterAs<Single>(NumberType.R4, cursor, scoreCol);
                    ValueGetter<Single> weightGetter = weightCol == -1 ? (ref float dst) => dst = 1 :
                        RowCursorUtils.GetGetterAs<Single>(NumberType.R4, cursor, weightCol);

                    int num = 0;
                    while (cursor.MoveNext())
                    {
                        Single label = 0;
                        labelGetter(ref label);
                        if (!FloatUtils.IsFinite(label))
                            continue;
                        Single score = 0;
                        scoreGetter(ref score);
                        if (!FloatUtils.IsFinite(score))
                            continue;
                        Single weight = 0;
                        weightGetter(ref weight);
                        if (!FloatUtils.IsFinite(weight))
                            continue;

                        caliTrainer.ProcessTrainingExample(score, label > 0, weight);

                        if (maxRows > 0 && ++num >= maxRows)
                            break;
                    }
                }
            }
            return caliTrainer.FinishTraining(ch);
        }

        public static IPredictorProducing<float> CreateCalibratedPredictor(IHostEnvironment env, IPredictorProducing<float> predictor, ICalibrator cali)
        {
            Contracts.Assert(predictor != null);
            if (cali == null)
                return predictor;
            for (; ; )
            {
                var p = predictor as CalibratedPredictorBase;
                if (p == null)
                    break;
                predictor = p.SubPredictor;
            }
            // REVIEW: Split the requirement for IPredictorWithFeatureWeights into a different class.
            var predWithFeatureScores = predictor as IPredictorWithFeatureWeights<float>;
            if (predWithFeatureScores != null && predictor is IParameterMixer<float> && cali is IParameterMixer)
                return new ParameterMixingCalibratedPredictor(env, predWithFeatureScores, cali);
            if (predictor is IValueMapper)
                return new CalibratedPredictor(env, predictor, cali);
            return new SchemaBindableCalibratedPredictor(env, predictor, cali);
        }
    }

    [TlcModule.Component(Name = "NaiveCalibrator", FriendlyName = "Naive Calibrator", Alias = "Naive")]
    internal sealed class NaiveCalibratorTrainerFactory : ICalibratorTrainerFactory
    {
        public ICalibratorTrainer CreateComponent(IHostEnvironment env)
        {
            return new NaiveCalibratorTrainer(env);
        }
    }

    /// <summary>
    /// Trains a <see cref="NaiveCalibrator"/> by dividing the range of the outputs into equally sized bins.
    /// The probability of belonging to a particular class, for example class 1, is the number of class 1 instances in the bin, divided by the total number
    /// of instances in that bin.
    /// </summary>
    public sealed class NaiveCalibratorTrainer : ICalibratorTrainer
    {
        private readonly IHost _host;

        private List<float> _cMargins;
        private List<float> _ncMargins;

        public int NumBins;
        public float BinSize;
        public float Min;
        public float Max;
        public float[] BinProbs;

        // REVIEW: The others have user/load names of calibraTION, but this has calibratOR.
        internal const string UserName = "Naive Calibrator";
        internal const string LoadName = "NaiveCalibrator";
        internal const string Summary = "Naive calibrator divides the range of the outputs into equally sized bins. In each bin, "
            + "the probability of belonging to class 1 is the number of class 1 instances in the bin, divided by the total number "
            + "of instances in the bin.";

        // REVIEW: does this need a ctor that initialized the parameters to given values?
        /// <summary>
        /// Initializes a new instance of <see cref="NaiveCalibratorTrainer"/>.
        /// </summary>
        public NaiveCalibratorTrainer(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName);
            _cMargins = new List<float>();
            _ncMargins = new List<float>();
            NumBins = 200;
            Min = float.MaxValue;
            Max = float.MinValue;
        }

        bool ICalibratorTrainer.NeedsTraining => true;

        public bool ProcessTrainingExample(float output, bool labelIs1, float weight)
        {
            //AP todo proper weighting here
            if (labelIs1)
            {
                _cMargins.Add(output);
            }
            else
            {
                _ncMargins.Add(output);
            }
            return true;
        }

        ICalibrator ICalibratorTrainer.FinishTraining(IChannel ch)
        {
            float[] cOutputs = _cMargins.ToArray();
            ch.Check(cOutputs.Length > 0, "Calibrator trained on zero instances.");

            float minC = MathUtils.Min(cOutputs);
            float maxC = MathUtils.Max(cOutputs);

            float[] ncOutputs = _ncMargins.ToArray();
            float minNC = MathUtils.Min(ncOutputs);
            float maxNC = MathUtils.Max(ncOutputs);

            Min = (minC < minNC) ? minC : minNC;
            Max = (maxC > maxNC) ? maxC : maxNC;
            BinSize = (Max - Min) / NumBins;

            float[] cBins = new float[NumBins];
            float[] ncBins = new float[NumBins];

            foreach (float xi in cOutputs)
            {
                int binIdx = NaiveCalibrator.GetBinIdx(xi, Min, BinSize, NumBins);
                cBins[binIdx]++;
            }

            foreach (float xi in ncOutputs)
            {
                int binIdx = NaiveCalibrator.GetBinIdx(xi, Min, BinSize, NumBins);
                ncBins[binIdx]++;
            }

            BinProbs = new float[NumBins];
            for (int i = 0; i < NumBins; i++)
            {
                if (cBins[i] + ncBins[i] == 0)
                    BinProbs[i] = 0;
                else
                    BinProbs[i] = cBins[i] / (cBins[i] + ncBins[i]);
            }

            return new NaiveCalibrator(_host, Min, BinSize, BinProbs);
        }
    }

    /// <summary>
    /// The naive binning-based calibrator.
    /// </summary>
    public sealed class NaiveCalibrator : ICalibrator, ICanSaveInBinaryFormat
    {
        internal const string LoaderSignature = "NaiveCaliExec";
        internal const string RegistrationName = "NaiveCalibrator";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NAIVECAL",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(NaiveCalibrator).Assembly.FullName);
        }

        private readonly IHost _host;

        /// <summary> The bin size.</summary>
        public readonly float BinSize;

        /// <summary> The minimum value in the first bin.</summary>
        public readonly float Min;

        /// <summary> The value of probability in each bin.</summary>
        public readonly float[] BinProbs;

        /// <summary> Initializes a new instance of <see cref="NaiveCalibrator"/>.</summary>
        /// <param name="env">The <see cref="IHostEnvironment"/> to use.</param>
        /// <param name="min">The minimum value in the first bin.</param>
        /// <param name="binProbs">The values of the probability in each bin.</param>
        /// <param name="binSize">The bin size.</param>
        public NaiveCalibrator(IHostEnvironment env, float min, float binSize, float[] binProbs)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            Min = min;
            BinSize = binSize;
            BinProbs = binProbs;
        }

        private NaiveCalibrator(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // int: sizeof(float)
            // float: bin size
            // float: minimum value of first bin
            // int: number of bins
            // float[]: probability in each bin
            int cbFloat = ctx.Reader.ReadInt32();
            _host.CheckDecode(cbFloat == sizeof(float));

            BinSize = ctx.Reader.ReadFloat();
            _host.CheckDecode(0 < BinSize && BinSize < float.PositiveInfinity);

            Min = ctx.Reader.ReadFloat();
            _host.CheckDecode(FloatUtils.IsFinite(Min));

            BinProbs = ctx.Reader.ReadFloatArray();
            _host.CheckDecode(Utils.Size(BinProbs) > 0);
            _host.CheckDecode(BinProbs.All(x => (0 <= x && x <= 1)));
        }

        private static NaiveCalibrator Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new NaiveCalibrator(env, ctx);
        }

        public void SaveAsBinary(BinaryWriter writer)
        {
            ModelSaveContext.Save(writer, SaveCore);
        }

        private void SaveCore(ModelSaveContext ctx)
        {
            _host.AssertValue(ctx);

            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(float)
            // float: bin size
            // float: minimum value of first bin
            // int: number of bins
            // float[]: probability in each bin
            ctx.Writer.Write(sizeof(float));
            ctx.Writer.Write(BinSize);
            ctx.Writer.Write(Min);
            ctx.Writer.WriteSingleArray(BinProbs);
        }

        /// <summary>
        /// Given a classifier output, produce the probability
        /// </summary>
        public float PredictProbability(float output)
        {
            if (float.IsNaN(output))
                return output;
            int binIdx = GetBinIdx(output, Min, BinSize, BinProbs.Length);
            return BinProbs[binIdx];
        }

        // get the bin for a given output
        internal static int GetBinIdx(float output, float min, float binSize, int numBins)
        {
            int binIdx = (int)((output - min) / binSize);
            if (binIdx >= numBins)
                binIdx = numBins - 1;
            if (binIdx < 0)
                binIdx = 0;
            return binIdx;
        }

        /// <summary> Get the summary of current calibrator settings </summary>
        public string GetSummary()
        {
            return string.Format("Naive Calibrator has {0} bins, starting at {1}, with bin size of {2}", BinProbs.Length, Min, BinSize);
        }
    }

    /// <summary>
    /// Base class for calibrator trainers.
    /// </summary>
    public abstract class CalibratorTrainerBase : ICalibratorTrainer
    {
        protected readonly IHost Host;
        protected CalibrationDataStore Data;
        protected const int DefaultMaxNumSamples = 1000000;
        protected int MaxNumSamples;

        protected CalibratorTrainerBase(IHostEnvironment env, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
            MaxNumSamples = DefaultMaxNumSamples;
        }

        bool ICalibratorTrainer.NeedsTraining => true;

        /// <summary>
        /// Training calibrators:  provide the classifier output and the class label
        /// </summary>
        bool ICalibratorTrainer.ProcessTrainingExample(float output, bool labelIs1, float weight)
        {
            if (Data == null)
                Data = new CalibrationDataStore(MaxNumSamples);
            Data.AddToStore(output, labelIs1, weight);
            return true;
        }

        ICalibrator ICalibratorTrainer.FinishTraining(IChannel ch)
        {
            ch.Check(Data != null, "Calibrator trained on zero instances.");
            var calibrator = CreateCalibrator(ch);
            Data = null;
            return calibrator;
        }

        public abstract ICalibrator CreateCalibrator(IChannel ch);
    }

    [TlcModule.Component(Name = "PlattCalibrator", FriendlyName = "Platt Calibrator", Aliases = new[] { "Platt", "Sigmoid" }, Desc = "Platt calibration.")]
    [BestFriend]
    internal sealed class PlattCalibratorTrainerFactory : ICalibratorTrainerFactory
    {
        public ICalibratorTrainer CreateComponent(IHostEnvironment env)
        {
            return new PlattCalibratorTrainer(env);
        }
    }

    public sealed class PlattCalibratorTrainer : CalibratorTrainerBase
    {
        internal const string UserName = "Sigmoid Calibration";
        internal const string LoadName = "PlattCalibration";
        internal const string Summary = "This model was introduced by Platt in the paper Probabilistic Outputs for Support Vector Machines "
            + "and Comparisons to Regularized Likelihood Methods";

        public PlattCalibratorTrainer(IHostEnvironment env)
            : base(env, LoadName)
        {
        }

        public override ICalibrator CreateCalibrator(IChannel ch)
        {
            Double slope = 0;
            Double offset = 0;

            Double prior0 = 0;
            Double prior1 = 0;
            long n = 0;
            foreach (var di in Data)
            {
                var weight = di.Weight;
                if (di.Target)
                    prior1 += weight;
                else
                    prior0 += weight;
                n++;
            }
            if (n == 0)
                return new PlattCalibrator(Host, slope, offset);

            slope = 0;
            // Initialize B to be the marginal probability of class
            // smoothed i.e. P(+ | x) = (N+ + 1) / (N + 2)
            offset = Math.Log((prior0 + 1) / (prior1 + 1));

            // OK. We're going to maximize the likelihood of the output by
            // minimizing the cross-entropy of the output. Here's a
            // magic special hack: make the target of the cross-entropy function
            Double hiTarget = (prior1 + 1) / (prior1 + 2);
            Double loTarget = 1 / (prior0 + 2);

            Double lambda = 0.001;
            Double olderr = Double.MaxValue / 2;
            // array to store current estimate of probability of training points
            float[] pp = new float[n];
            float defValue = (float)((prior1 + 1) / (prior0 + prior1 + 2));
            for (int i = 0; i < n; i++)
                pp[i] = defValue;

            int count = 0;

            // Don't go more than 100 iterations: LM is quadratically convergent, after all

            for (int it = 0; it < 100; it++)
            {
                Double a = 0;
                Double b = 0;
                Double c = 0;
                Double d = 0;
                Double e = 0;
                // Loop over all points, computing Hessian of cross-entropy [a c; c b]
                // and gradient of cross-entropy [d e]

                int i = 0;
                foreach (var d_i in Data)
                {
                    var weight = d_i.Weight;
                    var x = d_i.Score;
                    var t = (d_i.Target ? hiTarget : loTarget);
                    var p = pp[i];

                    Double deriv = p * (1 - p) * weight;
                    Double dd = (p - t) * weight;
                    a += x * x * deriv;
                    b += deriv;
                    c += x * deriv;
                    d += x * dd;
                    e += dd;
                    i++;
                }

                // If gradient is tiny, you're done
                if (d > -1e-9 && d < 1e-9 && e > -1e-9 && e < 1e-9)
                {
                    break;
                }
                Double err = 0;
                Double oldA = slope;
                Double oldB = offset;
                // Loop until you get a increase in the goodness of fit
                for (; ; )
                {
                    Double det = (a + lambda) * (b + lambda) - c * c;

                    if (det == 0.0)
                    {
                        lambda *= 10;
                        continue;
                    }
                    // This is the Newton-Raphson step (with lambda as stabilizer)
                    slope = oldA + ((b + lambda) * d - c * e) / det;
                    offset = oldB + ((a + lambda) * e - c * d) / det;
                    // Now, compute goodness of fit
                    err = 0;

                    i = 0;
                    foreach (var d_i in Data)
                    {
                        var y = d_i.Target ? d_i.Score : -d_i.Score;
                        var p = PlattCalibrator.PredictProbability(d_i.Score, slope, offset);
                        var t = d_i.Target ? hiTarget : loTarget;
                        var weight = d_i.Weight;
                        pp[i] = p;
                        Double logp = -200;
                        Double log1p = -200;
                        if (p > 0.0)
                            logp = Math.Log(p);
                        if (p < 1.0)
                            log1p = Math.Log(1 - p);
                        err -= (t * logp + (1 - t) * log1p) * weight;

                        i++;
                    }

                    // If goodness increased, you don't need so much stabilization
                    if (err < olderr * (1.0 + 1e-7))
                    {
                        lambda *= 0.1;
                        break;
                    }
                    // Oops. Goodness decreased. Newton-Raphson must be wigging out.
                    // Increase stabilizer by factor of 10 and try again
                    lambda *= 10;
                    // If stabilizer is bigger than 1e6, just give up now.
                    if (lambda >= 1e6)
                        break;
                }

                // Check to see if goodness of fit has improved more than
                // a factor of about 1e-3 (either relative or absolute)
                // You've converged if this happens more than twice in a row
                Double diff = err - olderr;

                Double scale = 0.5 * (err + olderr + 1);
                if (diff > (-1e-3) * scale && diff < 1e-7 * scale)
                    count++;
                else
                    count = 0;
                olderr = err;
                if (count == 3)
                    break;
            }

            return new PlattCalibrator(Host, slope, offset);
        }
    }

    public sealed class FixedPlattCalibratorTrainer : ICalibratorTrainer
    {
        [TlcModule.Component(Name = "FixedPlattCalibrator", FriendlyName = "Fixed Platt Calibrator", Aliases = new[] { "FixedPlatt", "FixedSigmoid" })]
        public sealed class Arguments : ICalibratorTrainerFactory
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The slope parameter of f(x) = 1 / (1 + exp(-slope * x + offset)", ShortName = "a")]
            public Double Slope = 1;

            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The offset parameter of f(x) = 1 / (1 + exp(-slope * x + offset)", ShortName = "b")]
            public Double Offset = 0;

            public ICalibratorTrainer CreateComponent(IHostEnvironment env)
            {
                return new FixedPlattCalibratorTrainer(env, this);
            }
        }

        internal const string UserName = "Fixed Sigmoid Calibration";
        internal const string LoadName = "FixedPlattCalibration";
        internal const string Summary = "Sigmoid calibrator with configurable slope and offset.";

        private readonly IHost _host;
        private readonly Double _slope;
        private readonly Double _offset;

        internal FixedPlattCalibratorTrainer(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName);
            _slope = args.Slope;
            _offset = args.Offset;
        }

        bool ICalibratorTrainer.NeedsTraining => false;

        bool ICalibratorTrainer.ProcessTrainingExample(float output, bool labelIs1, float weight) => false;

        ICalibrator ICalibratorTrainer.FinishTraining(IChannel ch) => new PlattCalibrator(_host, _slope, _offset);
    }

    ///<summary> The Platt calibrator calculates the probability following:
    /// P(x) = 1 / (1 + exp(-<see cref="PlattCalibrator.Slope"/> * x + <see cref="PlattCalibrator.Offset"/>) </summary>.
    public sealed class PlattCalibrator : ICalibrator, IParameterMixer, ICanSaveModel, ISingleCanSavePfa, ISingleCanSaveOnnx
    {
        internal const string LoaderSignature = "PlattCaliExec";
        internal const string RegistrationName = "PlattCalibrator";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PLATTCAL",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PlattCalibrator).Assembly.FullName);
        }

        private readonly IHost _host;

        public Double Slope { get; }
        public Double Offset { get; }
        bool ICanSavePfa.CanSavePfa => true;
        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => true;

        /// <summary>
        /// Initializes a new instance of <see cref="PlattCalibrator"/>.
        /// </summary>
        public PlattCalibrator(IHostEnvironment env, Double slope, Double offset)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            Slope = slope;
            Offset = offset;
        }

        private PlattCalibrator(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            _host = env.Register(RegistrationName);
            _host.AssertValue(ctx);

            // *** Binary format ***
            // Double: A
            // Double: B
            Slope = ctx.Reader.ReadDouble();
            _host.CheckDecode(FloatUtils.IsFinite(Slope));

            Offset = ctx.Reader.ReadDouble();
            _host.CheckDecode(FloatUtils.IsFinite(Offset));
        }

        private static PlattCalibrator Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PlattCalibrator(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        private void SaveCore(ModelSaveContext ctx)
        {
            _host.AssertValue(ctx);

            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // Double: A
            // Double: B
            ctx.Writer.Write(Slope);
            ctx.Writer.Write(Offset);

            if (ctx.InRepository)
            {
                ctx.SaveTextStream("Calibrator.txt", (Action<TextWriter>)(writer =>
                {
                    writer.WriteLine("Platt calibrator");
                    writer.WriteLine("P(y=1|x) = 1/1+exp(A*x + B)");
                    writer.WriteLine("A={0:R}", (object)Slope);
                    writer.WriteLine("B={0:R}", Offset);
                }));
            }
        }

        public float PredictProbability(float output)
        {
            if (float.IsNaN(output))
                return output;
            return PredictProbability(output, Slope, Offset);
        }

        public static float PredictProbability(float output, Double a, Double b)
        {
            return (float)(1 / (1 + Math.Exp(a * output + b)));
        }

        JToken ISingleCanSavePfa.SaveAsPfa(BoundPfaContext ctx, JToken input)
        {
            _host.CheckValue(ctx, nameof(ctx));
            _host.CheckValue(input, nameof(input));

            return PfaUtils.Call("m.link.logit",
                PfaUtils.Call("+", -Offset, PfaUtils.Call("*", -Slope, input)));
        }

        bool ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] scoreProbablityColumnNames, string featureColumnName)
        {
            _host.CheckValue(ctx, nameof(ctx));
            _host.CheckValue(scoreProbablityColumnNames, nameof(scoreProbablityColumnNames));
            _host.Check(Utils.Size(scoreProbablityColumnNames) == 2);

            string opType = "Affine";
            string linearOutput = ctx.AddIntermediateVariable(null, "linearOutput", true);
            var node = ctx.CreateNode(opType, new[] { scoreProbablityColumnNames[0] },
                new[] { linearOutput }, ctx.GetNodeName(opType), "");
            node.AddAttribute("alpha", Slope * -1);
            node.AddAttribute("beta", -0.0000001);

            opType = "Sigmoid";
            node = ctx.CreateNode(opType, new[] { linearOutput },
                new[] { scoreProbablityColumnNames[1] }, ctx.GetNodeName(opType), "");

            return true;
        }

        public string GetSummary()
        {
            return string.Format("Platt calibrator parameters: A={0}, B={1}", Slope, Offset);
        }

        IParameterMixer IParameterMixer.CombineParameters(IList<IParameterMixer> calibrators)
        {
            Double a = 0;
            Double b = 0;
            foreach (IParameterMixer calibrator in calibrators)
            {
                PlattCalibrator cal = calibrator as PlattCalibrator;

                a += cal.Slope;
                b += cal.Offset;
            }

            PlattCalibrator newCal = new PlattCalibrator(_host, a / calibrators.Count, b / calibrators.Count);
            return newCal;
        }
    }

    [TlcModule.Component(Name = "PavCalibrator", FriendlyName = "PAV Calibrator", Alias = "Pav")]
    internal sealed class PavCalibratorTrainerFactory : ICalibratorTrainerFactory
    {
        public ICalibratorTrainer CreateComponent(IHostEnvironment env)
        {
            return new PavCalibratorTrainer(env);
        }
    }

    public class PavCalibratorTrainer : CalibratorTrainerBase
    {
        // a piece of the piecwise function
        private readonly struct Piece
        {
            public readonly float MinX; // end of interval.
            public readonly float MaxX; // beginning of interval.
            public readonly float Value; // value of function in interval.
            public readonly float N; // number of points/sum of weights of interval.

            public Piece(float minX, float maxX, float value, float n)
            {
                Contracts.Assert(minX <= maxX);
                // REVIEW: Can this fail due to more innocent imprecision issues?
                Contracts.Assert(0 <= value && value <= 1);
                Contracts.Assert(n >= 0);
                MinX = minX;
                MaxX = maxX;
                Value = value;
                N = n;
            }
        }

        internal const string UserName = "PAV Calibration";
        internal const string LoadName = "PAVCalibration";
        internal const string Summary = "Piecewise linear calibrator.";

        // REVIEW: Do we need a ctor that initializes min, max, value, n?
        public PavCalibratorTrainer(IHostEnvironment env)
            : base(env, LoadName)
        {
        }

        public override ICalibrator CreateCalibrator(IChannel ch)
        {
            Stack<Piece> stack = new Stack<Piece>();
            Piece top = default(Piece);

            foreach (var di in Data) // this will iterate in sorted order
            {
                ch.Assert(stack.Count == 0 || di.Score >= top.MaxX);
                Piece curr = new Piece(di.Score, di.Score, di.Target ? 1 : 0, di.Weight);
                for (; stack.Count > 0 && ((top.MaxX >= curr.MinX) || curr.Value <= top.Value);)
                {
                    float newN = top.N + curr.N;
                    curr = new Piece(top.MinX, curr.MaxX, (top.Value * top.N + curr.Value * curr.N) / newN, newN);
                    stack.Pop();
                    if (stack.Count > 0)
                        top = stack.Peek();
                }
                ch.Assert(stack.Count == 0 || top.MaxX < curr.MinX);
                stack.Push(curr);
                top = curr;
            }

            ch.Info("PAV calibrator:  piecewise function approximation has {0} components.", stack.Count);
            float[] mins = new float[stack.Count];
            float[] maxes = new float[stack.Count];
            float[] values = new float[stack.Count];

            for (int i = stack.Count - 1; stack.Count > 0; --i)
            {
                top = stack.Pop();
                mins[i] = top.MinX;
                maxes[i] = top.MaxX;
                values[i] = top.Value;
            }

            return new PavCalibrator(Host, mins.ToImmutableArray(), maxes.ToImmutableArray(), values.ToImmutableArray());
        }
    }

    /// <summary>
    /// The function that is implemented by this calibrator is:
    /// f(x) = v_i, if minX_i &lt;= x &lt;= maxX_i
    ///      = linear interpolate between v_i and v_i+1, if maxX_i &lt; x &lt; minX_i+1
    ///      = v_0, if x &lt; minX_0
    ///      = v_n, if x &gt; maxX_n
    /// </summary>
    public sealed class PavCalibrator : ICalibrator, ICanSaveInBinaryFormat
    {
        internal const string LoaderSignature = "PAVCaliExec";
        internal const string RegistrationName = "PAVCalibrator";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PAV  CAL",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(PavCalibrator).Assembly.FullName);
        }

        // Epsilon for 0-comparisons
        private const float Epsilon = (float)1e-15;
        private const float MinToReturn = Epsilon; // max predicted is 1 - min;
        private const float MaxToReturn = 1 - Epsilon; // max predicted is 1 - min;

        private readonly IHost _host;
        public readonly ImmutableArray<float> Mins;
        public readonly ImmutableArray<float> Maxes;
        public readonly ImmutableArray<float> Values;

        /// <summary>
        /// Initializes a new instance of <see cref="PavCalibrator"/>.
        /// </summary>
        /// <param name="env">The <see cref="IHostEnvironment"/> to use.</param>
        /// <param name="mins">The minimum values for each piece.</param>
        /// <param name="maxes">The maximum values for each piece.</param>
        /// <param name="values">The actual values for each piece.</param>
        public PavCalibrator(IHostEnvironment env, ImmutableArray<float> mins, ImmutableArray<float> maxes, ImmutableArray<float> values)
        {
            Contracts.AssertValue(env);
            _host = env.Register(RegistrationName);
            _host.AssertNonEmpty(mins);
            _host.AssertNonEmpty(maxes);
            _host.AssertNonEmpty(values);
            _host.Assert(Utils.IsMonotonicallyIncreasing(mins));
            _host.Assert(Utils.IsMonotonicallyIncreasing(maxes));
            _host.Assert(Utils.IsMonotonicallyIncreasing(values));
            _host.Assert(values.Length == 0 || (0 <= values[0] && values[values.Length - 1] <= 1));
            _host.Assert(mins.Zip(maxes, (min, max) => min <= max).All(x => x));

            Mins = mins;
            Maxes = maxes;
            Values = values;
        }

        private PavCalibrator(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            _host = env.Register(RegistrationName);
            _host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(float)
            // int: number of pieces
            // for each piece:
            //      float: MinX
            //      float: MaxX
            //      float: Value
            int cbFloat = ctx.Reader.ReadInt32();
            _host.CheckDecode(cbFloat == sizeof(float));

            int numPieces = ctx.Reader.ReadInt32();
            _host.CheckDecode(numPieces >= 0);
            var mins = new float[numPieces];
            var maxes = new float[numPieces];
            var values = new float[numPieces];
            float valuePrev = 0;
            float maxPrev = float.NegativeInfinity;
            for (int i = 0; i < numPieces; ++i)
            {
                float minX = ctx.Reader.ReadFloat();
                float maxX = ctx.Reader.ReadFloat();
                float val = ctx.Reader.ReadFloat();
                _host.CheckDecode(minX <= maxX);
                _host.CheckDecode(minX > maxPrev);
                _host.CheckDecode(val > valuePrev || val == valuePrev && i == 0);
                valuePrev = val;
                maxPrev = maxX;
                mins[i] = minX;
                maxes[i] = maxX;
                values[i] = val;
            }

            Mins = mins.ToImmutableArray();
            Maxes = maxes.ToImmutableArray();
            Values = values.ToImmutableArray();
            _host.CheckDecode(valuePrev <= 1);
        }

        private static PavCalibrator Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PavCalibrator(env, ctx);
        }

        public void SaveAsBinary(BinaryWriter writer)
        {
            ModelSaveContext.Save(writer, SaveCore);
        }

        private void SaveCore(ModelSaveContext ctx)
        {
            _host.AssertValue(ctx);

            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(float)
            // int: number of pieces
            // for each piece:
            //      float: MinX
            //      float: MaxX
            //      float: Value
            ctx.Writer.Write(sizeof(float));

            _host.Assert(Mins.Length == Maxes.Length);
            _host.Assert(Mins.Length == Values.Length);
            ctx.Writer.Write(Mins.Length);
            float valuePrev = 0;
            float maxPrev = float.NegativeInfinity;
            for (int i = 0; i < Mins.Length; i++)
            {
                _host.Assert(Mins[i] <= Maxes[i]);
                _host.Assert(Mins[i] > maxPrev);
                _host.Assert(Values[i] > valuePrev || Values[i] == valuePrev && i == 0);
                valuePrev = Values[i];
                maxPrev = Maxes[i];
                ctx.Writer.Write(Mins[i]);
                ctx.Writer.Write(Maxes[i]);
                ctx.Writer.Write(Values[i]);
            }
            _host.CheckDecode(valuePrev <= 1);
        }

        public float PredictProbability(float output)
        {
            if (float.IsNaN(output))
                return output;
            float prob = FindValue(output);
            if (prob < MinToReturn)
                return MinToReturn;
            if (prob > MaxToReturn)
                return MaxToReturn;
            return prob;
        }

        private float FindValue(float score)
        {
            int p = Mins.Length;
            if (p == 0)
                return 0;
            if (score < Mins[0])
            {
                return Values[0];
                // tail off to zero exponentially
                // return Math.Exp(-(piecewise[0].MinX-score)) * piecewise[0].Value;
            }
            if (score > Maxes[p - 1])
            {
                return Values[p - 1];
                // tail off to one exponentially
                // return (1-Math.Exp(-(score - piecewise[P - 1].MaxX))) * (1 - piecewise[P - 1].Value) + piecewise[P - 1].Value;
            }

            int pos = Maxes.FindIndexSorted(score);
            _host.Assert(pos < p);
            // inside the piece, the value is constant
            if (score >= Mins[pos])
                return Values[pos];
            // between pieces, interpolate
            float t = (score - Maxes[pos - 1]) / (Mins[pos] - Maxes[pos - 1]);
            return Values[pos - 1] + t * (Values[pos] - Values[pos - 1]);
        }

        public string GetSummary()
        {
            return string.Format("PAV calibrator with {0} intervals", Mins.Length);
        }
    }

    public sealed class CalibrationDataStore : IEnumerable<CalibrationDataStore.DataItem>
    {
        public readonly struct DataItem
        {
            // The actual binary label of this example.
            public readonly bool Target;
            // The weight associated with this example.
            public readonly float Weight;
            // The output of the example.
            public readonly float Score;

            public DataItem(bool target, float weight, float score)
            {
                Target = target;
                Weight = weight;
                Score = score;
            }
        }

        // REVIEW: Should probably be a long.
        private int _itemsSeen;
        private readonly Random _random;

        private static int _randSeed;

        private readonly int _capacity;
        private DataItem[] _data;
        private bool _dataSorted;

        public CalibrationDataStore()
            : this(1000000)
        {
        }

        public CalibrationDataStore(int capacity)
        {
            Contracts.CheckParam(capacity > 0, nameof(capacity), "must be positive");

            _capacity = capacity;
            _data = new DataItem[Math.Min(4, capacity)];
            // REVIEW: Horrifying. At a point when we have the IHost stuff plumbed through
            // calibrator training and also have the appetite to change a bunch of baselines, this
            // should be seeded using the host random.
            _random = new System.Random(System.Threading.Interlocked.Increment(ref _randSeed) - 1);
        }

        /// <summary>
        /// An enumerator over the <see cref="DataItem"/> entries sorted by score.
        /// </summary>
        /// <returns></returns>
        public IEnumerator<DataItem> GetEnumerator()
        {
            if (!_dataSorted)
            {
                var comp = Comparer<DataItem>.Create((x, y) => x.Score.CompareTo(y.Score));
                Array.Sort(_data, 0, Math.Min(_itemsSeen, _capacity), comp);
                _dataSorted = true;
            }
            return _data.Take(_itemsSeen).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void AddToStore(float score, bool isPositive, float weight)
        {
            // Can't calibrate NaN scores.
            if (weight == 0 || float.IsNaN(score))
                return;
            int index = _itemsSeen++;
            if (_itemsSeen <= _capacity)
                Utils.EnsureSize(ref _data, _itemsSeen, _capacity);
            else
            {
                index = _random.Next(_itemsSeen); // 0 to items_seen - 1.
                if (index >= _capacity) // Don't keep it.
                    return;
            }
            _data[index] = new DataItem(isPositive, weight, score);
        }
    }

    internal static class Calibrate
    {
        [TlcModule.EntryPointKind(typeof(CommonInputs.ICalibratorInput))]
        public abstract class CalibrateInputBase : TransformInputBase
        {
            [Argument(ArgumentType.Required, ShortName = "uncalibratedPredictorModel", HelpText = "The predictor to calibrate", SortOrder = 2)]
            public PredictorModel UncalibratedPredictorModel;

            [Argument(ArgumentType.Required, ShortName = "maxRows", HelpText = "The maximum number of examples to train the calibrator on", SortOrder = 3)]
            [TlcModule.Range(Inf = 0, Max = int.MaxValue)]
            public int MaxRows = 1000000000;
        }

        public sealed class NoArgumentsInput : CalibrateInputBase
        {
        }

        public sealed class FixedPlattInput : CalibrateInputBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "slope", HelpText = "The slope parameter of the calibration function 1 / (1 + exp(-slope * x + offset)", SortOrder = 1)]
            public Double Slope = 1;

            [Argument(ArgumentType.AtMostOnce, ShortName = "offset", HelpText = "The offset parameter of the calibration function 1 / (1 + exp(-slope * x + offset)", SortOrder = 3)]
            public Double Offset = 0;
        }

        [TlcModule.EntryPoint(Name = "Models.PlattCalibrator", Desc = "Apply a Platt calibrator to an input model", UserName = PlattCalibratorTrainer.UserName)]
        public static CommonOutputs.CalibratorOutput Platt(IHostEnvironment env, NoArgumentsInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Platt");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return CalibratePredictor<CommonOutputs.CalibratorOutput>(host, input, new PlattCalibratorTrainer(host));
        }

        [TlcModule.EntryPoint(Name = "Models.NaiveCalibrator", Desc = "Apply a Naive calibrator to an input model", UserName = NaiveCalibratorTrainer.UserName)]
        public static CommonOutputs.CalibratorOutput Naive(IHostEnvironment env, NoArgumentsInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Naive");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return CalibratePredictor<CommonOutputs.CalibratorOutput>(host, input, new NaiveCalibratorTrainer(host));
        }

        [TlcModule.EntryPoint(Name = "Models.PAVCalibrator", Desc = "Apply a PAV calibrator to an input model", UserName = PavCalibratorTrainer.UserName)]
        public static CommonOutputs.CalibratorOutput Pav(IHostEnvironment env, NoArgumentsInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("PAV");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return CalibratePredictor<CommonOutputs.CalibratorOutput>(host, input, new PavCalibratorTrainer(host));
        }

        [TlcModule.EntryPoint(Name = "Models.FixedPlattCalibrator", Desc = "Apply a Platt calibrator with a fixed slope and offset to an input model", UserName = FixedPlattCalibratorTrainer.UserName)]
        public static CommonOutputs.CalibratorOutput FixedPlatt(IHostEnvironment env, FixedPlattInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("FixedPlatt");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return
                CalibratePredictor<CommonOutputs.CalibratorOutput>(host, input,
                    new FixedPlattCalibratorTrainer(host, new FixedPlattCalibratorTrainer.Arguments() { Offset = input.Offset, Slope = input.Slope }));
        }

        /// <summary>
        /// This method calibrates the specified predictor using the specified calibrator, training on the specified data.
        /// </summary>
        /// <param name="host">A host to pass to the components created in this method.</param>
        /// <param name="input">The input object, containing the predictor, the data and an integer indicating the maximum number
        /// of examples to use for training the calibrator.</param>
        /// <param name="calibratorTrainer">The kind of calibrator to use.</param>
        /// <returns>A <see cref="CommonOutputs.TrainerOutput"/> object, containing an <see cref="PredictorModel"/>.</returns>
        internal static TOut CalibratePredictor<TOut>(IHost host, CalibrateInputBase input,
            ICalibratorTrainer calibratorTrainer)
            where TOut : CommonOutputs.TrainerOutput, new()
        {
            Contracts.CheckValue(host, nameof(host));
            host.CheckValue(input, nameof(input));
            host.CheckUserArg(input.MaxRows >= 0, nameof(input.MaxRows), "Argument must be non-negative. specify 0 to use all available examples.");

            RoleMappedData data;
            IPredictor predictor;
            input.UncalibratedPredictorModel.PrepareData(host, input.Data, out data, out predictor);
            using (var ch = host.Start("Calibrating"))
            {
                // If the predictor is a pipeline ensemble where the label column is inside the pipelines, there may not be a global
                // label column. In that case the model has to be calibrated using ISelfCalibratingPredictor.
                IPredictor calibratedPredictor;
                var scp = predictor as ISelfCalibratingPredictor;
                if (data.Schema.Label == null && scp != null)
                    calibratedPredictor = scp.Calibrate(ch, data.Data, calibratorTrainer, input.MaxRows);
                else
                {
                    calibratedPredictor =
                        CalibratorUtils.GetCalibratedPredictor(host, ch, calibratorTrainer, predictor, data, input.MaxRows);
                }
                return new TOut() { PredictorModel = new PredictorModelImpl(host, data, input.Data, calibratedPredictor) };
            }
        }
    }
}