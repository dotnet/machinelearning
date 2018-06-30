// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Newtonsoft.Json.Linq;
using Microsoft.ML.Runtime.EntryPoints;

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

namespace Microsoft.ML.Runtime.Internal.Calibration
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
        bool ProcessTrainingExample(Float output, bool labelIs1, Float weight);

        /// <summary> Finish up training after seeing all examples </summary>
        ICalibrator FinishTraining(IChannel ch);
    }

    /// <summary>
    /// An interface for probability calibrators.
    /// </summary>
    public interface ICalibrator
    {
        /// <summary> Given a classifier output, produce the probability </summary>		
        Float PredictProbability(Float output);

        /// <summary> Get the summary of current calibrator settings </summary>
        string GetSummary();
    }

    /// <summary>
    /// An interface for predictors that take care of their own calibration given an input data view.
    /// </summary>
    public interface ISelfCalibratingPredictor
    {
        IPredictor Calibrate(IChannel ch, IDataView data, ICalibratorTrainer caliTrainer, int maxRows);
    }

    public abstract class CalibratedPredictorBase :
        IDistPredictorProducing<Float, Float>,
        ICanSaveInIniFormat,
        ICanSaveInTextFormat,
        ICanSaveInSourceCode,
        ICanSaveSummary,
        ICanGetSummaryInKeyValuePairs
    {
        protected readonly IHost Host;

        public IPredictorProducing<Float> SubPredictor { get; }
        public ICalibrator Calibrator { get; }
        public PredictionKind PredictionKind => SubPredictor.PredictionKind;

        protected CalibratedPredictorBase(IHostEnvironment env, string name, IPredictorProducing<Float> predictor, ICalibrator calibrator)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
            Host.CheckValue(predictor, nameof(predictor));
            Host.CheckValue(calibrator, nameof(calibrator));

            SubPredictor = predictor;
            Calibrator = calibrator;
        }

        public void SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator = null)
        {
            Host.Check(calibrator == null, "Too many calibrators.");
            var saver = SubPredictor as ICanSaveInIniFormat;
            saver?.SaveAsIni(writer, schema, Calibrator);
        }

        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubPredictor as ICanSaveInTextFormat;
            if (saver != null)
                saver.SaveAsText(writer, schema);
        }

        public void SaveAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubPredictor as ICanSaveInSourceCode;
            if (saver != null)
                saver.SaveAsCode(writer, schema);
        }

        public void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubPredictor as ICanSaveSummary;
            if (saver != null)
                saver.SaveSummary(writer, schema);
        }

        ///<inheritdoc/>
        public IList<KeyValuePair<string, object>> GetSummaryInKeyValuePairs(RoleMappedSchema schema)
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

        protected static IPredictorProducing<Float> GetPredictor(IHostEnvironment env, ModelLoadContext ctx)
        {
            IPredictorProducing<Float> predictor;
            ctx.LoadModel<IPredictorProducing<Float>, SignatureLoadModel>(env, out predictor, ModelFileUtils.DirPredictor);
            return predictor;
        }

        protected static ICalibrator GetCalibrator(IHostEnvironment env, ModelLoadContext ctx)
        {
            ICalibrator calibrator;
            ctx.LoadModel<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            return calibrator;
        }
    }

    public abstract class ValueMapperCalibratedPredictorBase : CalibratedPredictorBase, IValueMapperDist, IWhatTheFeatureValueMapper,
        IDistCanSavePfa, IDistCanSaveOnnx
    {
        private readonly IValueMapper _mapper;
        private readonly IWhatTheFeatureValueMapper _whatTheFeature;

        public ColumnType InputType => _mapper.InputType;
        public ColumnType OutputType => _mapper.OutputType;
        public ColumnType DistType => NumberType.Float;
        public bool CanSavePfa => (_mapper as ICanSavePfa)?.CanSavePfa == true;
        public bool CanSaveOnnx => (_mapper as ICanSaveOnnx)?.CanSaveOnnx == true;

        protected ValueMapperCalibratedPredictorBase(IHostEnvironment env, string name, IPredictorProducing<Float> predictor, ICalibrator calibrator)
            : base(env, name, predictor, calibrator)
        {
            Contracts.AssertValue(Host);

            _mapper = SubPredictor as IValueMapper;
            Host.Check(_mapper != null, "The predictor does not implement IValueMapper");
            Host.Check(_mapper.OutputType == NumberType.Float, "The output type of the predictor is expected to be Float");

            _whatTheFeature = predictor as IWhatTheFeatureValueMapper;
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            return _mapper.GetMapper<TIn, TOut>();
        }

        public ValueMapper<TIn, TOut, TDist> GetMapper<TIn, TOut, TDist>()
        {
            Host.Check(typeof(TOut) == typeof(Float));
            Host.Check(typeof(TDist) == typeof(Float));
            var map = GetMapper<TIn, Float>();
            ValueMapper<TIn, Float, Float> del =
                (ref TIn src, ref Float score, ref Float prob) =>
                {
                    map(ref src, ref score);
                    prob = Calibrator.PredictProbability(score);
                };
            return (ValueMapper<TIn, TOut, TDist>)(Delegate)del;
        }

        public ValueMapper<TSrc, VBuffer<Float>> GetWhatTheFeatureMapper<TSrc, TDst>(int top, int bottom, bool normalize)
        {
            // REVIEW: checking this a bit too late.
            Host.Check(_whatTheFeature != null, "Predictor does not implement IWhatTheFeatureValueMapper");
            return _whatTheFeature.GetWhatTheFeatureMapper<TSrc, TDst>(top, bottom, normalize);
        }

        public JToken SaveAsPfa(BoundPfaContext ctx, JToken input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(input, nameof(input));

            Host.Assert(_mapper is ISingleCanSavePfa);
            var mapper = (ISingleCanSavePfa)_mapper;
            return mapper.SaveAsPfa(ctx, input);
        }

        public void SaveAsPfa(BoundPfaContext ctx, JToken input,
            string score, out JToken scoreToken, string prob, out JToken probToken)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(input, nameof(input));
            Host.CheckValueOrNull(score);
            Host.CheckValueOrNull(prob);

            JToken scoreExpression = SaveAsPfa(ctx, input);
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

        public bool SaveAsOnnx(IOnnxContext ctx, string[] outputNames, string featureColumnName)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(outputNames, nameof(outputNames));

            Host.Assert(_mapper is ISingleCanSaveOnnx);

            var mapper = (ISingleCanSaveOnnx)_mapper;
            if (!mapper.SaveAsOnnx(ctx, new[] { outputNames[1] }, featureColumnName))
                return false;

            var calibrator = Calibrator as ISingleCanSaveOnnx;
            if (!(calibrator?.CanSaveOnnx == true && calibrator.SaveAsOnnx(ctx, new[] { outputNames[1], outputNames[2] }, featureColumnName)))
                ctx.RemoveVariable(outputNames[1], true);

            return true;
        }

    }
    public sealed class CalibratedPredictor : ValueMapperCalibratedPredictorBase, ICanSaveModel
    {
        public CalibratedPredictor(IHostEnvironment env, IPredictorProducing<Float> predictor, ICalibrator calibrator)
            : base(env, RegistrationName, predictor, calibrator)
        {
        }

        public const string LoaderSignature = "CaliPredExec";
        public const string RegistrationName = "CalibratedPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CALIPRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }
        private static VersionInfo GetVersionInfoBulk()
        {
            return new VersionInfo(
                modelSignature: "BCALPRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private CalibratedPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, GetPredictor(env, ctx), GetCalibrator(env, ctx))
        {
        }

        public static CalibratedPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
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

    public sealed class FeatureWeightsCalibratedPredictor :
        ValueMapperCalibratedPredictorBase,
        IPredictorWithFeatureWeights<Float>,
        ICanSaveModel
    {
        private readonly IPredictorWithFeatureWeights<Float> _featureWeights;

        public FeatureWeightsCalibratedPredictor(IHostEnvironment env, IPredictorWithFeatureWeights<Float> predictor,
            ICalibrator calibrator)
            : base(env, RegistrationName, predictor, calibrator)
        {
            _featureWeights = predictor;
        }

        public const string LoaderSignature = "FeatWCaliPredExec";
        public const string RegistrationName = "FeatureWeightsCalibratedPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FTWTCALP",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private FeatureWeightsCalibratedPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, GetPredictor(env, ctx), GetCalibrator(env, ctx))
        {
            Host.Check(SubPredictor is IPredictorWithFeatureWeights<Float>, "Predictor does not implement " + nameof(IPredictorWithFeatureWeights<Float>));
            _featureWeights = (IPredictorWithFeatureWeights<Float>)SubPredictor;
        }

        public static FeatureWeightsCalibratedPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
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

        public void GetFeatureWeights(ref VBuffer<Float> weights)
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
        IParameterMixer<Float>,
        IPredictorWithFeatureWeights<Float>,
        ICanSaveModel
    {
        private readonly IPredictorWithFeatureWeights<Float> _featureWeights;

        public ParameterMixingCalibratedPredictor(IHostEnvironment env, IPredictorWithFeatureWeights<Float> predictor, ICalibrator calibrator)
            : base(env, RegistrationName, predictor, calibrator)
        {
            Host.Check(predictor is IParameterMixer<Float>, "Predictor does not implement " + nameof(IParameterMixer<Float>));
            Host.Check(calibrator is IParameterMixer, "Calibrator does not implement " + nameof(IParameterMixer));
            _featureWeights = predictor;
        }

        public const string LoaderSignature = "PMixCaliPredExec";
        public const string RegistrationName = "ParameterMixingCalibratedPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PMIXCALP",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private ParameterMixingCalibratedPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, GetPredictor(env, ctx), GetCalibrator(env, ctx))
        {
            Host.Check(SubPredictor is IParameterMixer<Float>, "Predictor does not implement " + nameof(IParameterMixer));
            Host.Check(SubPredictor is IPredictorWithFeatureWeights<Float>, "Predictor does not implement " + nameof(IPredictorWithFeatureWeights<Float>));
            _featureWeights = (IPredictorWithFeatureWeights<Float>)SubPredictor;
        }

        public static ParameterMixingCalibratedPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
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

        public void GetFeatureWeights(ref VBuffer<Float> weights)
        {
            _featureWeights.GetFeatureWeights(ref weights);
        }

        public IParameterMixer<Float> CombineParameters(IList<IParameterMixer<Float>> models)
        {
            var predictors = models.Select(
                m =>
                {
                    var model = m as ParameterMixingCalibratedPredictor;
                    Contracts.Assert(model != null);
                    return (IParameterMixer<Float>)model.SubPredictor;
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
            return new ParameterMixingCalibratedPredictor(Host, (IPredictorWithFeatureWeights<Float>)combinedPredictor, (ICalibrator)combinedCalibrator);
        }
    }

    public sealed class SchemaBindableCalibratedPredictor : CalibratedPredictorBase, ISchemaBindableMapper, ICanSaveModel,
        IBindableCanSavePfa, IBindableCanSaveOnnx, IWhatTheFeatureValueMapper
    {
        private sealed class Bound : ISchemaBoundRowMapper
        {
            private readonly SchemaBindableCalibratedPredictor _parent;
            private readonly ISchemaBoundRowMapper _predictor;
            private readonly ISchema _outputSchema;
            private readonly int _scoreCol;

            public ISchemaBindableMapper Bindable => _parent;
            public RoleMappedSchema InputSchema => _predictor.InputSchema;
            public ISchema OutputSchema => _outputSchema;

            public Bound(IHostEnvironment env, SchemaBindableCalibratedPredictor parent, RoleMappedSchema schema)
            {
                Contracts.AssertValue(env);
                env.AssertValue(parent);
                _parent = parent;
                _predictor = _parent._bindable.Bind(env, schema) as ISchemaBoundRowMapper;
                env.Check(_predictor != null, "Predictor is not a row-to-row mapper");
                if (!_predictor.OutputSchema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out _scoreCol))
                    throw env.Except("Predictor does not output a score");
                var scoreType = _predictor.OutputSchema.GetColumnType(_scoreCol);
                env.Check(!scoreType.IsVector && scoreType.IsNumber);
                _outputSchema = new BinaryClassifierSchema();
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < OutputSchema.ColumnCount; i++)
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

            public IRow GetOutputRow(IRow input, Func<int, bool> predicate, out Action disposer)
            {
                Func<int, bool> predictorPredicate = col => false;
                for (int i = 0; i < _outputSchema.ColumnCount; i++)
                {
                    if (predicate(i))
                    {
                        predictorPredicate = col => true;
                        break;
                    }
                }
                var predictorRow = _predictor.GetOutputRow(input, predictorPredicate, out disposer);
                var getters = new Delegate[_outputSchema.ColumnCount];
                for (int i = 0; i < _outputSchema.ColumnCount - 1; i++)
                {
                    var type = predictorRow.Schema.GetColumnType(i);
                    if (!predicate(i))
                        continue;
                    getters[i] = Utils.MarshalInvoke(GetPredictorGetter<int>, type.RawType, predictorRow, i);
                }
                if (predicate(_outputSchema.ColumnCount - 1))
                    getters[_outputSchema.ColumnCount - 1] = GetProbGetter(predictorRow);
                return new SimpleRow(_outputSchema, predictorRow, getters);
            }

            private Delegate GetPredictorGetter<T>(IRow input, int col)
            {
                return input.GetGetter<T>(col);
            }

            private Delegate GetProbGetter(IRow input)
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
        private readonly IWhatTheFeatureValueMapper _whatTheFeature;

        public const string LoaderSignature = "SchemaBindableCalibrated";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "BINDCALI",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        /// <summary>
        /// Whether we can save as PFA. Note that this depends on whether the underlying predictor
        /// can save as PFA, since in the event that this in particular does not get saved,
        /// </summary>
        public bool CanSavePfa => (_bindable as ICanSavePfa)?.CanSavePfa == true;

        public bool CanSaveOnnx => (_bindable as ICanSaveOnnx)?.CanSaveOnnx == true;

        public SchemaBindableCalibratedPredictor(IHostEnvironment env, IPredictorProducing<Single> predictor, ICalibrator calibrator)
            : base(env, LoaderSignature, predictor, calibrator)
        {
            _bindable = ScoreUtils.GetSchemaBindableMapper(Host, SubPredictor, null);
            _whatTheFeature = SubPredictor as IWhatTheFeatureValueMapper;
        }

        private SchemaBindableCalibratedPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, GetPredictor(env, ctx), GetCalibrator(env, ctx))
        {
            _bindable = ScoreUtils.GetSchemaBindableMapper(Host, SubPredictor, null);
            _whatTheFeature = SubPredictor as IWhatTheFeatureValueMapper;
        }

        public static SchemaBindableCalibratedPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
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

        public void SaveAsPfa(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputs)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(schema, nameof(schema));
            Host.CheckParam(Utils.Size(outputs) == 2, nameof(outputs), "Expected this to have two outputs");
            Host.Check(CanSavePfa, "Called despite not being savable");

            ctx.Hide(outputs);
        }

        public bool SaveAsOnnx(IOnnxContext ctx, RoleMappedSchema schema, string[] outputs)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckParam(Utils.Size(outputs) == 2, nameof(outputs), "Expected this to have two outputs");
            Host.CheckValue(schema, nameof(schema));
            Host.Check(CanSaveOnnx, "Called despite not being savable");
            return false;
        }

        public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Host.CheckValue(env, nameof(env));
            env.CheckValue(schema, nameof(schema));
            return new Bound(Host, this, schema);
        }

        public ValueMapper<TSrc, VBuffer<float>> GetWhatTheFeatureMapper<TSrc, TDst>(int top, int bottom, bool normalize)
        {
            // REVIEW: checking this a bit too late.
            Host.Check(_whatTheFeature != null, "Predictor does not implement IWhatTheFeatureValueMapper");
            return _whatTheFeature.GetWhatTheFeatureMapper<TSrc, TDst>(top, bottom, normalize);
        }
    }

    public static class CalibratorUtils
    {
        private static bool NeedCalibration(IHostEnvironment env, IChannel ch, ICalibratorTrainer calibrator,
            ITrainer trainer, IPredictor predictor, RoleMappedSchema schema)
        {
            var trainerEx = trainer as ITrainerEx;
            if (trainerEx == null || !trainerEx.NeedCalibration)
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

            if (!(predictor is IPredictorProducing<Float>))
            {
                ch.Info("Not training a calibrator because the predictor does not implement IPredictorProducing<float>.");
                return false;
            }

            var bindable = ScoreUtils.GetSchemaBindableMapper(env, predictor, null);
            var bound = bindable.Bind(env, schema);
            var outputSchema = bound.OutputSchema;
            int scoreCol;
            if (!outputSchema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreCol))
            {
                ch.Info("Not training a calibrator because the predictor does not output a score column.");
                return false;
            }
            var type = outputSchema.GetColumnType(scoreCol);
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

            return TrainCalibrator(env, ch, calibrator, maxRows, predictor, data);
        }

        /// <summary>
        /// Trains a calibrator.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="ch">The channel.</param>
        /// <param name="caliTrainer">The calibrator trainer.</param>
        /// <param name="maxRows">The maximum rows to use for calibrator training.</param>
        /// <param name="predictor">The predictor that needs calibration.</param>
        /// <param name="data">The examples to used for calibrator training.</param>
        /// <returns>The original predictor, if no calibration is needed, 
        /// or a metapredictor that wraps the original predictor and the newly trained calibrator.</returns>
        public static IPredictor TrainCalibrator(IHostEnvironment env, IChannel ch, ICalibratorTrainer caliTrainer,
            int maxRows, IPredictor predictor, RoleMappedData data)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckValue(predictor, nameof(predictor));
            ch.CheckValue(data, nameof(data));
            ch.CheckParam(data.Schema.Label != null, nameof(data), "data must have a Label column");

            var scored = ScoreUtils.GetScorer(predictor, data, env, null);

            if (caliTrainer.NeedsTraining)
            {
                int labelCol;
                if (!scored.Schema.TryGetColumnIndex(data.Schema.Label.Name, out labelCol))
                    throw ch.Except("No label column found");
                int scoreCol;
                if (!scored.Schema.TryGetColumnIndex(MetadataUtils.Const.ScoreValueKind.Score, out scoreCol))
                    throw ch.Except("No score column found");
                int weightCol;
                if (data.Schema.Weight == null || !scored.Schema.TryGetColumnIndex(data.Schema.Weight.Name, out weightCol))
                    weightCol = -1;
                ch.Info("Training calibrator.");
                using (var cursor = scored.GetRowCursor(col => col == labelCol || col == scoreCol || col == weightCol))
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
            var cali = caliTrainer.FinishTraining(ch);
            return CreateCalibratedPredictor(env, (IPredictorProducing<Float>)predictor, cali);
        }

        public static IPredictorProducing<Float> CreateCalibratedPredictor(IHostEnvironment env, IPredictorProducing<Float> predictor, ICalibrator cali)
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
            var predWithFeatureScores = predictor as IPredictorWithFeatureWeights<Float>;
            if (predWithFeatureScores != null && predictor is IParameterMixer<Float> && cali is IParameterMixer)
                return new ParameterMixingCalibratedPredictor(env, predWithFeatureScores, cali);
            if (predictor is IValueMapper)
                return new CalibratedPredictor(env, predictor, cali);
            return new SchemaBindableCalibratedPredictor(env, predictor, cali);
        }
    }

    [TlcModule.Component(Name = "NaiveCalibrator", FriendlyName = "Naive Calibrator", Alias = "Naive")]
    public sealed class NaiveCalibratorTrainerFactory : ICalibratorTrainerFactory
    {
        public ICalibratorTrainer CreateComponent(IHostEnvironment env)
        {
            return new NaiveCalibratorTrainer(env);
        }
    }

    public sealed class NaiveCalibratorTrainer : ICalibratorTrainer
    {
        private readonly IHost _host;

        private List<Float> _cMargins;
        private List<Float> _ncMargins;

        private int _numBins;
        private Float _binSize;
        private Float _min;
        private Float _max;
        private Float[] _binProbs;

        // REVIEW: The others have user/load names of calibraTION, but this has calibratOR.
        public const string UserName = "Naive Calibrator";
        public const string LoadName = "NaiveCalibrator";
        internal const string Summary = "Naive calibrator divides the range of the outputs into equally sized bins. In each bin, "
            + "the probability of belonging to class 1 is the number of class 1 instances in the bin, divided by the total number "
            + "of instances in the bin.";

        public NaiveCalibratorTrainer(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName);
            _cMargins = new List<Float>();
            _ncMargins = new List<Float>();
            _numBins = 200;
            _min = Float.MaxValue;
            _max = Float.MinValue;
        }

        public bool NeedsTraining => true;

        public bool ProcessTrainingExample(Float output, bool labelIs1, Float weight)
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

        public ICalibrator FinishTraining(IChannel ch)
        {
            Float[] cOutputs = _cMargins.ToArray();
            ch.Check(cOutputs.Length > 0, "Calibrator trained on zero instances.");

            Float minC = MathUtils.Min(cOutputs);
            Float maxC = MathUtils.Max(cOutputs);

            Float[] ncOutputs = _ncMargins.ToArray();
            Float minNC = MathUtils.Min(ncOutputs);
            Float maxNC = MathUtils.Max(ncOutputs);

            _min = (minC < minNC) ? minC : minNC;
            _max = (maxC > maxNC) ? maxC : maxNC;
            _binSize = (_max - _min) / _numBins;

            Float[] cBins = new Float[_numBins];
            Float[] ncBins = new Float[_numBins];

            foreach (Float xi in cOutputs)
            {
                int binIdx = NaiveCalibrator.GetBinIdx(xi, _min, _binSize, _numBins);
                cBins[binIdx]++;
            }

            foreach (Float xi in ncOutputs)
            {
                int binIdx = NaiveCalibrator.GetBinIdx(xi, _min, _binSize, _numBins);
                ncBins[binIdx]++;
            }

            _binProbs = new Float[_numBins];
            for (int i = 0; i < _numBins; i++)
            {
                if (cBins[i] + ncBins[i] == 0)
                    _binProbs[i] = 0;
                else
                    _binProbs[i] = cBins[i] / (cBins[i] + ncBins[i]);
            }

            return new NaiveCalibrator(_host, _min, _binSize, _binProbs);
        }
    }

    /// <summary>
    /// The naive binning-based calibrator
    /// </summary>
    public sealed class NaiveCalibrator : ICalibrator, ICanSaveInBinaryFormat
    {
        public const string LoaderSignature = "NaiveCaliExec";
        public const string RegistrationName = "NaiveCalibrator";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NAIVECAL",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private readonly IHost _host;
        private readonly Float _binSize;
        private readonly Float _min;
        private readonly Float[] _binProbs;

        /// <summary> Create a default calibrator </summary>
        public NaiveCalibrator(IHostEnvironment env, Float min, Float binSize, Float[] binProbs)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _min = min;
            _binSize = binSize;
            _binProbs = binProbs;
        }

        private NaiveCalibrator(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // int: sizeof(Float)
            // Float: bin size
            // Float: minimum value of first bin
            // int: number of bins
            // Float[]: probability in each bin
            int cbFloat = ctx.Reader.ReadInt32();
            _host.CheckDecode(cbFloat == sizeof(Float));

            _binSize = ctx.Reader.ReadFloat();
            _host.CheckDecode(0 < _binSize && _binSize < Float.PositiveInfinity);

            _min = ctx.Reader.ReadFloat();
            _host.CheckDecode(FloatUtils.IsFinite(_min));

            _binProbs = ctx.Reader.ReadFloatArray();
            _host.CheckDecode(Utils.Size(_binProbs) > 0);
            _host.CheckDecode(_binProbs.All(x => (0 <= x && x <= 1)));
        }

        public static NaiveCalibrator Create(IHostEnvironment env, ModelLoadContext ctx)
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
            // int: sizeof(Float)
            // Float: bin size
            // Float: minimum value of first bin
            // int: number of bins
            // Float[]: probability in each bin
            ctx.Writer.Write(sizeof(Float));
            ctx.Writer.Write(_binSize);
            ctx.Writer.Write(_min);
            ctx.Writer.WriteFloatArray(_binProbs);
        }

        /// <summary>
        /// Given a classifier output, produce the probability
        /// </summary>
        public Float PredictProbability(Float output)
        {
            if (Float.IsNaN(output))
                return output;
            int binIdx = GetBinIdx(output, _min, _binSize, _binProbs.Length);
            return _binProbs[binIdx];
        }

        // get the bin for a given output
        internal static int GetBinIdx(Float output, Float min, Float binSize, int numBins)
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
            return string.Format("Naive Calibrator has {0} bins, starting at {1}, with bin size of {2}", _binProbs.Length, _min, _binSize);
        }
    }

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

        public bool NeedsTraining { get { return true; } }

        /// <summary>
        /// Training calibrators:  provide the classifier output and the class label
        /// </summary>
        public bool ProcessTrainingExample(Float output, bool labelIs1, Float weight)
        {
            if (Data == null)
                Data = new CalibrationDataStore(MaxNumSamples);
            Data.AddToStore(output, labelIs1, weight);
            return true;
        }

        public ICalibrator FinishTraining(IChannel ch)
        {
            ch.Check(Data != null, "Calibrator trained on zero instances.");
            return CreateCalibrator(ch);
        }

        public abstract ICalibrator CreateCalibrator(IChannel ch);
    }

    [TlcModule.Component(Name = "PlattCalibrator", FriendlyName = "Platt Calibrator", Aliases = new[] { "Platt", "Sigmoid" }, Desc = "Platt calibration.")]
    public sealed class PlattCalibratorTrainerFactory : ICalibratorTrainerFactory
    {
        public ICalibratorTrainer CreateComponent(IHostEnvironment env)
        {
            return new PlattCalibratorTrainer(env);
        }
    }

    public sealed class PlattCalibratorTrainer : CalibratorTrainerBase
    {
        private Double _paramA;
        private Double _paramB;

        public const string UserName = "Sigmoid Calibration";
        public const string LoadName = "PlattCalibration";
        internal const string Summary = "This model was introduced by Platt in the paper Probabilistic Outputs for Support Vector Machines "
            + "and Comparisons to Regularized Likelihood Methods";

        public PlattCalibratorTrainer(IHostEnvironment env)
            : base(env, LoadName)
        {

        }

        public override ICalibrator CreateCalibrator(IChannel ch)
        {
            _paramA = 0;
            _paramB = 0;
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
                return new PlattCalibrator(Host, _paramA, _paramB);

            _paramA = 0;
            // Initialize B to be the marginal probability of class
            // smoothed i.e. P(+ | x) = (N+ + 1) / (N + 2)
            _paramB = Math.Log((prior0 + 1) / (prior1 + 1));

            // OK. We're going to maximize the likelihood of the output by
            // minimizing the cross-entropy of the output. Here's a
            // magic special hack: make the target of the cross-entropy function
            Double hiTarget = (prior1 + 1) / (prior1 + 2);
            Double loTarget = 1 / (prior0 + 2);

            Double lambda = 0.001;
            Double olderr = Double.MaxValue / 2;
            // array to store current estimate of probability of training points
            Float[] pp = new Float[n];
            Float defValue = (Float)((prior1 + 1) / (prior0 + prior1 + 2));
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
                Double oldA = _paramA;
                Double oldB = _paramB;
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
                    _paramA = oldA + ((b + lambda) * d - c * e) / det;
                    _paramB = oldB + ((a + lambda) * e - c * d) / det;
                    // Now, compute goodness of fit
                    err = 0;

                    i = 0;
                    foreach (var d_i in Data)
                    {
                        var y = d_i.Target ? d_i.Score : -d_i.Score;
                        var p = PlattCalibrator.PredictProbability(d_i.Score, _paramA, _paramB);
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

            return new PlattCalibrator(Host, _paramA, _paramB);
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

        public const string UserName = "Fixed Sigmoid Calibration";
        public const string LoadName = "FixedPlattCalibration";
        internal const string Summary = "Sigmoid calibrator with configurable slope and offset.";

        private readonly IHost _host;
        private readonly Double _slope;
        private readonly Double _offset;

        public FixedPlattCalibratorTrainer(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName);
            _slope = args.Slope;
            _offset = args.Offset;
        }

        public bool NeedsTraining => false;

        public bool ProcessTrainingExample(Float output, bool labelIs1, Float weight)
            => false;

        public ICalibrator FinishTraining(IChannel ch)
        {
            return new PlattCalibrator(_host, _slope, _offset);
        }
    }

    public sealed class PlattCalibrator : ICalibrator, IParameterMixer, ICanSaveModel, ISingleCanSavePfa, ISingleCanSaveOnnx
    {
        public const string LoaderSignature = "PlattCaliExec";
        public const string RegistrationName = "PlattCalibrator";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PLATTCAL",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private readonly IHost _host;

        public Double ParamA { get; }
        public Double ParamB { get; }
        public bool CanSavePfa => true;
        public bool CanSaveOnnx => true;

        public PlattCalibrator(IHostEnvironment env, Double paramA, Double paramB)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            ParamA = paramA;
            ParamB = paramB;
        }

        private PlattCalibrator(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            _host = env.Register(RegistrationName);
            _host.AssertValue(ctx);

            // *** Binary format ***
            // Double: A
            // Double: B
            ParamA = ctx.Reader.ReadDouble();
            _host.CheckDecode(FloatUtils.IsFinite(ParamA));

            ParamB = ctx.Reader.ReadDouble();
            _host.CheckDecode(FloatUtils.IsFinite(ParamB));
        }

        public static PlattCalibrator Create(IHostEnvironment env, ModelLoadContext ctx)
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
            ctx.Writer.Write(ParamA);
            ctx.Writer.Write(ParamB);

            if (ctx.InRepository)
            {
                ctx.SaveTextStream("Calibrator.txt", (Action<TextWriter>)(writer =>
                {
                    writer.WriteLine("Platt calibrator");
                    writer.WriteLine("P(y=1|x) = 1/1+exp(A*x + B)");
                    writer.WriteLine("A={0:R}", (object)ParamA);
                    writer.WriteLine("B={0:R}", ParamB);
                }));
            }
        }

        public Float PredictProbability(Float output)
        {
            if (Float.IsNaN(output))
                return output;
            return PredictProbability(output, ParamA, ParamB);
        }

        public static Float PredictProbability(Float output, Double a, Double b)
        {
            return (Float)(1 / (1 + Math.Exp(a * output + b)));
        }

        public JToken SaveAsPfa(BoundPfaContext ctx, JToken input)
        {
            _host.CheckValue(ctx, nameof(ctx));
            _host.CheckValue(input, nameof(input));

            return PfaUtils.Call("m.link.logit",
                PfaUtils.Call("+", -ParamB, PfaUtils.Call("*", -ParamA, input)));
        }

        public bool SaveAsOnnx(IOnnxContext ctx, string[] scoreProbablityColumnNames, string featureColumnName)
        {
            _host.CheckValue(ctx, nameof(ctx));
            _host.CheckValue(scoreProbablityColumnNames, nameof(scoreProbablityColumnNames));
            _host.Check(Utils.Size(scoreProbablityColumnNames) == 2);

            string opType = "Affine";
            string linearOutput = ctx.AddIntermediateVariable(null, "linearOutput", true);
            var node = ctx.CreateNode(opType, new[] { scoreProbablityColumnNames[0] },
                new[] { linearOutput }, ctx.GetNodeName(opType), "ai.onnx");
            node.AddAttribute("alpha", ParamA * -1);
            node.AddAttribute("beta", -0.0000001);

            opType = "Sigmoid";
            node = ctx.CreateNode(opType, new[] { linearOutput },
                new[] { scoreProbablityColumnNames[1] }, ctx.GetNodeName(opType), "ai.onnx");

            return true;
        }

        public string GetSummary()
        {
            return string.Format("Platt calibrator parameters: A={0}, B={1}", ParamA, ParamB);
        }

        public IParameterMixer CombineParameters(IList<IParameterMixer> calibrators)
        {
            Double a = 0;
            Double b = 0;
            foreach (IParameterMixer calibrator in calibrators)
            {
                PlattCalibrator cal = calibrator as PlattCalibrator;

                a += cal.ParamA;
                b += cal.ParamB;
            }

            PlattCalibrator newCal = new PlattCalibrator(_host, a / calibrators.Count, b / calibrators.Count);
            return newCal;
        }
    }

    [TlcModule.Component(Name = "PavCalibrator", FriendlyName = "PAV Calibrator", Alias = "Pav")]
    public sealed class PavCalibratorTrainerFactory : ICalibratorTrainerFactory
    {
        public ICalibratorTrainer CreateComponent(IHostEnvironment env)
        {
            return new PavCalibratorTrainer(env);
        }
    }

    public class PavCalibratorTrainer : CalibratorTrainerBase
    {
        // a piece of the piecwise function
        private struct Piece
        {
            public readonly Float MinX; // end of interval.
            public readonly Float MaxX; // beginning of interval.
            public readonly Float Value; // value of function in interval.
            public readonly Float N; // number of points/sum of weights of interval.

            public Piece(Float minX, Float maxX, Float value, Float n)
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

        public const string UserName = "PAV Calibration";
        public const string LoadName = "PAVCalibration";
        internal const string Summary = "Piecewise linear calibrator.";

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
                    Float newN = top.N + curr.N;
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
            Float[] mins = new Float[stack.Count];
            Float[] maxes = new Float[stack.Count];
            Float[] values = new Float[stack.Count];

            for (int i = stack.Count - 1; stack.Count > 0; --i)
            {
                top = stack.Pop();
                mins[i] = top.MinX;
                maxes[i] = top.MaxX;
                values[i] = top.Value;
            }

            return new PavCalibrator(Host, mins, maxes, values);
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
        public const string LoaderSignature = "PAVCaliExec";
        public const string RegistrationName = "PAVCalibrator";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PAV  CAL",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        // Epsilon for 0-comparisons
        private const Float Epsilon = (Float)1e-15;
        private const Float MinToReturn = Epsilon; // max predicted is 1 - min;
        private const Float MaxToReturn = 1 - Epsilon; // max predicted is 1 - min;

        private readonly IHost _host;
        private readonly Float[] _mins;
        private readonly Float[] _maxes;
        private readonly Float[] _values;

        internal PavCalibrator(IHostEnvironment env, Float[] mins, Float[] maxes, Float[] values)
        {
            Contracts.AssertValue(env);
            _host = env.Register(RegistrationName);
            _host.AssertValue(mins);
            _host.AssertValue(maxes);
            _host.AssertValue(values);
            _host.Assert(Utils.IsSorted(mins));
            _host.Assert(Utils.IsSorted(maxes));
            _host.Assert(Utils.IsSorted(values));
            _host.Assert(values.Length == 0 || (0 <= values[0] && values[values.Length - 1] <= 1));
            _host.Assert(mins.Zip(maxes, (min, max) => min <= max).All(x => x));

            _mins = mins;
            _maxes = maxes;
            _values = values;
        }

        private PavCalibrator(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            _host = env.Register(RegistrationName);
            _host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // int: number of pieces
            // for each piece:
            //      Float: MinX
            //      Float: MaxX
            //      Float: Value
            int cbFloat = ctx.Reader.ReadInt32();
            _host.CheckDecode(cbFloat == sizeof(Float));

            int numPieces = ctx.Reader.ReadInt32();
            _host.CheckDecode(numPieces >= 0);
            _mins = new Float[numPieces];
            _maxes = new Float[numPieces];
            _values = new Float[numPieces];
            Float valuePrev = 0;
            Float maxPrev = Float.NegativeInfinity;
            for (int i = 0; i < numPieces; ++i)
            {
                Float minX = ctx.Reader.ReadFloat();
                Float maxX = ctx.Reader.ReadFloat();
                Float val = ctx.Reader.ReadFloat();
                _host.CheckDecode(minX <= maxX);
                _host.CheckDecode(minX > maxPrev);
                _host.CheckDecode(val > valuePrev || val == valuePrev && i == 0);
                valuePrev = val;
                maxPrev = maxX;
                _mins[i] = minX;
                _maxes[i] = maxX;
                _values[i] = val;
            }
            _host.CheckDecode(valuePrev <= 1);
        }

        public static PavCalibrator Create(IHostEnvironment env, ModelLoadContext ctx)
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
            // int: sizeof(Float)
            // int: number of pieces
            // for each piece:
            //      Float: MinX
            //      Float: MaxX
            //      Float: Value
            ctx.Writer.Write(sizeof(Float));

            _host.Assert(_mins.Length == _maxes.Length);
            _host.Assert(_mins.Length == _values.Length);
            ctx.Writer.Write(_mins.Length);
            Float valuePrev = 0;
            Float maxPrev = Float.NegativeInfinity;
            for (int i = 0; i < _mins.Length; i++)
            {
                _host.Assert(_mins[i] <= _maxes[i]);
                _host.Assert(_mins[i] > maxPrev);
                _host.Assert(_values[i] > valuePrev || _values[i] == valuePrev && i == 0);
                valuePrev = _values[i];
                maxPrev = _maxes[i];
                ctx.Writer.Write(_mins[i]);
                ctx.Writer.Write(_maxes[i]);
                ctx.Writer.Write(_values[i]);
            }
            _host.CheckDecode(valuePrev <= 1);
        }

        public Float PredictProbability(Float output)
        {
            if (Float.IsNaN(output))
                return output;
            Float prob = FindValue(output);
            if (prob < MinToReturn)
                return MinToReturn;
            if (prob > MaxToReturn)
                return MaxToReturn;
            return prob;
        }

        private Float FindValue(Float score)
        {
            int p = _mins.Length;
            if (p == 0)
                return 0;
            if (score < _mins[0])
            {
                return _values[0];
                // tail off to zero exponentially
                // return Math.Exp(-(piecewise[0].MinX-score)) * piecewise[0].Value;
            }
            if (score > _maxes[p - 1])
            {
                return _values[p - 1];
                // tail off to one exponentially
                // return (1-Math.Exp(-(score - piecewise[P - 1].MaxX))) * (1 - piecewise[P - 1].Value) + piecewise[P - 1].Value;
            }

            int pos = _maxes.FindIndexSorted(score);
            _host.Assert(pos < p);
            // inside the piece, the value is constant
            if (score >= _mins[pos])
                return _values[pos];
            // between pieces, interpolate
            Float t = (score - _maxes[pos - 1]) / (_mins[pos] - _maxes[pos - 1]);
            return _values[pos - 1] + t * (_values[pos] - _values[pos - 1]);
        }

        public string GetSummary()
        {
            return string.Format("PAV calibrator with {0} intervals", _mins.Length);
        }
    }

    public sealed class CalibrationDataStore : IEnumerable<CalibrationDataStore.DataItem>
    {
        public struct DataItem
        {
            // The actual binary label of this example.
            public readonly bool Target;
            // The weight associated with this example.
            public readonly Float Weight;
            // The output of the example.
            public readonly Float Score;

            public DataItem(bool target, Float weight, Float score)
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

        public void AddToStore(Float score, bool isPositive, Float weight)
        {
            // Can't calibrate NaN scores.
            if (weight == 0 || Float.IsNaN(score))
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

    public static class Calibrate
    {
        [TlcModule.EntryPointKind(typeof(CommonInputs.ICalibratorInput))]
        public abstract class CalibrateInputBase : TransformInputBase
        {
            [Argument(ArgumentType.Required, ShortName = "uncalibratedPredictorModel", HelpText = "The predictor to calibrate", SortOrder = 2)]
            public IPredictorModel UncalibratedPredictorModel;

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
        /// <returns>A <see cref="CommonOutputs.TrainerOutput"/> object, containing an <see cref="IPredictorModel"/>.</returns>
        public static TOut CalibratePredictor<TOut>(IHost host, CalibrateInputBase input,
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
                        CalibratorUtils.TrainCalibrator(host, ch, calibratorTrainer, input.MaxRows, predictor, data);
                }
                ch.Done();

                return new TOut() { PredictorModel = new PredictorModel(host, data, input.Data, calibratedPredictor) };
            }
        }
    }
}
