// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Learners;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Trainers;
using Microsoft.ML.Training;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(Ova.Summary, typeof(Ova), typeof(Ova.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    Ova.UserNameValue,
    Ova.LoadNameValue, DocName = "trainer/OvaPkpd.md")]

[assembly: LoadableClass(typeof(OvaModelParameters), null, typeof(SignatureLoadModel),
    "OVA Executor",
    OvaModelParameters.LoaderSignature)]

[assembly: EntryPointModule(typeof(OvaModelParameters))]
namespace Microsoft.ML.Trainers
{
    using CR = RoleMappedSchema.ColumnRole;
    using TDistPredictor = IDistPredictorProducing<float, float>;
    using TScalarPredictor = IPredictorProducing<float>;
    using TScalarTrainer = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;

    public sealed class Ova : MetaMulticlassTrainer<MulticlassPredictionTransformer<OvaModelParameters>, OvaModelParameters>
    {
        internal const string LoadNameValue = "OVA";
        internal const string UserNameValue = "One-vs-All";
        internal const string Summary = "In this strategy, a binary classification algorithm is used to train one classifier for each class, "
            + "which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers, "
            + "and choosing the prediction with the highest confidence score.";

        private readonly Arguments _args;

        /// <summary>
        /// Arguments passed to OVA.
        /// </summary>
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use probability or margins to determine max", ShortName = "useprob")]
            [TGUI(Label = "Use Probability", Description = "Use probabilities (vs. raw outputs) to identify top-score category")]
            public bool UseProbabilities = true;
        }

        /// <summary>
        /// Legacy constructor that builds the <see cref="Ova"/> trainer supplying the base trainer to use, for the classification task
        /// through the <see cref="Arguments"/>arguments.
        /// Developers should instantiate OVA by supplying the trainer argument directly to the OVA constructor
        /// using the other public constructor.
        /// </summary>
        /// <param name="env">The private <see cref="IHostEnvironment"/> for this estimator.</param>
        /// <param name="args">The legacy <see cref="Arguments"/></param>
        internal Ova(IHostEnvironment env, Arguments args)
            : base(env, args, LoadNameValue)
        {
            _args = args;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="Ova"/>.
        /// </summary>
        /// <param name="env">The <see cref="IHostEnvironment"/> instance.</param>
        /// <param name="binaryEstimator">An instance of a binary <see cref="ITrainerEstimator{TTransformer, TPredictor}"/> used as the base trainer.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not explicitely provided, it will default to <see cref="PlattCalibratorTrainer"/></param>
        /// <param name="labelColumn">The name of the label colum.</param>
        /// <param name="imputeMissingLabelsAsNegative">Whether to treat missing labels as having negative labels, instead of keeping them missing.</param>
        /// <param name="maxCalibrationExamples">Number of instances to train the calibrator.</param>
        /// <param name="useProbabilities">Use probabilities (vs. raw outputs) to identify top-score category.</param>
        public Ova(IHostEnvironment env,
            TScalarTrainer binaryEstimator,
            string labelColumn = DefaultColumnNames.Label,
            bool imputeMissingLabelsAsNegative = false,
            ICalibratorTrainer calibrator = null,
            int maxCalibrationExamples = 1000000000,
            bool useProbabilities = true)
         : base(env,
               new Arguments
               {
                   ImputeMissingLabelsAsNegative = imputeMissingLabelsAsNegative,
                   MaxCalibrationExamples = maxCalibrationExamples,
               },
               LoadNameValue, labelColumn, binaryEstimator, calibrator)
        {
            Host.CheckValue(labelColumn, nameof(labelColumn), "Label column should not be null.");
            _args = (Arguments)Args;
            _args.UseProbabilities = useProbabilities;
        }

        private protected override OvaModelParameters TrainCore(IChannel ch, RoleMappedData data, int count)
        {
            // Train one-vs-all models.
            var predictors = new TScalarPredictor[count];
            for (int i = 0; i < predictors.Length; i++)
            {
                ch.Info($"Training learner {i}");
                predictors[i] = TrainOne(ch, Trainer, data, i).Model;
            }
            return OvaModelParameters.Create(Host, _args.UseProbabilities, predictors);
        }

        private ISingleFeaturePredictionTransformer<TScalarPredictor> TrainOne(IChannel ch, TScalarTrainer trainer, RoleMappedData data, int cls)
        {
            var view = MapLabels(data, cls);

            string trainerLabel = data.Schema.Label.Value.Name;

            // REVIEW: In principle we could support validation sets and the like via the train context, but
            // this is currently unsupported.
            var transformer = trainer.Fit(view);

            if (_args.UseProbabilities)
            {
                var calibratedModel = transformer.Model as TDistPredictor;

                // REVIEW: restoring the RoleMappedData, as much as we can.
                // not having the weight column on the data passed to the TrainCalibrator should be addressed.
                var trainedData = new RoleMappedData(view, label: trainerLabel, feature: transformer.FeatureColumn);

                if (calibratedModel == null)
                    calibratedModel = CalibratorUtils.GetCalibratedPredictor(Host, ch, Calibrator, transformer.Model, trainedData, Args.MaxCalibrationExamples) as TDistPredictor;

                Host.Check(calibratedModel != null, "Calibrated predictor does not implement the expected interface");
                return new BinaryPredictionTransformer<TScalarPredictor>(Host, calibratedModel, trainedData.Data.Schema, transformer.FeatureColumn);
            }

            return new BinaryPredictionTransformer<TScalarPredictor>(Host, transformer.Model, view.Schema, transformer.FeatureColumn);
        }

        private IDataView MapLabels(RoleMappedData data, int cls)
        {
            var lab = data.Schema.Label.Value;
            Host.Assert(!lab.IsHidden);
            Host.Assert(lab.Type.KeyCount > 0 || lab.Type == NumberType.R4 || lab.Type == NumberType.R8);

            if (lab.Type.KeyCount > 0)
            {
                // Key values are 1-based.
                uint key = (uint)(cls + 1);
                return MapLabelsCore(NumberType.U4, (in uint val) => key == val, data);
            }
            if (lab.Type == NumberType.R4)
            {
                float key = cls;
                return MapLabelsCore(NumberType.R4, (in float val) => key == val, data);
            }
            if (lab.Type == NumberType.R8)
            {
                Double key = cls;
                return MapLabelsCore(NumberType.R8, (in double val) => key == val, data);
            }

            throw Host.ExceptNotSupp($"Label column type is not supported by OVA: {lab.Type}");
        }

        public override MulticlassPredictionTransformer<OvaModelParameters> Fit(IDataView input)
        {
            var roles = new KeyValuePair<CR, string>[1];
            roles[0] = new KeyValuePair<CR, string>(new CR(DefaultColumnNames.Label), LabelColumn.Name);
            var td = new RoleMappedData(input, roles);

            td.CheckMultiClassLabel(out var numClasses);

            var predictors = new TScalarPredictor[numClasses];
            string featureColumn = null;

            using (var ch = Host.Start("Fitting"))
            {
                for (int i = 0; i < predictors.Length; i++)
                {
                    ch.Info($"Training learner {i}");

                    if (i == 0)
                    {
                        var transformer = TrainOne(ch, Trainer, td, i);
                        featureColumn = transformer.FeatureColumn;
                    }

                    predictors[i] = TrainOne(ch, Trainer, td, i).Model;
                }
            }

            return new MulticlassPredictionTransformer<OvaModelParameters>(Host, OvaModelParameters.Create(Host, _args.UseProbabilities, predictors), input.Schema, featureColumn, LabelColumn.Name);
        }
    }

    public sealed class OvaModelParameters :
        ModelParametersBase<VBuffer<float>>,
        IValueMapper,
        ICanSaveInSourceCode,
        ICanSaveInTextFormat,
        ISingleCanSavePfa
    {
        internal const string LoaderSignature = "OVAExec";
        internal const string RegistrationName = "OVAPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TLC OVA ",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(OvaModelParameters).Assembly.FullName);
        }

        private const string SubPredictorFmt = "SubPredictor_{0:000}";

        private readonly ImplBase _impl;

        public ImmutableArray<object> SubModelParameters => _impl.Predictors.Cast<object>().ToImmutableArray();

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;

        /// <summary>
        /// Function applied to output of predictors. Assume that we have n predictors (one per class) and for the i-th predictor,
        /// y_i is its raw output and p_i is its probability output. Note that not all predictors are able to produce probability output.
        /// <para>
        /// <see cref="Raw"/>: output the result of predictors without post-processing. Output is [y_1, ..., y_n].
        /// <see cref="ProbabilityNormalization"/>: fetch probability output of each class probability from provided predictors and make sure the sume of class probabilities is one.
        /// Output is [p_1 / (p_1 + ... + p_n), ..., p_n / (p_1 + ... + p_n)].
        /// <see cref="Softmax"/>: Generate probability by feeding raw outputs to softmax function. Output is [z_1, ..., z_n], where z_i is exp(y_i) / (exp(y_1) + ... + exp(y_n)).
        /// </para>
        /// </summary>
        public enum OutputFormula { Raw = 0, ProbabilityNormalization = 1, Softmax = 2 };
        private readonly ColumnType _outputType;
        private ColumnType DistType => _outputType;
        bool ICanSavePfa.CanSavePfa => _impl.CanSavePfa;

        [BestFriend]
        internal static OvaModelParameters Create(IHost host,  OutputFormula outputFormula, TScalarPredictor[] predictors)
        {
            ImplBase impl;

            using (var ch = host.Start("Creating OVA predictor"))
            {
                if (outputFormula == OutputFormula.Softmax)
                {
                    impl = new ImplSoftmax(predictors);
                    return new OvaModelParameters(host, impl);
                }

                // Caller of this function asks for probability output. We check if input predictor can produce probability.
                // If that predictor can't produce probability, ivmd will be null.
                IValueMapperDist ivmd = null;
                if (outputFormula == OutputFormula.ProbabilityNormalization &&
                    ((ivmd = predictors[0] as IValueMapperDist) == null ||
                        ivmd.OutputType != NumberType.Float ||
                        ivmd.DistType != NumberType.Float))
                {
                    ch.Warning($"{nameof(Ova.Arguments.UseProbabilities)} specified with {nameof(Ova.Arguments.PredictorType)} that can't produce probabilities.");
                    ivmd = null;
                }

                // If ivmd is null, either the user didn't ask for probability or the provided predictors can't produce probability.
                if (ivmd != null)
                {
                    var dists = new IValueMapperDist[predictors.Length];
                    for (int i = 0; i < predictors.Length; ++i)
                        dists[i] = (IValueMapperDist)predictors[i];
                    impl = new ImplDist(dists);
                }
                else
                    impl = new ImplRaw(predictors);
            }

            return new OvaModelParameters(host, impl);
        }

        [BestFriend]
        internal static OvaModelParameters Create(IHost host, bool useProbability, TScalarPredictor[] predictors)
        {
            var outputFormula = useProbability ? OutputFormula.ProbabilityNormalization : OutputFormula.Raw;

            return Create(host, outputFormula, predictors);
        }

        /// <summary>
        /// Create a OVA predictor from an array of predictors.
        /// </summary>
        [BestFriend]
        internal static OvaModelParameters Create(IHost host, TScalarPredictor[] predictors)
        {
            Contracts.CheckValue(host, nameof(host));
            host.CheckNonEmpty(predictors, nameof(predictors));
            return Create(host, OutputFormula.ProbabilityNormalization, predictors);
        }

        private OvaModelParameters(IHostEnvironment env, ImplBase impl)
                : base(env, RegistrationName)
        {
            Host.AssertValue(impl, nameof(impl));
            Host.Assert(Utils.Size(impl.Predictors) > 0);

            _impl = impl;
            _outputType = new VectorType(NumberType.Float, _impl.Predictors.Length);
        }

        private OvaModelParameters(IHostEnvironment env, ModelLoadContext ctx)
                : base(env, RegistrationName, ctx)
        {
            // *** Binary format ***
            // bool: useDist
            // int: predictor count
            bool useDist = ctx.Reader.ReadBoolByte();
            int len = ctx.Reader.ReadInt32();
            Host.CheckDecode(len > 0);

            if (useDist)
            {
                var predictors = new IValueMapperDist[len];
                LoadPredictors(Host, predictors, ctx);
                _impl = new ImplDist(predictors);
            }
            else
            {
                var predictors = new TScalarPredictor[len];
                LoadPredictors(Host, predictors, ctx);
                _impl = new ImplRaw(predictors);
            }

            _outputType = new VectorType(NumberType.Float, _impl.Predictors.Length);
        }

        private static OvaModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new OvaModelParameters(env, ctx);
        }

        private static void LoadPredictors<TPredictor>(IHostEnvironment env, TPredictor[] predictors, ModelLoadContext ctx)
            where TPredictor : class
        {
            for (int i = 0; i < predictors.Length; i++)
                ctx.LoadModel<TPredictor, SignatureLoadModel>(env, out predictors[i], string.Format(SubPredictorFmt, i));
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            var preds = _impl.Predictors;

            // *** Binary format ***
            // bool: useDist
            // int: predictor count
            ctx.Writer.WriteBoolByte(_impl is ImplDist);
            ctx.Writer.Write(preds.Length);

            // Save other streams.
            for (int i = 0; i < preds.Length; i++)
                ctx.SaveModel(preds[i], string.Format(SubPredictorFmt, i));
        }

        JToken ISingleCanSavePfa.SaveAsPfa(BoundPfaContext ctx, JToken input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(input, nameof(input));
            return _impl.SaveAsPfa(ctx, input);
        }

        ColumnType IValueMapper.InputType
        {
            get { return _impl.InputType; }
        }

        ColumnType IValueMapper.OutputType
        {
            get { return _outputType; }
        }
        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<float>));

            return (ValueMapper<TIn, TOut>)(Delegate)_impl.GetMapper();
        }

        void ICanSaveInSourceCode.SaveAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            var preds = _impl.Predictors;
            writer.WriteLine("double[] outputs = new double[{0}];", preds.Length);

            for (int i = 0; i < preds.Length; i++)
            {
                var saveInSourceCode = preds[i] as ICanSaveInSourceCode;
                Host.Check(saveInSourceCode != null, "Saving in code is not supported.");

                writer.WriteLine("{");
                saveInSourceCode.SaveAsCode(writer, schema);
                writer.WriteLine("outputs[{0}] = output;", i);
                writer.WriteLine("}");
            }
        }

        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            Host.CheckValue(writer, nameof(writer));
            Host.CheckValue(schema, nameof(schema));

            var preds = _impl.Predictors;

            for (int i = 0; i < preds.Length; i++)
            {
                var saveInText = preds[i] as ICanSaveInTextFormat;
                Host.Check(saveInText != null, "Saving in text is not supported.");

                writer.WriteLine("#region: class-{0} classifier", i);
                saveInText.SaveAsText(writer, schema);

                writer.WriteLine("#endregion: class-{0} classifier", i);
                writer.WriteLine();
            }
        }

        private abstract class ImplBase : ISingleCanSavePfa
        {
            public abstract ColumnType InputType { get; }
            public abstract IValueMapper[] Predictors { get; }
            public abstract bool CanSavePfa { get; }
            public abstract ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper();
            public abstract JToken SaveAsPfa(BoundPfaContext ctx, JToken input);

            protected bool IsValid(IValueMapper mapper, ref ColumnType inputType)
            {
                Contracts.AssertValueOrNull(mapper);
                Contracts.AssertValueOrNull(inputType);

                if (mapper == null)
                    return false;
                if (mapper.OutputType != NumberType.Float)
                    return false;
                if (!mapper.InputType.IsVector || mapper.InputType.ItemType != NumberType.Float)
                    return false;
                if (inputType == null)
                    inputType = mapper.InputType;
                else if (inputType.VectorSize != mapper.InputType.VectorSize)
                {
                    if (inputType.VectorSize == 0)
                        inputType = mapper.InputType;
                    else if (mapper.InputType.VectorSize != 0)
                        return false;
                }
                return true;
            }
        }

        private sealed class ImplRaw : ImplBase
        {
            public override ColumnType InputType { get; }
            public override IValueMapper[] Predictors { get; }
            public override bool CanSavePfa { get; }

            internal ImplRaw(TScalarPredictor[] predictors)
            {
                Contracts.CheckNonEmpty(predictors, nameof(predictors));

                Predictors = new IValueMapper[predictors.Length];
                ColumnType inputType = null;
                for (int i = 0; i < predictors.Length; i++)
                {
                    var vm = predictors[i] as IValueMapper;
                    Contracts.Check(IsValid(vm, ref inputType), "Predictor doesn't implement the expected interface");
                    Predictors[i] = vm;
                }
                CanSavePfa = Predictors.All(m => (m as ISingleCanSavePfa)?.CanSavePfa == true);
                Contracts.AssertValue(inputType);
                InputType = inputType;
            }

            public override ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper()
            {
                var maps = new ValueMapper<VBuffer<float>, float>[Predictors.Length];
                for (int i = 0; i < Predictors.Length; i++)
                    maps[i] = Predictors[i].GetMapper<VBuffer<float>, float>();

                var buffer = new float[maps.Length];
                return
                    (in VBuffer<float> src, ref VBuffer<float> dst) =>
                    {
                        if (InputType.VectorSize > 0)
                            Contracts.Check(src.Length == InputType.VectorSize);

                        var tmp = src;
                        Parallel.For(0, maps.Length, i => maps[i](in tmp, ref buffer[i]));

                        var editor = VBufferEditor.Create(ref dst, maps.Length);
                        buffer.CopyTo(editor.Values);
                        dst = editor.Commit();
                    };
            }

            public override JToken SaveAsPfa(BoundPfaContext ctx, JToken input)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                Contracts.CheckValue(input, nameof(input));
                Contracts.Assert(CanSavePfa);

                JArray rootObjects = new JArray();
                for (int i = 0; i < Predictors.Length; ++i)
                {
                    var pred = (ISingleCanSavePfa)Predictors[i];
                    Contracts.Assert(pred.CanSavePfa);
                    rootObjects.Add(ctx.DeclareVar(null, pred.SaveAsPfa(ctx, input)));
                }
                JObject jobj = null;
                return jobj.AddReturn("type", PfaUtils.Type.Array(PfaUtils.Type.Double)).AddReturn("new", rootObjects);
            }
        }

        private sealed class ImplDist : ImplBase
        {
            private readonly IValueMapperDist[] _mappers;
            public override ColumnType InputType { get; }
            public override IValueMapper[] Predictors => _mappers;
            public override bool CanSavePfa { get; }

            internal ImplDist(IValueMapperDist[] predictors)
            {
                Contracts.Check(Utils.Size(predictors) > 0);

                _mappers = new IValueMapperDist[predictors.Length];
                ColumnType inputType = null;
                for (int i = 0; i < predictors.Length; i++)
                {
                    var vm = predictors[i];
                    Contracts.Check(IsValid(vm, ref inputType), "Predictor doesn't implement the expected interface");
                    _mappers[i] = vm;
                }
                CanSavePfa = Predictors.All(m => (m as IDistCanSavePfa)?.CanSavePfa == true);
                Contracts.AssertValue(inputType);
                InputType = inputType;
            }

            private bool IsValid(IValueMapperDist mapper, ref ColumnType inputType)
            {
                return base.IsValid(mapper, ref inputType) && mapper.DistType == NumberType.Float;
            }

            /// <summary>
            /// Each predictor produces a probability of a class. All classes' probabilities are normalized so that
            /// their sum is one.
            /// </summary>
            public override ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper()
            {
                var maps = new ValueMapper<VBuffer<float>, float, float>[Predictors.Length];
                for (int i = 0; i < Predictors.Length; i++)
                    maps[i] = _mappers[i].GetMapper<VBuffer<float>, float, float>();

                var buffer = new float[maps.Length];
                return
                    (in VBuffer<float> src, ref VBuffer<float> dst) =>
                    {
                        if (InputType.VectorSize > 0)
                            Contracts.Check(src.Length == InputType.VectorSize);

                        var tmp = src;
                        Parallel.For(0, maps.Length,
                            i =>
                            {
                                float score = 0;
                                // buffer[i] is the probability of the i-th class.
                                // score is the raw prediction score.
                                maps[i](in tmp, ref score, ref buffer[i]);
                            });

                        // buffer[i] is the probability of the i-th class.
                        // score is the raw prediction score.
                        NormalizeSumToOne(buffer, maps.Length);

                        var editor = VBufferEditor.Create(ref dst, maps.Length);
                        buffer.CopyTo(editor.Values);
                        dst = editor.Commit();
                    };
            }

            private void NormalizeSumToOne(float[] output, int count)
            {
                // Clamp to zero and normalize.
                Double sum = 0;
                for (int i = 0; i < count; i++)
                {
                    var value = output[i];
                    if (value >= 0)
                        sum += value;
                    else
                        output[i] = 0;
                }

                if (sum > 0)
                {
                    for (int i = 0; i < count; i++)
                        output[i] = (float)(output[i] / sum);
                }
            }

            public override JToken SaveAsPfa(BoundPfaContext ctx, JToken input)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                Contracts.CheckValue(input, nameof(input));
                Contracts.Assert(CanSavePfa);

                JArray rootObjects = new JArray();
                for (int i = 0; i < Predictors.Length; ++i)
                {
                    var pred = (IDistCanSavePfa)Predictors[i];
                    Contracts.Assert(pred.CanSavePfa);
                    pred.SaveAsPfa(ctx, input, null, out JToken scoreToken, null, out JToken probToken);
                    rootObjects.Add(probToken);
                }
                JObject jobj = null;
                var rootResult = jobj.AddReturn("type", PfaUtils.Type.Array(PfaUtils.Type.Double)).AddReturn("new", rootObjects);
                var resultVar = ctx.DeclareVar(null, rootResult);
                var factorVar = ctx.DeclareVar(null, PfaUtils.Call("/", 1.0, PfaUtils.Call("a.sum", resultVar)));
                return PfaUtils.Call("la.scale", resultVar, factorVar);
            }
        }

        private sealed class ImplSoftmax : ImplBase
        {
            public override ColumnType InputType { get; }
            public override IValueMapper[] Predictors { get; }
            public override bool CanSavePfa { get; }

            internal ImplSoftmax(TScalarPredictor[] predictors)
            {
                Contracts.CheckNonEmpty(predictors, nameof(predictors));

                Predictors = new IValueMapper[predictors.Length];
                ColumnType inputType = null;
                for (int i = 0; i < predictors.Length; i++)
                {
                    var vm = predictors[i] as IValueMapper;
                    Contracts.Check(IsValid(vm, ref inputType), "Predictor doesn't implement the expected interface");
                    Predictors[i] = vm;
                }
                CanSavePfa = false;
                Contracts.AssertValue(inputType);
                InputType = inputType;
            }

            public override ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper()
            {
                var maps = new ValueMapper<VBuffer<float>, float>[Predictors.Length];
                for (int i = 0; i < Predictors.Length; i++)
                    maps[i] = Predictors[i].GetMapper<VBuffer<float>, float>();

                var buffer = new float[maps.Length];
                return
                    (in VBuffer<float> src, ref VBuffer<float> dst) =>
                    {
                        if (InputType.VectorSize > 0)
                            Contracts.Check(src.Length == InputType.VectorSize);

                        var tmp = src;
                        Parallel.For(0, maps.Length, i => maps[i](in tmp, ref buffer[i]));
                        NormalizeSoftmax(buffer, maps.Length);

                        var editor = VBufferEditor.Create(ref dst, maps.Length);
                        buffer.CopyTo(editor.Values);
                        dst = editor.Commit();
                    };
            }

            private void NormalizeSoftmax(float[] scores, int count)
            {
                float sum = 0;
                for (int i = 0; i < count; i++)
                {
                    scores[i] = (float)Math.Exp(scores[i]);
                    sum += scores[i];
                }

                for (int i = 0; i < count; i++)
                    scores[i] = scores[i] / sum;
            }

            public override JToken SaveAsPfa(BoundPfaContext ctx, JToken input)
            {
                throw new NotImplementedException("Softmax's PFA exporter is not implemented yet.");
            }
        }
    }
}