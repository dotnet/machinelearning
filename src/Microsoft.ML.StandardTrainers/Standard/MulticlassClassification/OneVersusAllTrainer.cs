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
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(OneVersusAllTrainer.Summary, typeof(OneVersusAllTrainer), typeof(OneVersusAllTrainer.Options),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer) },
    OneVersusAllTrainer.UserNameValue,
    OneVersusAllTrainer.LoadNameValue)]

[assembly: LoadableClass(typeof(OneVersusAllModelParameters), null, typeof(SignatureLoadModel),
    "OVA Executor",
    OneVersusAllModelParameters.LoaderSignature)]

[assembly: EntryPointModule(typeof(OneVersusAllModelParameters))]
namespace Microsoft.ML.Trainers
{
    using CR = RoleMappedSchema.ColumnRole;
    using TDistPredictor = IDistPredictorProducing<float, float>;
    using TScalarPredictor = IPredictorProducing<float>;
    using TScalarTrainer = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a one-versus-all multi-class classifier that uses the specified binary classifier.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [OneVersusAll](xref:Microsoft.ML.StandardTrainersCatalog.OneVersusAll``1(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Microsoft.ML.Trainers.ITrainerEstimator{Microsoft.ML.Data.BinaryPredictionTransformer{``0},``0},System.String,System.Boolean,Microsoft.ML.IEstimator{Microsoft.ML.ISingleFeaturePredictionTransformer{Microsoft.ML.Calibrators.ICalibrator}},System.Int32,System.Boolean)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-multiclass-classification.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | Depends on the underlying binary classifier |
    /// | Is caching required? | Yes |
    /// | Required NuGet in addition to Microsoft.ML | None |
    /// | Exportable to ONNX | Yes |
    ///
    /// ### Training Algorithm Details
    /// In one-versus-all (OVA) strategy, a binary classification algorithm is used to train one classifier for each class,
    /// which distinguishes that class from all other classes. Prediction is then performed by running
    /// these binary classifiers and choosing the prediction with the highest confidence score.
    /// This algorithm can be used with any of the binary classifiers in ML.NET. A few binary classifiers
    /// already have implementation for multi-class problems, thus users can choose either one depending on the context.
    /// The OVA version of a binary classifier, such as wrapping a <xref:Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer>,
    /// can be different from <xref:Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer>, which develops a multi-class classifier directly.
    /// Note that even if the classifier indicates that it does not need caching, OneVersusAll will always
    /// request caching, as it will be performing multiple passes over the data set.
    /// This trainer will request normalization from the data pipeline if the classifier indicates it would benefit from it.
    ///
    /// This can allow you to exploit trainers that do not naturally have a
    /// multiclass option, for example, using the <xref:Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer>
    /// to solve a multiclass problem.
    /// Alternately, it can allow ML.NET to solve a "simpler" problem even in the cases
    /// where the trainer has a multiclass option, but using it directly is not
    /// practical due to, usually, memory constraints. For example, while a multiclass
    /// logistic regression is a more principled way to solve a multiclass problem, it
    /// requires that the trainer store a lot more intermediate state in the form of
    /// L-BFGS history for all classes *simultaneously*, rather than just one-by-one
    /// as would be needed for a one-versus-all classification model.
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="StandardTrainersCatalog.OneVersusAll{TModel}(MulticlassClassificationCatalog.MulticlassClassificationTrainers, ITrainerEstimator{BinaryPredictionTransformer{TModel}, TModel}, string, bool, IEstimator{ISingleFeaturePredictionTransformer{ICalibrator}}, int, bool)" />
    public sealed class OneVersusAllTrainer : MetaMulticlassTrainer<MulticlassPredictionTransformer<OneVersusAllModelParameters>, OneVersusAllModelParameters>
    {
        internal const string LoadNameValue = "OVA";
        internal const string UserNameValue = "One-vs-All";
        internal const string Summary = "In this strategy, a binary classification algorithm is used to train one classifier for each class, "
            + "which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers, "
            + "and choosing the prediction with the highest confidence score.";

        private readonly Options _options;

        /// <summary>
        /// Options passed to <see cref="OneVersusAllTrainer"/>
        /// </summary>
        internal sealed class Options : OptionsBase
        {
            /// <summary>
            /// Whether to use probabilities (vs. raw outputs) to identify top-score category.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use probability or margins to determine max", ShortName = "useprob")]
            [TGUI(Label = "Use Probability", Description = "Use probabilities (vs. raw outputs) to identify top-score category")]
            public bool UseProbabilities = true;
        }

        /// <summary>
        /// Constructs a <see cref="OneVersusAllTrainer"/> trainer supplying a <see cref="Options"/>.
        /// </summary>
        /// <param name="env">The private <see cref="IHostEnvironment"/> for this estimator.</param>
        /// <param name="options">The legacy <see cref="Options"/></param>
        internal OneVersusAllTrainer(IHostEnvironment env, Options options)
            : base(env, options, LoadNameValue)
        {
            _options = options;
        }

        /// <summary>
        /// Initializes a new instance of <see cref="OneVersusAllTrainer"/>.
        /// </summary>
        /// <param name="env">The <see cref="IHostEnvironment"/> instance.</param>
        /// <param name="binaryEstimator">An instance of a binary <see cref="ITrainerEstimator{TTransformer, TPredictor}"/> used as the base trainer.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not provided, it will default to <see cref="PlattCalibratorTrainer"/></param>
        /// <param name="labelColumnName">The name of the label colum.</param>
        /// <param name="imputeMissingLabelsAsNegative">If true will treat missing labels as negative labels.</param>
        /// <param name="maximumCalibrationExampleCount">Number of instances to train the calibrator.</param>
        /// <param name="useProbabilities">Use probabilities (vs. raw outputs) to identify top-score category.</param>
        internal OneVersusAllTrainer(IHostEnvironment env,
            TScalarTrainer binaryEstimator,
            string labelColumnName = DefaultColumnNames.Label,
            bool imputeMissingLabelsAsNegative = false,
            ICalibratorTrainer calibrator = null,
            int maximumCalibrationExampleCount = 1000000000,
            bool useProbabilities = true)
         : base(env,
               new Options
               {
                   ImputeMissingLabelsAsNegative = imputeMissingLabelsAsNegative,
                   MaxCalibrationExamples = maximumCalibrationExampleCount,
               },
               LoadNameValue, labelColumnName, binaryEstimator, calibrator)
        {
            Host.CheckValue(labelColumnName, nameof(labelColumnName), "Label column should not be null.");
            _options = (Options)Args;
            _options.UseProbabilities = useProbabilities;
        }

        private protected override OneVersusAllModelParameters TrainCore(IChannel ch, RoleMappedData data, int count)
        {
            // Train one-vs-all models.
            var predictors = new TScalarPredictor[count];
            for (int i = 0; i < predictors.Length; i++)
            {
                ch.Info($"Training learner {i}");
                predictors[i] = TrainOne(ch, Trainer, data, i).Model;
            }
            return OneVersusAllModelParameters.Create(Host, _options.UseProbabilities, predictors);
        }

        private ISingleFeaturePredictionTransformer<TScalarPredictor> TrainOne(IChannel ch, TScalarTrainer trainer, RoleMappedData data, int cls)
        {
            var view = MapLabels(data, cls);

            string trainerLabel = data.Schema.Label.Value.Name;

            // REVIEW: In principle we could support validation sets and the like via the train context, but
            // this is currently unsupported.
            var transformer = trainer.Fit(view);

            if (_options.UseProbabilities)
            {
                var calibratedModel = transformer.Model as TDistPredictor;

                // REVIEW: restoring the RoleMappedData, as much as we can.
                // not having the weight column on the data passed to the TrainCalibrator should be addressed.
                var trainedData = new RoleMappedData(view, label: trainerLabel, feature: transformer.FeatureColumnName);

                if (calibratedModel == null)
                    calibratedModel = CalibratorUtils.GetCalibratedPredictor(Host, ch, Calibrator, transformer.Model, trainedData, Args.MaxCalibrationExamples) as TDistPredictor;

                Host.Check(calibratedModel != null, "Calibrated predictor does not implement the expected interface");
                return new BinaryPredictionTransformer<TScalarPredictor>(Host, calibratedModel, trainedData.Data.Schema, transformer.FeatureColumnName);
            }

            return new BinaryPredictionTransformer<TScalarPredictor>(Host, transformer.Model, view.Schema, transformer.FeatureColumnName);
        }

        private IDataView MapLabels(RoleMappedData data, int cls)
        {
            var label = data.Schema.Label.Value;
            Host.Assert(!label.IsHidden);
            Host.Assert(label.Type.GetKeyCount() > 0 || label.Type == NumberDataViewType.Single || label.Type == NumberDataViewType.Double);

            if (label.Type.GetKeyCount() > 0)
            {
                // Key values are 1-based.
                uint key = (uint)(cls + 1);
                return MapLabelsCore(NumberDataViewType.UInt32, (in uint val) => key == val, data);
            }

            throw Host.ExceptNotSupp($"Label column type is not supported by OneVersusAllTrainer: {label.Type.RawType}");
        }

        /// <summary> Trains a <see cref="MulticlassPredictionTransformer{OneVersusAllModelParameters}"/> model.</summary>
        /// <param name="input">The input data.</param>
        /// <returns>A <see cref="MulticlassPredictionTransformer{OneVersusAllModelParameters}"/> model./></returns>
        public override MulticlassPredictionTransformer<OneVersusAllModelParameters> Fit(IDataView input)
        {
            var roles = new KeyValuePair<CR, string>[1];
            roles[0] = new KeyValuePair<CR, string>(new CR(DefaultColumnNames.Label), LabelColumn.Name);
            var td = new RoleMappedData(input, roles);

            td.CheckMulticlassLabel(out var numClasses);

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
                        featureColumn = transformer.FeatureColumnName;
                    }

                    predictors[i] = TrainOne(ch, Trainer, td, i).Model;
                }
            }

            return new MulticlassPredictionTransformer<OneVersusAllModelParameters>(Host, OneVersusAllModelParameters.Create(Host, _options.UseProbabilities, predictors), input.Schema, featureColumn, LabelColumn.Name);
        }
    }

    /// <summary>
    /// Model parameters for <see cref="OneVersusAllTrainer"/>.
    /// </summary>
    public sealed class OneVersusAllModelParameters :
        ModelParametersBase<VBuffer<float>>,
        IValueMapper,
        ICanSaveInSourceCode,
        ICanSaveInTextFormat,
        ISingleCanSavePfa,
        ISingleCanSaveOnnx
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
                loaderAssemblyName: typeof(OneVersusAllModelParameters).Assembly.FullName);
        }

        private const string SubPredictorFmt = "SubPredictor_{0:000}";

        private readonly ImplBase _impl;

        /// <summary>
        /// Retrieves the model parameters.
        /// </summary>
        internal ImmutableArray<object> SubModelParameters => _impl.Predictors.Cast<object>().ToImmutableArray();

        /// <summary>
        /// The type of the prediction task.
        /// </summary>
        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

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
        [BestFriend]
        internal enum OutputFormula { Raw = 0, ProbabilityNormalization = 1, Softmax = 2 };

        private DataViewType DistType { get; }

        bool ICanSavePfa.CanSavePfa => _impl.CanSavePfa;

        [BestFriend]
        internal static OneVersusAllModelParameters Create(IHost host, OutputFormula outputFormula, TScalarPredictor[] predictors)
        {
            ImplBase impl;

            using (var ch = host.Start("Creating OVA predictor"))
            {
                if (outputFormula == OutputFormula.Softmax)
                {
                    impl = new ImplSoftmax(predictors);
                    return new OneVersusAllModelParameters(host, impl);
                }

                // Caller of this function asks for probability output. We check if input predictor can produce probability.
                // If that predictor can't produce probability, ivmd will be null.
                IValueMapperDist ivmd = null;
                if (outputFormula == OutputFormula.ProbabilityNormalization &&
                    ((ivmd = predictors[0] as IValueMapperDist) == null ||
                        ivmd.OutputType != NumberDataViewType.Single ||
                        ivmd.DistType != NumberDataViewType.Single))
                {
                    ch.Warning($"{nameof(OneVersusAllTrainer.Options.UseProbabilities)} specified with {nameof(OneVersusAllTrainer.Options.PredictorType)} that can't produce probabilities.");
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

            return new OneVersusAllModelParameters(host, impl);
        }

        [BestFriend]
        internal static OneVersusAllModelParameters Create(IHost host, bool useProbability, TScalarPredictor[] predictors)
        {
            var outputFormula = useProbability ? OutputFormula.ProbabilityNormalization : OutputFormula.Raw;

            return Create(host, outputFormula, predictors);
        }

        /// <summary>
        /// Create a <see cref="OneVersusAllModelParameters"/> from an array of predictors.
        /// </summary>
        [BestFriend]
        internal static OneVersusAllModelParameters Create(IHost host, TScalarPredictor[] predictors)
        {
            Contracts.CheckValue(host, nameof(host));
            host.CheckNonEmpty(predictors, nameof(predictors));
            return Create(host, OutputFormula.ProbabilityNormalization, predictors);
        }

        private OneVersusAllModelParameters(IHostEnvironment env, ImplBase impl)
                : base(env, RegistrationName)
        {
            Host.AssertValue(impl, nameof(impl));
            Host.Assert(Utils.Size(impl.Predictors) > 0);

            _impl = impl;
            DistType = new VectorDataViewType(NumberDataViewType.Single, _impl.Predictors.Length);
        }

        private OneVersusAllModelParameters(IHostEnvironment env, ModelLoadContext ctx)
                : base(env, RegistrationName, ctx)
        {
            // *** Binary format ***
            // byte: OutputFormula as byte
            // int: predictor count
            OutputFormula outputFormula = (OutputFormula)ctx.Reader.ReadByte();
            int len = ctx.Reader.ReadInt32();
            Host.CheckDecode(len > 0);

            if (outputFormula == OutputFormula.Raw)
            {
                var predictors = new TScalarPredictor[len];
                LoadPredictors(Host, predictors, ctx);
                _impl = new ImplRaw(predictors);
            }
            else if (outputFormula == OutputFormula.ProbabilityNormalization)
            {
                var predictors = new IValueMapperDist[len];
                LoadPredictors(Host, predictors, ctx);
                _impl = new ImplDist(predictors);
            }
            else if (outputFormula == OutputFormula.Softmax)
            {
                var predictors = new TScalarPredictor[len];
                LoadPredictors(Host, predictors, ctx);
                _impl = new ImplSoftmax(predictors);
            }

            DistType = new VectorDataViewType(NumberDataViewType.Single, _impl.Predictors.Length);
        }

        internal static OneVersusAllModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new OneVersusAllModelParameters(env, ctx);
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
            // byte: _impl.OutputFormula as byte
            // int: predictor count
            byte[] outputFormula = { (byte)_impl.OutputFormula };
            ctx.Writer.WriteBytesNoCount(outputFormula, 1);
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

        DataViewType IValueMapper.InputType
        {
            get { return _impl.InputType; }
        }

        DataViewType IValueMapper.OutputType
        {
            get { return DistType; }
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

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => _impl.CanSaveOnnx(ctx);

        bool ISingleCanSaveOnnx.SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumn) => _impl.SaveAsOnnx(ctx, outputNames, featureColumn);

        private abstract class ImplBase : ISingleCanSavePfa, ISingleCanSaveOnnx
        {
            public OutputFormula OutputFormula;
            public abstract DataViewType InputType { get; }
            public abstract IValueMapper[] Predictors { get; }
            public abstract bool CanSavePfa { get; }
            public abstract ValueMapper<VBuffer<float>, VBuffer<float>> GetMapper();
            public abstract JToken SaveAsPfa(BoundPfaContext ctx, JToken input);

            public bool CanSaveOnnx(OnnxContext ctx) => Predictors.All(pred => (pred as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true);

            public abstract bool SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumn);

            protected bool IsValid(IValueMapper mapper, ref VectorDataViewType inputType)
            {
                Contracts.AssertValueOrNull(mapper);
                Contracts.AssertValueOrNull(inputType);

                if (mapper == null)
                    return false;
                if (mapper.OutputType != NumberDataViewType.Single)
                    return false;
                if (!(mapper.InputType is VectorDataViewType mapperVectorType) || mapperVectorType.ItemType != NumberDataViewType.Single)
                    return false;
                if (inputType == null)
                    inputType = mapperVectorType;
                else if (inputType.Size != mapperVectorType.Size)
                {
                    if (inputType.Size == 0)
                        inputType = mapperVectorType;
                    else if (mapperVectorType.Size != 0)
                        return false;
                }
                return true;
            }

            public string[] SaveAsOnnxPreProcess(OnnxContext ctx, string featureColumn, bool clipToZero)
            {
                string[] outputs = new string[Predictors.Length];

                string[] localOutputNames = { DefaultColumnNames.PredictedLabel, DefaultColumnNames.Score, DefaultColumnNames.Probability };

                for (int i = 0; i < Predictors.Length; i++)
                {
                    var predictorOutputNames = new string[localOutputNames.Length];

                    predictorOutputNames[0] = ctx.AddIntermediateVariable(NumberDataViewType.UInt32, $"{DefaultColumnNames.PredictedLabel}_{i}", true);
                    predictorOutputNames[1] = ctx.AddIntermediateVariable(NumberDataViewType.Single, $"{DefaultColumnNames.Score}_{i}", true);
                    predictorOutputNames[2] = ctx.AddIntermediateVariable(NumberDataViewType.Single, $"{DefaultColumnNames.Probability}_{i}", true);

                    string clipInput = predictorOutputNames[2];

                    var pred = Predictors[i] as ISingleCanSaveOnnx;
                    Contracts.AssertValue(pred);
                    pred.SaveAsOnnx(ctx, predictorOutputNames, featureColumn);

                    if (clipToZero)
                    {
                        var clipOutput = ctx.AddIntermediateVariable(NumberDataViewType.Single, $"ClipOutput_{i}", true);
                        outputs[i] = clipOutput;

                        string opType = "Clip";
                        var zeroVar = ctx.AddInitializer(0.0f, "Zero");
                        var clipNode = ctx.CreateNode(opType, new[] { clipInput, zeroVar }, new[] { outputs[i] }, ctx.GetNodeName(opType), "");
                    }
                    else
                        outputs[i] = predictorOutputNames[1];
                }
                return outputs;
            }

            public void SaveAsOnnxPostProcess(OnnxContext ctx, string inputName, string[] outputNames)
            {
                Contracts.Assert(outputNames.Length >= 2);

                string opType;
                opType = "ArgMax";
                var argMaxOutput = ctx.AddIntermediateVariable(NumberDataViewType.Int64, "ArgMaxOutput");
                var argMaxNode = ctx.CreateNode(opType, inputName, argMaxOutput, ctx.GetNodeName(opType), "");
                argMaxNode.AddAttribute("keepdims", 1);
                argMaxNode.AddAttribute("axis", 1);

                opType = "Add";
                var one = ctx.AddInitializer(1);
                var addOutput = ctx.AddIntermediateVariable(NumberDataViewType.Int64, "AddOutput");
                var addNode = ctx.CreateNode(opType, new[] { argMaxOutput, one }, new[] { addOutput }, ctx.GetNodeName(opType), "");

                opType = "Cast";
                var castToUint32Node = ctx.CreateNode(opType, addOutput, outputNames[0], ctx.GetNodeName(opType), "");
                var t2 = InternalDataKindExtensions.ToInternalDataKind(DataKind.UInt32).ToType();
                castToUint32Node.AddAttribute("to", t2);

                opType = "Max";
                ctx.CreateNode(opType, inputName, outputNames[1], ctx.GetNodeName(opType), "");
            }
        }

        private sealed class ImplRaw : ImplBase
        {
            public override DataViewType InputType { get; }
            public override IValueMapper[] Predictors { get; }
            public override bool CanSavePfa { get; }

            internal ImplRaw(TScalarPredictor[] predictors)
            {
                Contracts.CheckNonEmpty(predictors, nameof(predictors));

                Predictors = new IValueMapper[predictors.Length];
                VectorDataViewType inputType = null;
                for (int i = 0; i < predictors.Length; i++)
                {
                    var vm = predictors[i] as IValueMapper;
                    Contracts.Check(IsValid(vm, ref inputType), "Predictor doesn't implement the expected interface");
                    Predictors[i] = vm;
                }
                CanSavePfa = Predictors.All(m => (m as ISingleCanSavePfa)?.CanSavePfa == true);
                Contracts.AssertValue(inputType);
                InputType = inputType;
                OutputFormula = OutputFormula.Raw;
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
                        int inputSize = InputType.GetVectorSize();
                        if (inputSize > 0)
                            Contracts.Check(src.Length == inputSize);

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

            public override bool SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumn)
            {
                const int minimumOpSetVersion = 9;
                ctx.CheckOpSetVersion(minimumOpSetVersion, LoaderSignature);
                var probabilityOutputs = base.SaveAsOnnxPreProcess(ctx, featureColumn, false);

                string opType = "Concat";
                var type = new VectorDataViewType(NumberDataViewType.Single, probabilityOutputs.Length);
                var concatOutput = ctx.AddIntermediateVariable(type, "ConcatOutputRaw");
                var concatNode = ctx.CreateNode(opType, probabilityOutputs, new[] { concatOutput }, ctx.GetNodeName(opType), "");
                concatNode.AddAttribute("axis", 1);

                base.SaveAsOnnxPostProcess(ctx, concatOutput, outputNames);

                return true;

            }
        }

        private sealed class ImplDist : ImplBase
        {
            private readonly IValueMapperDist[] _mappers;
            public override DataViewType InputType { get; }
            public override IValueMapper[] Predictors => _mappers;
            public override bool CanSavePfa { get; }

            internal ImplDist(IValueMapperDist[] predictors)
            {
                Contracts.Check(Utils.Size(predictors) > 0);

                _mappers = new IValueMapperDist[predictors.Length];
                VectorDataViewType inputType = null;
                for (int i = 0; i < predictors.Length; i++)
                {
                    var vm = predictors[i];
                    Contracts.Check(IsValid(vm, ref inputType), "Predictor doesn't implement the expected interface");
                    _mappers[i] = vm;
                }
                CanSavePfa = Predictors.All(m => (m as IDistCanSavePfa)?.CanSavePfa == true);
                Contracts.AssertValue(inputType);
                InputType = inputType;
                OutputFormula = OutputFormula.ProbabilityNormalization;
            }

            private bool IsValid(IValueMapperDist mapper, ref VectorDataViewType inputType)
            {
                return base.IsValid(mapper, ref inputType) && mapper.DistType == NumberDataViewType.Single;
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
                        int inputSize = InputType.GetVectorSize();
                        if (inputSize > 0)
                            Contracts.Check(src.Length == inputSize);

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
                    if (float.IsNaN(value))
                        continue;

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

            public override bool SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumn)
            {
                Contracts.Assert(outputNames.Length >= 2);
                const int minimumOpSetVersion = 9;
                ctx.CheckOpSetVersion(minimumOpSetVersion, LoaderSignature);

                string opType;
                var probabilityOutputs = base.SaveAsOnnxPreProcess(ctx, featureColumn, true);

                opType = "Sum";
                var sumOutput = ctx.AddIntermediateVariable(NumberDataViewType.Single, "SumOfScores");
                ctx.CreateNode(opType, probabilityOutputs, new[] { sumOutput }, ctx.GetNodeName(opType), "");

                opType = "Cast";
                var castOutput = ctx.AddIntermediateVariable(BooleanDataViewType.Instance, "CastOutput");
                var castNode = ctx.CreateNode(opType, sumOutput, castOutput, ctx.GetNodeName(opType), "");
                var t = InternalDataKindExtensions.ToInternalDataKind(DataKind.Boolean).ToType();
                castNode.AddAttribute("to", t);

                opType = "Not";
                var notOutput = ctx.AddIntermediateVariable(BooleanDataViewType.Instance, "IsSumZero");
                ctx.CreateNode(opType, castOutput, notOutput, ctx.GetNodeName(opType), "");

                opType = "Cast";
                var castIsZeroSumToFloat = ctx.AddIntermediateVariable(NumberDataViewType.Single, "IsSumZeroAsFloat");
                var castIsZeroSumToFloatNode = ctx.CreateNode(opType, notOutput, castIsZeroSumToFloat, ctx.GetNodeName(opType), "");
                var t1 = InternalDataKindExtensions.ToInternalDataKind(DataKind.Single).ToType();
                castIsZeroSumToFloatNode.AddAttribute("to", t1);

                opType = "Sum";
                var sumOutputNonZero = ctx.AddIntermediateVariable(NumberDataViewType.Single, "SumOfScoresNonZero");
                ctx.CreateNode(opType, new[] { sumOutput, castIsZeroSumToFloat },
                    new[] { sumOutputNonZero }, ctx.GetNodeName(opType), "");

                string[] divOutputs = new string[Predictors.Length];
                for (int i = 0; i < Predictors.Length; i++)
                {
                    opType = "Div";
                    divOutputs[i] = ctx.AddIntermediateVariable(NumberDataViewType.Single, $"DivOutput_{i}");
                    ctx.CreateNode(opType, new[] { probabilityOutputs[i], sumOutputNonZero }, new[] { divOutputs[i] }, ctx.GetNodeName(opType), "");
                }

                opType = "Concat";
                var type = new VectorDataViewType(NumberDataViewType.Single, divOutputs.Length);
                var concatOutput = ctx.AddIntermediateVariable(type, "ConcatOutputDist");
                var concatNode = ctx.CreateNode(opType, divOutputs, new[] { concatOutput }, ctx.GetNodeName(opType), "");
                concatNode.AddAttribute("axis", 1);

                base.SaveAsOnnxPostProcess(ctx, concatOutput, outputNames);

                return true;
            }
        }

        private sealed class ImplSoftmax : ImplBase
        {
            public override DataViewType InputType { get; }
            public override IValueMapper[] Predictors { get; }
            public override bool CanSavePfa { get; }

            internal ImplSoftmax(TScalarPredictor[] predictors)
            {
                Contracts.CheckNonEmpty(predictors, nameof(predictors));

                Predictors = new IValueMapper[predictors.Length];
                VectorDataViewType inputType = null;
                for (int i = 0; i < predictors.Length; i++)
                {
                    var vm = predictors[i] as IValueMapper;
                    Contracts.Check(IsValid(vm, ref inputType), "Predictor doesn't implement the expected interface");
                    Predictors[i] = vm;
                }
                CanSavePfa = false;
                Contracts.AssertValue(inputType);
                InputType = inputType;
                OutputFormula = OutputFormula.Softmax;
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
                        int inputSize = InputType.GetVectorSize();
                        if (inputSize > 0)
                            Contracts.Check(src.Length == inputSize);

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
                double sum = 0;
                var score = new double[count];

                for (int i = 0; i < count; i++)
                {
                    score[i] = Math.Exp(scores[i]);
                    sum += score[i];
                }

                for (int i = 0; i < count; i++)
                    scores[i] = (float)(score[i] / sum);
            }

            public override JToken SaveAsPfa(BoundPfaContext ctx, JToken input)
            {
                throw new NotImplementedException("Softmax's PFA exporter is not implemented yet.");
            }

            public override bool SaveAsOnnx(OnnxContext ctx, string[] outputNames, string featureColumn)
            {
                Contracts.Assert(outputNames.Length >= 2);
                const int minimumOpSetVersion = 9;
                ctx.CheckOpSetVersion(minimumOpSetVersion, LoaderSignature);

                var probabilityOutputs = base.SaveAsOnnxPreProcess(ctx, featureColumn, false);

                string opType;
                opType = "Concat";
                var type = new VectorDataViewType(NumberDataViewType.Single, probabilityOutputs.Length);
                var concatOutput = ctx.AddIntermediateVariable(type, "ConcatOutputSoftMax");
                var concatNode = ctx.CreateNode(opType, probabilityOutputs, new[] { concatOutput }, ctx.GetNodeName(opType), "");
                concatNode.AddAttribute("axis", 1);

                opType = "Exp";
                var expOutput = ctx.AddIntermediateVariable(type, "ExpOutput");
                var expNode = ctx.CreateNode(opType, concatOutput, expOutput, ctx.GetNodeName(opType), "");

                opType = "ReduceSum";
                var sumOutput = ctx.AddIntermediateVariable(NumberDataViewType.Single, "SumOutput");
                var sumNode = ctx.CreateNode(opType, expOutput, sumOutput, ctx.GetNodeName(opType), "");
                sumNode.AddAttribute("keepdims", 1);
                long[] list = { 1 };
                sumNode.AddAttribute("axes", list);

                opType = "Div";
                var divOutput = ctx.AddIntermediateVariable(type, "DivOutput");
                var divNode = ctx.CreateNode(opType, new[] { expOutput, sumOutput }, new[] { divOutput }, ctx.GetNodeName(opType), "");

                base.SaveAsOnnxPostProcess(ctx, divOutput, outputNames);

                return true;
            }
        }
    }
}
