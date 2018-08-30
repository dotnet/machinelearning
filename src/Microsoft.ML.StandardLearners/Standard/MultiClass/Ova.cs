// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(Ova.Summary, typeof(Ova), typeof(Ova.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    Ova.UserNameValue,
    Ova.LoadNameValue, DocName = "trainer/OvaPkpd.md")]

[assembly: LoadableClass(typeof(OvaPredictor), null, typeof(SignatureLoadModel),
    "OVA Executor",
    OvaPredictor.LoaderSignature)]

[assembly: EntryPointModule(typeof(OvaPredictor))]
namespace Microsoft.ML.Runtime.Learners
{
    using CR = RoleMappedSchema.ColumnRole;
    using TScalarPredictor = IPredictorProducing<Float>;
    using TScalarTrainer = ITrainer<IPredictorProducing<Float>>;

    /// <include file='doc.xml' path='doc/members/member[@name="OVA"]' />
    public sealed class Ova : MetaMulticlassTrainer<OvaPredictor, Ova.Arguments>
    {
        internal const string LoadNameValue = "OVA";
        internal const string UserNameValue = "One-vs-All";
        internal const string Summary = "In this strategy, a binary classification algorithm is used to train one classifier for each class, "
            + "which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers, "
            + "and choosing the prediction with the highest confidence score.";

        /// <summary>
        /// Arguments passed to OVA.
        /// </summary>
        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use probability or margins to determine max", ShortName = "useprob")]
            [TGUI(Label = "Use Probability", Description = "Use probabilities (vs. raw outputs) to identify top-score category")]
            public bool UseProbabilities = true;
        }

        public Ova(IHostEnvironment env, Arguments args)
            : base(env, args, LoadNameValue)
        {
        }

        protected override OvaPredictor TrainCore(IChannel ch, RoleMappedData data, int count)
        {
            // Train one-vs-all models.
            var predictors = new TScalarPredictor[count];
            for (int i = 0; i < predictors.Length; i++)
            {
                ch.Info($"Training learner {i}");
                predictors[i] = TrainOne(ch, GetTrainer(), data, i);
            }
            return OvaPredictor.Create(Host, Args.UseProbabilities, predictors);
        }

        private TScalarPredictor TrainOne(IChannel ch, TScalarTrainer trainer, RoleMappedData data, int cls)
        {
            string dstName;
            var view = MapLabels(data, cls, out dstName);

            var roles = data.Schema.GetColumnRoleNames()
                .Where(kvp => kvp.Key.Value != CR.Label.Value)
                .Prepend(CR.Label.Bind(dstName));
            var td = new RoleMappedData(view, roles);

            // REVIEW: In principle we could support validation sets and the like via the train context, but
            // this is currently unsupported.
            var predictor = trainer.Train(td);

            if (Args.UseProbabilities)
            {
                ICalibratorTrainer calibrator;
                if (Args.Calibrator == null)
                    calibrator = null;
                else
                    calibrator = Args.Calibrator.CreateComponent(Host);
                var res = CalibratorUtils.TrainCalibratorIfNeeded(Host, ch, calibrator, Args.MaxCalibrationExamples,
                    trainer, predictor, td);
                predictor = res as TScalarPredictor;
                Host.Check(predictor != null, "Calibrated predictor does not implement the expected interface");
            }
            return predictor;
        }

        private IDataView MapLabels(RoleMappedData data, int cls, out string dstName)
        {
            var lab = data.Schema.Label;
            Host.Assert(!data.Schema.Schema.IsHidden(lab.Index));
            Host.Assert(lab.Type.KeyCount > 0 || lab.Type == NumberType.R4 || lab.Type == NumberType.R8);

            // Get the destination label column name.
            dstName = data.Schema.Schema.GetTempColumnName();

            if (lab.Type.KeyCount > 0)
            {
                // Key values are 1-based.
                uint key = (uint)(cls + 1);
                return MapLabelsCore(NumberType.U4, (ref uint val) => key == val, data, dstName);
            }
            if (lab.Type == NumberType.R4)
            {
                Float key = cls;
                return MapLabelsCore(NumberType.R4, (ref float val) => key == val, data, dstName);
            }
            if (lab.Type == NumberType.R8)
            {
                Double key = cls;
                return MapLabelsCore(NumberType.R8, (ref double val) => key == val, data, dstName);
            }

            throw Host.ExceptNotSupp($"Label column type is not supported by OVA: {lab.Type}");
        }
    }

    public sealed class OvaPredictor :
        PredictorBase<VBuffer<Float>>,
        IValueMapper,
        ICanSaveModel,
        ICanSaveInSourceCode,
        ICanSaveInTextFormat,
        ISingleCanSavePfa
    {
        public const string LoaderSignature = "OVAExec";
        public const string RegistrationName = "OVAPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TLC OVA ",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string SubPredictorFmt = "SubPredictor_{0:000}";

        private readonly ImplBase _impl;

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;
        public ColumnType InputType => _impl.InputType;
        public ColumnType OutputType { get; }
        public ColumnType DistType => OutputType;
        public bool CanSavePfa => _impl.CanSavePfa;

        internal static OvaPredictor Create(IHost host, bool useProb, TScalarPredictor[] predictors)
        {
            ImplBase impl;

            using (var ch = host.Start("Creating OVA predictor"))
            {
                IValueMapperDist ivmd = null;
                if (useProb &&
                    ((ivmd = predictors[0] as IValueMapperDist) == null ||
                        ivmd.OutputType != NumberType.Float ||
                        ivmd.DistType != NumberType.Float))
                {
                    ch.Warning($"{nameof(Ova.Arguments.UseProbabilities)} specified with {nameof(Ova.Arguments.PredictorType)} that can't produce probabilities.");
                    ivmd = null;
                }

                if (ivmd != null)
                {
                    var dists = new IValueMapperDist[predictors.Length];
                    for (int i = 0; i < predictors.Length; ++i)
                        dists[i] = (IValueMapperDist)predictors[i];
                    impl = new ImplDist(dists);
                }
                else
                    impl = new ImplRaw(predictors);

                ch.Done();
            }

            return new OvaPredictor(host, impl);
        }

        [TlcModule.EntryPoint(Name = "Models.OvaModelCombiner", Desc = "Combines a sequence of PredictorModels into a single model")]
        public static ModelOperations.PredictorModelOutput CombineOvaModels(IHostEnvironment env, ModelOperations.CombineOvaPredictorModelsInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineOvaModels");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            host.CheckNonEmpty(input.ModelArray, nameof(input.ModelArray));
            // Something tells me we should put normalization as part of macro expansion, but since i get
            // subgraph instead of learner it's a bit tricky to get learner and decide should we add
            // normalization node or not, plus everywhere in code we leave that reposnsibility to TransformModel.
            var normalizedView = input.ModelArray[0].TransformModel.Apply(host, input.TrainingData);
            using (var ch = host.Start("CombineOvaModels"))
            {
                ISchema schema = normalizedView.Schema;
                var label = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(input.LabelColumn),
                    input.LabelColumn,
                    DefaultColumnNames.Label);
                var feature = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(input.FeatureColumn),
                    input.FeatureColumn, DefaultColumnNames.Features);
                var weight = TrainUtils.MatchNameOrDefaultOrNull(ch, schema, nameof(input.WeightColumn),
                    input.WeightColumn, DefaultColumnNames.Weight);
                var data = new RoleMappedData(normalizedView, label, feature, null, weight);

                return new ModelOperations.PredictorModelOutput
                {
                    PredictorModel = new PredictorModel(env, data, input.TrainingData,
                    Create(host, input.UseProbabilities,
                            input.ModelArray.Select(p => p.Predictor as IPredictorProducing<float>).ToArray()))
                };
            }
        }

        /// <summary>
        /// Create a OVA predictor from an array of predictors.
        /// </summary>
        public static OvaPredictor Create(IHost host, TScalarPredictor[] predictors)
        {
            Contracts.CheckValue(host, nameof(host));
            host.CheckNonEmpty(predictors, nameof(predictors));
            return Create(host, true, predictors);
        }

        private OvaPredictor(IHostEnvironment env, ImplBase impl)
                : base(env, RegistrationName)
        {
            Host.AssertValue(impl, nameof(impl));
            Host.Assert(Utils.Size(impl.Predictors) > 0);

            _impl = impl;
            OutputType = new VectorType(NumberType.Float, _impl.Predictors.Length);
        }

        private OvaPredictor(IHostEnvironment env, ModelLoadContext ctx)
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

            OutputType = new VectorType(NumberType.Float, _impl.Predictors.Length);
        }

        public static OvaPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new OvaPredictor(env, ctx);
        }

        private static void LoadPredictors<TPredictor>(IHostEnvironment env, TPredictor[] predictors, ModelLoadContext ctx)
            where TPredictor : class
        {
            for (int i = 0; i < predictors.Length; i++)
                ctx.LoadModel<TPredictor, SignatureLoadModel>(env, out predictors[i], string.Format(SubPredictorFmt, i));
        }

        protected override void SaveCore(ModelSaveContext ctx)
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

        public JToken SaveAsPfa(BoundPfaContext ctx, JToken input)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.CheckValue(input, nameof(input));
            return _impl.SaveAsPfa(ctx, input);
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<Float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<Float>));

            return (ValueMapper<TIn, TOut>)(Delegate)_impl.GetMapper();
        }

        public void SaveAsCode(TextWriter writer, RoleMappedSchema schema)
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

        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
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
            public abstract ValueMapper<VBuffer<Float>, VBuffer<Float>> GetMapper();
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

            public override ValueMapper<VBuffer<Float>, VBuffer<Float>> GetMapper()
            {
                var maps = new ValueMapper<VBuffer<Float>, Float>[Predictors.Length];
                for (int i = 0; i < Predictors.Length; i++)
                    maps[i] = Predictors[i].GetMapper<VBuffer<Float>, Float>();

                return
                    (ref VBuffer<Float> src, ref VBuffer<Float> dst) =>
                    {
                        if (InputType.VectorSize > 0)
                            Contracts.Check(src.Length == InputType.VectorSize);

                        var values = dst.Values;
                        if (Utils.Size(values) < maps.Length)
                            values = new Float[maps.Length];

                        var tmp = src;
                        Parallel.For(0, maps.Length, i => maps[i](ref tmp, ref values[i]));
                        dst = new VBuffer<Float>(maps.Length, values, dst.Indices);
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

            public override ValueMapper<VBuffer<Float>, VBuffer<Float>> GetMapper()
            {
                var maps = new ValueMapper<VBuffer<Float>, Float, Float>[Predictors.Length];
                for (int i = 0; i < Predictors.Length; i++)
                    maps[i] = _mappers[i].GetMapper<VBuffer<Float>, Float, Float>();

                return
                    (ref VBuffer<Float> src, ref VBuffer<Float> dst) =>
                    {
                        if (InputType.VectorSize > 0)
                            Contracts.Check(src.Length == InputType.VectorSize);

                        var values = dst.Values;
                        if (Utils.Size(values) < maps.Length)
                            values = new Float[maps.Length];

                        var tmp = src;
                        Parallel.For(0, maps.Length,
                            i =>
                            {
                                Float score = 0;
                                maps[i](ref tmp, ref score, ref values[i]);
                            });
                        Normalize(values, maps.Length);
                        dst = new VBuffer<Float>(maps.Length, values, dst.Indices);
                    };
            }

            private void Normalize(Float[] output, int count)
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
                        output[i] = (Float)(output[i] / sum);
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
    }
}