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
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.SEAL
{
    /*
    /// <summary>
    /// Class for allowing a post-processing step, defined by <see cref="Calibrator"/>, to <see cref="SubModel"/>'s
    /// output.
    /// </summary>
    /// <typeparam name="TSubModel">Type being calibrated.</typeparam>
    /// <typeparam name="TCalibrator">Type used to calibrate.</typeparam>
    /// <remarks>
    /// For example, in binary classification, <see cref="Calibrator"/> can convert support vector machine's
    /// output value to the probability of belonging to the positive (or negative) class. Detailed math materials
    /// can be found at <a href="https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf">this paper</a>.
    /// </remarks>
    public abstract class EncryptedCalibratedModelParametersBase<TSubModel, TCalibrator> : CalibratedModelParametersBase,
        IDistPredictorProducing<float, float>,
        ICanSaveInIniFormat,
        ICanSaveInTextFormat,
        ICanSaveInSourceCode,
        ICanSaveSummary,
        ICanGetSummaryInKeyValuePairs,
        IWeaklyTypedCalibratedModelParameters
        where TSubModel : class
        where TCalibrator : class, ICalibrator
    {
        private protected readonly IHost Host;

        // Strongly-typed members.
        /// <summary>
        /// <see cref="SubModel"/>'s output would calibrated by <see cref="Calibrator"/>.
        /// </summary>
        public new TSubModel SubModel { get; }

        /// <summary>
        /// <see cref="Calibrator"/> is used to post-process score produced by <see cref="SubModel"/>.
        /// </summary>
        public new TCalibrator Calibrator { get; }

        // Type-unsafed accessors of strongly-typed members.
        IPredictorProducing<float> IWeaklyTypedCalibratedModelParameters.WeaklyTypedSubModel => (IPredictorProducing<float>)SubModel;
        ICalibrator IWeaklyTypedCalibratedModelParameters.WeaklyTypedCalibrator => Calibrator;

        PredictionKind IPredictor.PredictionKind => ((IPredictorProducing<float>)SubModel).PredictionKind;

        private protected CalibratedModelParametersBase(IHostEnvironment env, string name, TSubModel predictor, TCalibrator calibrator)
            : base(predictor, calibrator)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonWhiteSpace(name, nameof(name));
            Host = env.Register(name);
            Host.CheckValue(predictor, nameof(predictor));
            Host.CheckValue(calibrator, nameof(calibrator));
            Host.Assert(predictor is IPredictorProducing<float>);

            SubModel = predictor;
            Calibrator = calibrator;
        }

        void ICanSaveInIniFormat.SaveAsIni(TextWriter writer, RoleMappedSchema schema, ICalibrator calibrator)
        {
            Host.Check(calibrator == null, "Too many calibrators.");
            var saver = SubModel as ICanSaveInIniFormat;
            saver?.SaveAsIni(writer, schema, Calibrator);
        }

        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubModel as ICanSaveInTextFormat;
            if (saver != null)
                saver.SaveAsText(writer, schema);
        }

        void ICanSaveInSourceCode.SaveAsCode(TextWriter writer, RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubModel as ICanSaveInSourceCode;
            if (saver != null)
                saver.SaveAsCode(writer, schema);
        }

        void ICanSaveSummary.SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubModel as ICanSaveSummary;
            if (saver != null)
                saver.SaveSummary(writer, schema);
        }

        ///<inheritdoc/>
        IList<KeyValuePair<string, object>> ICanGetSummaryInKeyValuePairs.GetSummaryInKeyValuePairs(RoleMappedSchema schema)
        {
            // REVIEW: What about the calibrator?
            var saver = SubModel as ICanGetSummaryInKeyValuePairs;
            if (saver != null)
                return saver.GetSummaryInKeyValuePairs(schema);

            return null;
        }

        private protected void SaveCore(ModelSaveContext ctx)
        {
            ctx.SaveModel(SubModel, ModelFileUtils.DirPredictor);
            ctx.SaveModel(Calibrator, @"Calibrator");
        }

        private protected static TSubModel GetPredictor(IHostEnvironment env, ModelLoadContext ctx)
        {
            TSubModel predictor;
            ctx.LoadModel<TSubModel, SignatureLoadModel>(env, out predictor, ModelFileUtils.DirPredictor);
            return predictor;
        }

        private protected static TCalibrator GetCalibrator(IHostEnvironment env, ModelLoadContext ctx)
        {
            TCalibrator calibrator;
            ctx.LoadModel<TCalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            return calibrator;
        }
    }
    */

    /// <summary>
    /// Encapsulates a predictor and a calibrator that implement <see cref="IParameterMixer"/>.
    /// Its implementation of <see cref="IParameterMixer.CombineParameters"/> combines both the predictors and the calibrators.
    /// </summary>
    internal sealed class EncryptedParameterMixingCalibratedModelParameters<TSubModel, TCalibrator, TESubModel> :
        ValueMapperCalibratedModelParametersBase<TSubModel, TCalibrator>,
        IParameterMixer<float>,
        IPredictorWithFeatureWeights<float>,
        ICanSaveModel
        where TSubModel : class
        where TCalibrator : class, ICalibrator
    {
        private readonly IPredictorWithFeatureWeights<float> _featureWeights;

        internal EncryptedParameterMixingCalibratedModelParameters(IHostEnvironment env, TSubModel predictor, TCalibrator calibrator, TESubModel ePredictor)
            : base(env, RegistrationName, predictor, calibrator)
        {
            Host.Check(predictor is IParameterMixer<float>, "Predictor does not implement " + nameof(IParameterMixer<float>));
            Host.Check(calibrator is IParameterMixer, "Calibrator does not implement " + nameof(IParameterMixer));
            Host.Assert(predictor is IPredictorWithFeatureWeights<float>);
            _featureWeights = predictor as IPredictorWithFeatureWeights<float>;
        }

        internal const string LoaderSignature = "EPMixCaliPredExec";
        internal const string RegistrationName = "EncryptedParameterMixingCalibratedPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EPMIXCAP",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(EncryptedParameterMixingCalibratedModelParameters<TSubModel, TCalibrator, TESubModel>).Assembly.FullName);
        }

        private EncryptedParameterMixingCalibratedModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, GetPredictor(env, ctx), GetCalibrator(env, ctx))
        {
            Host.Check(SubModel is IParameterMixer<float>, "Predictor does not implement " + nameof(IParameterMixer));
            Host.Check(SubModel is IPredictorWithFeatureWeights<float>, "Predictor does not implement " + nameof(IPredictorWithFeatureWeights<float>));
            _featureWeights = SubModel as IPredictorWithFeatureWeights<float>;
        }

        private static CalibratedModelParametersBase Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new EncryptedParameterMixingCalibratedModelParameters<TSubModel, TCalibrator, TESubModel>(env, ctx);
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
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
                    var model = m as ParameterMixingCalibratedModelParameters<TSubModel, TCalibrator>;
                    Contracts.Assert(model != null);
                    return (IParameterMixer<float>)model.SubModel;
                }).ToArray();
            var calibrators = models.Select(
                m =>
                {
                    var model = m as ParameterMixingCalibratedModelParameters<TSubModel, TCalibrator>;
                    Contracts.Assert(model != null);
                    return (IParameterMixer)model.Calibrator;
                }).ToArray();
            var combinedPredictor = predictors[0].CombineParameters(predictors);
            var combinedCalibrator = calibrators[0].CombineParameters(calibrators);
            return new ParameterMixingCalibratedModelParameters<TSubModel, TCalibrator>(Host, (TSubModel)combinedPredictor, (TCalibrator)combinedCalibrator);
        }
    }
}