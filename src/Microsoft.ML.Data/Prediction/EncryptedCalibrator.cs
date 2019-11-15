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
using Microsoft.Research.SEAL;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.SEAL
{
    internal abstract class EncryptedValueMapperCalibratedModelParametersBase<TSubModel, TCalibrator, TESubModel> :
        CalibratedModelParametersBase<TSubModel, TCalibrator>,
        IValueMapperDist, IValueMapperTwoToOne, IFeatureContributionMapper, ICalculateFeatureContribution,
        IDistCanSavePfa, IDistCanSaveOnnx
        where TSubModel : class
        where TCalibrator : class, ICalibrator
    {
        private readonly IValueMapper _mapper;
        private readonly IValueMapperTwoToOne _mapperTwoToOne;
        private readonly IFeatureContributionMapper _featureContribution;

        DataViewType IValueMapper.InputType => _mapper.InputType;
        DataViewType IValueMapper.OutputType => _mapper.OutputType;
        DataViewType IValueMapperDist.DistType => NumberDataViewType.Single;
        DataViewType IValueMapperTwoToOne.InputType => _mapperTwoToOne.InputType;
        DataViewType IValueMapperTwoToOne.OutputType => _mapperTwoToOne.OutputType;
        bool ICanSavePfa.CanSavePfa => (_mapper as ICanSavePfa)?.CanSavePfa == true;

        FeatureContributionCalculator ICalculateFeatureContribution.FeatureContributionCalculator => new FeatureContributionCalculator(this);

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => (_mapper as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true;

        private protected EncryptedValueMapperCalibratedModelParametersBase(IHostEnvironment env, string name, TSubModel predictor, TCalibrator calibrator, TESubModel ePredictor)
            : base(env, name, predictor, calibrator)
        {
            Contracts.AssertValue(Host);

            _mapper = SubModel as IValueMapper;
            _mapperTwoToOne = ePredictor as IValueMapperTwoToOne;
            Host.Check(_mapper != null, "The predictor does not implement IValueMapper");
            Host.Check(_mapper.OutputType == NumberDataViewType.Single, "The output type of the predictor is expected to be float");

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

        ValueMapperTwoToOne<TSrc, TKey, TDst> IValueMapperTwoToOne.GetMapper<TSrc, TKey, TDst>()
        {
            Host.Check(typeof(TSrc) == typeof(Ciphertext[]));
            Host.Check(typeof(TKey) == typeof(GaloisKeys));
            Host.Check(typeof(TDst) == typeof(Ciphertext[]));
            return _mapperTwoToOne.GetMapper<TSrc, TKey, TDst>();
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

    /// <summary>
    /// Encapsulates a predictor and a calibrator that implement <see cref="IParameterMixer"/>.
    /// Its implementation of <see cref="IParameterMixer.CombineParameters"/> combines both the predictors and the calibrators.
    /// </summary>
    internal sealed class EncryptedParameterMixingCalibratedModelParameters<TSubModel, TCalibrator, TESubModel> :
        EncryptedValueMapperCalibratedModelParametersBase<TSubModel, TCalibrator, TESubModel>,
        IParameterMixer<float>,
        IPredictorWithFeatureWeights<float>,
        ICanSaveModel
        where TSubModel : class
        where TCalibrator : class, ICalibrator
        where TESubModel : class
    {
        private readonly IPredictorWithFeatureWeights<float> _featureWeights;

        internal EncryptedParameterMixingCalibratedModelParameters(IHostEnvironment env, TSubModel predictor, TCalibrator calibrator, TESubModel ePredictor)
            : base(env, RegistrationName, predictor, calibrator, ePredictor)
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

        private static TESubModel GetEPredictor(IHostEnvironment env, ModelLoadContext ctx)
        {
            TESubModel predictor;
            ctx.LoadModel<TESubModel, SignatureLoadModel>(env, out predictor, ModelFileUtils.DirPredictor);
            return predictor;
        }

        private EncryptedParameterMixingCalibratedModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, GetPredictor(env, ctx), GetCalibrator(env, ctx), GetEPredictor(env, ctx))
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