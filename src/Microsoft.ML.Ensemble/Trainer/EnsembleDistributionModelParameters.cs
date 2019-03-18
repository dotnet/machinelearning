// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

// These are for deserialization from a model repository.
[assembly: LoadableClass(typeof(EnsembleDistributionModelParameters), null, typeof(SignatureLoadModel),
    EnsembleDistributionModelParameters.UserName, EnsembleDistributionModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.Ensemble
{
    using TDistPredictor = IDistPredictorProducing<Single, Single>;

    internal sealed class EnsembleDistributionModelParameters : EnsembleModelParametersBase<Single>,
         TDistPredictor, IValueMapperDist
    {
        internal const string UserName = "Ensemble Distribution Executor";
        internal const string LoaderSignature = "EnsemDbExec";
        internal const string RegistrationName = "EnsembleDistributionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ENSEM DB",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Metrics and subset info into main stream, after each predictor
                verWrittenCur: 0x00010003, // Don't serialize the "IsAveraged" property of the metrics
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(EnsembleDistributionModelParameters).Assembly.FullName);
        }

        private readonly Single[] _averagedWeights;
        private readonly Median _probabilityCombiner;
        private readonly IValueMapperDist[] _mappers;

        private readonly VectorType _inputType;

        DataViewType IValueMapper.InputType => _inputType;
        DataViewType IValueMapper.OutputType => NumberDataViewType.Single;
        DataViewType IValueMapperDist.DistType => NumberDataViewType.Single;

        private protected override PredictionKind PredictionKind { get; }

        /// <summary>
        /// Instantiate new ensemble model from existing sub-models.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="kind">The prediction kind <see cref="PredictionKind"/></param>
        /// <param name="models">Array of sub-models that you want to ensemble together.</param>
        /// <param name="combiner">The combiner class to use to ensemble the models.</param>
        /// <param name="weights">The weights assigned to each model to be ensembled.</param>
        internal EnsembleDistributionModelParameters(IHostEnvironment env, PredictionKind kind,
            FeatureSubsetModel<float>[] models, IOutputCombiner<Single> combiner, Single[] weights = null)
            : base(env, RegistrationName, models, combiner, weights)
        {
            PredictionKind = kind;
            _probabilityCombiner = new Median(env);
            _inputType = InitializeMappers(out _mappers);
            ComputeAveragedWeights(out _averagedWeights);
        }

        private EnsembleDistributionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            PredictionKind = (PredictionKind)ctx.Reader.ReadInt32();
            _probabilityCombiner = new Median(env);
            _inputType = InitializeMappers(out _mappers);
            ComputeAveragedWeights(out _averagedWeights);
        }

        private VectorType InitializeMappers(out IValueMapperDist[] mappers)
        {
            Host.AssertNonEmpty(Models);

            mappers = new IValueMapperDist[Models.Length];
            VectorType inputType = null;
            for (int i = 0; i < Models.Length; i++)
            {
                var vmd = Models[i].Predictor as IValueMapperDist;
                if (!IsValid(vmd, out VectorType vmdInputType))
                    throw Host.Except("Predictor does not implement expected interface");
                if (vmdInputType.Size > 0)
                {
                    if (inputType == null)
                        inputType = vmdInputType;
                    else if (vmdInputType.Size != inputType.Size)
                        throw Host.Except("Predictor input type mismatch");
                }
                mappers[i] = vmd;
            }
            return inputType ?? new VectorType(NumberDataViewType.Single);
        }

        private bool IsValid(IValueMapperDist mapper, out VectorType inputType)
        {
            if (mapper != null
                && mapper.InputType is VectorType inVectorType && inVectorType.ItemType == NumberDataViewType.Single
                && mapper.OutputType == NumberDataViewType.Single
                && mapper.DistType == NumberDataViewType.Single)
            {
                inputType = inVectorType;
                return true;
            }
            else
            {
                inputType = null;
                return false;
            }
        }

        private static EnsembleDistributionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new EnsembleDistributionModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: PredictionKind
            ctx.Writer.Write((int)PredictionKind);
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<Single>));
            Host.Check(typeof(TOut) == typeof(Single));

            var combine = Combiner.GetCombiner();
            var maps = GetMaps();
            var predictions = new Single[_mappers.Length];
            var probabilities = new Single[_mappers.Length];
            var vBuffers = new VBuffer<Single>[_mappers.Length];
            ValueMapper<VBuffer<Single>, Single> del =
                (in VBuffer<Single> src, ref Single dst) =>
                {
                    if (_inputType.Size > 0)
                        Host.Check(src.Length == _inputType.Size);

                    var tmp = src;
                    Parallel.For(0, maps.Length, i =>
                    {
                        var model = Models[i];
                        if (model.SelectedFeatures != null)
                        {
                            EnsembleUtils.SelectFeatures(in tmp, model.SelectedFeatures, model.Cardinality, ref vBuffers[i]);
                            maps[i](in vBuffers[i], ref predictions[i], ref probabilities[i]);
                        }
                        else
                            maps[i](in tmp, ref predictions[i], ref probabilities[i]);
                    });

                    // REVIEW: DistributionEnsemble - AveragedWeights are used only in one of the two PredictDistributions overloads
                    combine(ref dst, predictions, Weights);
                };

            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        ValueMapper<TIn, TOut, TDist> IValueMapperDist.GetMapper<TIn, TOut, TDist>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<Single>));
            Host.Check(typeof(TOut) == typeof(Single));
            Host.Check(typeof(TDist) == typeof(Single));

            var combine = Combiner.GetCombiner();
            var combineProb = _probabilityCombiner.GetCombiner();
            var maps = GetMaps();
            var predictions = new Single[_mappers.Length];
            var probabilities = new Single[_mappers.Length];
            var vBuffers = new VBuffer<Single>[_mappers.Length];
            ValueMapper<VBuffer<Single>, Single, Single> del =
                (in VBuffer<Single> src, ref Single score, ref Single prob) =>
                {
                    if (_inputType.Size > 0)
                        Host.Check(src.Length == _inputType.Size);

                    var tmp = src;
                    Parallel.For(0, maps.Length, i =>
                    {
                        var model = Models[i];
                        if (model.SelectedFeatures != null)
                        {
                            EnsembleUtils.SelectFeatures(in tmp, model.SelectedFeatures, model.Cardinality, ref vBuffers[i]);
                            maps[i](in vBuffers[i], ref predictions[i], ref probabilities[i]);
                        }
                        else
                            maps[i](in tmp, ref predictions[i], ref probabilities[i]);
                    });

                    combine(ref score, predictions, _averagedWeights);
                    combineProb(ref prob, probabilities, _averagedWeights);
                };

            return (ValueMapper<TIn, TOut, TDist>)(Delegate)del;
        }

        private ValueMapper<VBuffer<Single>, Single, Single>[] GetMaps()
        {
            Host.AssertValue(_mappers);

            var maps = new ValueMapper<VBuffer<Single>, Single, Single>[_mappers.Length];
            for (int i = 0; i < _mappers.Length; i++)
                maps[i] = _mappers[i].GetMapper<VBuffer<Single>, Single, Single>();
            return maps;
        }

        private void ComputeAveragedWeights(out Single[] averagedWeights)
        {
            averagedWeights = Weights;
            if (Combiner is IWeightedAverager weightedAverager && averagedWeights == null && Models[0].Metrics != null)
            {
                var metric = default(KeyValuePair<string, double>);
                bool found = false;
                foreach (var m in Models[0].Metrics)
                {
                    metric = m;
                    if (Utils.ExtractLettersAndNumbers(m.Key).ToLower().Equals(weightedAverager.WeightageMetricName.ToLower()))
                    {
                        found = true;
                        break;
                    }
                }
                if (found)
                    averagedWeights = Models.SelectMany(model => model.Metrics).Where(m => m.Key == metric.Key).Select(m => (Single)m.Value).ToArray();
            }
        }
    }
}
