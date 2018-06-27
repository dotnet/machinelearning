// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

// These are for deserialization from a model repository.
[assembly: LoadableClass(typeof(EnsembleDistributionPredictor), null, typeof(SignatureLoadModel),
    EnsembleDistributionPredictor.UserName, EnsembleDistributionPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.Ensemble
{
    using TDistPredictor = IDistPredictorProducing<Single, Single>;

    public sealed class EnsembleDistributionPredictor : EnsemblePredictorBase<TDistPredictor, Single>,
         TDistPredictor, IValueMapperDist
    {
        public const string UserName = "Ensemble Distribution Executor";
        public const string LoaderSignature = "EnsemDbExec";
        public const string RegistrationName = "EnsembleDistributionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ENSEM DB",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Metrics and subset info into main stream, after each predictor
                verWrittenCur: 0x00010003, // Don't serialize the "IsAveraged" property of the metrics
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature);
        }

        private readonly Single[] _averagedWeights;
        private readonly Median _probabilityCombiner;
        private readonly IValueMapperDist[] _mappers;

        public ColumnType InputType { get; }
        public ColumnType OutputType => NumberType.Float;
        public ColumnType DistType => NumberType.Float;

        public override PredictionKind PredictionKind { get; }

        internal EnsembleDistributionPredictor(IHostEnvironment env, PredictionKind kind,
            FeatureSubsetModel<TDistPredictor>[] models, IOutputCombiner<Single> combiner, Single[] weights = null)
            : base(env, RegistrationName, models, combiner, weights)
        {
            PredictionKind = kind;
            _probabilityCombiner = new Median(env);
            InputType = InitializeMappers(out _mappers);
            ComputeAveragedWeights(out _averagedWeights);
        }

        private EnsembleDistributionPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            PredictionKind = (PredictionKind)ctx.Reader.ReadInt32();
            _probabilityCombiner = new Median(env);
            InputType = InitializeMappers(out _mappers);
            ComputeAveragedWeights(out _averagedWeights);
        }

        private ColumnType InitializeMappers(out IValueMapperDist[] mappers)
        {
            Host.AssertNonEmpty(Models);

            mappers = new IValueMapperDist[Models.Length];
            ColumnType inputType = null;
            for (int i = 0; i < Models.Length; i++)
            {
                var vmd = Models[i].Predictor as IValueMapperDist;
                if (!IsValid(vmd))
                    throw Host.Except("Predictor does not implement expected interface");
                if (vmd.InputType.VectorSize > 0)
                {
                    if (inputType == null)
                        inputType = vmd.InputType;
                    else if (vmd.InputType.VectorSize != inputType.VectorSize)
                        throw Host.Except("Predictor input type mismatch");
                }
                mappers[i] = vmd;
            }
            return inputType ?? new VectorType(NumberType.Float);
        }

        private bool IsValid(IValueMapperDist mapper)
        {
            return mapper != null
                && mapper.InputType.IsVector && mapper.InputType.ItemType == NumberType.Float
                && mapper.OutputType == NumberType.Float
                && mapper.DistType == NumberType.Float;
        }

        public static EnsembleDistributionPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new EnsembleDistributionPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: PredictionKind
            ctx.Writer.Write((int)PredictionKind);
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<Single>));
            Host.Check(typeof(TOut) == typeof(Single));

            var combine = Combiner.GetCombiner();
            var maps = GetMaps();
            var predictions = new Single[_mappers.Length];
            var probabilities = new Single[_mappers.Length];
            var vBuffers = new VBuffer<Single>[_mappers.Length];
            ValueMapper<VBuffer<Single>, Single> del =
                (ref VBuffer<Single> src, ref Single dst) =>
                {
                    if (InputType.VectorSize > 0)
                        Host.Check(src.Length == InputType.VectorSize);

                    var tmp = src;
                    Parallel.For(0, maps.Length, i =>
                    {
                        var model = Models[i];
                        if (model.SelectedFeatures != null)
                        {
                            EnsembleUtils.SelectFeatures(ref tmp, model.SelectedFeatures, model.Cardinality, ref vBuffers[i]);
                            maps[i](ref vBuffers[i], ref predictions[i], ref probabilities[i]);
                        }
                        else
                            maps[i](ref tmp, ref predictions[i], ref probabilities[i]);
                    });

                    // REVIEW: DistributionEnsemble - AveragedWeights are used only in one of the two PredictDistributions overloads
                    combine(ref dst, predictions, Weights);
                };

            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        public ValueMapper<TIn, TOut, TDist> GetMapper<TIn, TOut, TDist>()
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
                (ref VBuffer<Single> src, ref Single score, ref Single prob) =>
                {
                    if (InputType.VectorSize > 0)
                        Host.Check(src.Length == InputType.VectorSize);

                    var tmp = src;
                    Parallel.For(0, maps.Length, i =>
                    {
                        var model = Models[i];
                        if (model.SelectedFeatures != null)
                        {
                            EnsembleUtils.SelectFeatures(ref tmp, model.SelectedFeatures, model.Cardinality, ref vBuffers[i]);
                            maps[i](ref vBuffers[i], ref predictions[i], ref probabilities[i]);
                        }
                        else
                            maps[i](ref tmp, ref predictions[i], ref probabilities[i]);
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
