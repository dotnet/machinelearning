// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(EnsemblePredictor), null, typeof(SignatureLoadModel),
    EnsemblePredictor.UserName, EnsemblePredictor.LoaderSignature)]
[assembly: EntryPointModule(typeof(EnsemblePredictor))]

namespace Microsoft.ML.Runtime.Ensemble
{
    using TScalarPredictor = IPredictorProducing<Single>;
    public sealed class EnsemblePredictor :
        EnsemblePredictorBase<TScalarPredictor, Single>,
        IValueMapper
    {
        public const string UserName = "Ensemble Executor";
        public const string LoaderSignature = "EnsembleFloatExec";
        public const string RegistrationName = "EnsemblePredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ENSEM XX",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Metrics and subset info into main stream, after each predictor
                verWrittenCur: 0x00010003, // Don't serialize the "IsAveraged" property of the metrics
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature);
        }

        private readonly IValueMapper[] _mappers;

        public ColumnType InputType { get; }
        public ColumnType OutputType { get { return NumberType.Float; } }
        public override PredictionKind PredictionKind { get; }

        internal EnsemblePredictor(IHostEnvironment env, PredictionKind kind,
            FeatureSubsetModel<TScalarPredictor>[] models, IOutputCombiner<Single> combiner, Single[] weights = null)
            : base(env, LoaderSignature, models, combiner, weights)
        {
            PredictionKind = kind;
            InputType = InitializeMappers(out _mappers);
        }

        private EnsemblePredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            PredictionKind = (PredictionKind)ctx.Reader.ReadInt32();
            InputType = InitializeMappers(out _mappers);
        }

        private ColumnType InitializeMappers(out IValueMapper[] mappers)
        {
            Host.AssertNonEmpty(Models);

            mappers = new IValueMapper[Models.Length];
            ColumnType inputType = null;
            for (int i = 0; i < Models.Length; i++)
            {
                var vm = Models[i].Predictor as IValueMapper;
                if (!IsValid(vm))
                    throw Host.Except("Predictor does not implement expected interface");
                if (vm.InputType.VectorSize > 0)
                {
                    if (inputType == null)
                        inputType = vm.InputType;
                    else if (vm.InputType.VectorSize != inputType.VectorSize)
                        throw Host.Except("Predictor input type mismatch");
                }
                mappers[i] = vm;
            }

            return inputType ?? new VectorType(NumberType.Float);
        }

        private bool IsValid(IValueMapper mapper)
        {
            return mapper != null
                && mapper.InputType.IsVector && mapper.InputType.ItemType == NumberType.Float
                && mapper.OutputType == NumberType.Float;
        }

        public static EnsemblePredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new EnsemblePredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: _kind
            ctx.Writer.Write((int)PredictionKind);
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<Single>));
            Host.Check(typeof(TOut) == typeof(Single));

            var combine = Combiner.GetCombiner();
            var predictions = new Single[_mappers.Length];
            var buffers = new VBuffer<Single>[_mappers.Length];
            var maps = new ValueMapper<VBuffer<Single>, Single>[_mappers.Length];
            for (int i = 0; i < _mappers.Length; i++)
                maps[i] = _mappers[i].GetMapper<VBuffer<Single>, Single>();

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
                            EnsembleUtils.SelectFeatures(ref tmp, model.SelectedFeatures, model.Cardinality, ref buffers[i]);
                            maps[i](ref buffers[i], ref predictions[i]);
                        }
                        else
                            maps[i](ref tmp, ref predictions[i]);
                    });

                    combine(ref dst, predictions, Weights);
                };

            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }
    }
}