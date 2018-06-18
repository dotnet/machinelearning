// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Ensemble
{
    using TVectorPredictor = IPredictorProducing<VBuffer<Single>>;
    public sealed class EnsembleMultiClassPredictor :
        EnsemblePredictorBase<TVectorPredictor, VBuffer<Single>>,
        IValueMapper
    {
        public const string LoaderSignature = "EnsemMcExec";
        public const string RegistrationName = "EnsembleMultiClassPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ENSEM MC",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Metrics and subset info into main stream, after each predictor
                verWrittenCur: 0x00010003, // Don't serialize the "IsAveraged" property of the metrics
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature);
        }

        private readonly ColumnType _inputType;
        private readonly ColumnType _outputType;
        private readonly IValueMapper[] _mappers;

        public ColumnType InputType { get { return _inputType; } }
        public ColumnType OutputType { get { return _outputType; } }

        internal EnsembleMultiClassPredictor(IHostEnvironment env, FeatureSubsetModel<TVectorPredictor>[] models,
            IOutputCombiner<VBuffer<Single>> combiner, Single[] weights = null)
            : base(env, RegistrationName, models, combiner, weights)
        {
            InitializeMappers(out _mappers, out _inputType, out _outputType);
        }

        private EnsembleMultiClassPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            InitializeMappers(out _mappers, out _inputType, out _outputType);
        }

        private void InitializeMappers(out IValueMapper[] mappers, out ColumnType inputType, out ColumnType outputType)
        {
            Host.AssertNonEmpty(Models);

            mappers = new IValueMapper[Models.Length];
            inputType = null;
            outputType = null;
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

                if (outputType == null || vm.OutputType.VectorSize > outputType.VectorSize)
                    outputType = vm.OutputType;

                mappers[i] = vm;
            }
            Host.AssertValue(outputType);

            if (inputType == null)
                inputType = new VectorType(NumberType.Float);
        }

        public static EnsembleMultiClassPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new EnsembleMultiClassPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        public override PredictionKind PredictionKind { get { return PredictionKind.MultiClassClassification; } }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<Single>));
            Host.Check(typeof(TOut) == typeof(VBuffer<Single>));

            var combine = Combiner.GetCombiner();
            var features = new VBuffer<Single>[_mappers.Length];
            var predictions = new VBuffer<Single>[_mappers.Length];
            var maps = new ValueMapper<VBuffer<Single>, VBuffer<Single>>[_mappers.Length];
            for (int i = 0; i < _mappers.Length; i++)
            {
                // IsValid method ensures we go this else path only if the OutputType.VectorSize of
                // all _mappers is greater than zero
                Host.Assert(_mappers[i].OutputType.VectorSize > 0);
                maps[i] = _mappers[i].GetMapper<VBuffer<Single>, VBuffer<Single>>();
            }

            ValueMapper<VBuffer<Single>, VBuffer<Single>> del =
                (ref VBuffer<Single> src, ref VBuffer<Single> dst) =>
                {
                    if (_inputType.VectorSize > 0)
                        Host.Check(src.Length == _inputType.VectorSize);

                    var tmp = src;
                    Parallel.For(0, maps.Length, i =>
                    {
                        var model = Models[i];
                        if (model.SelectedFeatures != null)
                        {
                            EnsembleUtils.SelectFeatures(ref tmp, model.SelectedFeatures, model.Cardinality, ref features[i]);
                            maps[i](ref features[i], ref predictions[i]);
                        }
                        else
                            maps[i](ref tmp, ref predictions[i]);

                        // individual maps delegates will return always the same VBuffer length
                        Host.Check(predictions[i].Length == _mappers[i].OutputType.VectorSize);
                    });

                    combine(ref dst, predictions, Weights);
                };

            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private bool IsValid(IValueMapper mapper)
        {
            return mapper != null
                && mapper.InputType.IsVector && mapper.InputType.ItemType == NumberType.Float
                && mapper.OutputType.VectorSize > 0 && mapper.OutputType.ItemType == NumberType.Float;
        }
    }
}
