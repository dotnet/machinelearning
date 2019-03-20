// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(EnsembleModelParameters), null, typeof(SignatureLoadModel), EnsembleModelParameters.UserName,
    EnsembleModelParameters.LoaderSignature)]

[assembly: EntryPointModule(typeof(EnsembleModelParameters))]

namespace Microsoft.ML.Trainers.Ensemble
{
    /// <summary>
    /// A class for artifacts of ensembled models.
    /// </summary>
    internal sealed class EnsembleModelParameters : EnsembleModelParametersBase<Single>, IValueMapper
    {
        internal const string UserName = "Ensemble Executor";
        internal const string LoaderSignature = "EnsembleFloatExec";
        internal const string RegistrationName = "EnsemblePredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ENSEM XX",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Metrics and subset info into main stream, after each predictor
                verWrittenCur: 0x00010003, // Don't serialize the "IsAveraged" property of the metrics
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(EnsembleModelParameters).Assembly.FullName);
        }

        private readonly IValueMapper[] _mappers;

        private readonly VectorType _inputType;
        DataViewType IValueMapper.InputType => _inputType;
        DataViewType IValueMapper.OutputType => NumberDataViewType.Single;
        private protected override PredictionKind PredictionKind { get; }

        /// <summary>
        /// Instantiate new ensemble model from existing sub-models.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="kind">The prediction kind <see cref="PredictionKind"/></param>
        /// <param name="models">Array of sub-models that you want to ensemble together.</param>
        /// <param name="combiner">The combiner class to use to ensemble the models.</param>
        /// <param name="weights">The weights assigned to each model to be ensembled.</param>
        internal EnsembleModelParameters(IHostEnvironment env, PredictionKind kind,
            FeatureSubsetModel<float>[] models, IOutputCombiner<Single> combiner, Single[] weights = null)
            : base(env, LoaderSignature, models, combiner, weights)
        {
            PredictionKind = kind;
            _inputType = InitializeMappers(out _mappers);
        }

        private EnsembleModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            PredictionKind = (PredictionKind)ctx.Reader.ReadInt32();
            _inputType = InitializeMappers(out _mappers);
        }

        private VectorType InitializeMappers(out IValueMapper[] mappers)
        {
            Host.AssertNonEmpty(Models);

            mappers = new IValueMapper[Models.Length];
            VectorType inputType = null;
            for (int i = 0; i < Models.Length; i++)
            {
                var vm = Models[i].Predictor as IValueMapper;
                if (!IsValid(vm, out VectorType vmInputType))
                    throw Host.Except("Predictor does not implement expected interface");
                if (vmInputType.Size > 0)
                {
                    if (inputType == null)
                        inputType = vmInputType;
                    else if (vmInputType.Size != inputType.Size)
                        throw Host.Except("Predictor input type mismatch");
                }
                mappers[i] = vm;
            }

            return inputType ?? new VectorType(NumberDataViewType.Single);
        }

        private bool IsValid(IValueMapper mapper, out VectorType inputType)
        {
            if (mapper != null
                && mapper.InputType is VectorType inputVectorType && inputVectorType.ItemType == NumberDataViewType.Single
                && mapper.OutputType == NumberDataViewType.Single)
            {
                inputType = inputVectorType;
                return true;
            }
            else
            {
                inputType = null;
                return false;
            }
        }

        private static EnsembleModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new EnsembleModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: _kind
            ctx.Writer.Write((int)PredictionKind);
        }

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
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
                            EnsembleUtils.SelectFeatures(in tmp, model.SelectedFeatures, model.Cardinality, ref buffers[i]);
                            maps[i](in buffers[i], ref predictions[i]);
                        }
                        else
                            maps[i](in tmp, ref predictions[i]);
                    });

                    combine(ref dst, predictions, Weights);
                };

            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }
    }
}