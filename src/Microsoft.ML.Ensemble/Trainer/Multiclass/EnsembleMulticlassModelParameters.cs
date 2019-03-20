// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(EnsembleMulticlassModelParameters), null, typeof(SignatureLoadModel),
    EnsembleMulticlassModelParameters.UserName, EnsembleMulticlassModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class EnsembleMulticlassModelParameters : EnsembleModelParametersBase<VBuffer<Single>>, IValueMapper
    {
        internal const string UserName = "Ensemble Multiclass Executor";
        internal const string LoaderSignature = "EnsemMcExec";
        internal const string RegistrationName = "EnsembleMultiClassPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ENSEM MC",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Metrics and subset info into main stream, after each predictor
                verWrittenCur: 0x00010003, // Don't serialize the "IsAveraged" property of the metrics
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(EnsembleMulticlassModelParameters).Assembly.FullName);
        }

        private readonly VectorType _inputType;
        private readonly VectorType _outputType;
        private readonly IValueMapper[] _mappers;

        DataViewType IValueMapper.InputType => _inputType;
        DataViewType IValueMapper.OutputType => _outputType;

        /// <summary>
        /// Instantiate new ensemble model from existing sub-models.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="models">Array of sub-models that you want to ensemble together.</param>
        /// <param name="combiner">The combiner class to use to ensemble the models.</param>
        /// <param name="weights">The weights assigned to each model to be ensembled.</param>
        internal EnsembleMulticlassModelParameters(IHostEnvironment env, FeatureSubsetModel<VBuffer<float>>[] models,
            IMulticlassOutputCombiner combiner, Single[] weights = null)
            : base(env, RegistrationName, models, combiner, weights)
        {
            InitializeMappers(out _mappers, out _inputType, out _outputType);
        }

        private EnsembleMulticlassModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            InitializeMappers(out _mappers, out _inputType, out _outputType);
        }

        private void InitializeMappers(out IValueMapper[] mappers, out VectorType inputType, out VectorType outputType)
        {
            Host.AssertNonEmpty(Models);

            mappers = new IValueMapper[Models.Length];
            inputType = null;
            outputType = null;
            for (int i = 0; i < Models.Length; i++)
            {
                var vm = Models[i].Predictor as IValueMapper;
                if (!IsValid(vm, out VectorType vmInputType, out VectorType vmOutputType))
                    throw Host.Except("Predictor does not implement expected interface");
                if (vmInputType.Size > 0)
                {
                    if (inputType == null)
                        inputType = vmInputType;
                    else if (vmInputType.Size != inputType.Size)
                        throw Host.Except("Predictor input type mismatch");
                }

                if (outputType == null || vmOutputType.Size > outputType.Size)
                    outputType = vmOutputType;

                mappers[i] = vm;
            }
            Host.AssertValue(outputType);

            if (inputType == null)
                inputType = new VectorType(NumberDataViewType.Single);
        }

        private static EnsembleMulticlassModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new EnsembleMulticlassModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        ValueMapper<TIn, TOut> IValueMapper.GetMapper<TIn, TOut>()
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
                Host.Assert(_mappers[i].OutputType.GetVectorSize() > 0);
                maps[i] = _mappers[i].GetMapper<VBuffer<Single>, VBuffer<Single>>();
            }

            ValueMapper<VBuffer<Single>, VBuffer<Single>> del =
                (in VBuffer<Single> src, ref VBuffer<Single> dst) =>
                {
                    if (_inputType.Size > 0)
                        Host.Check(src.Length == _inputType.Size);

                    var tmp = src;
                    Parallel.For(0, maps.Length, i =>
                    {
                        var model = Models[i];
                        if (model.SelectedFeatures != null)
                        {
                            EnsembleUtils.SelectFeatures(in tmp, model.SelectedFeatures, model.Cardinality, ref features[i]);
                            maps[i](in features[i], ref predictions[i]);
                        }
                        else
                            maps[i](in tmp, ref predictions[i]);

                        // individual maps delegates will return always the same VBuffer length
                        Host.Check(predictions[i].Length == _mappers[i].OutputType.GetVectorSize());
                    });

                    combine(ref dst, predictions, Weights);
                };

            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }

        private bool IsValid(IValueMapper mapper, out VectorType inputType, out VectorType outputType)
        {
            if (mapper != null
                && mapper.InputType is VectorType inVectorType && inVectorType.ItemType == NumberDataViewType.Single
                && mapper.OutputType is VectorType outVectorType
                && outVectorType.Size > 0 && outVectorType.ItemType == NumberDataViewType.Single)
            {
                inputType = inVectorType;
                outputType = outVectorType;
                return true;
            }
            else
            {
                inputType = null;
                outputType = null;
                return false;
            }
        }
    }
}
