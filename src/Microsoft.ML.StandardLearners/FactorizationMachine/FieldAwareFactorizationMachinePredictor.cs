// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.FactorizationMachine;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(typeof(FieldAwareFactorizationMachinePredictor), null, typeof(SignatureLoadModel), "Field Aware Factorization Machine", FieldAwareFactorizationMachinePredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.FactorizationMachine
{
    public sealed class FieldAwareFactorizationMachinePredictor : PredictorBase<float>, ISchemaBindableMapper, ICanSaveModel
    {
        public const string LoaderSignature = "FieldAwareFactMacPredict";
        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        private bool _norm;
        internal int FieldCount { get; }
        internal int FeatureCount { get; }
        internal int LatentDim { get; }
        internal int LatentDimAligned { get; }
        private readonly float[] _linearWeights;
        private readonly AlignedArray _latentWeightsAligned;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FAFAMAPD",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        internal FieldAwareFactorizationMachinePredictor(IHostEnvironment env, bool norm, int fieldCount, int featureCount, int latentDim,
            float[] linearWeights, AlignedArray latentWeightsAligned) : base(env, LoaderSignature)
        {
            Host.Assert(fieldCount > 0);
            Host.Assert(featureCount > 0);
            Host.Assert(latentDim > 0);
            Host.Assert(Utils.Size(linearWeights) == featureCount);
            LatentDimAligned = FieldAwareFactorizationMachineUtils.GetAlignedVectorLength(latentDim);
            Host.Assert(latentWeightsAligned.Size == checked(featureCount * fieldCount * LatentDimAligned));

            _norm = norm;
            FieldCount = fieldCount;
            FeatureCount = featureCount;
            LatentDim = latentDim;
            _linearWeights = linearWeights;
            _latentWeightsAligned = latentWeightsAligned;
        }

        private FieldAwareFactorizationMachinePredictor(IHostEnvironment env, ModelLoadContext ctx) : base(env, LoaderSignature)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // bool: whether to normalize feature vectors
            // int: number of fields
            // int: number of features
            // int: latent dimension
            // float[]: linear coefficients
            // float[]: latent representation of features

            var norm = ctx.Reader.ReadBoolean();
            var fieldCount = ctx.Reader.ReadInt32();
            Host.CheckDecode(fieldCount > 0);
            var featureCount = ctx.Reader.ReadInt32();
            Host.CheckDecode(featureCount > 0);
            var latentDim = ctx.Reader.ReadInt32();
            Host.CheckDecode(latentDim > 0);
            LatentDimAligned = FieldAwareFactorizationMachineUtils.GetAlignedVectorLength(latentDim);
            Host.Check(checked(featureCount * fieldCount * LatentDimAligned) <= Utils.ArrayMaxSize, "Latent dimension too large");
            var linearWeights = ctx.Reader.ReadFloatArray();
            Host.CheckDecode(Utils.Size(linearWeights) == featureCount);
            var latentWeights = ctx.Reader.ReadFloatArray();
            Host.CheckDecode(Utils.Size(latentWeights) == featureCount * fieldCount * latentDim);

            _norm = norm;
            FieldCount = fieldCount;
            FeatureCount = featureCount;
            LatentDim = latentDim;
            _linearWeights = linearWeights;
            _latentWeightsAligned = new AlignedArray(FeatureCount * FieldCount * LatentDimAligned, 16);
            for (int j = 0; j < FeatureCount; j++)
            {
                for (int f = 0; f < FieldCount; f++)
                {
                    int vBias = j * FieldCount * LatentDim + f * LatentDim;
                    int vBiasAligned = j * FieldCount * LatentDimAligned + f * LatentDimAligned;
                    for (int k = 0; k < LatentDimAligned; k++)
                    {
                        if (k < LatentDim)
                            _latentWeightsAligned[vBiasAligned + k] = latentWeights[vBias + k];
                        else
                            _latentWeightsAligned[vBiasAligned + k] = 0;
                    }
                }
            }
        }

        public static FieldAwareFactorizationMachinePredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new FieldAwareFactorizationMachinePredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bool: whether to normalize feature vectors
            // int: number of fields
            // int: number of features
            // int: latent dimension
            // float[]: linear coefficients
            // float[]: latent representation of features

            // REVIEW:FAFM needs to store the names of the features, so that they prediction data does not have the
            // restriciton of the columns needing to be ordered the same as the training data.

            Host.Assert(FieldCount > 0);
            Host.Assert(FeatureCount > 0);
            Host.Assert(LatentDim > 0);
            Host.Assert(Utils.Size(_linearWeights) == FeatureCount);
            Host.Assert(_latentWeightsAligned.Size == FeatureCount * FieldCount * LatentDimAligned);

            ctx.Writer.Write(_norm);
            ctx.Writer.Write(FieldCount);
            ctx.Writer.Write(FeatureCount);
            ctx.Writer.Write(LatentDim);
            ctx.Writer.WriteFloatArray(_linearWeights);
            float[] latentWeights = new float[FeatureCount * FieldCount * LatentDim];
            for (int j = 0; j < FeatureCount; j++)
            {
                for (int f = 0; f < FieldCount; f++)
                {
                    int vBias = j * FieldCount * LatentDim + f * LatentDim;
                    int vBiasAligned = j * FieldCount * LatentDimAligned + f * LatentDimAligned;
                    for (int k = 0; k < LatentDim; k++)
                        latentWeights[vBias + k] = _latentWeightsAligned[vBiasAligned + k];
                }
            }
            ctx.Writer.WriteFloatArray(latentWeights);
        }

        internal float CalculateResponse(ValueGetter<VBuffer<float>>[] getters, VBuffer<float> featureBuffer,
            int[] featureFieldBuffer, int[] featureIndexBuffer, float[] featureValueBuffer, AlignedArray latentSum)
        {
            int count = 0;
            float modelResponse = 0;
            FieldAwareFactorizationMachineUtils.LoadOneExampleIntoBuffer(getters, featureBuffer, _norm, ref count,
                featureFieldBuffer, featureIndexBuffer, featureValueBuffer);
            FieldAwareFactorizationMachineInterface.CalculateIntermediateVariables(FieldCount, LatentDimAligned, count,
                featureFieldBuffer, featureIndexBuffer, featureValueBuffer, _linearWeights, _latentWeightsAligned, latentSum, ref modelResponse);
            return modelResponse;
        }

        public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
            => new FieldAwareFactorizationMachineScalarRowMapper(env, schema, new BinaryClassifierSchema(), this);

        internal void CopyLinearWeightsTo(float[] linearWeights)
        {
            Host.AssertValue(_linearWeights);
            Host.AssertValue(linearWeights);
            Array.Copy(_linearWeights, linearWeights, _linearWeights.Length);
        }

        internal void CopyLatentWeightsTo(AlignedArray latentWeights)
        {
            Host.AssertValue(_latentWeightsAligned);
            Host.AssertValue(latentWeights);
            latentWeights.CopyFrom(_latentWeightsAligned);
        }
    }
}
