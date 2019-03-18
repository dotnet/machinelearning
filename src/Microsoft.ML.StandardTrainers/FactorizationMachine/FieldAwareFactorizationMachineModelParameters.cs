// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(typeof(FieldAwareFactorizationMachineModelParameters), null, typeof(SignatureLoadModel), "Field Aware Factorization Machine", FieldAwareFactorizationMachineModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(FieldAwareFactorizationMachinePredictionTransformer), typeof(FieldAwareFactorizationMachinePredictionTransformer), null, typeof(SignatureLoadModel),
    "", FieldAwareFactorizationMachinePredictionTransformer.LoaderSignature)]

namespace Microsoft.ML.Trainers
{
    public sealed class FieldAwareFactorizationMachineModelParameters : ModelParametersBase<float>, ISchemaBindableMapper
    {
        internal const string LoaderSignature = "FieldAwareFactMacPredict";
        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
        private bool _norm;

        /// <summary>
        /// Get the number of fields. It's the symbol `m` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
        /// </summary>
        public int FieldCount { get; }

        /// <summary>
        /// Get the number of features. It's the symbol `n` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
        /// </summary>
        public int FeatureCount { get; }

        /// <summary>
        /// Get the latent dimension. It's the tlngth of `v_{j, f}` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
        /// </summary>
        public int LatentDimension { get; }

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
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FieldAwareFactorizationMachineModelParameters).Assembly.FullName);
        }

        /// <summary>
        /// Initialize model parameters with a trained model.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="norm">True if user wants to normalize feature vector to unit length.</param>
        /// <param name="fieldCount">The number of fileds, which is the symbol `m` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf </param>
        /// <param name="featureCount">The number of features, which is the symbol `n` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf </param>
        /// <param name="latentDim">The latent dimensions, which is the length of `v_{j, f}` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf </param>
        /// <param name="linearWeights">The linear coefficients of the features, which is the symbol `w` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf </param>
        /// <param name="latentWeights">Latent representation of each feature. Note that one feature may have <see cref="FieldCount"/> latent vectors
        /// and each latent vector contains <see cref="LatentDimension"/> values. In the f-th field, the j-th feature's latent vector, `v_{j, f}` in the doc
        /// https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf, starts at latentWeights[j * fieldCount * latentDim + f * latentDim].
        /// The k-th element in v_{j, f} is latentWeights[j * fieldCount * latentDim + f * latentDim + k]. The size of the array must be featureCount x fieldCount x latentDim.</param>
        internal FieldAwareFactorizationMachineModelParameters(IHostEnvironment env, bool norm, int fieldCount, int featureCount, int latentDim,
            float[] linearWeights, float[] latentWeights) : base(env, LoaderSignature)
        {
            Host.Assert(fieldCount > 0);
            Host.Assert(featureCount > 0);
            Host.Assert(latentDim > 0);
            Host.Assert(Utils.Size(linearWeights) == featureCount);
            LatentDimAligned = FieldAwareFactorizationMachineUtils.GetAlignedVectorLength(latentDim);
            Host.Assert(Utils.Size(latentWeights) == checked(featureCount * fieldCount * LatentDimAligned));

            _norm = norm;
            FieldCount = fieldCount;
            FeatureCount = featureCount;
            LatentDimension = latentDim;
            _linearWeights = linearWeights;

            _latentWeightsAligned = new AlignedArray(FeatureCount * FieldCount * LatentDimAligned, 16);

            for (int j = 0; j < FeatureCount; j++)
            {
                for (int f = 0; f < FieldCount; f++)
                {
                    int index = j * FieldCount * LatentDimension + f * LatentDimension;
                    int indexAligned = j * FieldCount * LatentDimAligned + f * LatentDimAligned;
                    for (int k = 0; k < LatentDimAligned; k++)
                    {
                        if (k < LatentDimension)
                            _latentWeightsAligned[indexAligned + k] = latentWeights[index + k];
                        else
                            _latentWeightsAligned[indexAligned + k] = 0;
                    }
                }
            }
        }

        internal FieldAwareFactorizationMachineModelParameters(IHostEnvironment env, bool norm, int fieldCount, int featureCount, int latentDim,
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
            LatentDimension = latentDim;
            _linearWeights = linearWeights;
            _latentWeightsAligned = latentWeightsAligned;
        }

        private FieldAwareFactorizationMachineModelParameters(IHostEnvironment env, ModelLoadContext ctx) : base(env, LoaderSignature)
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
            LatentDimension = latentDim;
            _linearWeights = linearWeights;
            _latentWeightsAligned = new AlignedArray(FeatureCount * FieldCount * LatentDimAligned, 16);
            for (int j = 0; j < FeatureCount; j++)
            {
                for (int f = 0; f < FieldCount; f++)
                {
                    int vBias = j * FieldCount * LatentDimension + f * LatentDimension;
                    int vBiasAligned = j * FieldCount * LatentDimAligned + f * LatentDimAligned;
                    for (int k = 0; k < LatentDimAligned; k++)
                    {
                        if (k < LatentDimension)
                            _latentWeightsAligned[vBiasAligned + k] = latentWeights[vBias + k];
                        else
                            _latentWeightsAligned[vBiasAligned + k] = 0;
                    }
                }
            }
        }

        private static FieldAwareFactorizationMachineModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new FieldAwareFactorizationMachineModelParameters(env, ctx);
        }

        private protected override void SaveCore(ModelSaveContext ctx)
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
            Host.Assert(LatentDimension > 0);
            Host.Assert(Utils.Size(_linearWeights) == FeatureCount);
            Host.Assert(_latentWeightsAligned.Size == FeatureCount * FieldCount * LatentDimAligned);

            ctx.Writer.Write(_norm);
            ctx.Writer.Write(FieldCount);
            ctx.Writer.Write(FeatureCount);
            ctx.Writer.Write(LatentDimension);
            ctx.Writer.WriteSingleArray(_linearWeights);
            float[] latentWeights = new float[FeatureCount * FieldCount * LatentDimension];
            for (int j = 0; j < FeatureCount; j++)
            {
                for (int f = 0; f < FieldCount; f++)
                {
                    int vBias = j * FieldCount * LatentDimension + f * LatentDimension;
                    int vBiasAligned = j * FieldCount * LatentDimAligned + f * LatentDimAligned;
                    for (int k = 0; k < LatentDimension; k++)
                        latentWeights[vBias + k] = _latentWeightsAligned[vBiasAligned + k];
                }
            }
            ctx.Writer.WriteSingleArray(latentWeights);
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

        ISchemaBoundMapper ISchemaBindableMapper.Bind(IHostEnvironment env, RoleMappedSchema schema)
            => new FieldAwareFactorizationMachineScalarRowMapper(env, schema, ScoreSchemaFactory.CreateBinaryClassificationSchema(), this);

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

        /// <summary>
        /// The linear coefficients of the features. It's the symbol `w` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
        /// </summary>
        public IReadOnlyList<float> GetLinearWeights() => _linearWeights;

        /// <summary>
        /// Latent representation of each feature. Note that one feature may have <see cref="FieldCount"/> latent vectors
        /// and each latent vector contains <see cref="LatentDimension"/> values. In the f-th field, the j-th feature's latent vector, `v_{j, f}` in the doc
        /// https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf, starts at latentWeights[j * fieldCount * latentDim + f * latentDim].
        /// The k-th element in v_{j, f} is latentWeights[j * fieldCount * latentDim + f * latentDim + k].
        /// The size of the returned value is featureCount x fieldCount x latentDim.
        /// </summary>
        public IReadOnlyList<float> GetLatentWeights()
        {
            var latentWeights = new float[FeatureCount * FieldCount * LatentDimension];
            for (int j = 0; j < FeatureCount; j++)
            {
                for (int f = 0; f < FieldCount; f++)
                {
                    int index = j * FieldCount * LatentDimension + f * LatentDimension;
                    int indexAligned = j * FieldCount * LatentDimAligned + f * LatentDimAligned;
                    for (int k = 0; k < LatentDimension; k++)
                    {
                        latentWeights[index + k] = _latentWeightsAligned[indexAligned + k];
                    }
                }
            }
            return latentWeights;
        }
    }

    public sealed class FieldAwareFactorizationMachinePredictionTransformer : PredictionTransformerBase<FieldAwareFactorizationMachineModelParameters>
    {
        internal const string LoaderSignature = "FAFMPredXfer";

        /// <summary>
        /// The name of the feature column used by the prediction transformer.
        /// </summary>
        private IReadOnlyList<string> FeatureColumns { get; }

        /// <summary>
        /// The type of the feature columns.
        /// </summary>
        private IReadOnlyList<DataViewType> FeatureColumnTypes { get; }

        private readonly string _thresholdColumn;
        private readonly float _threshold;

        internal FieldAwareFactorizationMachinePredictionTransformer(IHostEnvironment host, FieldAwareFactorizationMachineModelParameters model, DataViewSchema trainSchema,
            string[] featureColumns, float threshold = 0f, string thresholdColumn = DefaultColumnNames.Score)
            : base(Contracts.CheckRef(host, nameof(host)).Register(nameof(FieldAwareFactorizationMachinePredictionTransformer)), model, trainSchema)
        {
            Host.CheckNonEmpty(thresholdColumn, nameof(thresholdColumn));
            Host.CheckNonEmpty(featureColumns, nameof(featureColumns));

            _threshold = threshold;
            _thresholdColumn = thresholdColumn;
            FeatureColumns = featureColumns;
            var featureColumnTypes = new DataViewType[featureColumns.Length];

            int i = 0;
            foreach (var feat in featureColumns)
            {
                if (!trainSchema.TryGetColumnIndex(feat, out int col))
                    throw Host.ExceptSchemaMismatch(nameof(featureColumns), "feature", feat);
                featureColumnTypes[i++] = trainSchema[col].Type;
            }
            FeatureColumnTypes = featureColumnTypes;

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, model);

            var schema = GetSchema();
            var args = new BinaryClassifierScorer.Arguments { Threshold = _threshold, ThresholdColumn = _thresholdColumn };
            Scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, trainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        internal FieldAwareFactorizationMachinePredictionTransformer(IHostEnvironment host, ModelLoadContext ctx)
            : base(Contracts.CheckRef(host, nameof(host)).Register(nameof(FieldAwareFactorizationMachinePredictionTransformer)), ctx)
        {
            // *** Binary format ***
            // <base info>
            // ids of strings: feature columns.
            // float: scorer threshold
            // id of string: scorer threshold column

            // count of feature columns. FAFM uses more than one.
            int featCount = Model.FieldCount;

            var featureColumns = new string[featCount];
            var featureColumnTypes = new DataViewType[featCount];

            for (int i = 0; i < featCount; i++)
            {
                featureColumns[i] = ctx.LoadString();
                if (!TrainSchema.TryGetColumnIndex(featureColumns[i], out int col))
                    throw Host.ExceptSchemaMismatch(nameof(FeatureColumns), "feature", featureColumns[i]);
                featureColumnTypes[i] = TrainSchema[col].Type;
            }
            FeatureColumns = featureColumns;
            FeatureColumnTypes = featureColumnTypes;

            _threshold = ctx.Reader.ReadSingle();
            _thresholdColumn = ctx.LoadString();

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, Model);

            var schema = GetSchema();
            var args = new BinaryClassifierScorer.Arguments { Threshold = _threshold, ThresholdColumn = _thresholdColumn };
            Scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        /// <summary>
        /// Gets the <see cref="DataViewSchema"/> result after transformation.
        /// </summary>
        /// <param name="inputSchema">The <see cref="DataViewSchema"/> of the input data.</param>
        /// <returns>The post transformation <see cref="DataViewSchema"/>.</returns>
        public override DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            for (int i = 0; i < FeatureColumns.Count; i++)
            {
                var feat = FeatureColumns[i];
                if (!inputSchema.TryGetColumnIndex(feat, out int col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "feature", feat, FeatureColumnTypes[i].ToString(), null);

                if (!inputSchema[col].Type.Equals(FeatureColumnTypes[i]))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "feature", feat, FeatureColumnTypes[i].ToString(), inputSchema[col].Type.ToString());
            }

            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        /// <summary>
        /// Saves the transformer to file.
        /// </summary>
        /// <param name="ctx">The <see cref="ModelSaveContext"/> that facilitates saving to the <see cref="Repository"/>.</param>
        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // model: prediction model.
            // stream: empty data view that contains train schema.
            // ids of strings: feature columns.
            // float: scorer threshold
            // id of string: scorer threshold column

            ctx.SaveModel(Model, DirModel);
            ctx.SaveBinaryStream(DirTransSchema, writer =>
            {
                using (var ch = Host.Start("Saving train schema"))
                {
                    var saver = new BinarySaver(Host, new BinarySaver.Arguments { Silent = true });
                    DataSaverUtils.SaveDataView(ch, saver, new EmptyDataView(Host, TrainSchema), writer.BaseStream);
                }
            });

            for (int i = 0; i < Model.FieldCount; i++)
                ctx.SaveString(FeatureColumns[i]);

            ctx.Writer.Write(_threshold);
            ctx.SaveString(_thresholdColumn);
        }

        private RoleMappedSchema GetSchema()
        {
            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            foreach (var feat in FeatureColumns)
                roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, feat));

            var schema = new RoleMappedSchema(TrainSchema, roles);
            return schema;
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FAFMPRED",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(FieldAwareFactorizationMachinePredictionTransformer).Assembly.FullName);
        }

        private static FieldAwareFactorizationMachinePredictionTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
            => new FieldAwareFactorizationMachinePredictionTransformer(env, ctx);
    }
}
