// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.FactorizationMachine;
using Microsoft.ML.Internal.CpuMath;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;

[assembly: LoadableClass(typeof(FieldAwareFactorizationMachineModelParameters), null, typeof(SignatureLoadModel), "Field Aware Factorization Machine", FieldAwareFactorizationMachineModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(FieldAwareFactorizationMachinePredictionTransformer), typeof(FieldAwareFactorizationMachinePredictionTransformer), null, typeof(SignatureLoadModel),
    "", FieldAwareFactorizationMachinePredictionTransformer.LoaderSignature)]

namespace Microsoft.ML.FactorizationMachine
{
    public sealed class FieldAwareFactorizationMachineModelParameters : ModelParametersBase<float>, ISchemaBindableMapper
    {
        internal const string LoaderSignature = "FieldAwareFactMacPredict";
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
        /// and each latent vector contains <see cref="LatentDim"/> values. In the f-th field, the j-th feature's latent vector, `v_{j, f}` in the doc
        /// https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf, starts at latentWeights[j * fieldCount * latentDim + f * latentDim].
        /// The k-th element in v_{j, f} is latentWeights[j * fieldCount * latentDim + f * latentDim + k]. The size of the array must be featureCount x fieldCount x latentDim.</param>
        public FieldAwareFactorizationMachineModelParameters(IHostEnvironment env, bool norm, int fieldCount, int featureCount, int latentDim,
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
            LatentDim = latentDim;
            _linearWeights = linearWeights;

            _latentWeightsAligned = new AlignedArray(FeatureCount * FieldCount * LatentDimAligned, 16);

            for (int j = 0; j < FeatureCount; j++)
            {
                for (int f = 0; f < FieldCount; f++)
                {
                    int index = j * FieldCount * LatentDim + f * LatentDim;
                    int indexAligned = j * FieldCount * LatentDimAligned + f * LatentDimAligned;
                    for (int k = 0; k < LatentDimAligned; k++)
                    {
                        if (k < LatentDim)
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
            LatentDim = latentDim;
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
            Host.Assert(LatentDim > 0);
            Host.Assert(Utils.Size(_linearWeights) == FeatureCount);
            Host.Assert(_latentWeightsAligned.Size == FeatureCount * FieldCount * LatentDimAligned);

            ctx.Writer.Write(_norm);
            ctx.Writer.Write(FieldCount);
            ctx.Writer.Write(FeatureCount);
            ctx.Writer.Write(LatentDim);
            ctx.Writer.WriteSingleArray(_linearWeights);
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
        /// Get the number of fields. It's the symbol `m` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
        /// </summary>
        public int GetFieldCount() => FieldCount;

        /// <summary>
        /// Get the number of features. It's the symbol `n` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
        /// </summary>
        public int GetFeatureCount() => FeatureCount;

        /// <summary>
        /// Get the latent dimension. It's the tlngth of `v_{j, f}` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
        /// </summary>
        public int GetLatentDim() => LatentDim;

        /// <summary>
        /// The linear coefficients of the features. It's the symbol `w` in the doc: https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf
        /// </summary>
        public float[] GetLinearWeights() => _linearWeights;

        /// <summary>
        /// Latent representation of each feature. Note that one feature may have <see cref="FieldCount"/> latent vectors
        /// and each latent vector contains <see cref="LatentDim"/> values. In the f-th field, the j-th feature's latent vector, `v_{j, f}` in the doc
        /// https://github.com/wschin/fast-ffm/blob/master/fast-ffm.pdf, starts at latentWeights[j * fieldCount * latentDim + f * latentDim].
        /// The k-th element in v_{j, f} is latentWeights[j * fieldCount * latentDim + f * latentDim + k].
        /// The size of the returned value is featureCount x fieldCount x latentDim.
        /// </summary>
        public float[] GetLatentWeights()
        {
            var latentWeights = new float[FeatureCount * FieldCount * LatentDim];
            for (int j = 0; j < FeatureCount; j++)
            {
                for (int f = 0; f < FieldCount; f++)
                {
                    int index = j * FieldCount * LatentDim + f * LatentDim;
                    int indexAligned = j * FieldCount * LatentDimAligned + f * LatentDimAligned;
                    for (int k = 0; k < LatentDim; k++)
                    {
                        latentWeights[index + k] = _latentWeightsAligned[indexAligned + k];
                    }
                }
            }
            return latentWeights;
        }
    }

    public sealed class FieldAwareFactorizationMachinePredictionTransformer : PredictionTransformerBase<FieldAwareFactorizationMachineModelParameters, BinaryClassifierScorer>, ICanSaveModel
    {
        public const string LoaderSignature = "FAFMPredXfer";

        /// <summary>
        /// The name of the feature column used by the prediction transformer.
        /// </summary>
        public string[] FeatureColumns { get; }

        /// <summary>
        /// The type of the feature columns.
        /// </summary>
        public ColumnType[] FeatureColumnTypes { get; }

        protected override BinaryClassifierScorer Scorer { get; set; }

        private readonly string _thresholdColumn;
        private readonly float _threshold;

        public FieldAwareFactorizationMachinePredictionTransformer(IHostEnvironment host, FieldAwareFactorizationMachineModelParameters model, Schema trainSchema,
            string[] featureColumns, float threshold = 0f, string thresholdColumn = DefaultColumnNames.Score)
            :base(Contracts.CheckRef(host, nameof(host)).Register(nameof(FieldAwareFactorizationMachinePredictionTransformer)), model, trainSchema)
        {
            Host.CheckNonEmpty(thresholdColumn, nameof(thresholdColumn));
            _threshold = threshold;
            _thresholdColumn = thresholdColumn;

            Host.CheckValue(featureColumns, nameof(featureColumns));
            int featCount = featureColumns.Length;
            Host.Check(featCount >= 0, "Empty features column.");

            FeatureColumns = featureColumns;
            FeatureColumnTypes = new ColumnType[featCount];

            int i = 0;
            foreach (var feat in featureColumns)
            {
                if (!trainSchema.TryGetColumnIndex(feat, out int col))
                    throw Host.ExceptSchemaMismatch(nameof(featureColumns), "feature", feat);
                FeatureColumnTypes[i++] = trainSchema[col].Type;
            }

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, model);

            var schema = GetSchema();
            var args = new BinaryClassifierScorer.Arguments { Threshold = _threshold, ThresholdColumn = _thresholdColumn };
            Scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, trainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        public FieldAwareFactorizationMachinePredictionTransformer(IHostEnvironment host, ModelLoadContext ctx)
            :base(Contracts.CheckRef(host, nameof(host)).Register(nameof(FieldAwareFactorizationMachinePredictionTransformer)), ctx)
        {
            // *** Binary format ***
            // <base info>
            // ids of strings: feature columns.
            // float: scorer threshold
            // id of string: scorer threshold column

            // count of feature columns. FAFM uses more than one.
            int featCount = Model.FieldCount;

            FeatureColumns = new string[featCount];
            FeatureColumnTypes = new ColumnType[featCount];

            for (int i = 0; i < featCount; i++)
            {
                FeatureColumns[i] = ctx.LoadString();
                if (!TrainSchema.TryGetColumnIndex(FeatureColumns[i], out int col))
                    throw Host.ExceptSchemaMismatch(nameof(FeatureColumns), "feature", FeatureColumns[i]);
                FeatureColumnTypes[i] = TrainSchema[col].Type;
            }

            _threshold = ctx.Reader.ReadSingle();
            _thresholdColumn = ctx.LoadString();

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, Model);

            var schema = GetSchema();
            var args = new BinaryClassifierScorer.Arguments { Threshold = _threshold, ThresholdColumn = _thresholdColumn };
            Scorer = new BinaryClassifierScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        /// <summary>
        /// Gets the <see cref="Schema"/> result after transformation.
        /// </summary>
        /// <param name="inputSchema">The <see cref="Schema"/> of the input data.</param>
        /// <returns>The post transformation <see cref="Schema"/>.</returns>
        public override Schema GetOutputSchema(Schema inputSchema)
        {
            for (int i = 0; i < FeatureColumns.Length; i++)
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
        public void Save(ModelSaveContext ctx)
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
