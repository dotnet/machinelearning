// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(EncryptedSchemaBindableBinaryPredictorWrapper), null, typeof(SignatureLoadModel),
    "Encrypted Binary Classification Bindable Mapper", EncryptedSchemaBindableBinaryPredictorWrapper.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This is an <see cref="ISchemaBindableMapper"/> wrapper for calibrated binary classification predictors.
    /// They need a separate wrapper because they return two values instead of one: the raw score and the probability.
    /// </summary>
    internal sealed class EncryptedSchemaBindableBinaryPredictorWrapper : SchemaBindablePredictorWrapperBase
    {
        public const string LoaderSignature = "EncryptedBinarySchemaBindable";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "EBNSCHBD",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // ISchemaBindableWrapper update
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(EncryptedSchemaBindableBinaryPredictorWrapper).Assembly.FullName);
        }

        private readonly IValueMapperDist _distMapper;

        public EncryptedSchemaBindableBinaryPredictorWrapper(IPredictor predictor)
            : base(predictor)
        {
            CheckValid(out _distMapper);
        }

        private EncryptedSchemaBindableBinaryPredictorWrapper(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx)
        {
            CheckValid(out _distMapper);
        }

        public static EncryptedSchemaBindableBinaryPredictorWrapper Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new EncryptedSchemaBindableBinaryPredictorWrapper(env, ctx);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            base.SaveModel(ctx);
        }

        private protected override void SaveAsPfaCore(BoundPfaContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Assert(ValueMapper is IDistCanSavePfa);
            Contracts.Assert(schema.Feature.HasValue);
            Contracts.Assert(Utils.Size(outputNames) == 2); // Score and prob.
            var mapper = (IDistCanSavePfa)ValueMapper;
            // If the features column was not produced, we must hide the outputs.
            string featureToken = ctx.TokenOrNullForName(schema.Feature.Value.Name);
            if (featureToken == null)
                ctx.Hide(outputNames);

            JToken scoreToken;
            JToken probToken;
            mapper.SaveAsPfa(ctx, featureToken, outputNames[0], out scoreToken, outputNames[1], out probToken);
            Contracts.Assert(ctx.TokenOrNullForName(outputNames[0]) == scoreToken.ToString());
            Contracts.Assert(ctx.TokenOrNullForName(outputNames[1]) == probToken.ToString());
        }

        private protected override bool SaveAsOnnxCore(OnnxContext ctx, RoleMappedSchema schema, string[] outputNames)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(schema, nameof(schema));

            var mapper = ValueMapper as ISingleCanSaveOnnx;
            Contracts.CheckValue(mapper, nameof(mapper));
            Contracts.Assert(schema.Feature.HasValue);
            Contracts.Assert(Utils.Size(outputNames) == 3); // Predicted Label, Score and Probablity.

            var featName = schema.Feature.Value.Name;
            if (!ctx.ContainsColumn(featName))
                return false;
            Contracts.Assert(ctx.ContainsColumn(featName));
            return mapper.SaveAsOnnx(ctx, outputNames, ctx.GetVariableName(featName));
        }

        private void CheckValid(out IValueMapperDist distMapper)
        {
            Contracts.Check(ScoreType == NumberDataViewType.Single, "Expected predictor result type to be float");

            distMapper = Predictor as IValueMapperDist;
            if (distMapper == null)
                throw Contracts.Except("Predictor does not provide probabilities");

            // REVIEW: In theory the restriction on input type could be relaxed at the expense
            // of more complicated code in CalibratedRowMapper.GetGetters. Not worth it at this point
            // and no good way to test it.
            Contracts.Check(distMapper.InputType is VectorDataViewType vectorType && vectorType.ItemType == NumberDataViewType.Single,
                "Invalid input type for the IValueMapperDist");
            Contracts.Check(distMapper.DistType == NumberDataViewType.Single,
                "Invalid probability type for the IValueMapperDist");
        }

        private protected override ISchemaBoundMapper BindCore(IChannel ch, RoleMappedSchema schema)
        {
            if (Predictor.PredictionKind != PredictionKind.BinaryClassification)
                ch.Warning("Scoring predictor of kind '{0}' as '{1}'.", Predictor.PredictionKind, PredictionKind.BinaryClassification);

            // For distribution mappers, produce both score and probability.
            Contracts.AssertValue(_distMapper);
            return new EncryptedCalibratedRowMapper(schema, this);
        }

        /// <summary>
        /// The <see cref="ISchemaBoundRowMapper"/> implementation for distribution predictor wrappers that produce
        /// two float-valued output columns. Note that the Bindable wrapper does input schema validation.
        /// </summary>
        private sealed class EncryptedCalibratedRowMapper : ISchemaBoundRowMapper
        {
            private readonly EncryptedSchemaBindableBinaryPredictorWrapper _parent;

            public RoleMappedSchema InputRoleMappedSchema { get; }
            public DataViewSchema InputSchema => InputRoleMappedSchema.Schema;

            public DataViewSchema OutputSchema { get; }

            public ISchemaBindableMapper Bindable => _parent;

            public EncryptedCalibratedRowMapper(RoleMappedSchema schema, EncryptedSchemaBindableBinaryPredictorWrapper parent)
            {
                Contracts.AssertValue(parent);
                Contracts.Assert(parent._distMapper != null);
                Contracts.AssertValue(schema);

                _parent = parent;
                InputRoleMappedSchema = schema;
                OutputSchema = ScoreSchemaFactory.CreateBinaryClassificationSchema();

                if (schema.Feature?.Type is DataViewType typeSrc)
                {
                    Contracts.Check(typeSrc is VectorDataViewType vectorType
                        && vectorType.IsKnownSize
                        && vectorType.ItemType == NumberDataViewType.Single,
                        "Invalid feature column type");
                }
            }

            /// <summary>
            /// Given a set of columns, return the input columns that are needed to generate those output columns.
            /// </summary>
            IEnumerable<DataViewSchema.Column> ISchemaBoundRowMapper.GetDependenciesForNewColumns(IEnumerable<DataViewSchema.Column> dependingColumns)
            {

                if (dependingColumns.Count() == 0 || !InputRoleMappedSchema.Feature.HasValue)
                    return Enumerable.Empty<DataViewSchema.Column>();

                return Enumerable.Repeat(InputRoleMappedSchema.Feature.Value, 1);
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RoleMappedSchema.ColumnRole.Feature.Bind(InputRoleMappedSchema.Feature?.Name);
            }

            private Delegate[] CreateGetters(DataViewRow input, bool[] active)
            {
                Contracts.Assert(Utils.Size(active) == 2);
                Contracts.Assert(_parent._distMapper != null);

                var getters = new Delegate[2];
                if (active[0] || active[1])
                {
                    // Put all captured locals at this scope.
                    var featureGetter = InputRoleMappedSchema.Feature.HasValue ? input.GetGetter<VBuffer<float>>(InputRoleMappedSchema.Feature.Value) : null;
                    float prob = 0;
                    float score = 0;
                    long cachedPosition = -1;
                    var features = default(VBuffer<float>);
                    ValueMapper<VBuffer<float>, float, float> mapper;

                    mapper = _parent._distMapper.GetMapper<VBuffer<float>, float, float>();
                    if (active[0])
                    {
                        ValueGetter<float> getScore =
                            (ref float dst) =>
                            {
                                EnsureCachedResultValueMapper(mapper, ref cachedPosition, featureGetter, ref features, ref score, ref prob, input);
                                dst = score;
                            };
                        getters[0] = getScore;
                    }
                    if (active[1])
                    {
                        ValueGetter<float> getProb =
                            (ref float dst) =>
                            {
                                EnsureCachedResultValueMapper(mapper, ref cachedPosition, featureGetter, ref features, ref score, ref prob, input);
                                dst = prob;
                            };
                        getters[1] = getProb;
                    }
                }
                return getters;
            }

            private static void EnsureCachedResultValueMapper(ValueMapper<VBuffer<float>, float, float> mapper,
                ref long cachedPosition, ValueGetter<VBuffer<float>> featureGetter, ref VBuffer<float> features,
                ref float score, ref float prob, DataViewRow input)
            {
                Contracts.AssertValue(mapper);
                if (cachedPosition != input.Position)
                {
                    if (featureGetter != null)
                        featureGetter(ref features);

                    mapper(in features, ref score, ref prob);
                    cachedPosition = input.Position;
                }
            }

            DataViewRow ISchemaBoundRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                Contracts.AssertValue(input);
                var active = Utils.BuildArray(OutputSchema.Count, activeColumns);
                var getters = CreateGetters(input, active);
                return new SimpleRow(OutputSchema, input, getters);
            }
        }
    }
}