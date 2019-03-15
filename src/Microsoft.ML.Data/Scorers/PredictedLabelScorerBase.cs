// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Class for scorers that compute on additional "PredictedLabel" column from the score column.
    /// Currently, this scorer is used for binary classification, multi-class classification, and clustering.
    /// </summary>
    internal abstract class PredictedLabelScorerBase : RowToRowScorerBase, ITransformCanSavePfa, ITransformCanSaveOnnx
    {
        public abstract class ThresholdArgumentsBase : ScorerArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Value for classification thresholding", ShortName = "t")]
            public float Threshold;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Specify which predictor output to use for classification thresholding", ShortName = "tcol")]
            public string ThresholdColumn = AnnotationUtils.Const.ScoreValueKind.Score;
        }

        [BestFriend]
        private protected sealed class BindingsImpl : BindingsBase
        {
            // Column index of the score column in Mapper's schema.
            public readonly int ScoreColumnIndex;
            // The type of the derived column.
            public readonly DataViewType PredColType;
            // The ScoreColumnKind metadata value for all score columns.
            public readonly string ScoreColumnKind;

            private readonly AnnotationUtils.AnnotationGetter<ReadOnlyMemory<char>> _getScoreColumnKind;
            private readonly AnnotationUtils.AnnotationGetter<ReadOnlyMemory<char>> _getScoreValueKind;
            private readonly DataViewSchema.Annotations _predColMetadata;
            private BindingsImpl(DataViewSchema input, ISchemaBoundRowMapper mapper, string suffix, string scoreColumnKind,
                bool user, int scoreColIndex, DataViewType predColType)
                : base(input, mapper, suffix, user, DefaultColumnNames.PredictedLabel)
            {
                Contracts.AssertNonEmpty(scoreColumnKind);
                Contracts.Assert(DerivedColumnCount == 1);

                ScoreColumnIndex = scoreColIndex;
                ScoreColumnKind = scoreColumnKind;
                PredColType = predColType;

                _getScoreColumnKind = GetScoreColumnKind;
                _getScoreValueKind = GetScoreValueKind;

                // REVIEW: This logic is very specific to multiclass, which is deeply
                // regrettable, but the class structure as designed and the status of this schema
                // bearing object makes pushing the logic into the multiclass scorer almost impossible.
                if (predColType is KeyType predColKeyType && predColKeyType.Count > 0)
                {
                    var scoreColMetadata = mapper.OutputSchema[scoreColIndex].Annotations;

                    var slotColumn = scoreColMetadata.Schema.GetColumnOrNull(AnnotationUtils.Kinds.SlotNames);
                    if (slotColumn?.Type is VectorType slotColVecType && (ulong)slotColVecType.Size == predColKeyType.Count)
                    {
                        Contracts.Assert(slotColVecType.Size > 0);
                        _predColMetadata = Utils.MarshalInvoke(KeyValueMetadataFromMetadata<int>, slotColVecType.RawType,
                            scoreColMetadata, slotColumn.Value);
                    }
                    else
                    {
                        var trainLabelColumn = scoreColMetadata.Schema.GetColumnOrNull(AnnotationUtils.Kinds.TrainingLabelValues);
                        if (trainLabelColumn?.Type is VectorType trainLabelColVecType && (ulong)trainLabelColVecType.Size == predColKeyType.Count)
                        {
                            Contracts.Assert(trainLabelColVecType.Size > 0);
                            _predColMetadata = Utils.MarshalInvoke(KeyValueMetadataFromMetadata<int>, trainLabelColVecType.RawType,
                                scoreColMetadata, trainLabelColumn.Value);
                        }
                    }
                }
            }

            private static DataViewSchema.Annotations KeyValueMetadataFromMetadata<T>(DataViewSchema.Annotations meta, DataViewSchema.Column metaCol)
            {
                Contracts.AssertValue(meta);
                Contracts.Assert(metaCol.Type.RawType == typeof(T));
                var builder = new DataViewSchema.Annotations.Builder();
                builder.Add(AnnotationUtils.Kinds.KeyValues, metaCol.Type, meta.GetGetter<T>(metaCol));
                return builder.ToAnnotations();
            }

            public static BindingsImpl Create(DataViewSchema input, ISchemaBoundRowMapper mapper, string suffix,
                string scoreColKind, int scoreColIndex, DataViewType predColType)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(mapper);
                Contracts.AssertValueOrNull(suffix);
                Contracts.AssertNonEmpty(scoreColKind);

                return new BindingsImpl(input, mapper, suffix, scoreColKind, true,
                    scoreColIndex, predColType);
            }

            public BindingsImpl ApplyToSchema(DataViewSchema input, ISchemaBindableMapper bindable, IHostEnvironment env)
            {
                Contracts.AssertValue(env);
                env.AssertValue(input);
                env.AssertValue(bindable);

                string scoreCol = RowMapper.OutputSchema[ScoreColumnIndex].Name;
                var schema = new RoleMappedSchema(input, RowMapper.GetInputColumnRoles());

                // Checks compatibility of the predictor input types.
                var mapper = bindable.Bind(env, schema);
                var rowMapper = mapper as ISchemaBoundRowMapper;
                env.CheckParam(rowMapper != null, nameof(bindable), "Mapper must implement ISchemaBoundRowMapper");
                int mapperScoreColumn;
                bool tmp = rowMapper.OutputSchema.TryGetColumnIndex(scoreCol, out mapperScoreColumn);
                env.Check(tmp, "Mapper doesn't have expected score column");

                return new BindingsImpl(input, rowMapper, Suffix, ScoreColumnKind, true, mapperScoreColumn, PredColType);
            }

            public static BindingsImpl Create(ModelLoadContext ctx, DataViewSchema input,
                IHostEnvironment env, ISchemaBindableMapper bindable,
                Func<DataViewType, bool> outputTypeMatches, Func<DataViewType, ISchemaBoundRowMapper, DataViewType> getPredColType)
            {
                Contracts.AssertValue(env);
                env.AssertValue(ctx);

                // *** Binary format ***
                // <base info>
                // int: id of the scores column kind (metadata output)
                // int: id of the column used for deriving the predicted label column

                string suffix;
                var roles = LoadBaseInfo(ctx, out suffix);

                string scoreKind = ctx.LoadNonEmptyString();
                string scoreCol = ctx.LoadNonEmptyString();

                var mapper = bindable.Bind(env, new RoleMappedSchema(input, roles));
                var rowMapper = mapper as ISchemaBoundRowMapper;
                env.CheckParam(rowMapper != null, nameof(bindable), "Bindable expected to be an " + nameof(ISchemaBindableMapper) + "!");

                // Find the score column of the mapper.
                int scoreColIndex;
                env.CheckDecode(mapper.OutputSchema.TryGetColumnIndex(scoreCol, out scoreColIndex));

                var scoreType = mapper.OutputSchema[scoreColIndex].Type;
                env.CheckDecode(outputTypeMatches(scoreType));
                var predColType = getPredColType(scoreType, rowMapper);

                return new BindingsImpl(input, rowMapper, suffix, scoreKind, false, scoreColIndex, predColType);
            }

            internal override void SaveModel(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // <base info>
                // int: id of the scores column kind (metadata output)
                // int: id of the column used for deriving the predicted label column
                SaveBase(ctx);
                ctx.SaveNonEmptyString(ScoreColumnKind);
                ctx.SaveNonEmptyString(RowMapper.OutputSchema[ScoreColumnIndex].Name);
            }

            protected override DataViewType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
                if (iinfo < DerivedColumnCount)
                    return PredColType;
                return base.GetColumnTypeCore(iinfo);
            }

            protected override IEnumerable<KeyValuePair<string, DataViewType>> GetAnnotationTypesCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo && iinfo < InfoCount);

                // This sets the score column kind for all columns.
                yield return TextDataViewType.Instance.GetPair(AnnotationUtils.Kinds.ScoreColumnKind);
                if (iinfo < DerivedColumnCount)
                {
                    yield return TextDataViewType.Instance.GetPair(AnnotationUtils.Kinds.ScoreValueKind);
                    if (_predColMetadata != null)
                    {
                        var sch = _predColMetadata.Schema;
                        for (int i = 0; i < sch.Count; ++i)
                            yield return new KeyValuePair<string, DataViewType>(sch[i].Name, sch[i].Type);
                    }
                }
                foreach (var pair in base.GetAnnotationTypesCore(iinfo))
                {
                    if (pair.Key != AnnotationUtils.Kinds.ScoreColumnKind)
                        yield return pair;
                }
            }

            protected override DataViewType GetAnnotationTypeCore(string kind, int iinfo)
            {
                Contracts.Assert(0 <= iinfo && iinfo < InfoCount);

                if (kind == AnnotationUtils.Kinds.ScoreColumnKind)
                    return TextDataViewType.Instance;
                if (iinfo < DerivedColumnCount && kind == AnnotationUtils.Kinds.ScoreValueKind)
                    return TextDataViewType.Instance;
                if (iinfo < DerivedColumnCount && _predColMetadata != null)
                {
                    int mcol;
                    if (_predColMetadata.Schema.TryGetColumnIndex(kind, out mcol))
                        return _predColMetadata.Schema[mcol].Type;
                }
                return base.GetAnnotationTypeCore(kind, iinfo);
            }

            protected override void GetAnnotationCore<TValue>(string kind, int iinfo, ref TValue value)
            {
                if (kind == AnnotationUtils.Kinds.ScoreColumnKind)
                {
                    _getScoreColumnKind.Marshal(iinfo, ref value);
                    return;
                }
                if (iinfo < DerivedColumnCount && kind == AnnotationUtils.Kinds.ScoreValueKind)
                {
                    _getScoreValueKind.Marshal(iinfo, ref value);
                    return;
                }
                if (iinfo < DerivedColumnCount && _predColMetadata != null)
                {
                    var mcol = _predColMetadata.Schema.GetColumnOrNull(kind);
                    if (mcol.HasValue)
                    {
                        // REVIEW: In the event that TValue is not the right type, it won't really be
                        // the "right" type of exception. However considering that I consider the metadata
                        // schema as it stands right now to be temporary, let's suppose we don't really care.
                        _predColMetadata.GetGetter<TValue>(mcol.Value)(ref value);
                        return;
                    }
                }
                base.GetAnnotationCore<TValue>(kind, iinfo, ref value);
            }

            private void GetScoreColumnKind(int iinfo, ref ReadOnlyMemory<char> dst)
            {
                Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
                dst = ScoreColumnKind.AsMemory();
            }

            private void GetScoreValueKind(int iinfo, ref ReadOnlyMemory<char> dst)
            {
                // This should only get called for the derived column.
                Contracts.Assert(0 <= iinfo && iinfo < DerivedColumnCount);
                dst = AnnotationUtils.Const.ScoreValueKind.PredictedLabel.AsMemory();
            }

            public override Func<int, bool> GetActiveMapperColumns(bool[] active)
            {
                Contracts.Assert(DerivedColumnCount == 1);

                // Return true in two cases:
                // 1. col is active directly.
                // 2. col is the score column and the derived column is active.
                var pred = base.GetActiveMapperColumns(active);
                return col => pred(col) || col == ScoreColumnIndex && active[MapIinfoToCol(0)];
            }
        }

        [BestFriend]
        private protected readonly BindingsImpl Bindings;
        [BestFriend]
        private protected sealed override BindingsBase GetBindings() => Bindings;
        public override DataViewSchema OutputSchema { get; }

        bool ICanSavePfa.CanSavePfa => (Bindable as ICanSavePfa)?.CanSavePfa == true;

        bool ICanSaveOnnx.CanSaveOnnx(OnnxContext ctx) => (Bindable as ICanSaveOnnx)?.CanSaveOnnx(ctx) == true;

        [BestFriend]
        private protected PredictedLabelScorerBase(ScorerArgumentsBase args, IHostEnvironment env, IDataView data,
            ISchemaBoundMapper mapper, RoleMappedSchema trainSchema, string registrationName, string scoreColKind, string scoreColName,
            Func<DataViewType, bool> outputTypeMatches, Func<DataViewType, ISchemaBoundRowMapper, DataViewType> getPredColType)
            : base(env, data, registrationName, Contracts.CheckRef(mapper, nameof(mapper)).Bindable)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckNonEmpty(scoreColKind, nameof(scoreColKind));
            Host.CheckNonEmpty(scoreColName, nameof(scoreColName));
            Host.CheckValue(outputTypeMatches, nameof(outputTypeMatches));
            Host.CheckValue(getPredColType, nameof(getPredColType));

            var rowMapper = mapper as ISchemaBoundRowMapper;
            Host.CheckParam(rowMapper != null, nameof(mapper), "mapper should implement " + nameof(ISchemaBoundRowMapper));

            int scoreColIndex;
            if (!mapper.OutputSchema.TryGetColumnIndex(scoreColName, out scoreColIndex))
                throw Host.ExceptParam(nameof(scoreColName), "mapper does not contain a column '{0}'", scoreColName);

            var scoreType = mapper.OutputSchema[scoreColIndex].Type;
            Host.Check(outputTypeMatches(scoreType), "Unexpected predictor output type");
            var predColType = getPredColType(scoreType, rowMapper);

            Bindings = BindingsImpl.Create(data.Schema, rowMapper, args.Suffix, scoreColKind, scoreColIndex, predColType);
            OutputSchema = Bindings.AsSchema;
        }

        protected PredictedLabelScorerBase(IHostEnvironment env, PredictedLabelScorerBase transform,
            IDataView newSource, string registrationName)
            : base(env, newSource, registrationName, transform.Bindable)
        {
            Bindings = transform.Bindings.ApplyToSchema(newSource.Schema, Bindable, env);
            OutputSchema = Bindings.AsSchema;
        }

        [BestFriend]
        private protected PredictedLabelScorerBase(IHost host, ModelLoadContext ctx, IDataView input,
            Func<DataViewType, bool> outputTypeMatches, Func<DataViewType, ISchemaBoundRowMapper, DataViewType> getPredColType)
            : base(host, ctx, input)
        {
            Host.AssertValue(ctx);
            Host.AssertValue(host);
            Host.AssertValue(outputTypeMatches);
            Host.AssertValue(getPredColType);

            Bindings = BindingsImpl.Create(ctx, input.Schema, host, Bindable, outputTypeMatches, getPredColType);
            OutputSchema = Bindings.AsSchema;
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            Bindings.SaveModel(ctx);
        }

        void ISaveAsPfa.SaveAsPfa(BoundPfaContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(Bindable is IBindableCanSavePfa);
            var pfaBindable = (IBindableCanSavePfa)Bindable as IBindableCanSavePfa;

            var schema = Bindings.RowMapper.InputRoleMappedSchema;
            int delta = Bindings.DerivedColumnCount;
            Host.Assert(delta == 1);
            string[] outColNames = new string[Bindings.InfoCount - delta];
            for (int iinfo = delta; iinfo < Bindings.InfoCount; ++iinfo)
                outColNames[iinfo - delta] = Bindings.GetColumnName(Bindings.MapIinfoToCol(iinfo));

            pfaBindable.SaveAsPfa(ctx, schema, outColNames);
            for (int i = 0; i < outColNames.Length; ++i)
                outColNames[i] = ctx.TokenOrNullForName(outColNames[i]);

            var predictedLabelExpression = PredictedLabelPfa(outColNames);
            string derivedName = Bindings.GetColumnName(Bindings.MapIinfoToCol(0));
            if (predictedLabelExpression == null)
            {
                ctx.Hide(derivedName);
                return;
            }
            ctx.DeclareVar(derivedName, predictedLabelExpression);
        }

        [BestFriend]
        private protected abstract JToken PredictedLabelPfa(string[] mapperOutputs);

        void ISaveAsOnnx.SaveAsOnnx(OnnxContext ctx) => SaveAsOnnxCore(ctx);

        [BestFriend]
        private protected virtual void SaveAsOnnxCore(OnnxContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            Host.Assert(Bindable is IBindableCanSaveOnnx);
            var onnxBindable = (IBindableCanSaveOnnx)Bindable;

            var schema = Bindings.RowMapper.InputRoleMappedSchema;
            int delta = Bindings.DerivedColumnCount;

            Host.Assert(delta == 1);

            string[] outVariableNames = new string[Bindings.InfoCount];
            for (int iinfo = 0; iinfo < Bindings.InfoCount; ++iinfo)
            {
                int colIndex = Bindings.MapIinfoToCol(iinfo);
                string colName = Bindings.GetColumnName(colIndex);
                colName = ctx.AddIntermediateVariable(Bindings.GetColumnType(colIndex), colName, true);
                outVariableNames[iinfo] = colName;
            }

            if (!onnxBindable.SaveAsOnnx(ctx, schema, outVariableNames))
            {
                foreach (var name in outVariableNames)
                    ctx.RemoveVariable(name, true);
            }
        }

        protected override bool WantParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);

            // Prefer parallel cursors iff some of our columns are active, otherwise, don't care.
            return Bindings.AnyNewColumnsActive(predicate);
        }

        protected override Delegate[] GetGetters(DataViewRow output, Func<int, bool> predicate)
        {
            Host.Assert(Bindings.DerivedColumnCount == 1);
            Host.AssertValue(output);
            Host.AssertValue(predicate);
            Host.Assert(output.Schema == Bindings.RowMapper.OutputSchema);
            Host.Assert(Bindings.InfoCount == output.Schema.Count + 1);

            var getters = new Delegate[Bindings.InfoCount];

            // Deal with the predicted label column.
            int delta = Bindings.DerivedColumnCount;
            Delegate delScore = null;
            if (predicate(0))
            {
                Host.Assert(output.IsColumnActive(output.Schema[Bindings.ScoreColumnIndex]));
                getters[0] = GetPredictedLabelGetter(output, out delScore);
            }

            for (int iinfo = delta; iinfo < getters.Length; iinfo++)
            {
                if (!predicate(iinfo))
                    continue;
                if (iinfo == delta + Bindings.ScoreColumnIndex && delScore != null)
                    getters[iinfo] = delScore;
                else
                    getters[iinfo] = GetGetterFromRow(output, iinfo - delta);
            }

            return getters;
        }

        protected abstract Delegate GetPredictedLabelGetter(DataViewRow output, out Delegate scoreGetter);

        protected void EnsureCachedPosition<TScore>(ref long cachedPosition, ref TScore score,
            DataViewRow boundRow, ValueGetter<TScore> scoreGetter)
        {
            if (cachedPosition != boundRow.Position)
            {
                scoreGetter(ref score);
                cachedPosition = boundRow.Position;
            }
        }
    }
}