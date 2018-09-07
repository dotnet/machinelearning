// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;
using System.Collections.Generic;
using System;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A base class for schemas for ISchemaBoundMappers. Takes care of all the metadata that has to do with
    /// the score column. If the predictor schema has more than one output column, then the GetColumnType(),
    /// TryGetColumnIndex() and GetColumnName() methods should be overridden. If additional metadata is
    /// needed, the metadata methods can also be overridden.
    /// </summary>
    public abstract class ScoreMapperSchemaBase : ISchema
    {
        protected readonly ColumnType ScoreType;
        protected readonly string ScoreColumnKind;
        protected readonly MetadataUtils.MetadataGetter<ReadOnlyMemory<char>> ScoreValueKindGetter;
        protected readonly MetadataUtils.MetadataGetter<ReadOnlyMemory<char>> ScoreColumnKindGetter;

        public ScoreMapperSchemaBase(ColumnType scoreType, string scoreColumnKind)
        {
            Contracts.CheckValue(scoreType, nameof(scoreType));
            Contracts.CheckNonEmpty(scoreColumnKind, nameof(scoreColumnKind));

            ScoreType = scoreType;
            ScoreColumnKind = scoreColumnKind;
            ScoreValueKindGetter = GetScoreValueKind;
            ScoreColumnKindGetter = GetScoreColumnKind;
        }

        public virtual int ColumnCount { get { return 1; } }

        private void CheckColZero(int col, string methName)
        {
            Contracts.Assert(0 <= col && col < ColumnCount);
            if (col == 0)
                return;
            throw Contracts.Except("Derived class should have overriden {0} to handle all columns except zero",
                methName);
        }

        public virtual ColumnType GetColumnType(int col)
        {
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            CheckColZero(col, "GetColumnType");
            return ScoreType;
        }

        /// <summary>
        /// This only knows about column zero, the Score column. Derived classes should handle all others.
        /// </summary>
        public virtual bool TryGetColumnIndex(string name, out int col)
        {
            Contracts.CheckValueOrNull(name);
            col = 0;
            return name == MetadataUtils.Const.ScoreValueKind.Score;
        }

        /// <summary>
        /// This only knows about column zero, the Score column. Derived classes should handle all others.
        /// </summary>
        public virtual string GetColumnName(int col)
        {
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            CheckColZero(col, "GetColumnName");
            return MetadataUtils.Const.ScoreValueKind.Score;
        }

        /// <summary>
        /// Assumes all columns have ScoreColumnKind and ScoreValueKind.
        /// </summary>
        public virtual IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
        {
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            return new[] {
                TextType.Instance.GetPair(MetadataUtils.Kinds.ScoreColumnKind),
                TextType.Instance.GetPair(MetadataUtils.Kinds.ScoreValueKind)
            };
        }

        /// <summary>
        /// Assumes all columns have ScoreColumnKind and ScoreValueKind of type Text.
        /// </summary>
        public virtual ColumnType GetMetadataTypeOrNull(string kind, int col)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            switch (kind)
            {
            case MetadataUtils.Kinds.ScoreColumnKind:
            case MetadataUtils.Kinds.ScoreValueKind:
                return TextType.Instance;
            }
            return null;
        }

        /// <summary>
        /// Assumes all columns have ScoreColumnKind and ScoreValueKind.
        /// </summary>
        public virtual void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            switch (kind)
            {
            case MetadataUtils.Kinds.ScoreColumnKind:
                ScoreColumnKindGetter.Marshal(col, ref value);
                break;
            case MetadataUtils.Kinds.ScoreValueKind:
                ScoreValueKindGetter.Marshal(col, ref value);
                break;
            default:
                throw MetadataUtils.ExceptGetMetadata();
            }
        }

        protected virtual void GetScoreValueKind(int col, ref ReadOnlyMemory<char> dst)
        {
            Contracts.Assert(0 <= col && col < ColumnCount);
            CheckColZero(col, "GetScoreValueKind");
            dst = MetadataUtils.Const.ScoreValueKind.Score.AsMemory();
        }

        private void GetScoreColumnKind(int col, ref ReadOnlyMemory<char> dst)
        {
            dst = ScoreColumnKind.AsMemory();
        }
    }

    /// <summary>
    /// Schema implementation for an ISchemaBoundMapper that produces a single column named Score.
    /// </summary>
    public sealed class ScoreMapperSchema : ScoreMapperSchemaBase
    {
        public ScoreMapperSchema(ColumnType scoreType, string scoreColumnKind)
            : base(scoreType, scoreColumnKind)
        {
        }
    }

    /// <summary>
    /// The base class handles the score column (index zero). This class handles the probability column (index one).
    /// </summary>
    public sealed class BinaryClassifierSchema : ScoreMapperSchemaBase
    {
        public override int ColumnCount { get { return base.ColumnCount + 1; } }

        public BinaryClassifierSchema()
            : base(NumberType.Float, MetadataUtils.Const.ScoreColumnKind.BinaryClassification)
        {
        }

        public override ColumnType GetColumnType(int col)
        {
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            if (col == base.ColumnCount)
                return NumberType.Float;
            return base.GetColumnType(col);
        }

        public override bool TryGetColumnIndex(string name, out int col)
        {
            Contracts.CheckValue(name, nameof(name));
            if (name == MetadataUtils.Const.ScoreValueKind.Probability)
            {
                col = base.ColumnCount;
                return true;
            }
            return base.TryGetColumnIndex(name, out col);
        }

        public override string GetColumnName(int col)
        {
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            if (col == base.ColumnCount)
                return MetadataUtils.Const.ScoreValueKind.Probability;
            return base.GetColumnName(col);
        }

        public override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
        {
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));
            var items = base.GetMetadataTypes(col);
            if (col == base.ColumnCount)
                items = items.Prepend(BoolType.Instance.GetPair(MetadataUtils.Kinds.IsNormalized));
            return items;
        }

        public override ColumnType GetMetadataTypeOrNull(string kind, int col)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

            if (col == base.ColumnCount && kind == MetadataUtils.Kinds.IsNormalized)
                return BoolType.Instance;
            return base.GetMetadataTypeOrNull(kind, col);
        }

        public override void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

            if (col == base.ColumnCount && kind == MetadataUtils.Kinds.IsNormalized)
                MetadataUtils.Marshal<DvBool, TValue>(IsNormalized, col, ref value);
            else
                base.GetMetadata<TValue>(kind, col, ref value);
        }

        private void IsNormalized(int col, ref DvBool dst)
        {
            dst = DvBool.True;
        }

        protected override void GetScoreValueKind(int col, ref ReadOnlyMemory<char> dst)
        {
            Contracts.Assert(0 <= col && col < ColumnCount);
            if (col == base.ColumnCount)
                dst = MetadataUtils.Const.ScoreValueKind.Probability.AsMemory();
            else
                base.GetScoreValueKind(col, ref dst);
        }
    }

    public sealed class SequencePredictorSchema : ScoreMapperSchemaBase
    {
        private readonly VectorType _keyNamesType;
        private readonly VBuffer<ReadOnlyMemory<char>> _keyNames;
        private readonly MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>> _getKeyNames;

        private bool HasKeyNames { get { return _keyNamesType != null; } }

        // REVIEW: Future iterations may want slot names.

        /// <summary>
        /// Constructs an <see cref="ISchemaBoundMapper"/> with one column of a given type, called PredictedLabel.
        /// If the input <paramref name="keyNames"/> has positive length, it is exposed as
        /// <see cref="MetadataUtils.Kinds.KeyValues"/> metadata. Note that we do not copy
        /// the input key names, but instead take a reference to it.
        /// </summary>
        public SequencePredictorSchema(ColumnType type, ref VBuffer<ReadOnlyMemory<char>> keyNames, string scoreColumnKind)
            : base(type, scoreColumnKind)
        {
            if (keyNames.Length > 0)
            {
                Contracts.CheckParam(type.ItemType.IsKey,
                    nameof(keyNames), "keyNames valid only for key type");
                Contracts.CheckParam(keyNames.Length == type.ItemType.KeyCount,
                    nameof(keyNames), "keyNames length must match type's key count");
                // REVIEW: Assuming the caller takes some care, it seems
                // like we can get away with
                _keyNames = keyNames;
                _keyNamesType = new VectorType(TextType.Instance, keyNames.Length);
                _getKeyNames = GetKeyNames;
            }
        }

        public override int ColumnCount { get { return 1; } }

        public override bool TryGetColumnIndex(string name, out int col)
        {
            Contracts.CheckValueOrNull(name);
            col = 0;
            return name == MetadataUtils.Const.ScoreValueKind.PredictedLabel;
        }

        public override string GetColumnName(int col)
        {
            Contracts.CheckParam(col == 0, nameof(col));
            return MetadataUtils.Const.ScoreValueKind.PredictedLabel;
        }

        private void GetKeyNames(int col, ref VBuffer<ReadOnlyMemory<char>> dst)
        {
            Contracts.Assert(col == 0);
            Contracts.AssertValue(_keyNamesType);
            _keyNames.CopyTo(ref dst);
        }

        public override void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckParam(col == 0, nameof(col));
            switch (kind)
            {
            case MetadataUtils.Kinds.KeyValues:
                if (!HasKeyNames)
                    throw MetadataUtils.ExceptGetMetadata();
                _getKeyNames.Marshal(col, ref value);
                break;
            default:
                base.GetMetadata(kind, col, ref value);
                break;
            }
        }

        public override IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
        {
            Contracts.CheckParam(col == 0, nameof(col));
            var items = base.GetMetadataTypes(col);
            if (HasKeyNames)
                items = items.Prepend(new KeyValuePair<string, ColumnType>(MetadataUtils.Kinds.KeyValues, _keyNamesType));
            return items;
        }

        public override ColumnType GetMetadataTypeOrNull(string kind, int col)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckParam(col == 0, nameof(col));
            switch (kind)
            {
            case MetadataUtils.Kinds.KeyValues:
                if (!HasKeyNames)
                    return null;
                return _keyNamesType;
            default:
                return base.GetMetadataTypeOrNull(kind, col);
            }
        }

        protected override void GetScoreValueKind(int col, ref ReadOnlyMemory<char> dst)
        {
            Contracts.Assert(col == 0);
            dst = MetadataUtils.Const.ScoreValueKind.PredictedLabel.AsMemory();
        }
    }
}
