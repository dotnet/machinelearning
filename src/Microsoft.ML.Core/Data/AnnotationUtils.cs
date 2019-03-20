// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Utilities for implementing and using the annotation API of <see cref="DataViewSchema"/>.
    /// </summary>
    [BestFriend]
    internal static class AnnotationUtils
    {
        /// <summary>
        /// This class lists the canonical annotation kinds
        /// </summary>
        public static class Kinds
        {
            /// <summary>
            /// Annotation kind for names associated with slots/positions in a vector-valued column.
            /// The associated annotation type is typically fixed-sized vector of Text.
            /// </summary>
            public const string SlotNames = "SlotNames";

            /// <summary>
            /// Annotation kind for values associated with the key indices when the column type's item type
            /// is a key type. The associated annotation type is typically fixed-sized vector of a primitive
            /// type. The primitive type is frequently Text, but can be anything.
            /// </summary>
            public const string KeyValues = "KeyValues";

            /// <summary>
            /// Annotation kind for sets of score columns. The value is typically a KeyType with raw type U4.
            /// </summary>
            public const string ScoreColumnSetId = "ScoreColumnSetId";

            /// <summary>
            /// Annotation kind that indicates the prediction kind as a string. For example, "BinaryClassification".
            /// The value is typically a ReadOnlyMemory&lt;char&gt;.
            /// </summary>
            public const string ScoreColumnKind = "ScoreColumnKind";

            /// <summary>
            /// Annotation kind that indicates the value kind of the score column as a string. For example, "Score", "PredictedLabel", "Probability". The value is typically a ReadOnlyMemory.
            /// </summary>
            public const string ScoreValueKind = "ScoreValueKind";

            /// <summary>
            /// Annotation kind that indicates if a column is normalized. The value is typically a Bool.
            /// </summary>
            public const string IsNormalized = "IsNormalized";

            /// <summary>
            /// Annotation kind that indicates if a column is visible to the users. The value is typically a Bool.
            /// Not to be confused with IsHidden() that determines if a column is masked.
            /// </summary>
            public const string IsUserVisible = "IsUserVisible";

            /// <summary>
            /// Annotation kind for the label values used in training to be used for the predicted label.
            /// The value is typically a fixed-sized vector of Text.
            /// </summary>
            public const string TrainingLabelValues = "TrainingLabelValues";

            /// <summary>
            /// Annotation kind that indicates the ranges within a column that are categorical features.
            /// The value is a vector type of ints with dimension of two. The first dimension
            /// represents the number of categorical features and second dimension represents the range
            /// and is of size two. The range has start and end index(both inclusive) of categorical
            /// slots within that column.
            /// </summary>
            public const string CategoricalSlotRanges = "CategoricalSlotRanges";
        }

        /// <summary>
        /// This class holds all pre-defined string values that can be found in canonical annotations
        /// </summary>
        public static class Const
        {
            public static class ScoreColumnKind
            {
                public const string BinaryClassification = "BinaryClassification";
                public const string MulticlassClassification = "MulticlassClassification";
                public const string Regression = "Regression";
                public const string Ranking = "Ranking";
                public const string Clustering = "Clustering";
                public const string MultiOutputRegression = "MultiOutputRegression";
                public const string AnomalyDetection = "AnomalyDetection";
                public const string SequenceClassification = "SequenceClassification";
                public const string QuantileRegression = "QuantileRegression";
                public const string Recommender = "Recommender";
                public const string ItemSimilarity = "ItemSimilarity";
                public const string FeatureContribution = "FeatureContribution";
            }

            public static class ScoreValueKind
            {
                public const string Score = "Score";
                public const string PredictedLabel = "PredictedLabel";
                public const string Probability = "Probability";
            }
        }

        /// <summary>
        /// Helper delegate for marshaling from generic land to specific types. Used by the Marshal method below.
        /// </summary>
        public delegate void AnnotationGetter<TValue>(int col, ref TValue dst);

        /// <summary>
        /// Returns a standard exception for responding to an invalid call to GetAnnotation.
        /// </summary>
        public static Exception ExceptGetAnnotation() => Contracts.Except("Invalid call to GetAnnotation");

        /// <summary>
        /// Returns a standard exception for responding to an invalid call to GetAnnotation.
        /// </summary>
        public static Exception ExceptGetAnnotation(this IExceptionContext ctx) => ctx.Except("Invalid call to GetAnnotation");

        /// <summary>
        /// Helper to marshal a call to GetAnnotation{TValue} to a specific type.
        /// </summary>
        public static void Marshal<THave, TNeed>(this AnnotationGetter<THave> getter, int col, ref TNeed dst)
        {
            Contracts.CheckValue(getter, nameof(getter));

            if (typeof(TNeed) != typeof(THave))
                throw ExceptGetAnnotation();
            var get = (AnnotationGetter<TNeed>)(Delegate)getter;
            get(col, ref dst);
        }

        /// <summary>
        /// Returns a vector type with item type text and the given size. The size must be positive.
        /// This is a standard type for annotation consisting of multiple text values, eg SlotNames.
        /// </summary>
        public static VectorType GetNamesType(int size)
        {
            Contracts.CheckParam(size > 0, nameof(size), "must be known size");
            return new VectorType(TextDataViewType.Instance, size);
        }

        /// <summary>
        /// Returns a vector type with item type int and the given size.
        /// The range count must be a positive integer.
        /// This is a standard type for annotation consisting of multiple int values that represent
        /// categorical slot ranges with in a column.
        /// </summary>
        public static VectorType GetCategoricalType(int rangeCount)
        {
            Contracts.CheckParam(rangeCount > 0, nameof(rangeCount), "must be known size");
            return new VectorType(NumberDataViewType.Int32, rangeCount, 2);
        }

        private static volatile KeyType _scoreColumnSetIdType;

        /// <summary>
        /// The type of the ScoreColumnSetId annotation.
        /// </summary>
        public static KeyType ScoreColumnSetIdType
        {
            get
            {
                return _scoreColumnSetIdType ??
                    Interlocked.CompareExchange(ref _scoreColumnSetIdType, new KeyType(typeof(uint), int.MaxValue), null) ??
                    _scoreColumnSetIdType;
            }
        }

        /// <summary>
        /// Returns a key-value pair useful when implementing GetAnnotationTypes(col).
        /// </summary>
        public static KeyValuePair<string, DataViewType> GetSlotNamesPair(int size)
        {
            return GetNamesType(size).GetPair(Kinds.SlotNames);
        }

        /// <summary>
        /// Returns a key-value pair useful when implementing GetAnnotationTypes(col). This assumes
        /// that the values of the key type are Text.
        /// </summary>
        public static KeyValuePair<string, DataViewType> GetKeyNamesPair(int size)
        {
            return GetNamesType(size).GetPair(Kinds.KeyValues);
        }

        /// <summary>
        /// Given a type and annotation kind string, returns a key-value pair. This is useful when
        /// implementing GetAnnotationTypes(col).
        /// </summary>
        public static KeyValuePair<string, DataViewType> GetPair(this DataViewType type, string kind)
        {
            Contracts.CheckValue(type, nameof(type));
            return new KeyValuePair<string, DataViewType>(kind, type);
        }

        // REVIEW: This should be in some general utility code.

        /// <summary>
        /// Prepends a params array to an enumerable. Useful when implementing GetAnnotationTypes.
        /// </summary>
        public static IEnumerable<T> Prepend<T>(this IEnumerable<T> tail, params T[] head)
        {
            return head.Concat(tail);
        }

        /// <summary>
        /// Returns the max value for the specified annotation kind.
        /// The annotation type should be a KeyType with raw type U4.
        /// colMax will be set to the first column that has the max value for the specified annotation.
        /// If no column has the specified annotation, colMax is set to -1 and the method returns zero.
        /// The filter function is called for each column, passing in the schema and the column index, and returns
        /// true if the column should be considered, false if the column should be skipped.
        /// </summary>
        public static uint GetMaxAnnotationKind(this DataViewSchema schema, out int colMax, string annotationKind, Func<DataViewSchema, int, bool> filterFunc = null)
        {
            uint max = 0;
            colMax = -1;
            for (int col = 0; col < schema.Count; col++)
            {
                var columnType = schema[col].Annotations.Schema.GetColumnOrNull(annotationKind)?.Type;
                if (!(columnType is KeyType) || columnType.RawType != typeof(uint))
                    continue;
                if (filterFunc != null && !filterFunc(schema, col))
                    continue;
                uint value = 0;
                schema[col].Annotations.GetValue(annotationKind, ref value);
                if (max < value)
                {
                    max = value;
                    colMax = col;
                }
            }
            return max;
        }

        /// <summary>
        /// Returns the set of column ids which match the value of specified annotation kind.
        /// The annotation type should be a KeyType with raw type U4.
        /// </summary>
        public static IEnumerable<int> GetColumnSet(this DataViewSchema schema, string annotationKind, uint value)
        {
            for (int col = 0; col < schema.Count; col++)
            {
                var columnType = schema[col].Annotations.Schema.GetColumnOrNull(annotationKind)?.Type;
                if (columnType is KeyType && columnType.RawType == typeof(uint))
                {
                    uint val = 0;
                    schema[col].Annotations.GetValue(annotationKind, ref val);
                    if (val == value)
                        yield return col;
                }
            }
        }

        /// <summary>
        /// Returns the set of column ids which match the value of specified annotation kind.
        /// The annotation type should be of type text.
        /// </summary>
        public static IEnumerable<int> GetColumnSet(this DataViewSchema schema, string annotationKind, string value)
        {
            for (int col = 0; col < schema.Count; col++)
            {
                var columnType = schema[col].Annotations.Schema.GetColumnOrNull(annotationKind)?.Type;
                if (columnType is TextDataViewType)
                {
                    ReadOnlyMemory<char> val = default;
                    schema[col].Annotations.GetValue(annotationKind, ref val);
                    if (ReadOnlyMemoryUtils.EqualsStr(value, val))
                        yield return col;
                }
            }
        }

        /// <summary>
        /// Returns <c>true</c> if the specified column:
        ///  * has a SlotNames annotation
        ///  * annotation type is VBuffer&lt;ReadOnlyMemory&lt;char&gt;&gt; of length <paramref name="vectorSize"/>.
        /// </summary>
        public static bool HasSlotNames(this DataViewSchema.Column column, int vectorSize)
        {
            if (vectorSize == 0)
                return false;

            var metaColumn = column.Annotations.Schema.GetColumnOrNull(Kinds.SlotNames);
            return
                metaColumn != null
                && metaColumn.Value.Type is VectorType vectorType
                && vectorType.Size == vectorSize
                && vectorType.ItemType is TextDataViewType;
        }

        public static void GetSlotNames(RoleMappedSchema schema, RoleMappedSchema.ColumnRole role, int vectorSize, ref VBuffer<ReadOnlyMemory<char>> slotNames)
        {
            Contracts.CheckValueOrNull(schema);
            Contracts.CheckParam(vectorSize >= 0, nameof(vectorSize));

            IReadOnlyList<DataViewSchema.Column> list = schema?.GetColumns(role);
            if (list?.Count != 1 || !schema.Schema[list[0].Index].HasSlotNames(vectorSize))
                VBufferUtils.Resize(ref slotNames, vectorSize, 0);
            else
                schema.Schema[list[0].Index].Annotations.GetValue(Kinds.SlotNames, ref slotNames);
        }

        public static bool NeedsSlotNames(this SchemaShape.Column col)
        {
            return col.Annotations.TryFindColumn(Kinds.KeyValues, out var metaCol)
                && metaCol.Kind == SchemaShape.Column.VectorKind.Vector
                && metaCol.ItemType is TextDataViewType;
        }

        /// <summary>
        /// Returns whether a column has the <see cref="Kinds.IsNormalized"/> annotation indicated by
        /// the schema shape.
        /// </summary>
        /// <param name="column">The schema shape column to query</param>
        /// <returns>True if and only if the column has the <see cref="Kinds.IsNormalized"/> annotation
        /// of a scalar <see cref="BooleanDataViewType"/> type, which we assume, if set, should be <c>true</c>.</returns>
        public static bool IsNormalized(this SchemaShape.Column column)
        {
            Contracts.CheckParam(column.IsValid, nameof(column), "struct not initialized properly");
            return column.Annotations.TryFindColumn(Kinds.IsNormalized, out var metaCol)
                && metaCol.Kind == SchemaShape.Column.VectorKind.Scalar && !metaCol.IsKey
                && metaCol.ItemType == BooleanDataViewType.Instance;
        }

        /// <summary>
        /// Returns whether a column has the <see cref="Kinds.SlotNames"/> annotation indicated by
        /// the schema shape.
        /// </summary>
        /// <param name="col">The schema shape column to query</param>
        /// <returns>True if and only if the column is a definite sized vector type, has the
        /// <see cref="Kinds.SlotNames"/> annotation of definite sized vectors of text.</returns>
        public static bool HasSlotNames(this SchemaShape.Column col)
        {
            Contracts.CheckParam(col.IsValid, nameof(col), "struct not initialized properly");
            return col.Kind == SchemaShape.Column.VectorKind.Vector
                && col.Annotations.TryFindColumn(Kinds.SlotNames, out var metaCol)
                && metaCol.Kind == SchemaShape.Column.VectorKind.Vector && !metaCol.IsKey
                && metaCol.ItemType == TextDataViewType.Instance;
        }

        /// <summary>
        /// Tries to get the annotation kind of the specified type for a column.
        /// </summary>
        /// <typeparam name="T">The raw type of the annotation, should match the PrimitiveType type</typeparam>
        /// <param name="schema">The schema</param>
        /// <param name="type">The type of the annotation</param>
        /// <param name="kind">The annotation kind</param>
        /// <param name="col">The column</param>
        /// <param name="value">The value to return, if successful</param>
        /// <returns>True if the annotation of the right type exists, false otherwise</returns>
        public static bool TryGetAnnotation<T>(this DataViewSchema schema, PrimitiveDataViewType type, string kind, int col, ref T value)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(type, nameof(type));

            var annotationType = schema[col].Annotations.Schema.GetColumnOrNull(kind)?.Type;
            if (!type.Equals(annotationType))
                return false;
            schema[col].Annotations.GetValue(kind, ref value);
            return true;
        }

        /// <summary>
        /// The categoricalFeatures is a vector of the indices of categorical features slots.
        /// This vector should always have an even number of elements, and the elements should be parsed in groups of two consecutive numbers.
        /// So if its value is the range of numbers: 0,2,3,4,8,9
        /// look at it as [0,2],[3,4],[8,9].
        /// The way to interpret that is: feature with indices 0, 1, and 2 are one categorical
        /// Features with indices 3 and 4 are another categorical. Features 5 and 6 don't appear there, so they are not categoricals.
        /// </summary>
        public static bool TryGetCategoricalFeatureIndices(DataViewSchema schema, int colIndex, out int[] categoricalFeatures)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Check(colIndex >= 0, nameof(colIndex));

            bool isValid = false;
            categoricalFeatures = null;
            if (!(schema[colIndex].Type is VectorType vecType && vecType.Size > 0))
                return isValid;

            var type = schema[colIndex].Annotations.Schema.GetColumnOrNull(Kinds.CategoricalSlotRanges)?.Type;
            if (type?.RawType == typeof(VBuffer<int>))
            {
                VBuffer<int> catIndices = default(VBuffer<int>);
                schema[colIndex].Annotations.GetValue(Kinds.CategoricalSlotRanges, ref catIndices);
                VBufferUtils.Densify(ref catIndices);
                int columnSlotsCount = vecType.Size;
                if (catIndices.Length > 0 && catIndices.Length % 2 == 0 && catIndices.Length <= columnSlotsCount * 2)
                {
                    int previousEndIndex = -1;
                    isValid = true;
                    var catIndicesValues = catIndices.GetValues();
                    for (int i = 0; i < catIndicesValues.Length; i += 2)
                    {
                        if (catIndicesValues[i] > catIndicesValues[i + 1] ||
                            catIndicesValues[i] <= previousEndIndex ||
                            catIndicesValues[i] >= columnSlotsCount ||
                            catIndicesValues[i + 1] >= columnSlotsCount)
                        {
                            isValid = false;
                            break;
                        }

                        previousEndIndex = catIndicesValues[i + 1];
                    }
                    if (isValid)
                        categoricalFeatures = catIndicesValues.ToArray();
                }
            }

            return isValid;
        }

        /// <summary>
        /// Produces sequence of columns that are generated by trainer estimators.
        /// </summary>
        /// <param name="isNormalized">whether we should also append 'IsNormalized' (typically for probability column)</param>
        public static IEnumerable<SchemaShape.Column> GetTrainerOutputAnnotation(bool isNormalized = false)
        {
            var cols = new List<SchemaShape.Column>();
            cols.Add(new SchemaShape.Column(Kinds.ScoreColumnSetId, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true));
            cols.Add(new SchemaShape.Column(Kinds.ScoreColumnKind, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false));
            cols.Add(new SchemaShape.Column(Kinds.ScoreValueKind, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false));
            if (isNormalized)
                cols.Add(new SchemaShape.Column(Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false));
            return cols;
        }

        /// <summary>
        /// Produces annotations for the score column generated by trainer estimators for multiclass classification.
        /// If input LabelColumn is not available it produces slotnames annotation by default.
        /// </summary>
        /// <param name="labelColumn">Label column.</param>
        public static IEnumerable<SchemaShape.Column> AnnotationsForMulticlassScoreColumn(SchemaShape.Column? labelColumn = null)
        {
            var cols = new List<SchemaShape.Column>();
            if (labelColumn != null && labelColumn.Value.IsKey && NeedsSlotNames(labelColumn.Value))
                cols.Add(new SchemaShape.Column(Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false));
            cols.AddRange(GetTrainerOutputAnnotation());
            return cols;
        }

        private sealed class AnnotationRow : DataViewRow
        {
            private readonly DataViewSchema.Annotations _annotations;

            public AnnotationRow(DataViewSchema.Annotations annotations)
            {
                Contracts.AssertValue(annotations);
                _annotations = annotations;
            }

            public override DataViewSchema Schema => _annotations.Schema;
            public override long Position => 0;
            public override long Batch => 0;

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column) => _annotations.GetGetter<TValue>(column);

            public override ValueGetter<DataViewRowId> GetIdGetter() => (ref DataViewRowId dst) => dst = default;

            /// <summary>
            /// Returns whether the given column is active in this row.
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column) => true;
        }

        /// <summary>
        /// Presents a <see cref="DataViewSchema.Annotations"/> as a an <see cref="DataViewRow"/>.
        /// </summary>
        /// <param name="annotations">The annotations to wrap.</param>
        /// <returns>A row that wraps an input annotations.</returns>
        [BestFriend]
        internal static DataViewRow AnnotationsAsRow(DataViewSchema.Annotations annotations)
        {
            Contracts.CheckValue(annotations, nameof(annotations));
            return new AnnotationRow(annotations);
        }
    }
}