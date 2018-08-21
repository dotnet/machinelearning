// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Utilities for implementing and using the metadata API of ISchema.
    /// </summary>
    public static class MetadataUtils
    {
        /// <summary>
        /// This class lists the canonical metadata kinds
        /// </summary>
        public static class Kinds
        {
            /// <summary>
            /// Metadata kind for names associated with slots/positions in a vector-valued column.
            /// The associated metadata type is typically fixed-sized vector of Text.
            /// </summary>
            public const string SlotNames = "SlotNames";

            /// <summary>
            /// Metadata kind for values associated with the key indices when the column type's item type
            /// is a key type. The associated metadata type is typically fixed-sized vector of a primitive
            /// type. The primitive type is frequently Text, but can be anything.
            /// </summary>
            public const string KeyValues = "KeyValues";

            /// <summary>
            /// Metadata kind for sets of score columns. The value is typically a KeyType with raw type U4.
            /// </summary>
            public const string ScoreColumnSetId = "ScoreColumnSetId";

            /// <summary>
            /// Metadata kind that indicates the prediction kind as a string. E.g. "BinaryClassification". The value is typically a ReadOnlyMemory.
            /// </summary>
            public const string ScoreColumnKind = "ScoreColumnKind";

            /// <summary>
            /// Metadata kind that indicates the value kind of the score column as a string. E.g. "Score", "PredictedLabel", "Probability". The value is typically a ReadOnlyMemory.
            /// </summary>
            public const string ScoreValueKind = "ScoreValueKind";

            /// <summary>
            /// Metadata kind that indicates if a column is normalized. The value is typically a Bool.
            /// </summary>
            public const string IsNormalized = "IsNormalized";

            /// <summary>
            /// Metadata kind that indicates if a column is visible to the users. The value is typically a Bool.
            /// Not to be confused with IsHidden() that determines if a column is masked.
            /// </summary>
            public const string IsUserVisible = "IsUserVisible";

            /// <summary>
            /// Metadata kind that indicates if a column has missing values. The value is typically a Bool to allow for unknown status.
            /// </summary>
            public const string HasMissingValues = "HasMissingValues";

            /// <summary>
            /// Metadata kind for the label values used in training to be used for the predicted label.
            /// The value is typically a fixed-sized vector of Text.
            /// </summary>
            public const string TrainingLabelValues = "TrainingLabelValues";

            /// <summary>
            /// Metadata kind that indicates the ranges within a column that are categorical features.
            /// The value is a vector type of ints with dimension of two. The first dimension
            /// represents the number of categorical features and second dimension represents the range
            /// and is of size two. The range has start and end index(both inclusive) of categorical
            /// slots within that column.
            /// </summary>
            public const string CategoricalSlotRanges = "CategoricalSlotRanges";
        }

        /// <summary>
        /// This class holds all pre-defined string values that can be found in canonical metadata
        /// </summary>
        public static class Const
        {
            public static class ScoreColumnKind
            {
                public const string BinaryClassification = "BinaryClassification";
                public const string MultiClassClassification = "MultiClassClassification";
                public const string Regression = "Regression";
                public const string Ranking = "Ranking";
                public const string Clustering = "Clustering";
                public const string MultiOutputRegression = "MultiOutputRegression";
                public const string AnomalyDetection = "AnomalyDetection";
                public const string SequenceClassification = "SequenceClassification";
                public const string QuantileRegression = "QuantileRegression";
                public const string Recommender = "Recommender";
                public const string ItemSimilarity = "ItemSimilarity";
                public const string WhatTheFeature = "WhatTheFeature";
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
        public delegate void MetadataGetter<TValue>(int col, ref TValue dst);

        /// <summary>
        /// Returns a standard exception for responding to an invalid call to GetMetadata.
        /// </summary>
        public static Exception ExceptGetMetadata()
        {
            return Contracts.Except("Invalid call to GetMetadata");
        }

        /// <summary>
        /// Returns a standard exception for responding to an invalid call to GetMetadata.
        /// </summary>
        public static Exception ExceptGetMetadata(this IExceptionContext ctx)
        {
            return ctx.Except("Invalid call to GetMetadata");
        }

        /// <summary>
        /// Helper to marshal a call to GetMetadata{TValue} to a specific type.
        /// </summary>
        public static void Marshal<THave, TNeed>(this MetadataGetter<THave> getter, int col, ref TNeed dst)
        {
            Contracts.CheckValue(getter, nameof(getter));

            if (typeof(TNeed) != typeof(THave))
                throw ExceptGetMetadata();
            var get = (MetadataGetter<TNeed>)(Delegate)getter;
            get(col, ref dst);
        }

        /// <summary>
        /// Returns a vector type with item type text and the given size. The size must be positive.
        /// This is a standard type for metadata consisting of multiple text values, eg SlotNames.
        /// </summary>
        public static VectorType GetNamesType(int size)
        {
            Contracts.CheckParam(size > 0, nameof(size), "must be known size");
            return new VectorType(TextType.Instance, size);
        }

        /// <summary>
        /// Returns a vector type with item type int and the given size.
        /// The range count must be a positive integer.
        /// This is a standard type for metadata consisting of multiple int values that represent
        /// categorical slot ranges with in a column.
        /// </summary>
        public static VectorType GetCategoricalType(int rangeCount)
        {
            Contracts.CheckParam(rangeCount > 0, nameof(rangeCount), "must be known size");
            return new VectorType(NumberType.I4, rangeCount, 2);
        }

        private static volatile ColumnType _scoreColumnSetIdType;

        /// <summary>
        /// The type of the ScoreColumnSetId metadata.
        /// </summary>
        public static ColumnType ScoreColumnSetIdType
        {
            get
            {
                if (_scoreColumnSetIdType == null)
                {
                    var type = new KeyType(DataKind.U4, 0, 0);
                    Interlocked.CompareExchange(ref _scoreColumnSetIdType, type, null);
                }
                return _scoreColumnSetIdType;
            }
        }

        /// <summary>
        /// Returns a key-value pair useful when implementing GetMetadataTypes(col).
        /// </summary>
        public static KeyValuePair<string, ColumnType> GetSlotNamesPair(int size)
        {
            return GetNamesType(size).GetPair(Kinds.SlotNames);
        }

        /// <summary>
        /// Returns a key-value pair useful when implementing GetMetadataTypes(col). This assumes
        /// that the values of the key type are Text.
        /// </summary>
        public static KeyValuePair<string, ColumnType> GetKeyNamesPair(int size)
        {
            return GetNamesType(size).GetPair(Kinds.KeyValues);
        }

        /// <summary>
        /// Given a type and metadata kind string, returns a key-value pair. This is useful when
        /// implementing GetMetadataTypes(col).
        /// </summary>
        public static KeyValuePair<string, ColumnType> GetPair(this ColumnType type, string kind)
        {
            Contracts.CheckValue(type, nameof(type));
            return new KeyValuePair<string, ColumnType>(kind, type);
        }

        // REVIEW: This should be in some general utility code.

        /// <summary>
        /// Prepends a params array to an enumerable. Useful when implementing GetMetadataTypes.
        /// </summary>
        public static IEnumerable<T> Prepend<T>(this IEnumerable<T> tail, params T[] head)
        {
            return head.Concat(tail);
        }

        /// <summary>
        /// Returns the max value for the specified metadata kind.
        /// The metadata type should be a KeyType with raw type U4.
        /// colMax will be set to the first column that has the max value for the specified metadata.
        /// If no column has the specified metadata, colMax is set to -1 and the method returns zero.
        /// The filter function is called for each column, passing in the schema and the column index, and returns
        /// true if the column should be considered, false if the column should be skipped.
        /// </summary>
        public static uint GetMaxMetadataKind(this ISchema schema, out int colMax, string metadataKind, Func<ISchema, int, bool> filterFunc = null)
        {
            uint max = 0;
            colMax = -1;
            for (int col = 0; col < schema.ColumnCount; col++)
            {
                var columnType = schema.GetMetadataTypeOrNull(metadataKind, col);
                if (columnType == null || !columnType.IsKey || columnType.RawKind != DataKind.U4)
                    continue;
                if (filterFunc != null && !filterFunc(schema, col))
                    continue;
                uint value = 0;
                schema.GetMetadata(metadataKind, col, ref value);
                if (max < value)
                {
                    max = value;
                    colMax = col;
                }
            }
            return max;
        }

        /// <summary>
        /// Returns the set of column ids which match the value of specified metadata kind.
        /// The metadata type should be a KeyType with raw type U4.
        /// </summary>
        public static IEnumerable<int> GetColumnSet(this ISchema schema, string metadataKind, uint value)
        {
            for (int col = 0; col < schema.ColumnCount; col++)
            {
                var columnType = schema.GetMetadataTypeOrNull(metadataKind, col);
                if (columnType != null && columnType.IsKey && columnType.RawKind == DataKind.U4)
                {
                    uint val = 0;
                    schema.GetMetadata(metadataKind, col, ref val);
                    if (val == value)
                        yield return col;
                }
            }
        }

        /// <summary>
        /// Returns the set of column ids which match the value of specified metadata kind.
        /// The metadata type should be of type text.
        /// </summary>
        public static IEnumerable<int> GetColumnSet(this ISchema schema, string metadataKind, string value)
        {
            for (int col = 0; col < schema.ColumnCount; col++)
            {
                var columnType = schema.GetMetadataTypeOrNull(metadataKind, col);
                if (columnType != null && columnType.IsText)
                {
                    ReadOnlyMemory<char> val = default;
                    schema.GetMetadata(metadataKind, col, ref val);
                    if (ReadOnlyMemoryUtils.EqualsStr(value, val))
                        yield return col;
                }
            }
        }

        /// <summary>
        /// Returns <c>true</c> if the specified column:
        ///  * is a vector of length N (including 0)
        ///  * has a SlotNames metadata
        ///  * metadata type is VBuffer&lt;ReadOnlyMemory&gt; of length N
        /// </summary>
        public static bool HasSlotNames(this ISchema schema, int col, int vectorSize)
        {
            if (vectorSize == 0)
                return false;

            var type = schema.GetMetadataTypeOrNull(Kinds.SlotNames, col);
            return
                type != null
                && type.IsVector
                && type.VectorSize == vectorSize
                && type.ItemType.IsText;
        }

        public static void GetSlotNames(RoleMappedSchema schema, RoleMappedSchema.ColumnRole role, int vectorSize, ref VBuffer<ReadOnlyMemory<char>> slotNames)
        {
            Contracts.CheckValueOrNull(schema);
            Contracts.CheckParam(vectorSize >= 0, nameof(vectorSize));

            IReadOnlyList<ColumnInfo> list;
            if ((list = schema?.GetColumns(role)) == null || list.Count != 1 || !schema.Schema.HasSlotNames(list[0].Index, vectorSize))
                slotNames = new VBuffer<ReadOnlyMemory<char>>(vectorSize, 0, slotNames.Values, slotNames.Indices);
            else
                schema.Schema.GetMetadata(Kinds.SlotNames, list[0].Index, ref slotNames);
        }

        public static bool HasKeyNames(this ISchema schema, int col, int keyCount)
        {
            if (keyCount == 0)
                return false;

            var type = schema.GetMetadataTypeOrNull(Kinds.KeyValues, col);
            return
                type != null
                && type.IsVector
                && type.VectorSize == keyCount
                && type.ItemType.IsText;
        }

        /// <summary>
        /// Returns whether a column has the <see cref="Kinds.IsNormalized"/> metadata set to true.
        /// That metadata should be set when the data has undergone transforms that would render it
        /// "normalized."
        /// </summary>
        /// <param name="schema">The schema to query</param>
        /// <param name="col">Which column in the schema to query</param>
        /// <returns>True if and only if the column has the <see cref="Kinds.IsNormalized"/> metadata
        /// set to the scalar value <see cref="DvBool.True"/></returns>
        public static bool IsNormalized(this ISchema schema, int col)
        {
            Contracts.CheckValue(schema, nameof(schema));
            var value = default(DvBool);
            return schema.TryGetMetadata(BoolType.Instance, Kinds.IsNormalized, col, ref value) && value.IsTrue;
        }

        /// <summary>
        /// Tries to get the metadata kind of the specified type for a column.
        /// </summary>
        /// <typeparam name="T">The raw type of the metadata, should match the PrimitiveType type</typeparam>
        /// <param name="schema">The schema</param>
        /// <param name="type">The type of the metadata</param>
        /// <param name="kind">The metadata kind</param>
        /// <param name="col">The column</param>
        /// <param name="value">The value to return, if successful</param>
        /// <returns>True if the metadata of the right type exists, false otherwise</returns>
        public static bool TryGetMetadata<T>(this ISchema schema, PrimitiveType type, string kind, int col, ref T value)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(type, nameof(type));

            var metadataType = schema.GetMetadataTypeOrNull(kind, col);
            if (!type.Equals(metadataType))
                return false;
            schema.GetMetadata(kind, col, ref value);
            return true;
        }

        /// <summary>
        /// Return whether the given column index is hidden in the given schema.
        /// </summary>
        public static bool IsHidden(this ISchema schema, int col)
        {
            Contracts.CheckValue(schema, nameof(schema));
            string name = schema.GetColumnName(col);
            int top;
            bool tmp = schema.TryGetColumnIndex(name, out top);
            Contracts.Assert(tmp); // This would only be false if the implementation of schema were buggy.
            return !tmp || top != col;
        }

        /// <summary>
        /// The categoricalFeatures is a vector of the indices of categorical features slots.
        /// This vector should always have an even number of elements, and the elements should be parsed in groups of two consecutive numbers.
        /// So if its value is the range of numbers: 0,2,3,4,8,9
        /// look at it as [0,2],[3,4],[8,9].
        /// The way to interpret that is: feature with indices 0, 1, and 2 are one categorical
        /// Features with indices 3 and 4 are another categorical. Features 5 and 6 don't appear there, so they are not categoricals.
        /// </summary>
        public static bool TryGetCategoricalFeatureIndices(ISchema schema, int colIndex, out int[] categoricalFeatures)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Check(colIndex >= 0, nameof(colIndex));

            bool isValid = false;
            categoricalFeatures = null;
            if (!schema.GetColumnType(colIndex).IsKnownSizeVector)
                return isValid;

            var type = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.CategoricalSlotRanges, colIndex);
            if (type?.RawType == typeof(VBuffer<DvInt4>))
            {
                VBuffer<DvInt4> catIndices = default(VBuffer<DvInt4>);
                schema.GetMetadata(MetadataUtils.Kinds.CategoricalSlotRanges, colIndex, ref catIndices);
                VBufferUtils.Densify(ref catIndices);
                int columnSlotsCount = schema.GetColumnType(colIndex).AsVector.VectorSizeCore;
                if (catIndices.Length > 0 && catIndices.Length % 2 == 0 && catIndices.Length <= columnSlotsCount * 2)
                {
                    int previousEndIndex = -1;
                    isValid = true;
                    for (int i = 0; i < catIndices.Values.Length; i += 2)
                    {
                        if (catIndices.Values[i].RawValue > catIndices.Values[i + 1].RawValue ||
                            catIndices.Values[i].RawValue <= previousEndIndex ||
                            catIndices.Values[i].RawValue >= columnSlotsCount ||
                            catIndices.Values[i + 1].RawValue >= columnSlotsCount)
                        {
                            isValid = false;
                            break;
                        }

                        previousEndIndex = catIndices.Values[i + 1].RawValue;
                    }
                    if (isValid)
                        categoricalFeatures = catIndices.Values.Select(val => val.RawValue).ToArray();
                }
            }

            return isValid;
        }
    }
}