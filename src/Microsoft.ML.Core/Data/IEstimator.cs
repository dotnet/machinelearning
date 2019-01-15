// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;

namespace Microsoft.ML.Core.Data
{
    /// <summary>
    /// A set of 'requirements' to the incoming schema, as well as a set of 'promises' of the outgoing schema.
    /// This is more relaxed than the proper <see cref="Schema"/>, since it's only a subset of the columns,
    /// and also since it doesn't specify exact <see cref="ColumnType"/>'s for vectors and keys.
    /// </summary>
    public sealed class SchemaShape : IReadOnlyList<SchemaShape.Column>
    {
        private readonly Column[] _columns;

        private static readonly SchemaShape _empty = new SchemaShape(Enumerable.Empty<Column>());

        public int Count => _columns.Count();

        public Column this[int index] => _columns[index];

        public struct Column
        {
            public enum VectorKind
            {
                Scalar,
                Vector,
                VariableVector
            }

            /// <summary>
            /// The column name.
            /// </summary>
            public readonly string Name;

            /// <summary>
            /// The type of the column: scalar, fixed vector or variable vector.
            /// </summary>
            public readonly VectorKind Kind;

            /// <summary>
            /// The 'raw' type of column item: must be a primitive type or a structured type.
            /// </summary>
            public readonly ColumnType ItemType;
            /// <summary>
            /// The flag whether the column is actually a key. If yes, <see cref="ItemType"/> is representing
            /// the underlying primitive type.
            /// </summary>
            public readonly bool IsKey;
            /// <summary>
            /// The metadata that is present for this column.
            /// </summary>
            public readonly SchemaShape Metadata;

            [BestFriend]
            internal Column(string name, VectorKind vecKind, ColumnType itemType, bool isKey, SchemaShape metadata = null)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValueOrNull(metadata);
                Contracts.CheckParam(!(itemType is KeyType), nameof(itemType), "Item type cannot be a key");
                Contracts.CheckParam(!itemType.IsVector, nameof(itemType), "Item type cannot be a vector");
                Contracts.CheckParam(!isKey || KeyType.IsValidDataKind(itemType.RawKind), nameof(itemType), "The item type must be valid for a key");

                Name = name;
                Kind = vecKind;
                ItemType = itemType;
                IsKey = isKey;
                Metadata = metadata ?? _empty;
            }

            /// <summary>
            /// Returns whether <paramref name="inputColumn"/> is a valid input, if this object represents a
            /// requirement.
            ///
            /// Namely, it returns true iff:
            ///  - The <see cref="Name"/>, <see cref="Kind"/>, <see cref="ItemType"/>, <see cref="IsKey"/> fields match.
            ///  - The columns of <see cref="Metadata"/> of <paramref name="inputColumn"/> is a superset of our <see cref="Metadata"/> columns.
            ///  - Each such metadata column is itself compatible with the input metadata column.
            /// </summary>
            [BestFriend]
            internal bool IsCompatibleWith(Column inputColumn)
            {
                Contracts.Check(inputColumn.IsValid, nameof(inputColumn));
                if (Name != inputColumn.Name)
                    return false;
                if (Kind != inputColumn.Kind)
                    return false;
                if (!ItemType.Equals(inputColumn.ItemType))
                    return false;
                if (IsKey != inputColumn.IsKey)
                    return false;
                foreach (var metaCol in Metadata)
                {
                    if (!inputColumn.Metadata.TryFindColumn(metaCol.Name, out var inputMetaCol))
                        return false;
                    if (!metaCol.IsCompatibleWith(inputMetaCol))
                        return false;
                }
                return true;
            }

            [BestFriend]
            internal string GetTypeString()
            {
                string result = ItemType.ToString();
                if (IsKey)
                    result = $"Key<{result}>";
                if (Kind == VectorKind.Vector)
                    result = $"Vector<{result}>";
                else if (Kind == VectorKind.VariableVector)
                    result = $"VarVector<{result}>";
                return result;
            }

            /// <summary>
            /// Return if this structure is not identical to the default value of <see cref="Column"/>. If true,
            /// it means this structure is initialized properly and therefore considered as valid.
            /// </summary>
            [BestFriend]
            internal bool IsValid => Name != null;
        }

        public SchemaShape(IEnumerable<Column> columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            _columns = columns.ToArray();
            Contracts.CheckParam(columns.All(c => c.IsValid), nameof(columns), "Some items are not initialized properly.");
        }

        /// <summary>
        /// Given a <paramref name="type"/>, extract the type parameters that describe this type
        /// as a <see cref="SchemaShape"/>'s column type.
        /// </summary>
        /// <param name="type">The actual column type to process.</param>
        /// <param name="vecKind">The vector kind of <paramref name="type"/>.</param>
        /// <param name="itemType">The item type of <paramref name="type"/>.</param>
        /// <param name="isKey">Whether <paramref name="type"/> (or its item type) is a key.</param>
        [BestFriend]
        internal static void GetColumnTypeShape(ColumnType type,
            out Column.VectorKind vecKind,
            out ColumnType itemType,
            out bool isKey)
        {
            if (type.IsKnownSizeVector)
                vecKind = Column.VectorKind.Vector;
            else if (type.IsVector)
                vecKind = Column.VectorKind.VariableVector;
            else
                vecKind = Column.VectorKind.Scalar;

            itemType = type.ItemType;
            isKey = type.ItemType is KeyType;
            if (isKey)
                itemType = PrimitiveType.FromKind(type.ItemType.RawKind);
        }

        /// <summary>
        /// Create a schema shape out of the fully defined schema.
        /// </summary>
        [BestFriend]
        internal static SchemaShape Create(Schema schema)
        {
            Contracts.CheckValue(schema, nameof(schema));
            var cols = new List<Column>();

            for (int iCol = 0; iCol < schema.Count; iCol++)
            {
                if (!schema[iCol].IsHidden)
                {
                    // First create the metadata.
                    var mCols = new List<Column>();
                    foreach (var metaColumn in schema[iCol].Metadata.Schema)
                    {
                        GetColumnTypeShape(metaColumn.Type, out var mVecKind, out var mItemType, out var mIsKey);
                        mCols.Add(new Column(metaColumn.Name, mVecKind, mItemType, mIsKey));
                    }
                    var metadata = mCols.Count > 0 ? new SchemaShape(mCols) : _empty;
                    // Next create the single column.
                    GetColumnTypeShape(schema[iCol].Type, out var vecKind, out var itemType, out var isKey);
                    cols.Add(new Column(schema[iCol].Name, vecKind, itemType, isKey, metadata));
                }
            }
            return new SchemaShape(cols);
        }

        /// <summary>
        /// Returns if there is a column with a specified <paramref name="name"/> and if so stores it in <paramref name="column"/>.
        /// </summary>
        [BestFriend]
        internal bool TryFindColumn(string name, out Column column)
        {
            Contracts.CheckValue(name, nameof(name));
            column = _columns.FirstOrDefault(x => x.Name == name);
            return column.IsValid;
        }

        public IEnumerator<Column> GetEnumerator() => ((IEnumerable<Column>)_columns).GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        // REVIEW: I think we should have an IsCompatible method to check if it's OK to use one schema shape
        // as an input to another schema shape. I started writing, but realized that there's more than one way to check for
        // the 'compatibility': as in, 'CAN be compatible' vs. 'WILL be compatible'.
    }

    /// <summary>
    /// The 'data reader' takes a certain kind of input and turns it into an <see cref="IDataView"/>.
    /// </summary>
    /// <typeparam name="TSource">The type of input the reader takes.</typeparam>
    public interface IDataReader<in TSource>
    {
        /// <summary>
        /// Produce the data view from the specified input.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual reading happens here, just schema validation.
        /// </summary>
        IDataView Read(TSource input);

        /// <summary>
        /// The output schema of the reader.
        /// </summary>
        Schema GetOutputSchema();
    }

    /// <summary>
    /// Sometimes we need to 'fit' an <see cref="IDataReader{TIn}"/>.
    /// A DataReader estimator is the object that does it.
    /// </summary>
    public interface IDataReaderEstimator<in TSource, out TReader>
        where TReader : IDataReader<TSource>
    {
        // REVIEW: you could consider the transformer to take a different <typeparamref name="TSource"/>, but we don't have such components
        // yet, so why complicate matters?
        /// <summary>
        /// Train and return a data reader.
        /// </summary>
        TReader Fit(TSource input);

        /// <summary>
        /// The 'promise' of the output schema.
        /// It will be used for schema propagation.
        /// </summary>
        SchemaShape GetOutputSchema();
    }

    /// <summary>
    /// The transformer is a component that transforms data.
    /// It also supports 'schema propagation' to answer the question of 'how will the data with this schema look, after you transform it?'.
    /// </summary>
    public interface ITransformer
    {
        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// </summary>
        Schema GetOutputSchema(Schema inputSchema);

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        IDataView Transform(IDataView input);

        /// <summary>
        /// Whether a call to <see cref="GetRowToRowMapper(Schema)"/> should succeed, on an
        /// appropriate schema.
        /// </summary>
        bool IsRowToRowMapper { get; }

        /// <summary>
        /// Constructs a row-to-row mapper based on an input schema. If <see cref="IsRowToRowMapper"/>
        /// is <c>false</c>, then an exception should be thrown. If the input schema is in any way
        /// unsuitable for constructing the mapper, an exception should likewise be thrown.
        /// </summary>
        /// <param name="inputSchema">The input schema for which we should get the mapper.</param>
        /// <returns>The row to row mapper.</returns>
        IRowToRowMapper GetRowToRowMapper(Schema inputSchema);
    }

    /// <summary>
    /// The estimator (in Spark terminology) is an 'untrained transformer'. It needs to 'fit' on the data to manufacture
    /// a transformer.
    /// It also provides the 'schema propagation' like transformers do, but over <see cref="SchemaShape"/> instead of <see cref="Schema"/>.
    /// </summary>
    public interface IEstimator<out TTransformer>
        where TTransformer : ITransformer
    {
        /// <summary>
        /// Train and return a transformer.
        /// </summary>
        TTransformer Fit(IDataView input);

        /// <summary>
        /// Schema propagation for estimators.
        /// Returns the output schema shape of the estimator, if the input schema shape is like the one provided.
        /// </summary>
        SchemaShape GetOutputSchema(SchemaShape inputSchema);
    }
}
