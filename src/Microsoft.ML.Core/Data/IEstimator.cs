// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Core.Data
{
    /// <summary>
    /// A set of 'requirements' to the incoming schema, as well as a set of 'promises' of the outgoing schema.
    /// This is more relaxed than the proper <see cref="ISchema"/>, since it's only a subset of the columns,
    /// and also since it doesn't specify exact <see cref="ColumnType"/>'s for vectors and keys.
    /// </summary>
    public sealed class SchemaShape
    {
        public readonly Column[] Columns;

        public sealed class Column
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
            /// The metadata kinds that are present for this column.
            /// </summary>
            public readonly string[] MetadataKinds;

            public Column(string name, VectorKind vecKind, ColumnType itemType, bool isKey, string[] metadataKinds = null)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValueOrNull(metadataKinds);
                Contracts.CheckParam(!itemType.IsKey, nameof(itemType), "Item type cannot be a key");
                Contracts.CheckParam(!itemType.IsVector, nameof(itemType), "Item type cannot be a vector");

                Contracts.CheckParam(!isKey || KeyType.IsValidDataKind(itemType.RawKind), nameof(itemType), "The item type must be valid for a key");

                Name = name;
                Kind = vecKind;
                ItemType = itemType;
                IsKey = isKey;
                MetadataKinds = metadataKinds ?? new string[0];
            }

            /// <summary>
            /// Returns whether <paramref name="inputColumn"/> is a valid input, if this object represents a
            /// requirement.
            ///
            /// Namely, it returns true iff:
            ///  - The <see cref="Name"/>, <see cref="Kind"/>, <see cref="ItemType"/>, <see cref="IsKey"/> fields match.
            ///  - The <see cref="MetadataKinds"/> of <paramref name="inputColumn"/> is a superset of our <see cref="MetadataKinds"/>.
            /// </summary>
            public bool IsCompatibleWith(Column inputColumn)
            {
                Contracts.CheckValue(inputColumn, nameof(inputColumn));
                if (Name != inputColumn.Name)
                    return false;
                if (Kind != inputColumn.Kind)
                    return false;
                if (!ItemType.Equals(inputColumn.ItemType))
                    return false;
                if (IsKey != inputColumn.IsKey)
                    return false;
                if (inputColumn.MetadataKinds.Except(MetadataKinds).Any())
                    return false;
                return true;
            }

            public string GetTypeString()
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
        }

        public SchemaShape(IEnumerable<Column> columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            Columns = columns.ToArray();
        }

        /// <summary>
        /// Create a schema shape out of the fully defined schema.
        /// </summary>
        public static SchemaShape Create(ISchema schema)
        {
            Contracts.CheckValue(schema, nameof(schema));
            var cols = new List<Column>();

            for (int iCol = 0; iCol < schema.ColumnCount; iCol++)
            {
                if (!schema.IsHidden(iCol))
                {
                    Column.VectorKind vecKind;
                    var type = schema.GetColumnType(iCol);
                    if (type.IsKnownSizeVector)
                        vecKind = Column.VectorKind.Vector;
                    else if (type.IsVector)
                        vecKind = Column.VectorKind.VariableVector;
                    else
                        vecKind = Column.VectorKind.Scalar;

                    ColumnType itemType = type.ItemType;
                    if (type.ItemType.IsKey)
                        itemType = PrimitiveType.FromKind(type.ItemType.RawKind);
                    var isKey = type.ItemType.IsKey;

                    var metadataNames = schema.GetMetadataTypes(iCol)
                        .Select(kvp => kvp.Key)
                        .ToArray();
                    cols.Add(new Column(schema.GetColumnName(iCol), vecKind, itemType, isKey, metadataNames));
                }
            }
            return new SchemaShape(cols.ToArray());
        }

        /// <summary>
        /// Returns the column with a specified <paramref name="name"/>, and <c>null</c> if there is no such column.
        /// </summary>
        public Column FindColumn(string name)
        {
            Contracts.CheckValue(name, nameof(name));
            return Columns.FirstOrDefault(x => x.Name == name);
        }

        // REVIEW: I think we should have an IsCompatible method to check if it's OK to use one schema shape
        // as an input to another schema shape. I started writing, but realized that there's more than one way to check for
        // the 'compatibility': as in, 'CAN be compatible' vs. 'WILL be compatible'.
    }

    /// <summary>
    /// Exception class for schema validation errors.
    /// </summary>
    public class SchemaException : Exception
    {
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
        ISchema GetOutputSchema();
    }

    /// <summary>
    /// Sometimes we need to 'fit' an <see cref="IDataReader{TIn}"/>.
    /// A DataReader estimator is the object that does it.
    /// </summary>
    public interface IDataReaderEstimator<in TSource, out TReader>
        where TReader : IDataReader<TSource>
    {
        /// <summary>
        /// Train and return a data reader.
        ///
        /// REVIEW: you could consider the transformer to take a different <typeparamref name="TSource"/>, but we don't have such components
        /// yet, so why complicate matters?
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
    /// It also supports 'schema propagation' to answer the question of 'how the data with this schema look after you transform it?'.
    /// </summary>
    public interface ITransformer
    {
        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// Throws <see cref="SchemaException"/> iff the input schema is not valid for the transformer.
        /// </summary>
        ISchema GetOutputSchema(ISchema inputSchema);

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        IDataView Transform(IDataView input);
    }

    /// <summary>
    /// The estimator (in Spark terminology) is an 'untrained transformer'. It needs to 'fit' on the data to manufacture
    /// a transformer.
    /// It also provides the 'schema propagation' like transformers do, but over <see cref="SchemaShape"/> instead of <see cref="ISchema"/>.
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
        /// Throws <see cref="SchemaException"/> iff the input schema is not valid for the estimator.
        /// </summary>
        SchemaShape GetOutputSchema(SchemaShape inputSchema);
    }
}
