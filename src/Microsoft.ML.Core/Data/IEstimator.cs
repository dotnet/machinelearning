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

            public readonly string Name;
            public readonly VectorKind Kind;
            public readonly DataKind ItemKind;
            public readonly bool IsKey;
            public readonly string[] MetadataKinds;

            public Column(string name, VectorKind vecKind, DataKind itemKind, bool isKey, string[] metadataKinds)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Contracts.CheckValue(metadataKinds, nameof(metadataKinds));

                Name = name;
                Kind = vecKind;
                ItemKind = itemKind;
                IsKey = isKey;
                MetadataKinds = metadataKinds;
            }
        }

        public SchemaShape(Column[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            Columns = columns;
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

                    var kind = type.ItemType.RawKind;
                    var isKey = type.ItemType.IsKey;

                    var metadataNames = schema.GetMetadataTypes(iCol)
                        .Select(kvp => kvp.Key)
                        .ToArray();
                    cols.Add(new Column(schema.GetColumnName(iCol), vecKind, kind, isKey, metadataNames));
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
