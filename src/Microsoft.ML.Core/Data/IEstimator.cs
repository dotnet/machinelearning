// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
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

            public Column(string name, VectorKind vecKind, DataKind itemKind, bool isKey)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Name = name;
                Kind = vecKind;
                ItemKind = itemKind;
                IsKey = isKey;
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
                    cols.Add(new Column(schema.GetColumnName(iCol), vecKind, kind, isKey));
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
    /// The generic transformer takes any kind of input and turns it into an <see cref="IDataView"/>.
    /// Think of this as data loaders. Data transformers are also these, but they also implement <see cref="IDataTransformer"/>.
    /// </summary>
    /// <typeparam name="TIn">The type of input the transformer takes.</typeparam>
    public interface ITransformer<TIn>
    {
        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        IDataView Transform(TIn input);

        /// <summary>
        /// The output schema of the transformer.
        /// </summary>
        ISchema GetOutputSchema();
    }

    /// <summary>
    /// Estimator is a Spark name for 'trainable component'. Like a normalizer, or an SvmLightLoader.
    /// It needs to be 'fitted' to create a <see cref="ITransformer{TIn}"/>.
    /// </summary>
    /// <typeparam name="TIn">The type of input the estimator (and eventually transformer) takes.</typeparam>
    public interface IEstimator<TIn>
    {
        /// <summary>
        /// Train and return a transformer.
        /// 
        /// REVIEW: you could consider the transformer to take a different <typeparamref name="TIn"/>, but we don't have such components
        /// yet, so why complicate matters?
        /// </summary>
        ITransformer<TIn> Fit(TIn input);

        /// <summary>
        /// The 'promise' of the output schema.
        /// It will be used for schema propagation.
        /// </summary>
        SchemaShape GetOutputSchema();
    }

    /// <summary>
    /// An estimator that provides more details about the produced transformer, in the form of <typeparamref name="TTransformer"/>.
    /// </summary>
    public interface IEstimator<TIn, TTransformer>: IEstimator<TIn>
        where TTransformer: ITransformer<TIn>
    {
        new TTransformer Fit(TIn input);
    }

    /// <summary>
    /// The data transformer, in addition to being a transformer, also exposes the input schema shape. It is handy for
    /// evaluating what kind of columns the transformer expects.
    /// </summary>
    public interface IDataTransformer
    {
        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// Returns <c>null</c> iff the schema is invalid (then a call to Transform with this data will fail).
        /// </summary>
        ISchema GetOutputSchema(ISchema inputSchema);

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// Note that <see cref="IDataView"/>'s are lazy, so no actual transformations happen here, just schema validation.
        /// </summary>
        IDataView Transform(IDataView input);
    }

    public interface IDataEstimator
    {
        /// <summary>
        /// Train and return a transformer.
        /// </summary>
        IDataTransformer Fit(IDataView input);

        /// <summary>
        /// Schema propagation for estimators.
        /// Returns the output schema shape of the estimator, if the input schema shape is like the one provided.
        /// Returns <c>null</c> iff the schema shape is invalid (then a call to <see cref="Fit"/> with this data will fail).
        /// </summary>
        SchemaShape GetOutputSchema(SchemaShape inputSchema);
    }

    /// <summary>
    /// A data estimator that provides more details about the produced transformer, in the form of <typeparamref name="TTransformer"/>.
    /// </summary>
    public interface IDataEstimator<TTransformer>: IDataEstimator
        where TTransformer: IDataTransformer
    {
        new TTransformer Fit(IDataView input);
    }
}
