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
        public readonly ColumnBase[] Columns;

        public abstract class ColumnBase
        {
            public readonly string Name;
            public ColumnBase(string name)
            {
                Contracts.CheckNonEmpty(name, nameof(name));
                Name = name;
            }
        }

        public sealed class RelaxedColumn : ColumnBase
        {
            public enum VectorKind
            {
                Scalar,
                Vector,
                VariableVector
            }

            public readonly VectorKind Kind;
            public readonly DataKind ItemKind;
            public readonly bool IsKey;

            public RelaxedColumn(string name, VectorKind kind, DataKind itemKind, bool isKey)
                : base(name)
            {
                Kind = kind;
                ItemKind = itemKind;
                IsKey = isKey;
            }
        }

        public sealed class StrictColumn : ColumnBase
        {
            // REVIEW: do we ever need strict columns? Maybe we should only have relaxed?
            public readonly ColumnType ColumnType;

            public StrictColumn(string name, ColumnType columnType)
                : base(name)
            {
                Contracts.CheckValue(columnType, nameof(columnType));
                ColumnType = columnType;
            }
        }

        public SchemaShape(ColumnBase[] columns)
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
            var cols = new List<ColumnBase>();

            for (int iCol = 0; iCol < schema.ColumnCount; iCol++)
            {
                if (!schema.IsHidden(iCol))
                    cols.Append(new StrictColumn(schema.GetColumnName(iCol), schema.GetColumnType(iCol)));
            }
            return new SchemaShape(cols.ToArray());
        }

        /// <summary>
        /// Returns the column with a specified <paramref name="name"/>, and <c>null</c> if there is no such column.
        /// </summary>
        public ColumnBase FindColumn(string name)
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
    /// The data transformer, in addition to being a transformer, also exposes the input schema shape. It is handy for
    /// evaluating what kind of columns the transformer expects.
    /// </summary>
    public interface IDataTransformer : ITransformer<IDataView>
    {
        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// Returns <c>null</c> iff the schema is invalid (then a call to Transform with this data will fail).
        /// </summary>
        ISchema GetOutputSchema(ISchema inputSchema);
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
}
