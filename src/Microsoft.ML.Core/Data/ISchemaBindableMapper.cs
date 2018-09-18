// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// A mapper that can be bound to a <see cref="RoleMappedSchema"/> (which is an ISchema, with mappings from column kinds
    /// to columns). Binding an <see cref="ISchemaBindableMapper"/> to a <see cref="RoleMappedSchema"/> produces an
    /// <see cref="ISchemaBoundMapper"/>, which is an interface that has methods to return the names and indices of the input columns
    /// needed by the mapper to compute its output. The <see cref="ISchemaBoundRowMapper"/> is an extention to this interface, that
    /// can also produce an output IRow given an input IRow. The IRow produced generally contains only the output columns of the mapper, and not
    /// the input columns (but there is nothing preventing an <see cref="ISchemaBoundRowMapper"/> from mapping input columns directly to outputs).
    /// This interface is implemented by wrappers of IValueMapper based predictors, which are predictors that take a single
    /// features column. New predictors can implement <see cref="ISchemaBindableMapper"/> directly. Implementing <see cref="ISchemaBindableMapper"/>
    /// includes implementing a corresponding <see cref="ISchemaBoundMapper"/> (or <see cref="ISchemaBoundRowMapper"/>) and a corresponding ISchema
    /// for the output schema of the <see cref="ISchemaBoundMapper"/>. In case the <see cref="ISchemaBoundRowMapper"/> interface is implemented,
    /// the SimpleRow class can be used in the <see cref="IRowToRowMapper.GetRow"/> method.
    /// </summary>
    public interface ISchemaBindableMapper
    {
        ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema);
    }

    /// <summary>
    /// This interface is used to map a schema from input columns to output columns. The <see cref="ISchemaBoundMapper"/> should keep track
    /// of the input columns that are needed for the mapping.
    /// </summary>
    public interface ISchemaBoundMapper : ISchematized
    {
        /// <summary>
        /// The <see cref="RoleMappedSchema"/> that was passed to the <see cref="ISchemaBoundMapper"/> in the binding process.
        /// </summary>
        RoleMappedSchema InputRoleMappedSchema { get; }

        /// <summary>
        /// A property to get back the <see cref="ISchemaBindableMapper"/> that produced this <see cref="ISchemaBoundMapper"/>.
        /// </summary>
        ISchemaBindableMapper Bindable { get; }

        /// <summary>
        /// This method returns the binding information: which input columns are used and in what roles.
        /// </summary>
        IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles();
    }

    /// <summary>
    /// This interface combines <see cref="ISchemaBoundMapper"/> with <see cref="IRowToRowMapper"/>.
    /// </summary>
    public interface ISchemaBoundRowMapper : ISchemaBoundMapper, IRowToRowMapper
    {
    }

    /// <summary>
    /// This interface maps an input <see cref="IRow"/> to an output <see cref="IRow"/>. Typically, the output contains
    /// both the input columns and new columns added by the implementing class, although some implementations may
    /// return a subset of the input columns.
    /// This interface is similar to <see cref="ISchemaBoundRowMapper"/>, except it does not have any input role mappings,
    /// so to rebind, the same input column names must be used.
    /// Implementing of this object are typically created using a definie input <see cref="ISchema"/>.
    /// </summary>
    public interface IRowToRowMapper : ISchematized
    {
        /// <summary>
        /// Mappers are defined as accepting inputs with this very specific schema.
        /// </summary>
        ISchema InputSchema { get; }

        /// <summary>
        /// Given a predicate specifying which columns are needed, return a predicate indicating which input columns are
        /// needed. The domain of the function is defined over the indices of the columns of <see cref="ISchema.ColumnCount"/>
        /// for <see cref="InputSchema"/>.
        /// </summary>
        Func<int, bool> GetDependencies(Func<int, bool> predicate);

        /// <summary>
        /// Get an <see cref="IRow"/> with the indicated active columns, based on the input <paramref name="input"/>.
        /// The active columns are those for which <paramref name="active"/> returns true. Getting values on inactive
        /// columns of the returned row will throw. Null predicates are disallowed.
        ///
        /// The <see cref="ISchematized.Schema"/> of <paramref name="input"/> should be the same object as
        /// <see cref="InputSchema"/>. Implementors of this method should throw if that is not the case. Conversely,
        /// the returned value must have the same schema as <see cref="ISchematized.Schema"/>.
        ///
        /// This method creates a live connection between the input <see cref="IRow"/> and the output <see
        /// cref="IRow"/>. In particular, when the getters of the output <see cref="IRow"/> are invoked, they invoke the
        /// getters of the input row and base the output values on the current values of the input <see cref="IRow"/>.
        /// The output <see cref="IRow"/> values are re-computed when requested through the getters.
        ///
        /// The optional <paramref name="disposer"/> should be invoked by any user of this row mapping, once it no
        /// longer needs the <see cref="IRow"/>. If no action is needed when the cursor is Disposed, the implementation
        /// should set <paramref name="disposer"/> to <c>null</c>, otherwise it should be set to a delegate to be
        /// invoked by the code calling this object. (E.g., a wrapping cursor's <see cref="IDisposable.Dispose"/>
        /// method. It's best for this action to be idempotent - calling it multiple times should be equivalent to
        /// calling it once.
        /// </summary>
        IRow GetRow(IRow input, Func<int, bool> active, out Action disposer);
    }
}
