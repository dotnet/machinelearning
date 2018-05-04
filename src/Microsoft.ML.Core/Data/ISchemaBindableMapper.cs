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
    /// the SimpleRow class can be used in the <see cref="ISchemaBoundRowMapper.GetOutputRow"/> method.
    /// </summary>
    public interface ISchemaBindableMapper
    {
        ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema);
    }

    /// <summary>
    /// This interface is used to map a schema from input columns to output columns. The <see cref="ISchemaBoundMapper"/> should keep track
    /// of the input columns that are needed for the mapping.
    /// </summary>
    public interface ISchemaBoundMapper
    {
        /// <summary>
        /// The <see cref="RoleMappedSchema"/> that was passed to the <see cref="ISchemaBoundMapper"/> in the binding process.
        /// </summary>
        RoleMappedSchema InputSchema { get; }

        /// <summary>
        /// The output schema of the <see cref="ISchemaBoundMapper"/>.
        /// </summary>
        ISchema OutputSchema { get; }

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
    /// This interface extends <see cref="ISchemaBoundMapper"/> with an additional method: <see cref="GetOutputRow"/>. This method
    /// takes an input IRow and a predicate indicating which output columns are active, and returns a new IRow 
    /// containing the output columns.
    /// </summary>
    public interface ISchemaBoundRowMapper : ISchemaBoundMapper
    {
        /// <summary>
        /// Given a predicate specifying which output columns are needed, return a predicate
        /// indicating which input columns are needed.
        /// </summary>
        Func<int, bool> GetDependencies(Func<int, bool> predicate);

        /// <summary>
        /// Get an IRow based on the input IRow with the indicated active columns. The active columns are those for which
        /// predicate(col) returns true. The schema of the returned IRow will be the same as the OutputSchema, but getting 
        /// values on inactive columns will throw. Null predicates are disallowed.
        /// The schema of input should match the InputSchema.
        /// This method creates a live connection between the input IRow and the output IRow. In particular, when the
        /// getters of the output IRow are invoked, they invoke the getters of the input row and base the output values on 
        /// the current values of the input IRow. The output IRow values are re-computed when requested through the getters.
        /// The optional disposer is invoked by the cursor wrapping, when it no longer needs the IRow.
        /// If no action is needed when the cursor is Disposed, the override should set disposer to null,
        /// otherwise it should be set to a delegate to be invoked by the cursor's Dispose method. It's best
        /// for this action to be idempotent - calling it multiple times should be equivalent to calling it once.
        /// </summary>
        IRow GetOutputRow(IRow input, Func<int, bool> predicate, out Action disposer);
    }

    /// <summary>
    /// This interface maps an input <see cref="IRow"/> to an output <see cref="IRow"/>. Typically, the output contains
    /// both the input columns and new columns added by the implementing class, although some implementations may
    /// return a subset of the input columns.
    /// This interface is similar to <see cref="ISchemaBoundRowMapper"/>, except it does not have any input role mappings,
    /// so to rebind, the same input column names must be used.
    /// </summary>
    public interface IRowToRowMapper
    {
        /// <summary>
        /// Given a predicate specifying which columns are needed, return a predicate
        /// indicating which input columns are needed.
        /// </summary>
        Func<int, bool> GetDependencies(Func<int, bool> predicate);

        /// <summary>
        /// Get an IRow based on the input IRow with the indicated active columns. The active columns are those for which
        /// predicate(col) returns true. Getting values on inactive columns will throw. Null predicates are disallowed.
        /// The schema of input should match the InputSchema.
        /// This method creates a live connection between the input IRow and the output IRow. In particular, when the
        /// getters of the output IRow are invoked, they invoke the getters of the input row and base the output values on 
        /// the current values of the input IRow. The output IRow values are re-computed when requested through the getters.
        /// The optional disposer is invoked by the cursor wrapping, when it no longer needs the IRow.
        /// If no action is needed when the cursor is Disposed, the override should set disposer to null,
        /// otherwise it should be set to a delegate to be invoked by the cursor's Dispose method. It's best
        /// for this action to be idempotent - calling it multiple times should be equivalent to calling it once.
        /// </summary>
        IRow GetRow(IRow input, Func<int, bool> active, out Action disposer);
    }
}
