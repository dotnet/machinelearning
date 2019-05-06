// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
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
    [BestFriend]
    internal interface ISchemaBindableMapper
    {
        ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema);
    }

    /// <summary>
    /// This interface is used to map a schema from input columns to output columns. The <see cref="ISchemaBoundMapper"/> should keep track
    /// of the input columns that are needed for the mapping.
    /// </summary>
    [BestFriend]
    internal interface ISchemaBoundMapper
    {
        /// <summary>
        /// The <see cref="RoleMappedSchema"/> that was passed to the <see cref="ISchemaBoundMapper"/> in the binding process.
        /// </summary>
        RoleMappedSchema InputRoleMappedSchema { get; }

        /// <summary>
        /// Gets schema of this mapper's output.
        /// </summary>
        DataViewSchema OutputSchema { get; }

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
    /// This interface extends <see cref="ISchemaBoundMapper"/>.
    /// </summary>
    [BestFriend]
    internal interface ISchemaBoundRowMapper : ISchemaBoundMapper
    {
        /// <summary>
        /// Input schema accepted.
        /// </summary>
        DataViewSchema InputSchema { get; }

        /// <summary>
        /// Given a set of columns, from the newly generated ones, return the input columns that are needed to generate those output columns.
        /// </summary>
        IEnumerable<DataViewSchema.Column> GetDependenciesForNewColumns(IEnumerable<DataViewSchema.Column> dependingColumns);

        /// <summary>
        /// Get an <see cref="DataViewRow"/> with the indicated active columns, based on the input <paramref name="input"/>.
        /// Getting values on inactive columns of the returned row will throw.
        ///
        /// The <see cref="DataViewRow.Schema"/> of <paramref name="input"/> should be the same object as
        /// <see cref="InputSchema"/>. Implementors of this method should throw if that is not the case. Conversely,
        /// the returned value must have the same schema as <see cref="ISchemaBoundMapper.OutputSchema"/>.
        ///
        /// This method creates a live connection between the input <see cref="DataViewRow"/> and the output <see
        /// cref="DataViewRow"/>. In particular, when the getters of the output <see cref="DataViewRow"/> are invoked, they invoke the
        /// getters of the input row and base the output values on the current values of the input <see cref="DataViewRow"/>.
        /// The output <see cref="DataViewRow"/> values are re-computed when requested through the getters. Also, the returned
        /// <see cref="DataViewRow"/> will dispose <paramref name="input"/> when it is disposed.
        /// </summary>
        DataViewRow GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns);
    }
}
