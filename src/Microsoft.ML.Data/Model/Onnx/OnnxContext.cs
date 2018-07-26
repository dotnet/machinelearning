// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Onnx
{
    /// <summary>
    /// A context for defining a ONNX output. The context internally contains the model-in-progress being built. This
    /// same context object is iteratively given to exportable components via the <see cref="ICanSaveOnnx"/> interface
    /// and subinterfaces, that attempt to express their operations as ONNX nodes, if they can. At the point that it is
    /// given to a component, all other components up to that component have already attempted to express themselves in
    /// this context, with their outputs possibly available in the ONNX graph.
    /// </summary>
    public abstract class OnnxContext
    {
        /// <summary>
        /// Generates a unique name for the node based on a prefix.
        /// </summary>
        /// <param name="prefix">The prefix for the node</param>
        /// <returns>A name that has not yet been returned from this function, starting with <paramref name="prefix"/></returns>
        public abstract string GetNodeName(string prefix);

        /// <summary>
        /// Looks up whether a given data view column has a mapping in the ONNX context. Once confirmed, callers can
        /// safely call <see cref="GetVariableName(string)"/>.
        /// </summary>
        /// <param name="colName">The data view column name</param>
        /// <returns>Whether the column is mapped in this context</returns>
        public abstract bool ContainsColumn(string colName);

        /// <summary>
        /// Stops tracking a column.
        /// </summary>
        /// <param name="colName">Column name to stop tracking</param>
        /// <param name="removeVariable">Remove associated ONNX variable. This is useful in the event where an output
        /// variable is created through <see cref="AddIntermediateVariable(ColumnType, string, bool)"/>before realizing
        /// the transform cannot actually save as ONNX.</param>
        public abstract void RemoveColumn(string colName, bool removeVariable = false);

        /// <summary>
        /// Removes an ONNX variable. If removeColumn is true then it also removes the tracking for the <see
        /// cref="IDataView"/> column associated with it.
        /// </summary>
        /// <param name="variableName">ONNX variable to remove. Note that this is an ONNX variable name, not an <see
        /// cref="IDataView"/> column name</param>
        /// <param name="removeColumn">IDataView column to stop tracking</param>
        public abstract void RemoveVariable(string variableName, bool removeColumn);

        /// <summary>
        /// ONNX variables are referred to by name. At each stage of a ML.NET pipeline, the corresponding
        /// <see cref="IDataView"/>'s column names will map to a variable in the ONNX graph if the intermediate steps
        /// used to calculate that value are things we knew how to save as ONNX. Retrieves the variable name that maps
        /// to the <see cref="IDataView"/> column name at a given point in the pipeline execution. Callers should
        /// probably confirm with <see cref="ContainsColumn(string)"/> whether a mapping for that data view column
        /// already exists.
        /// </summary>
        /// <param name="colName">The data view column name</param>
        /// <returns>The ONNX variable name corresponding to that data view column</returns>
        public abstract string GetVariableName(string colName);

        /// <summary>
        /// Establishes a new mapping from an data view column in the context, if necessary generates a unique name, and
        /// returns that newly allocated name.
        /// </summary>
        /// <param name="type">The data view type associated with this column name</param>
        /// <param name="colName">The data view column name</param>
        /// <param name="skip">Whether we should skip the process of establishing the mapping from data view column to
        /// ONNX variable name.</param>
        /// <returns>The returned value is the name of the variable corresponding </returns>
        public abstract string AddIntermediateVariable(ColumnType type, string colName, bool skip = false);

        /// <summary>
        /// Creates an ONNX node
        /// </summary>
        /// <param name="opType">The name of the ONNX operator to apply</param>
        /// <param name="inputs">The names of the variables as inputs</param>
        /// <param name="outputs">The names of the variables to create as outputs,
        /// which ought to have been something returned from <see cref="AddIntermediateVariable(ColumnType, string, bool)"/></param>
        /// <param name="name">The name of the operator, which ought to be something returned from <see cref="GetNodeName(string)"/></param>
        /// <param name="domain">The domain of the ONNX operator, if non-default</param>
        /// <returns>A node added to the in-progress ONNX graph, that attributes can be set on</returns>
        public abstract OnnxNode CreateNode(string opType, IEnumerable<string> inputs,
            IEnumerable<string> outputs, string name, string domain = null);

        /// <summary>
        /// Convenience alternative to <see cref="CreateNode(string, IEnumerable{string}, IEnumerable{string}, string, string)"/>
        /// for the case where there is exactly one input and output.
        /// </summary>
        /// <param name="opType">The name of the ONNX operator to apply</param>
        /// <param name="input">The name of the variable as input</param>
        /// <param name="output">The name of the variable as output,
        /// which ought to have been something returned from <see cref="OnnxContext.AddIntermediateVariable(ColumnType, string, bool)"/></param>
        /// <param name="name">The name of the operator, which ought to be something returned from <see cref="OnnxContext.GetNodeName(string)"/></param>
        /// <param name="domain">The domain of the ONNX operator, if non-default</param>
        /// <returns>A node added to the in-progress ONNX graph, that attributes can be set on</returns>
        public OnnxNode CreateNode(string opType, string input, string output, string name, string domain = null)
            => CreateNode(opType, new[] { input }, new[] { output }, name, domain);
    }
}
