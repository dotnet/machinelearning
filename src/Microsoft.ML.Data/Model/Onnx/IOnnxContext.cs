// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Onnx
{
    /// <summary>
    /// A context for defining a ONNX output. This is iteratively given to exportable components
    /// via the <see cref="ICanSaveOnnx"/> interface and subinterfaces, that attempt to express
    /// their operations as ONNX nodes, if they can. At the point that it is given to a component,
    /// all other components up to that component have already attempted to express themselves in
    /// this context, with their outputs possibly available in the ONNX graph.
    /// </summary>
    public interface IOnnxContext
    {
        bool ContainsColumn(string colName);

        /// <summary>
        /// Stops tracking a column. If removeVariable is true then it also removes the 
        /// variable associated with it, this is useful in the event where an output variable is 
        /// created before realizing the transform cannot actually save as ONNX.
        /// </summary>
        /// <param name="colName">IDataView column name to stop tracking</param>
        /// <param name="removeVariable">Remove associated ONNX variable at the time.</param>
        void RemoveColumn(string colName, bool removeVariable);

        /// <summary>
        /// Removes an ONNX variable. If removeColumn is true then it also removes the 
        /// tracking for the <see cref="IDataView"/> column associated with it.
        /// </summary>
        /// <param name="variableName">ONNX variable to remove.</param>
        /// <param name="removeColumn">IDataView column to stop tracking</param>
        void RemoveVariable(string variableName, bool removeColumn);

        /// <summary>
        /// Generates a unique name for the node based on a prefix.
        /// </summary>
        string GetNodeName(string prefix);

        /// <summary>
        /// Retrieves the variable name that maps to the IDataView column name at a 
        /// given point in the pipeline execution.
        /// </summary>
        /// <returns>Column Name mapping.</returns>
        string GetVariableName(string colName);

        /// <summary>
        /// Retrieves the variable name that maps to the IDataView column name at a 
        /// given point in the pipeline execution.
        /// </summary>
        /// <returns>Column Name mapping.</returns>
        string TryGetVariableName(string colName);

        /// <summary>
        /// Adds an intermediate column to the list.
        /// </summary>
        string AddIntermediateVariable(ColumnType type, string colName, bool skip = false);

        /// <summary>
        /// Creates an ONNX node of
        /// </summary>
        /// <param name="opType">The name of the ONNX operator to apply</param>
        /// <param name="inputs">The names of the variables as inputs</param>
        /// <param name="outputs">The names of the variables to create as outputs,
        /// which ought to have been something </param>
        /// <param name="name"></param>
        /// <param name="domain"></param>
        /// <returns></returns>
        IOnnxNode CreateNode(string opType, List<string> inputs,
            List<string> outputs, string name, string domain = null);
    }

    public static class OnnxContextExtensions
    {
        public static IOnnxNode CreateNode(this IOnnxContext ctx,
            string opType, string inputs, string outputs, string name)
            => ctx.CreateNode(opType, new List<string>() { inputs }, new List<string>() { outputs }, name);
    }
}
