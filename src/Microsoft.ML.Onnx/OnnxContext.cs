// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.UniversalModelFormat.Onnx;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Onnx
{
    /// <summary>
    /// A context for defining a ONNX output.
    /// </summary>
    internal sealed class OnnxContext : IOnnxContext
    {
        private readonly List<NodeProto> _nodes;
        private readonly List<OnnxUtils.ModelArgs> _inputs;
        // The map from IDataView column names to variable names.
        private readonly List<OnnxUtils.ModelArgs> _intermediateValues;
        private readonly List<OnnxUtils.ModelArgs> _outputs;
        private readonly Dictionary<string, string> _columnNameMap;
        // All existing variable names. New variables must not exist in this set.
        private readonly HashSet<string> _variableNames;
        // All existing node names. New node names must not alrady exist in this set.
        private readonly HashSet<string> _nodeNames;
        private readonly string _name;
        private readonly string _producerName;
        private readonly IHost _host;
        private readonly string _domain;
        private readonly string _producerVersion;
        private readonly long _modelVersion;

        public OnnxContext(IHostEnvironment env, string name, string producerName,
            string producerVersion, long modelVersion, string domain)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckValue(name, nameof(name));
            Contracts.CheckValue(name, nameof(domain));

            _host = env.Register(nameof(OnnxContext));
            _nodes = new List<NodeProto>();
            _intermediateValues = new List<OnnxUtils.ModelArgs>();
            _inputs = new List<OnnxUtils.ModelArgs>();
            _outputs = new List<OnnxUtils.ModelArgs>();
            _columnNameMap = new Dictionary<string, string>();
            _variableNames = new HashSet<string>();
            _nodeNames = new HashSet<string>();
            _name = name;
            _producerName = producerName;
            _producerVersion = producerVersion;
            _modelVersion = modelVersion;
            _domain = domain;
        }

        public bool ContainsColumn(string colName) => _columnNameMap.ContainsKey(colName);

        /// <summary>
        /// Stops tracking a column. If removeVariable is true then it also removes the 
        /// variable associated with it, this is useful in the event where an output variable is 
        /// created before realizing the transform cannot actually save as ONNX.
        /// </summary>
        /// <param name="colName">IDataView column name to stop tracking</param>
        /// <param name="removeVariable">Remove associated ONNX variable at the time.</param>
        public void RemoveColumn(string colName, bool removeVariable)
        {

            if (removeVariable)
            {
                foreach (var val in _intermediateValues)
                {
                    if (val.Name == _columnNameMap[colName])
                    {
                        _intermediateValues.Remove(val);
                        break;
                    }
                }
            }

            if (_columnNameMap.ContainsKey(colName))
                _columnNameMap.Remove(colName);
        }

        /// <summary>
        /// Removes an ONNX variable. If removeColumn is true then it also removes the 
        /// IDataView column associated with it.
        /// </summary>
        /// <param name="variableName">ONNX variable to remove.</param>
        /// <param name="removeColumn">IDataView column to stop tracking</param>
        public void RemoveVariable(string variableName, bool removeColumn)
        {
            _host.Assert(_columnNameMap.ContainsValue(variableName));
            if (removeColumn)
            {
                foreach (var val in _intermediateValues)
                {
                    if (val.Name == variableName)
                    {
                        _intermediateValues.Remove(val);
                        break;
                    }
                }
            }

            string columnName = _columnNameMap.Single(kvp => string.Compare(kvp.Value, variableName) == 0).Key;

            Contracts.Assert(_variableNames.Contains(columnName));

            _columnNameMap.Remove(columnName);
            _variableNames.Remove(columnName);
        }

        /// <summary>
        /// Generates a unique name for the node based on a prefix.
        /// </summary>
        public string GetNodeName(string prefix)
        {
            _host.CheckValue(prefix, nameof(prefix));
            return GetUniqueName(prefix, _nodeNames.Contains);
        }

        /// <summary>
        /// Adds a node to the node list of the graph.
        /// </summary>
        /// <param name="node"></param>
        private void AddNode(NodeProto node)
        {
            _host.CheckValue(node, nameof(node));
            _host.Assert(!_nodeNames.Contains(node.Name));

            _nodeNames.Add(node.Name);
            _nodes.Add(node);
        }

        public IOnnxNode CreateNode(string opType, IEnumerable<string> inputs,
            IEnumerable<string> outputs, string name, string domain = null)
        {
            var innerNode = OnnxUtils.MakeNode(opType, inputs, outputs, name, domain);
            AddNode(innerNode);
            return new OnnxNode(innerNode);
        }

        /// <summary>
        /// Generates a unique name based on a prefix.
        /// </summary>
        private string GetUniqueName(string prefix, Func<string, bool> pred)
        {
            _host.CheckValue(prefix, nameof(prefix));
            _host.CheckValue(pred, nameof(pred));

            if (!pred(prefix))
                return prefix;

            int count = 0;
            while (pred(prefix + count++)) ;
            return prefix + --count;
        }

        /// <summary>
        /// Retrieves the variable name that maps to the IDataView column name at a 
        /// given point in the pipeline execution.
        /// </summary>
        /// <returns>Column Name mapping.</returns>
        public string GetVariableName(string colName)
        {
            _host.CheckValue(colName, nameof(colName));
            _host.Assert(_columnNameMap.ContainsKey(colName));

            return _columnNameMap[colName];
        }

        /// <summary>
        /// Retrieves the variable name that maps to the IDataView column name at a 
        /// given point in the pipeline execution.
        /// </summary>
        /// <returns>Column Name mapping.</returns>
        public string TryGetVariableName(string colName)
        {
            if (_columnNameMap.ContainsKey(colName))
                return GetVariableName(colName);
            return null;
        }

        /// <summary>
        /// Generates a unique column name based on the IDataView column name if
        /// there is a collision between names in the pipeline at any point.
        /// </summary>
        /// <param name="colName">IDataView column name.</param>
        /// <returns>Unique variable name.</returns>
        private string AddVariable(string colName)
        {
            _host.CheckValue(colName, nameof(colName));
            _columnNameMap[colName] = GetUniqueName(colName, _variableNames.Contains);
            _variableNames.Add(_columnNameMap[colName]);
            return _columnNameMap[colName];
        }

        /// <summary>
        /// Adds an intermediate column to the list.
        /// </summary>
        public string AddIntermediateVariable(ColumnType type, string colName, bool skip = false)
        {
            colName = AddVariable(colName);
            // Let the runtime figure the shape.
            if (!skip)
            {
                _host.CheckValue(type, nameof(type));
                _intermediateValues.Add(OnnxUtils.GetModelArgs(type, colName));
            }
            return colName;
        }

        /// <summary>
        /// Adds an output variable to the list.
        /// </summary>
        public string AddOutputVariable(ColumnType type, string colName, List<long> dim = null)
        {
            _host.CheckValue(type, nameof(type));

            if (!ContainsColumn(colName))
                AddVariable(colName);

            colName = GetVariableName(colName);
            _outputs.Add(OnnxUtils.GetModelArgs(type, colName, dim));
            return colName;
        }

        /// <summary>
        /// Adds an input variable to the list.
        /// </summary>
        public void AddInputVariable(ColumnType type, string colName)
        {
            _host.CheckValue(type, nameof(type));
            _host.CheckValue(colName, nameof(colName));

            colName = AddVariable(colName);
            _inputs.Add(OnnxUtils.GetModelArgs(type, colName));
        }

        /// <summary>
        /// Makes the ONNX model based on the context.
        /// </summary>
        public ModelProto MakeModel()
            => OnnxUtils.MakeModel(_nodes, _producerName, _name, _domain, _producerVersion, _modelVersion, _inputs, _outputs, _intermediateValues);
    }
}
