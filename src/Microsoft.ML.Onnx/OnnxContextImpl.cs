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
    internal sealed class OnnxContextImpl : OnnxContext
    {
        private readonly List<NodeProto> _nodes;
        private readonly List<OnnxUtils.ModelArgs> _inputs;
        // The map from IDataView column names to variable names.
        private readonly List<TensorProto> _initializers;
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
        private readonly OnnxVersion _onnxVersion;

        public OnnxContextImpl(IHostEnvironment env, string name, string producerName,
            string producerVersion, long modelVersion, string domain, OnnxVersion onnxVersion)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(OnnxContext));
            _host.CheckValue(name, nameof(name));
            _host.CheckValue(name, nameof(domain));

            _nodes = new List<NodeProto>();
            _intermediateValues = new List<OnnxUtils.ModelArgs>();
            _inputs = new List<OnnxUtils.ModelArgs>();
            _initializers = new List<TensorProto>();
            _outputs = new List<OnnxUtils.ModelArgs>();
            _columnNameMap = new Dictionary<string, string>();
            _variableNames = new HashSet<string>();
            _nodeNames = new HashSet<string>();
            _name = name;
            _producerName = producerName;
            _producerVersion = producerVersion;
            _modelVersion = modelVersion;
            _domain = domain;
            _onnxVersion = onnxVersion;
        }

        public override bool ContainsColumn(string colName) => _columnNameMap.ContainsKey(colName);

        /// <summary>
        /// Stops tracking a column. If removeVariable is true then it also removes the
        /// variable associated with it, this is useful in the event where an output variable is
        /// created before realizing the transform cannot actually save as ONNX.
        /// </summary>
        /// <param name="colName">IDataView column name to stop tracking</param>
        /// <param name="removeVariable">Remove associated ONNX variable at the time.</param>
        public override void RemoveColumn(string colName, bool removeVariable)
        {
            _host.CheckNonEmpty(colName, nameof(colName));

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
            _columnNameMap.Remove(colName);
        }

        /// <summary>
        /// Removes an ONNX variable. If removeColumn is true then it also removes the
        /// IDataView column associated with it.
        /// </summary>
        /// <param name="variableName">ONNX variable to remove.</param>
        /// <param name="removeColumn">IDataView column to stop tracking</param>
        public override void RemoveVariable(string variableName, bool removeColumn)
        {
            _host.CheckNonEmpty(variableName, nameof(variableName));
            if (!_columnNameMap.ContainsValue(variableName))
                throw _host.ExceptParam(nameof(variableName), $"Could not find '{variableName}' declared in ONNX graph");

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

            string columnName = _columnNameMap.Single(kvp => kvp.Value == variableName).Key;

            Contracts.Assert(_variableNames.Contains(columnName));

            _columnNameMap.Remove(columnName);
            _variableNames.Remove(columnName);
        }

        /// <summary>
        /// Generates a unique name for the node based on a prefix.
        /// </summary>
        public override string GetNodeName(string prefix)
        {
            _host.CheckNonEmpty(prefix, nameof(prefix));
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

        public override OnnxNode CreateNode(string opType, IEnumerable<string> inputs,
            IEnumerable<string> outputs, string name, string domain = null)
        {
            _host.CheckNonEmpty(opType, nameof(opType));
            _host.CheckValue(inputs, nameof(inputs));
            _host.CheckValue(outputs, nameof(outputs));
            _host.CheckNonEmpty(name, nameof(name));

            var innerNode = OnnxUtils.MakeNode(opType, inputs, outputs, name, domain);
            AddNode(innerNode);
            return new OnnxNodeImpl(innerNode);
        }

        /// <summary>
        /// Generates a unique name based on a prefix.
        /// </summary>
        private string GetUniqueName(string prefix, Func<string, bool> pred)
        {
            _host.CheckNonEmpty(prefix, nameof(prefix));
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
        public override string GetVariableName(string colName)
        {
            _host.CheckNonEmpty(colName, nameof(colName));
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
            _host.CheckNonEmpty(colName, nameof(colName));
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
            _host.CheckNonEmpty(colName, nameof(colName));
            _columnNameMap[colName] = GetUniqueName(colName, _variableNames.Contains);
            _variableNames.Add(_columnNameMap[colName]);
            return _columnNameMap[colName];
        }

        /// <summary>
        /// Adds an intermediate column to the list.
        /// </summary>
        public override string AddIntermediateVariable(ColumnType type, string colName, bool skip = false)
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
        /// Retrieve the shape of an ONNX variable. Returns null if no shape for the specified variable can be found.
        /// </summary>
        /// <param name="variableName">The ONNX name of the returned shape</param>
        /// <returns>The shape of the retrieved variable</returns>
        public override List<long> RetrieveShapeOrNull(string variableName)
        {
            foreach (var arg in _inputs)
                if (arg.Name == variableName)
                    return arg.Dims;

            foreach (var arg in _intermediateValues)
                if (arg.Name == variableName)
                    return arg.Dims;

            foreach (var arg in _outputs)
                if (arg.Name == variableName)
                    return arg.Dims;

            return null;
        }

        /// Adds constant tensor into the graph.
        public override string AddInitializer(float value, string name = null)
        {
            name = AddVariable(name ?? "float");
            _initializers.Add(OnnxUtils.MakeFloat(name, value));
            return name;
        }

        public override string AddInitializer(string value, string name = null)
        {
            name = AddVariable(name ?? "string");
            _initializers.Add(OnnxUtils.MakeString(name, value));
            return name;
        }

        public override string AddInitializer(long value, string name = null)
        {
            name = AddVariable(name ?? "int64");
            _initializers.Add(OnnxUtils.MakeInt64(name, value));
            return name;
        }

        public override string AddInitializer(IEnumerable<float> values, IEnumerable<long> dims, string name = null)
        {
            _host.CheckValue(values, nameof(values));
            if (dims != null)
                _host.Check(dims.Aggregate((x, y) => x * y) == values.Count(), "Number of elements doesn't match tensor size");

            name = AddVariable(name ?? "floats");
            _initializers.Add(OnnxUtils.MakeFloats(name, values, dims));
            return name;
        }

        public override string AddInitializer(IEnumerable<long> values, IEnumerable<long> dims, string name = null)
        {
            _host.CheckValue(values, nameof(values));
            if (dims != null)
                _host.Check(dims.Aggregate((x, y) => x * y) == values.Count(), "Number of elements doesn't match tensor size");

            name = AddVariable(name ?? "int64s");
            _initializers.Add(OnnxUtils.MakeInt64s(name, values, dims));
            return name;
        }

        public override string AddInitializer(IEnumerable<string> values, IEnumerable<long> dims, string name = null)
        {
            _host.CheckValue(values, nameof(values));
            if (dims != null)
                _host.Check(dims.Aggregate((x, y) => x * y) == values.Count(), "Number of elements doesn't match tensor size");

            name = AddVariable(name ?? "strings");
            _initializers.Add(OnnxUtils.MakeStrings(name, values, dims));
            return name;
        }

        /// <summary>
        /// Makes the ONNX model based on the context.
        /// </summary>
        public ModelProto MakeModel()
            => OnnxUtils.MakeModel(_nodes, _producerName, _name, _domain, _producerVersion, _modelVersion, _inputs, _outputs, _intermediateValues, _initializers);

        /// <summary>
        /// Return either "Experimental" or "Stable". The string "Experimental" indicates that some experimental features which are
        /// not officially supported in the official ONNX standard. Otherwise, only official ONNX features should be used.
        /// </summary>
        public override OnnxVersion GetOnnxVersion() => _onnxVersion;
    }
}
