// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.UniversalModelFormat.Onnx;

namespace Microsoft.ML.Runtime.Model.Onnx
{
    internal sealed class OnnxNode : IOnnxNode
    {
        private readonly NodeProto _node;

        public OnnxNode(NodeProto node)
        {
            Contracts.AssertValue(node);
            _node = node;
        }

        public void AddAttribute(string argName, double value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, IEnumerable<double> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, IEnumerable<float> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, IEnumerable<bool> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, long value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, IEnumerable<long> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, DvText value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, string[] value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, IEnumerable<DvText> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, IEnumerable<string> value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, string value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
        public void AddAttribute(string argName, bool value)
            => OnnxUtils.NodeAddAttributes(_node, argName, value);
    }
}
