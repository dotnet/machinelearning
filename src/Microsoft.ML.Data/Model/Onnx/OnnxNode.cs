// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Model.Onnx
{
    /// <summary>
    /// An abstraction for an ONNX node as created by
    /// <see cref="OnnxContext.CreateNode(string, IEnumerable{string}, IEnumerable{string}, string, string)"/>.
    /// That method creates a with inputs and outputs, but this object can modify the node further
    /// by adding attributes (in ONNX parlance, attributes are more or less constant parameterizations).
    /// </summary>
    public abstract class OnnxNode
    {
        public abstract void AddAttribute(string argName, double value);
        public abstract void AddAttribute(string argName, long value);
        public abstract void AddAttribute(string argName, DvText value);
        public abstract void AddAttribute(string argName, string value);
        public abstract void AddAttribute(string argName, bool value);

        public abstract void AddAttribute(string argName, IEnumerable<double> value);
        public abstract void AddAttribute(string argName, IEnumerable<float> value);
        public abstract void AddAttribute(string argName, IEnumerable<long> value);
        public abstract void AddAttribute(string argName, IEnumerable<DvText> value);
        public abstract void AddAttribute(string argName, string[] value);
        public abstract void AddAttribute(string argName, IEnumerable<string> value);
        public abstract void AddAttribute(string argName, IEnumerable<bool> value);
    }
}
