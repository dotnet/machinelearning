// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Runtime.Model.Onnx
{
    /// <summary>
    /// An abstraction for an ONNX node as created by
    /// <see cref="IOnnxContext.CreateNode(string, IEnumerable{string}, IEnumerable{string}, string, string)"/>.
    /// That method creates a with inputs and outputs, but this object can modify the node further
    /// by adding attributes (in ONNX parlance, attributes are more or less constant parameterizations).
    /// </summary>
    public interface IOnnxNode
    {
        void AddAttribute(string argName, double value);
        void AddAttribute(string argName, IEnumerable<double> value);
        void AddAttribute(string argName, IEnumerable<float> value);
        void AddAttribute(string argName, IEnumerable<bool> value);
        void AddAttribute(string argName, long value);
        void AddAttribute(string argName, IEnumerable<long> value);
        void AddAttribute(string argName, DvText value);
        void AddAttribute(string argName, string[] value);
        void AddAttribute(string argName, IEnumerable<DvText> value);
        void AddAttribute(string argName, IEnumerable<string> value);
        void AddAttribute(string argName, string value);
        void AddAttribute(string argName, bool value);
    }
}
