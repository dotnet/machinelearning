// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.ML.SearchSpace
{
    internal class Schema
    {
        public string EstimatorType { get; set; }

        public Parameter Parameter { get; set; }

        public override string ToString()
        {
            string parameterToString(Parameter p)
            {
                return p?.ParameterType switch
                {
                    ParameterType.String => $"\"{p.AsType<string>()}\"",
                    ParameterType.Integer => p.AsType<int>().ToString(CultureInfo.InvariantCulture),
                    ParameterType.Number => p.AsType<float>().ToString(CultureInfo.InvariantCulture),
                    ParameterType.Bool => p.AsType<bool>().ToString(CultureInfo.InvariantCulture),
                    ParameterType.Object => $"({string.Join(",", p.Select(kv => $"{kv.Key}={parameterToString(kv.Value)}"))})",
                    _ => String.Empty,
                };
            }

            return $"{EstimatorType}{parameterToString(Parameter)}";
        }
    }

    internal interface ICanSummarize
    {
        /// <summary>
        /// create an summary for given estimator, which includes estimator name and hyper parameter option.
        /// </summary>
        Schema Summarize();
    }
}
