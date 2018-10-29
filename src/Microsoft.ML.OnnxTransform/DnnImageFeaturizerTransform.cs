// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Transforms
{
    public sealed class DnnImageFeaturizerEstimator : IEstimator<TransformerChain<OnnxTransform>>
    {
        private readonly EstimatorChain<OnnxTransform> _modelChain;

        public DnnImageFeaturizerEstimator(IHostEnvironment env, string input, string output, Func<DnnImageModelSelector, EstimatorChain<OnnxTransform>> model)
        {
            _modelChain = model(new DnnImageModelSelector(env, input, output));
        }

        public TransformerChain<OnnxTransform> Fit(IDataView input)
        {
            return _modelChain.Fit(input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            return _modelChain.GetOutputSchema(inputSchema);
        }
    }

    public partial class DnnImageModelSelector
    {
        private readonly IHostEnvironment _env;
        private readonly string _input;
        private readonly string _output;

        public DnnImageModelSelector(IHostEnvironment env, string input, string output)
        {
            _env = env;
            _input = input;
            _output = output;
        }
    }
}
