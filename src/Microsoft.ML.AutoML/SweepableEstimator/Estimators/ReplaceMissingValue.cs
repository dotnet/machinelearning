// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class ReplaceMissingValues
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, ReplaceMissingValueOption param)
        {
            var inputOutputPairs = AutoMlUtils.CreateInputOutputColumnPairsFromStrings(param.InputColumnNames, param.OutputColumnNames);
            return context.Transforms.ReplaceMissingValues(inputOutputPairs);
        }
    }
}
