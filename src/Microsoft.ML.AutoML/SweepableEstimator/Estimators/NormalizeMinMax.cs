// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class NormalizeMinMax
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, NormalizeMinMaxOption param)
        {
            var inputOutputPairs = AutoMlUtils.CreateInputOutputColumnPairsFromStrings(param.OutputColumnNames, param.InputColumnNames);

            return context.Transforms.NormalizeMinMax(inputOutputPairs);
        }
    }
}
