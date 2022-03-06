// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class MapValueToKey
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, MapValueToKeyOption param)
        {
            return context.Transforms.Conversion.MapValueToKey(param.OutputColumnName, param.InputColumnName);
        }
    }

    internal partial class MapKeyToValue
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, MapKeyToValueOption param)
        {
            return context.Transforms.Conversion.MapKeyToValue(param.OutputColumnName, param.InputColumnName);
        }
    }
}
