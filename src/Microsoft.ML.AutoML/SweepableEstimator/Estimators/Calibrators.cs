// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class Naive
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, NaiveOption param)
        {
            return context.BinaryClassification.Calibrators.Naive(param.LabelColumnName, param.ScoreColumnName);
        }
    }
}
