// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class MatrixFactorization
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, MatrixFactorizationOption param)
        {
            return context.Recommendation().Trainers.MatrixFactorization(param.LabelColumnName, param.MatrixColumnIndexColumnName, param.MatrixRowIndexColumnName, param.ApproximationRank, param.LearningRate, param.NumberOfIterations);
        }
    }
}
