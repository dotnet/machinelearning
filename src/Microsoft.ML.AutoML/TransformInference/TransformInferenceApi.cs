// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.AutoML
{
    internal static class TransformInferenceApi
    {
        public static IEnumerable<SuggestedTransform> InferTransforms(MLContext context, TaskKind task, DatasetColumnInfo[] columns)
        {
            return TransformInference.InferTransforms(context, task, columns);
        }

        public static IEnumerable<SuggestedTransform> InferTransformsPostTrainer(MLContext context, TaskKind task, DatasetColumnInfo[] columns)
        {
            return TransformPostTrainerInference.InferTransforms(context, task, columns);
        }
    }
}
