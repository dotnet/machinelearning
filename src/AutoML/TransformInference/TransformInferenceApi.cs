using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class TransformInferenceApi
    {
        public static IEnumerable<SuggestedTransform> InferTransforms(MLContext context, (string, ColumnType, ColumnPurpose, ColumnDimensions)[] columns)
        {
            return TransformInference.InferTransforms(context, columns);
        }
    }
}
