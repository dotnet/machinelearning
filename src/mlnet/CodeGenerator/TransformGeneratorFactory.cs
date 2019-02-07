// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Auto;


namespace Microsoft.ML.CLI
{
    internal interface ITransformGenerator
    {
        string GenerateTransformer();

        string GenerateUsings();
    }

    internal static class TransformGeneratorFactory
    {
        internal static ITransformGenerator GetInstance(PipelineNode node)
        {
            ITransformGenerator result = null;
            if (Enum.TryParse(node.Name, out EstimatorName trainer))
            {
                switch (trainer)
                {
                    case EstimatorName.Normalizing:
                        result = new Normalizer(node);
                        break;
                    case EstimatorName.OneHotEncoding:
                        result = new OneHotEncoding(node);
                        break;
                    case EstimatorName.ColumnConcatenating:
                        result = new ColumnConcat(node);
                        break;
                    case EstimatorName.ColumnCopying:
                        result = new ColumnCopying(node);
                        break;
                    case EstimatorName.MissingValueIndicating:
                        result = new MissingValueIndicator(node);
                        break;
                    //todo : add missing value replacing too.
                    case EstimatorName.OneHotHashEncoding:
                        result = new OneHotHashEncoding(node);
                        break;
                    case EstimatorName.TextFeaturizing:
                        result = new TextFeaturizing(node);
                        break;
                    case EstimatorName.TypeConverting:
                        result = new TypeConverting(node);
                        break;
                    case EstimatorName.ValueToKeyMapping:
                        result = new ValueToKeyMapping(node);
                        break;
                    default:
                        return null;

                }
            }
            return result;
        }
    }
}
