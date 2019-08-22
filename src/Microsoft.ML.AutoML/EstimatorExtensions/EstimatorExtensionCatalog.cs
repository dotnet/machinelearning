// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.AutoML
{
    internal enum EstimatorName
    {
        ColumnConcatenating,
        ColumnCopying,
        KeyToValueMapping,
        MissingValueIndicating,
        MissingValueReplacing,
        Normalizing,
        OneHotEncoding,
        OneHotHashEncoding,
        TextFeaturizing,
        ImageLoading,
        ImageResizing,
        PixelExtracting,
        ResNet18Featurizing,
        TypeConverting,
        ValueToKeyMapping
    }

    internal class EstimatorExtensionCatalog
    {
        private static readonly IDictionary<EstimatorName, Type> _namesToExtensionTypes = new
            Dictionary<EstimatorName, Type>()
        {
            { EstimatorName.ColumnConcatenating, typeof(ColumnConcatenatingExtension) },
            { EstimatorName.ColumnCopying, typeof(ColumnCopyingExtension) },
            { EstimatorName.KeyToValueMapping, typeof(KeyToValueMappingExtension) },
            { EstimatorName.MissingValueIndicating, typeof(MissingValueIndicatingExtension) },
            { EstimatorName.MissingValueReplacing, typeof(MissingValueReplacingExtension) },
            { EstimatorName.Normalizing, typeof(NormalizingExtension) },
            { EstimatorName.OneHotEncoding, typeof(OneHotEncodingExtension) },
            { EstimatorName.OneHotHashEncoding, typeof(OneHotHashEncodingExtension) },
            { EstimatorName.TextFeaturizing, typeof(TextFeaturizingExtension) },
            { EstimatorName.ImageLoading, typeof(ImageLoadingExtension) },
            { EstimatorName.ImageResizing, typeof(ImageResizingExtension) },
            { EstimatorName.PixelExtracting, typeof(PixelExtractingExtension) },
            { EstimatorName.ResNet18Featurizing, typeof(ResNet18FeaturizingExtension) },
            { EstimatorName.TypeConverting, typeof(TypeConvertingExtension) },
            { EstimatorName.ValueToKeyMapping, typeof(ValueToKeyMappingExtension) },
        };

        public static IEstimatorExtension GetExtension(EstimatorName estimatorName)
        {
            var extType = _namesToExtensionTypes[estimatorName];
            return (IEstimatorExtension)Activator.CreateInstance(extType);
        }
    }
}
