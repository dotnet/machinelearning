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
        Hashing,
        KeyToValueMapping,
        MissingValueIndicating,
        MissingValueReplacing,
        Normalizing,
        OneHotEncoding,
        OneHotHashEncoding,
        TextFeaturizing,
        TypeConverting,
        ValueToKeyMapping,
        RawByteImageLoading,
        ImageLoading
    }

    internal class EstimatorExtensionCatalog
    {
        private static readonly IDictionary<EstimatorName, Type> _namesToExtensionTypes = new
            Dictionary<EstimatorName, Type>()
        {
            { EstimatorName.ColumnConcatenating, typeof(ColumnConcatenatingExtension) },
            { EstimatorName.ColumnCopying, typeof(ColumnCopyingExtension) },
            { EstimatorName.KeyToValueMapping, typeof(KeyToValueMappingExtension) },
            { EstimatorName.Hashing, typeof(HashingExtension) },
            { EstimatorName.MissingValueIndicating, typeof(MissingValueIndicatingExtension) },
            { EstimatorName.MissingValueReplacing, typeof(MissingValueReplacingExtension) },
            { EstimatorName.Normalizing, typeof(NormalizingExtension) },
            { EstimatorName.OneHotEncoding, typeof(OneHotEncodingExtension) },
            { EstimatorName.OneHotHashEncoding, typeof(OneHotHashEncodingExtension) },
            { EstimatorName.TextFeaturizing, typeof(TextFeaturizingExtension) },
            { EstimatorName.TypeConverting, typeof(TypeConvertingExtension) },
            { EstimatorName.ValueToKeyMapping, typeof(ValueToKeyMappingExtension) },
            { EstimatorName.RawByteImageLoading, typeof(RawByteImageLoading) },
            { EstimatorName.ImageLoading, typeof(ImageLoading) }
        };

        public static IEstimatorExtension GetExtension(EstimatorName estimatorName)
        {
            var extType = _namesToExtensionTypes[estimatorName];
            return (IEstimatorExtension)Activator.CreateInstance(extType);
        }
    }
}
