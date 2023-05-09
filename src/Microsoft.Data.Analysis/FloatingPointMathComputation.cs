

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveColumnComputations.tt. Do not modify directly

using System;
using System.Collections.Generic;
using System.Runtime.Versioning;

namespace Microsoft.Data.Analysis
{
    [RequiresPreviewFeatures]
    internal class FloatingPointMathComputation<T> : NumberMathComputation<T>
        where T : unmanaged, INumber<T>, IFloatingPoint<T>
    {
        public override void Round(PrimitiveColumnContainer<T> column)
        {
            Apply(column, T.Round);
        }
    }
}
