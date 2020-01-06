// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.CompilerServices;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    using TX = ReadOnlyMemory<char>;

    /// <summary>
    /// This class contains static helper methods needed for execution.
    /// </summary>
    internal sealed class Exec
    {
        /// <summary>
        /// Currently this class is not intended to be instantiated. However the methods generated
        /// by ExprTransform need to be associated with some public type. This one serves that
        /// purpose as well as containing static helpers.
        /// </summary>
        private Exec()
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX ToTX(string str)
        {
            // We shouldn't allow a null in here.
            Contracts.AssertValue(str);
            return str.AsMemory();
        }
    }
}
