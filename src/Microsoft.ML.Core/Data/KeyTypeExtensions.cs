// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Extension methods related to the KeyType class.
    /// </summary>
    [BestFriend]
    internal static class KeyTypeExtensions
    {
        /// <summary>
        /// Sometimes it is necessary to cast the Count to an int. This performs overflow check.
        /// </summary>
        public static int GetCountAsInt32(this KeyType key, IExceptionContext ectx = null)
        {
            ectx.Check(key.Count <= int.MaxValue, nameof(KeyType) + "." + nameof(KeyType.Count) + " exceeds int.MaxValue.");
            return (int)key.Count;
        }
    }
}
