// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.SearchSpace
{
    /// <summary>
    /// attribution class for nest option.
    /// </summary>
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false, AllowMultiple = false)]
    public sealed class NestOptionAttribute : Attribute
    {
        /// <summary>
        /// Create an <see cref="NestOptionAttribute"/>.
        /// </summary>
        public NestOptionAttribute()
        {
        }
    }
}
