// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Allow member to be ignored in mapping to text file.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class LoadColumnIgnoreAttribute : Attribute
    {
        /// <summary>
        /// Ignore member
        /// </summary>
        /// <param name="ignore">If the member should be ignored</param>
        public LoadColumnIgnoreAttribute(bool ignore = true)
        {
            Ignore = ignore;
        }

        internal bool Ignore;
    }
}
