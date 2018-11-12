// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Runtime.CommandLine
{
    [BestFriend]
    internal static class SpecialPurpose
    {
        /// <summary>
        /// This is used to specify a column mapping of a data transform.
        /// </summary>
        public const string ColumnSelector = "ColumnSelector";

        /// <summary>
        /// This is meant to be a large text (like a c# code block, for example).
        /// </summary>
        public const string MultilineText = "MultilineText";

        /// <summary>
        /// This is used to specify a column mapping of a data transform.
        /// </summary>
        public const string ColumnName = "ColumnName";
    }
}
