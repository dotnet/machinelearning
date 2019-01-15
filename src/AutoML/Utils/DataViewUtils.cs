// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Auto
{
    internal static class DataViewUtils
    {
        /// <summary>
        /// Generate a unique temporary column name for the given schema.
        /// Use tag to independently create multiple temporary, unique column
        /// names for a single transform.
        /// </summary>
        public static string GetTemporaryColumnName(this Schema schema, string tag = null)
        {
            if (!string.IsNullOrWhiteSpace(tag) && schema.GetColumnOrNull(tag) == null)
            {
                return tag;
            }

            for (int i = 0; ; i++)
            {
                string name = string.IsNullOrWhiteSpace(tag) ?
                    string.Format("temp_{0:000}", i) :
                    string.Format("temp_{0}_{1:000}", tag, i);

                if (schema.GetColumnOrNull(name) == null)
                {
                    return name;
                }
            }
        }
    }
}