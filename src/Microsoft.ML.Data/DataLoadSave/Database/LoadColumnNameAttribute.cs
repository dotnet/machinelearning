// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Allow member to specify mapping to field(s) in database.
    /// To override name of <see cref="IDataView"/> column use <see cref="ColumnNameAttribute"/>.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class LoadColumnNameAttribute : Attribute
    {
        /// <summary>
        /// Maps member to specific field in database.
        /// </summary>
        /// <param name="fieldName">The name of the field in the database.</param>
        public LoadColumnNameAttribute(string fieldName)
        {
            var sources = new List<string>(1);
            sources.Add(fieldName);
            Sources = sources;
        }

        /// <summary>
        /// Maps member to set of fields in database.
        /// </summary>
        /// <param name="fieldNames">Distinct database field names to load as part of this column.</param>
        public LoadColumnNameAttribute(params string[] fieldNames)
        {
            Sources = new List<string>(fieldNames);
        }

        [BestFriend]
        internal readonly IReadOnlyList<string> Sources;
    }
}
