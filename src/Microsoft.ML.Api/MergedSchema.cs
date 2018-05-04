// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Api
{
    /// <summary>
    /// This is the schema that is obtained by adding columns from <see cref="InternalSchemaDefinition"/> to the
    /// existing schema with auto-hiding.
    /// </summary>
    internal sealed class MergedSchema : ColumnBindingsBase
    {
        public readonly InternalSchemaDefinition AddedSchema;

        private MergedSchema(ISchema inputSchema, InternalSchemaDefinition addedSchema)
            : base(inputSchema)
        {
            Contracts.AssertValue(inputSchema);
            Contracts.AssertValue(addedSchema);
            Contracts.Assert(addedSchema.Columns.Length == 0);

            AddedSchema = addedSchema;
        }

        private MergedSchema(ISchema inputSchema, InternalSchemaDefinition addedSchema, string[] newColumnNames)
            : base(inputSchema, true, newColumnNames)
        {
            Contracts.AssertValue(inputSchema);
            Contracts.AssertValue(addedSchema);
            Contracts.AssertNonEmpty(newColumnNames);
            Contracts.Assert(addedSchema.Columns.Length == newColumnNames.Length);

            AddedSchema = addedSchema;
        }

        public static MergedSchema Create(ISchema inputSchema, InternalSchemaDefinition addedSchema)
        {
            Contracts.AssertValue(inputSchema);
            Contracts.AssertValue(addedSchema);
            if (addedSchema.Columns.Length > 0)
            {
                var newColumnNames = addedSchema.Columns.Select(x => x.ColumnName).ToArray();
                return new MergedSchema(inputSchema, addedSchema, newColumnNames);
            }

            return new MergedSchema(inputSchema, addedSchema);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < AddedSchema.Columns.Length);
            return AddedSchema.Columns[iinfo].ColumnType;
        }
    }
}