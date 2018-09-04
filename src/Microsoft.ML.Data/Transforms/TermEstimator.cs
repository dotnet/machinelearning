// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using System.Linq;

namespace Microsoft.ML.Runtime.Data
{
    public sealed class TermEstimator : IEstimator<TermTransform>
    {
        private readonly IHost _host;
        private readonly TermTransform.ColumnInfo[] _columns;
        public TermEstimator(IHostEnvironment env, string name, string source = null, int maxNumTerms = TermTransform.Defaults.MaxNumTerms, TermTransform.SortOrder sort = TermTransform.Defaults.Sort) :
           this(env, new TermTransform.ColumnInfo(name, source ?? name, maxNumTerms, sort))
        {
        }

        public TermEstimator(IHostEnvironment env, params TermTransform.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TermEstimator));
            _columns = columns;
        }

        public TermTransform Fit(IDataView input) => new TermTransform(_host, input, _columns);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                var col = inputSchema.FindColumn(colInfo.Input);

                if (col == null)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);

                if ((col.ItemType.ItemType.RawKind == default) || !(col.ItemType.IsVector || col.ItemType.IsPrimitive))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                string[] metadata;
                if (col.MetadataKinds.Contains(MetadataUtils.Kinds.SlotNames))
                    metadata = new[] { MetadataUtils.Kinds.SlotNames, MetadataUtils.Kinds.KeyValues };
                else
                    metadata = new[] { MetadataUtils.Kinds.KeyValues };
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, col.Kind, NumberType.U4, true, metadata);
            }

            return new SchemaShape(result.Values);
        }
    }
}
