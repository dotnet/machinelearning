// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Base class for transformer which produce new columns, but doesn't affect existing ones.
    /// </summary>
    public abstract class RowToRowTransformerBase: ITransformer, ICanSaveModel
    {
        protected readonly IHost Host;

        protected RowToRowTransformerBase(IHost host)
        {
            Contracts.AssertValue(host);
            Host = host;
        }

        public abstract void Save(ModelSaveContext ctx);

        public bool IsRowToRowMapper => true;

        public IRowToRowMapper GetRowToRowMapper(Schema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            return new RowToRowMapperTransform(Host, new EmptyDataView(Host, inputSchema), MakeRowMapper(inputSchema), MakeRowMapper);
        }

        protected abstract IRowMapper MakeRowMapper(Schema schema);

        public Schema GetOutputSchema(Schema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var mapper = MakeRowMapper(inputSchema);
            return RowToRowMapperTransform.GetOutputSchema(inputSchema, mapper);
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        protected RowToRowMapperTransform MakeDataTransform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return new RowToRowMapperTransform(Host, input, MakeRowMapper(input.Schema), MakeRowMapper);
        }

    }
}
