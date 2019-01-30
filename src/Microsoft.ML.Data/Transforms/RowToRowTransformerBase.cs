// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Model;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Base class for transformer which produce new columns, but doesn't affect existing ones.
    /// </summary>
    public abstract class RowToRowTransformerBase : ITransformer, ICanSaveModel
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

        [BestFriend]
        private protected abstract IRowMapper MakeRowMapper(Schema schema);

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

        protected abstract class MapperBase : IRowMapper
        {
            protected readonly IHost Host;
            protected readonly Schema InputSchema;
            private readonly Lazy<Schema.DetachedColumn[]> _outputColumns;
            private readonly RowToRowTransformerBase _parent;

            protected MapperBase(IHost host, Schema inputSchema, RowToRowTransformerBase parent)
            {
                Contracts.CheckValue(host, nameof(host));
                Contracts.CheckValue(inputSchema, nameof(inputSchema));
                Host = host;
                InputSchema = inputSchema;
                _parent = parent;
                _outputColumns = new Lazy<Schema.DetachedColumn[]>(GetOutputColumnsCore);
            }

            protected abstract Schema.DetachedColumn[] GetOutputColumnsCore();

            Schema.DetachedColumn[] IRowMapper.GetOutputColumns() => _outputColumns.Value;

            Delegate[] IRowMapper.CreateGetters(Row input, Func<int, bool> activeOutput, out Action disposer)
            {
                // REVIEW: it used to be that the mapper's input schema in the constructor was required to be reference-equal to the schema
                // of the input row.
                // It still has to be the same schema, but because we may make a transition from lazy to eager schema, the reference-equality
                // is no longer always possible. So, we relax the assert as below.
                Contracts.Assert(input.Schema == InputSchema);
                int n = _outputColumns.Value.Length;
                var result = new Delegate[n];
                var disposers = new Action[n];
                for (int i = 0; i < n; i++)
                {
                    if (!activeOutput(i))
                        continue;
                    result[i] = MakeGetter(input, i, activeOutput, out disposers[i]);
                }
                if (disposers.Any(x => x != null))
                {
                    disposer = () =>
                    {
                        foreach (var act in disposers)
                            act();
                    };
                }
                else
                    disposer = null;
                return result;
            }

            protected abstract Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer);

            Func<int, bool> IRowMapper.GetDependencies(Func<int, bool> activeOutput)
                => GetDependenciesCore(activeOutput);

            [BestFriend]
            private protected abstract Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput);

            public abstract void Save(ModelSaveContext ctx);

            public ITransformer GetTransformer() => _parent;
        }
    }
}
