// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Runtime;
namespace Microsoft.ML.Data
{
    /// <summary>
    /// Base class for transformer which produce new columns, but doesn't affect existing ones.
    /// </summary>
    public abstract class RowToRowTransformerBase : ITransformer
    {
        [BestFriend]
        private protected readonly IHost Host;

        [BestFriend]
        private protected RowToRowTransformerBase(IHost host)
        {
            Contracts.AssertValue(host);
            Host = host;
        }

        void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

        private protected abstract void SaveModel(ModelSaveContext ctx);

        bool ITransformer.IsRowToRowMapper => true;

        IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            return new RowToRowMapperTransform(Host, new EmptyDataView(Host, inputSchema), MakeRowMapper(inputSchema), MakeRowMapper);
        }

        [BestFriend]
        private protected abstract IRowMapper MakeRowMapper(DataViewSchema schema);

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var mapper = MakeRowMapper(inputSchema);
            return RowToRowMapperTransform.GetOutputSchema(inputSchema, mapper);
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        [BestFriend]
        private protected RowToRowMapperTransform MakeDataTransform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return new RowToRowMapperTransform(Host, input, MakeRowMapper(input.Schema), MakeRowMapper);
        }

        [BestFriend]
        private protected abstract class MapperBase : IRowMapper
        {
            protected readonly IHost Host;
            protected readonly DataViewSchema InputSchema;
            private readonly Lazy<DataViewSchema.DetachedColumn[]> _outputColumns;
            private readonly RowToRowTransformerBase _parent;

            protected MapperBase(IHost host, DataViewSchema inputSchema, RowToRowTransformerBase parent)
            {
                Contracts.CheckValue(host, nameof(host));
                Contracts.CheckValue(inputSchema, nameof(inputSchema));
                Host = host;
                InputSchema = inputSchema;
                _parent = parent;
                _outputColumns = new Lazy<DataViewSchema.DetachedColumn[]>(GetOutputColumnsCore);
            }

            protected abstract DataViewSchema.DetachedColumn[] GetOutputColumnsCore();

            DataViewSchema.DetachedColumn[] IRowMapper.GetOutputColumns() => _outputColumns.Value;

            Delegate[] IRowMapper.CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
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

            protected abstract Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer);

            Func<int, bool> IRowMapper.GetDependencies(Func<int, bool> activeOutput)
                => GetDependenciesCore(activeOutput);

            [BestFriend]
            private protected abstract Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput);

            void ICanSaveModel.Save(ModelSaveContext ctx) => SaveModel(ctx);

            private protected abstract void SaveModel(ModelSaveContext ctx);

            public ITransformer GetTransformer() => _parent;
        }
    }
}
