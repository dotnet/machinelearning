﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Model;
using System;
using System.Linq;

namespace Microsoft.ML.Runtime.Data
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

        protected abstract class MapperBase : IRowMapper
        {
            protected readonly IHost Host;
            protected readonly Schema InputSchema;
            private Schema.Column[] _outputColumns;

            protected MapperBase(IHost host, Schema inputSchema)
            {
                Contracts.CheckValue(host, nameof(host));
                Contracts.CheckValue(inputSchema, nameof(inputSchema));
                Host = host;
                InputSchema = inputSchema;
                _outputColumns = null;
            }

            protected abstract Schema.Column[] GetOutputColumnsCore();

            public Schema.Column[] GetOutputColumns()
            {
                if (_outputColumns == null)
                    _outputColumns = GetOutputColumnsCore();
                return _outputColumns;
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                // make sure _outputColumns populated.
                GetOutputColumns();
                // REVIEW: it used to be that the mapper's input schema in the constructor was required to be reference-equal to the schema
                // of the input row.
                // It still has to be the same schema, but because we may make a transition from lazy to eager schema, the reference-equality
                // is no longer always possible. So, we relax the assert as below.
                if (input.Schema is Schema s)
                    Contracts.Assert(s == InputSchema);
                var result = new Delegate[_outputColumns.Length];
                var disposers = new Action[_outputColumns.Length];
                for (int i = 0; i < _outputColumns.Length; i++)
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

            protected abstract Delegate MakeGetter(IRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer);

            public abstract Func<int, bool> GetDependencies(Func<int, bool> activeOutput);

            public abstract void Save(ModelSaveContext ctx);
        }
    }
}
