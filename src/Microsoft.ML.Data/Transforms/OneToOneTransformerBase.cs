// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Data
{
    public abstract class OneToOneTransformerBase : ITransformer, ICanSaveModel
    {
        protected readonly IHost Host;
        protected readonly (string input, string output)[] ColumnPairs;

        protected OneToOneTransformerBase(IHost host, (string input, string output)[] columns)
        {
            Contracts.AssertValue(host);
            host.CheckValue(columns, nameof(columns));

            var newNames = new HashSet<string>();
            foreach (var column in columns)
            {
                host.CheckNonEmpty(column.input, nameof(columns));
                host.CheckNonEmpty(column.output, nameof(columns));

                if (!newNames.Add(column.output))
                    throw Contracts.ExceptParam(nameof(columns), $"Output column '{column.output}' specified multiple times");
            }

            Host = host;
            ColumnPairs = columns;
        }

        protected OneToOneTransformerBase(IHost host, ModelLoadContext ctx)
        {
            Host = host;
            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name

            int n = ctx.Reader.ReadInt32();
            ColumnPairs = new (string input, string output)[n];
            for (int i = 0; i < n; i++)
            {
                string output = ctx.LoadNonEmptyString();
                string input = ctx.LoadNonEmptyString();
                ColumnPairs[i] = (input, output);
            }
        }

        public abstract void Save(ModelSaveContext ctx);

        protected void SaveColumns(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name

            ctx.Writer.Write(ColumnPairs.Length);
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                ctx.SaveNonEmptyString(ColumnPairs[i].output);
                ctx.SaveNonEmptyString(ColumnPairs[i].input);
            }
        }

        private void CheckInput(ISchema inputSchema, int col, out int srcCol)
        {
            Contracts.AssertValue(inputSchema);
            Contracts.Assert(0 <= col && col < ColumnPairs.Length);

            if (!inputSchema.TryGetColumnIndex(ColumnPairs[col].input, out srcCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input);
            CheckInputColumn(inputSchema, col, srcCol);
        }

        protected virtual void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            // By default, there are no extra checks.
        }

        protected abstract IRowMapper MakeRowMapper(ISchema schema);

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            // Check that all the input columns are present and correct.
            for (int i = 0; i < ColumnPairs.Length; i++)
                CheckInput(inputSchema, i, out int col);

            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        protected RowToRowMapperTransform MakeDataTransform(IDataView input)
        {
            Host.CheckValue(input, nameof(input));
            return new RowToRowMapperTransform(Host, input, MakeRowMapper(input.Schema));
        }

        protected abstract class MapperBase : IRowMapper
        {
            protected readonly IHost Host;
            protected readonly Dictionary<int, int> ColMapNewToOld;
            protected readonly ISchema InputSchema;
            private readonly OneToOneTransformerBase _parent;

            protected MapperBase(IHost host, OneToOneTransformerBase parent, ISchema inputSchema)
            {
                Contracts.AssertValue(host);
                Contracts.AssertValue(parent);
                Contracts.AssertValue(inputSchema);

                Host = host;
                _parent = parent;

                ColMapNewToOld = new Dictionary<int, int>();
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    _parent.CheckInput(inputSchema, i, out int srcCol);
                    ColMapNewToOld.Add(i, srcCol);
                }
                InputSchema = inputSchema;
            }
            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.ColumnCount];
                foreach (var pair in ColMapNewToOld)
                    if (activeOutput(pair.Key))
                        active[pair.Value] = true;
                return col => active[col];
            }

            public abstract RowMapperColumnInfo[] GetOutputColumns();

            public void Save(ModelSaveContext ctx) => _parent.Save(ctx);

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.Assert(input.Schema == InputSchema);
                var result = new Delegate[_parent.ColumnPairs.Length];
                var disposers = new Action[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!activeOutput(i))
                        continue;
                    int srcCol = ColMapNewToOld[i];
                    result[i] = MakeGetter(input, i, out disposers[i]);
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

            protected abstract Delegate MakeGetter(IRow input, int iinfo, out Action disposer);

            protected int AddMetaGetter<T>(ColumnMetadataInfo colMetaInfo, ISchema schema, string kind, ColumnType ct, Dictionary<int, int> colMap)
            {
                MetadataUtils.MetadataGetter<T> getter = (int col, ref T dst) =>
                {
                    var originalCol = colMap[col];
                    schema.GetMetadata<T>(kind, originalCol, ref dst);
                };
                var info = new MetadataInfo<T>(ct, getter);
                colMetaInfo.Add(kind, info);
                return 0;
            }
        }
    }
}
