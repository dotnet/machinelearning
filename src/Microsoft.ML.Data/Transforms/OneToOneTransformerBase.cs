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
    public abstract class OneToOneTransformerBase: ITransformer
    {
        protected readonly IHost Host;
        protected readonly (string input, string output)[] ColumnPairs;

        protected OneToOneTransformerBase(IHost host, (string input, string output)[] columns)
        {
            Contracts.AssertValue(host);
            Contracts.AssertValue(columns);

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

        protected void Save(ModelSaveContext ctx) => SaveContents(Host, ctx, ColumnPairs);

        private static void SaveContents(IHostEnvironment env, ModelSaveContext ctx, (string input, string output)[] columns)
        {
            Contracts.AssertValue(env);
            env.CheckValue(ctx, nameof(ctx));
            Contracts.AssertValue(columns);

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name

            ctx.Writer.Write(columns.Length);
            for (int i = 0; i < columns.Length; i++)
            {
                ctx.SaveNonEmptyString(columns[i].output);
                ctx.SaveNonEmptyString(columns[i].input);
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
            // By default, no extra checks.
        }

        protected abstract MapperBase MakeRowMapper(ISchema schema);

        protected abstract class MapperBase: IRowMapper
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
                var active = new bool[_inputSchema.ColumnCount];
                foreach (var pair in _colMapNewToOld)
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
            }
        }
}
