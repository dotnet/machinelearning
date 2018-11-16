// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Base class for transformer which operates on pairs input and output columns.
    /// </summary>
    public abstract class OneToOneTransformerBase : RowToRowTransformerBase
    {
        protected readonly (string input, string output)[] ColumnPairs;

        protected OneToOneTransformerBase(IHost host, (string input, string output)[] columns) : base(host)
        {
            host.CheckValue(columns, nameof(columns));
            var newNames = new HashSet<string>();
            foreach (var column in columns)
            {
                host.CheckNonEmpty(column.input, nameof(columns));
                host.CheckNonEmpty(column.output, nameof(columns));

                if (!newNames.Add(column.output))
                    throw Contracts.ExceptParam(nameof(columns), $"Output column '{column.output}' specified multiple times");
            }

            ColumnPairs = columns;
        }

        protected OneToOneTransformerBase(IHost host, ModelLoadContext ctx) : base(host)
        {
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

        protected abstract class MapperBase : IRowMapper
        {
            protected readonly IHost Host;
            protected readonly Dictionary<int, int> ColMapNewToOld;
            protected readonly Schema InputSchema;
            private readonly OneToOneTransformerBase _parent;

            protected MapperBase(IHost host, OneToOneTransformerBase parent, Schema inputSchema)
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

            public abstract Schema.Column[] GetOutputColumns();

            public void Save(ModelSaveContext ctx) => _parent.Save(ctx);

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                // REVIEW: it used to be that the mapper's input schema in the constructor was required to be reference-equal to the schema
                // of the input row.
                // It still has to be the same schema, but because we may make a transition from lazy to eager schema, the reference-equality
                // is no longer always possible. So, we relax the assert as below.
                if (input.Schema is Schema s)
                    Contracts.Assert(s == InputSchema);
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
