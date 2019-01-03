// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Model;

namespace Microsoft.ML.Data
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

        private void CheckInput(Schema inputSchema, int col, out int srcCol)
        {
            Contracts.AssertValue(inputSchema);
            Contracts.Assert(0 <= col && col < ColumnPairs.Length);

            if (!inputSchema.TryGetColumnIndex(ColumnPairs[col].input, out srcCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input);
            CheckInputColumn(inputSchema, col, srcCol);
        }

        protected virtual void CheckInputColumn(Schema inputSchema, int col, int srcCol)
        {
            // By default, there are no extra checks.
        }

        protected abstract class OneToOneMapperBase : MapperBase
        {
            protected readonly Dictionary<int, int> ColMapNewToOld;
            private readonly OneToOneTransformerBase _parent;

            protected OneToOneMapperBase(IHost host, OneToOneTransformerBase parent, Schema inputSchema) : base(host, inputSchema)
            {
                Contracts.AssertValue(parent);
                _parent = parent;

                ColMapNewToOld = new Dictionary<int, int>();
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    _parent.CheckInput(inputSchema, i, out int srcCol);
                    ColMapNewToOld.Add(i, srcCol);
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                foreach (var pair in ColMapNewToOld)
                    if (activeOutput(pair.Key))
                        active[pair.Value] = true;
                return col => active[col];
            }

            public override void Save(ModelSaveContext ctx) => _parent.Save(ctx);
        }
    }
}
