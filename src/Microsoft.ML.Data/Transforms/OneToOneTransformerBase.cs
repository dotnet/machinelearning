// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Base class for transformer which operates on pairs input and output columns.
    /// </summary>
    public abstract class OneToOneTransformerBase : RowToRowTransformerBase
    {
        [BestFriend]
        private protected readonly (string outputColumnName, string inputColumnName)[] ColumnPairs;

        [BestFriend]
        private protected OneToOneTransformerBase(IHost host, params (string outputColumnName, string inputColumnName)[] columns) : base(host)
        {
            host.CheckValue(columns, nameof(columns));
            var newNames = new HashSet<string>();
            foreach (var column in columns)
            {
                host.CheckNonEmpty(column.inputColumnName, nameof(columns));
                host.CheckNonEmpty(column.outputColumnName, nameof(columns));

                if (!newNames.Add(column.outputColumnName))
                    throw Contracts.ExceptParam(nameof(columns), $"Name of the result column '{column.outputColumnName}' specified multiple times");
            }

            ColumnPairs = columns;
        }

        [BestFriend]
        private protected OneToOneTransformerBase(IHost host, ModelLoadContext ctx) : base(host)
        {
            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name

            int n = ctx.Reader.ReadInt32();
            ColumnPairs = new (string outputColumnName, string inputColumnName)[n];
            for (int i = 0; i < n; i++)
            {
                string outputColumnName = ctx.LoadNonEmptyString();
                string inputColumnName = ctx.LoadNonEmptyString();
                ColumnPairs[i] = (outputColumnName, inputColumnName);
            }
        }

        [BestFriend]
        private protected void SaveColumns(ModelSaveContext ctx)
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
                ctx.SaveNonEmptyString(ColumnPairs[i].outputColumnName);
                ctx.SaveNonEmptyString(ColumnPairs[i].inputColumnName);
            }
        }

        private void CheckInput(DataViewSchema inputSchema, int col, out int srcCol)
        {
            Contracts.AssertValue(inputSchema);
            Contracts.Assert(0 <= col && col < ColumnPairs.Length);

            if (!inputSchema.TryGetColumnIndex(ColumnPairs[col].inputColumnName, out srcCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName);
            CheckInputColumn(inputSchema, col, srcCol);
        }

        [BestFriend]
        private protected virtual void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            // By default, there are no extra checks.
        }

        [BestFriend]
        private protected abstract class OneToOneMapperBase : MapperBase
        {
            protected readonly Dictionary<int, int> ColMapNewToOld;
            private readonly OneToOneTransformerBase _parent;

            protected OneToOneMapperBase(IHost host, OneToOneTransformerBase parent, DataViewSchema inputSchema) : base(host, inputSchema, parent)
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

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }
}
