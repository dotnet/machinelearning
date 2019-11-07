// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(ParallelMultiCountTable), null, typeof(SignatureLoadModel),
    "Parallel Multi Count Table", ParallelMultiCountTable.LoaderSignature)]

[assembly: LoadableClass(typeof(SharedMultiCountTable), null, typeof(SignatureLoadModel),
    "Shared Multi Count Table", SharedMultiCountTable.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Handles simultaneous counting for multiple columns and slots. Incapsulates the counting strategy:
    /// either keep everything in one big count table, or have a count table per column and slot
    /// </summary>
    public interface IMultiCountTableBuilder
    {
        /// <summary>
        /// Increment counts for a scalar column
        /// </summary>
        /// <param name="iCol">Column index</param>
        /// <param name="key">Key to increment count for</param>
        /// <param name="labelKey">Label to increment count for</param>
        /// <param name="value">Increment value</param>
        void IncrementOne(int iCol, uint key, uint labelKey, Double value);

        /// <summary>
        /// Increment counts for a given slot in the vector column
        /// </summary>
        /// <param name="iCol">Column index</param>
        /// <param name="iSlot">Slot index</param>
        /// <param name="key">Key to increment count for</param>
        /// <param name="labelKey">Label to increment count for</param>
        /// <param name="value">Increment value</param>
        void IncrementSlot(int iCol, int iSlot, uint key, uint labelKey, Double value);

        /// <summary>
        /// Completes the table building, creates and returns an IMultiCountTable
        /// </summary>
        IMultiCountTable CreateMultiCountTable();
    }

    /// <summary>
    /// Incapsulates count tables (or one count table) for multiple columns and slots.
    /// Handles (de)serialization and featurization.
    /// </summary>
    public interface IMultiCountTable : ICanSaveModel
    {
        // REVIEW petelu: may want to add methods to inspect number of columns and slots per column

        /// <summary>
        ///  Returns an ICountTable interface for a given column and slot index
        /// </summary>
        ICountTable GetCountTable(int iCol, int iSlot);
    }

    /// <summary>
    /// Implements the multi count table builder logic by keeping a count table per column and per slot
    /// </summary>
    internal sealed class ParallelMultiCountTableBuilder : IMultiCountTableBuilder
    {
        private readonly IHost _host;
        private readonly CountTableBuilderHelperBase[][] _countTableBuilders;
        //private readonly ICountTableBuilder[][] _countTableBuilders; // dimensions: number of columns X number of slots of each column (or 1 if non-vector)

        public const string RegistrationName = "ParallelMultiCountTableBuilder";

        public ParallelMultiCountTableBuilder(IHostEnvironment env,
            DataViewSchema.Column[] inputColumns,
            CountTableBuilderBase[] builders,
            long labelCardinality,
            string externalCountsFile = null, string externalCountsSchema = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(inputColumns, nameof(inputColumns));
            _host = env.Register(RegistrationName);

            var n = inputColumns.Length;
            _countTableBuilders = new CountTableBuilderHelperBase[n][];
            for (int i = 0; i < _countTableBuilders.Length; i++)
            {
                var col = inputColumns[i];
                var size = col.Type.GetValueCount();
                _host.Check(size > 0, "vectors of unknown length are not supported");
                _countTableBuilders[i] = new CountTableBuilderHelperBase[size];

                for (int j = 0; j < size; j++)
                    _countTableBuilders[i][j] = builders[i].GetBuilderHelper(labelCardinality);
            }

            if (!string.IsNullOrEmpty(externalCountsFile))
                LoadExternalCounts(externalCountsFile, externalCountsSchema, (int)labelCardinality);
        }

        public void IncrementOne(int iCol, uint key, uint labelKey, double value)
        {
            IncrementSlot(iCol, 0, key, labelKey, value);
        }

        public void IncrementSlot(int iCol, int iSlot, uint key, uint labelKey, double value)
        {
            _host.Assert(0 <= iCol && iCol < _countTableBuilders.Length);
            _host.Assert(0 <= iSlot && iSlot < _countTableBuilders[iCol].Length);
            _countTableBuilders[iCol][iSlot].Increment(key, labelKey, value);
        }

        public IMultiCountTable CreateMultiCountTable()
        {
            var n = _countTableBuilders.Length;
            var countTables = new ICountTable[n][];

            for (int i = 0; i < n; i++)
            {
                int size = _countTableBuilders[i].Length;
                countTables[i] = new ICountTable[size];
                for (int j = 0; j < size; j++)
                    countTables[i][j] = _countTableBuilders[i][j].CreateCountTable();
            }

            return new ParallelMultiCountTable(_host, countTables);
        }

        private void LoadExternalCounts(string externalCountsFile, string externalCountsSchema, int labelCardinality)
        {
            _host.AssertNonWhiteSpace(externalCountsFile);
            _host.CheckParam(File.Exists(externalCountsFile), nameof(externalCountsFile), "File does not exist");
            _host.AssertNonWhiteSpace(externalCountsSchema, "Schema must be specified");

            var allBuilders = new List<CountTableBuilderHelperBase>();
            for (int i = 0; i < _countTableBuilders.Length; i++)
            {
                for (int j = 0; j < _countTableBuilders[i].Length; j++)
                    allBuilders.Add(_countTableBuilders[i][j]);
            }

            // analyzing schema
            var columnIds = externalCountsSchema.Split(',').Select(x => x.Trim()).ToArray();
            if (columnIds.Length != allBuilders.Count)
            {
                throw _host.Except(
                    "External count schema doesn't match columns: expected {0} columns in schema, got {1}",
                    allBuilders.Count,
                    columnIds.Length);
            }

            _host.CheckParam(columnIds.Distinct().Count() == allBuilders.Count, nameof(externalCountsSchema),
                "Duplicate column IDs encountered in the schema");

            // reverse lookup
            var columnIdLookup = new Dictionary<string, int>();
            for (int i = 0; i < allBuilders.Count; i++)
                columnIdLookup[columnIds[i]] = i;

            Single[] counts = new float[labelCardinality];

            // The below parsing is throwing errors if the count file is badly formed.
            // If the well-formed count file contains counts that we don't care about
            // (column ID not listed in the schema), this is a valid case for us, we
            // just ignore these counts and move on. This way we can use the same count
            // file even if we change the columns, or we can use the same count file with
            // multiple transforms
            var columnMismatchMessage = $"Expected number of columns: {labelCardinality + 3}";
            foreach (var line in File.ReadLines(externalCountsFile))
            {
                var lineSpan = line.AsMemory();

                // column ID
                var isMore = ReadOnlyMemoryUtils.SplitOne(lineSpan, '\t', out var val, out lineSpan);
                _host.CheckDecode(isMore, columnMismatchMessage);
                int tableIndex;
                if (!columnIdLookup.TryGetValue(val.ToString(), out tableIndex))
                    continue;
                _host.Assert(tableIndex >= 0 && tableIndex < allBuilders.Count);

                // hash ID. No longer need to be zero.
                isMore = ReadOnlyMemoryUtils.SplitOne(lineSpan, '\t', out val, out lineSpan);
                _host.CheckDecode(isMore, columnMismatchMessage);
                int id = 0;
                Conversions.Instance.Convert(in val, ref id);
                _host.CheckDecode(id >= 0, "Fail to parse hash id.");

                // hash value
                isMore = ReadOnlyMemoryUtils.SplitOne(lineSpan, '\t', out val, out lineSpan);
                _host.CheckDecode(isMore, columnMismatchMessage);
                long key = 0;
                Conversions.Instance.Convert(in val, ref key);
                _host.CheckDecode(key >= 0, "Fail to parse hash value.");

                for (int classKey = 0; classKey < labelCardinality; classKey++)
                {
                    isMore = ReadOnlyMemoryUtils.SplitOne(lineSpan, '\t', out val, out lineSpan);
                    _host.CheckDecode(isMore == (classKey < labelCardinality - 1), columnMismatchMessage);
                    var result = DoubleParser.Parse(val.Span, out float value);
                    _host.Check(result == DoubleParser.Result.Good);
                    counts[classKey] = value;
                }

                // REVIEW petelu: adding 1 is a workaround for the fact that Hash is a key type,
                // with 'real' values differing from their string representation. Think of some other one?
                allBuilders[tableIndex].InsertOrUpdateRawCounts(id, key + 1, counts);
            }
        }
    }

    /// <summary>
    /// A multi count table that holds a count table for each column and slot
    /// </summary>
    internal sealed class ParallelMultiCountTable : IMultiCountTable
    {
        private readonly IHost _host;
        private readonly ICountTable[][] _countTables;

        public ParallelMultiCountTable(IHostEnvironment env, ICountTable[][] countTables)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _countTables = countTables;
        }

        public const string LoaderSignature = "ParallelMultiCountTable";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PAR  MCT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ParallelMultiCountTable).Assembly.FullName);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // number of columns
            // for each column, number of slots
            // Sub-models:
            // count tables (each in a separate folder)

            _host.Assert(_countTables.Length > 0);
            ctx.Writer.Write(_countTables.Length);

            for (int i = 0; i < _countTables.Length; i++)
            {
                var size = _countTables[i].Length;
                _host.Assert(size > 0);
                ctx.Writer.Write(size);
                for (int j = 0; j < size; j++)
                {
                    var tableName = string.Format("Table_{0:000}_{1:000}", i, j);
                    ctx.SaveModel(_countTables[i][j], tableName);
                }
            }
        }

        public ParallelMultiCountTable(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            _host = env.Register(LoaderSignature);
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // number of columns
            // for each column, number of slots
            // Sub-models:
            // count tables (each in a separate folder)

            int n = ctx.Reader.ReadInt32();
            _host.CheckDecode(n > 0);
            _countTables = new ICountTable[n][];
            for (int i = 0; i < n; i++)
            {
                var size = ctx.Reader.ReadInt32();
                _host.CheckDecode(size > 0);
                _countTables[i] = new ICountTable[size];

                for (int j = 0; j < size; j++)
                {
                    var tableName = string.Format("Table_{0:000}_{1:000}", i, j);
                    ctx.LoadModel<ICountTable, SignatureLoadModel>(_host, out _countTables[i][j], tableName);
                }
            }
        }

        public ICountTable GetCountTable(int iCol, int iSlot)
        {
            _host.Assert(0 <= iCol && iCol < _countTables.Length);
            _host.Assert(0 <= iSlot && iSlot < _countTables[iCol].Length);
            return _countTables[iCol][iSlot];
        }
    }

    /// <summary>
    /// Implements the multi count table builder by creating one count table for everything
    /// </summary>
    public sealed class BagMultiCountTableBuilder : IMultiCountTableBuilder
    {
        private readonly IHost _host;
        private readonly CountTableBuilderHelperBase _builder;

        public const string LoaderSignature = "BagMultiCountTableBuilder";

        public BagMultiCountTableBuilder(IHostEnvironment env, CountTableBuilderBase builder, long labelCardinality)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            // REVIEW petelu: how to disallow non-zero garbage bin for bag dict count table? Or maybe just ignore?
            _builder = builder.GetBuilderHelper(labelCardinality);
        }

        public void IncrementOne(int iCol, uint key, uint labelKey, double value)
        {
            IncrementSlot(iCol, 0, key, labelKey, value);
        }

        public void IncrementSlot(int iCol, int iSlot, uint key, uint labelKey, double value)
        {
            var mixin = Hashing.MurmurRound((uint)iCol, (uint)iSlot);
            var newKey = Hashing.MurmurRound(mixin, key);
            _builder.Increment(newKey, labelKey, value);
        }

        public IMultiCountTable CreateMultiCountTable()
        {
            return new SharedMultiCountTable(_host, _builder.CreateCountTable());
        }
    }

    /// <summary>
    /// A multi count table that works over one underlying count table and mixes column and slot index into the hash key
    /// </summary>
    internal sealed class SharedMultiCountTable : IMultiCountTable
    {
        private readonly IHost _host;
        private readonly ICountTable _baseTable;

        public SharedMultiCountTable(IHostEnvironment env, ICountTable baseBaseTable)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _baseTable = baseBaseTable;
        }

        public const string LoaderSignature = "SharedMultiCountTable";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SHRD MCT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SharedMultiCountTable).Assembly.FullName);
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: 0 (since empty data is not allowed)
            // count table (in a separate folder)

            ctx.Writer.Write(0);
            ctx.SaveModel(_baseTable, "BaseTable");
        }

        public SharedMultiCountTable(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            _host = env.Register(LoaderSignature);

            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: 0 (since empty data is not allowed)
            // count table (in a separate folder)

            int n = ctx.Reader.ReadInt32();
            _host.CheckDecode(n == 0);
            ctx.LoadModel<ICountTable, SignatureLoadModel>(_host, out _baseTable, "BaseTable");
        }

        public ICountTable GetCountTable(int iCol, int iSlot)
        {
            return new ProxyCountTable(iCol, iSlot, _baseTable);
        }

        /// <summary>
        /// Mixes the column and slot index into the key, and then passes it to the wrapped count table
        /// </summary>
        private class ProxyCountTable : ICountTable
        {
            private readonly uint _mixin;
            private readonly ICountTable _table;

            public ProxyCountTable(int iCol, int iSlot, ICountTable baseCountTable)
            {
                Contracts.CheckValue(baseCountTable, nameof(baseCountTable));
                Contracts.Check(baseCountTable.GarbageThreshold == 0, "Garbage bin not supported for shared table");

                _mixin = Hashing.MurmurRound((uint)iCol, (uint)iSlot);
                _table = baseCountTable;
            }

            public void GetCounts(long key, Span<float> counts)
            {
                var newKey = (long)Hashing.MurmurRound(_mixin, (uint)key);
                _table.GetCounts(newKey, counts);
            }

            public Single GarbageThreshold
            {
                get { return 0; }
            }

            public void GetPriors(float[] priorCounts, float[] garbageCounts)
            {
                _table.GetPriors(priorCounts, garbageCounts);
            }
        }
    }
}