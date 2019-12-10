// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(ParallelMultiCountTableBuilder.MultiCountTable), null, typeof(SignatureLoadModel),
    "Parallel Multi Count Table", ParallelMultiCountTableBuilder.MultiCountTable.LoaderSignature)]

[assembly: LoadableClass(typeof(BagMultiCountTableBuilder.MultiCountTable), null, typeof(SignatureLoadModel),
    "Shared Multi Count Table", BagMultiCountTableBuilder.MultiCountTable.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Handles simultaneous counting for multiple columns and slots. Incapsulates the counting strategy:
    /// either keep everything in one big count table, or have a count table per column and slot
    /// </summary>
    internal abstract class MultiCountTableBuilderBase
    {
        public abstract void IncrementSlot(int iCol, int iSlot, uint key, uint labelKey);
        public abstract MultiCountTableBase CreateMultiCountTable();
    }

    /// <summary>
    /// Incapsulates count tables (or one count table) for multiple columns and slots.
    /// Handles (de)serialization and featurization.
    /// </summary>
    internal abstract class MultiCountTableBase : ICanSaveModel
    {
        protected readonly IHost Host;

        public abstract int ColCount { get; }
        public abstract int[] SlotCount { get; }
        public abstract ICountTable this[int iCol, int iSlot] { get; }

        protected MultiCountTableBase(IHostEnvironment env, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register(registrationName);
        }

        public abstract void Save(ModelSaveContext ctx);

        public abstract IDataView ToDataView();

        public abstract MultiCountTableBuilderBase ToBuilder(IHostEnvironment env);
    }

    /// <summary>
    /// Implements the multi count table builder logic by keeping a count table per column and per slot
    /// </summary>
    internal sealed class ParallelMultiCountTableBuilder : MultiCountTableBuilderBase
    {
        private readonly IHost _host;
        private readonly InternalCountTableBuilderBase[][] _countTableBuilders;

        public const string RegistrationName = "ParallelMultiCountTableBuilder";

        public ParallelMultiCountTableBuilder(IHostEnvironment env,
            DataViewSchema.Column[] inputColumns,
            CountTableBuilderBase[] builders,
            long labelCardinality,
            string externalCountsFile = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(inputColumns, nameof(inputColumns));
            _host = env.Register(RegistrationName);

            var n = inputColumns.Length;
            _countTableBuilders = new InternalCountTableBuilderBase[n][];
            for (int i = 0; i < _countTableBuilders.Length; i++)
            {
                var col = inputColumns[i];
                var size = col.Type.GetValueCount();
                _host.Check(size > 0, "vectors of unknown length are not supported");
                _countTableBuilders[i] = new InternalCountTableBuilderBase[size];

                for (int j = 0; j < size; j++)
                    _countTableBuilders[i][j] = builders[i].GetInternalBuilder(labelCardinality);
            }

            if (!string.IsNullOrEmpty(externalCountsFile))
                LoadExternalCounts(externalCountsFile, (int)labelCardinality);
        }

        private ParallelMultiCountTableBuilder(IHostEnvironment env, MultiCountTable table)
        {
            Contracts.AssertValue(env, nameof(env));
            env.AssertValue(table, nameof(table));
            _host = env.Register(RegistrationName);

            var n = table.ColCount;
            _countTableBuilders = new InternalCountTableBuilderBase[n][];
            var slotCounts = table.SlotCount;
            for (int i = 0; i < _countTableBuilders.Length; i++)
            {
                var size = slotCounts[i];
                _host.Assert(size > 0);
                _countTableBuilders[i] = new InternalCountTableBuilderBase[size];

                for (int j = 0; j < size; j++)
                    _countTableBuilders[i][j] = ((CountTableBase)table[i, j]).ToBuilder();
            }
        }

        public override void IncrementSlot(int iCol, int iSlot, uint key, uint labelKey)
        {
            _host.Assert(0 <= iCol && iCol < _countTableBuilders.Length);
            _host.Assert(0 <= iSlot && iSlot < _countTableBuilders[iCol].Length);
            _countTableBuilders[iCol][iSlot].Increment(key, labelKey);
        }

        public override MultiCountTableBase CreateMultiCountTable()
        {
            var n = _countTableBuilders.Length;
            var countTables = new CountTableBase[n][];

            for (int i = 0; i < n; i++)
            {
                int size = _countTableBuilders[i].Length;
                countTables[i] = new CountTableBase[size];
                for (int j = 0; j < size; j++)
                    countTables[i][j] = _countTableBuilders[i][j].CreateCountTable();
            }

            return new MultiCountTable(_host, countTables);
        }

        private void LoadExternalCounts(string externalCountsFile, int labelCardinality)
        {
            _host.AssertNonWhiteSpace(externalCountsFile);
            _host.CheckParam(File.Exists(externalCountsFile), nameof(externalCountsFile), "File does not exist");

            var allBuilders = new List<InternalCountTableBuilderBase>();
            for (int i = 0; i < _countTableBuilders.Length; i++)
            {
                for (int j = 0; j < _countTableBuilders[i].Length; j++)
                    allBuilders.Add(_countTableBuilders[i][j]);
            }

            // Generate schema of externalCountsFile.
            var cols = new[]
            {
                new TextLoader.Column("TableId", DataKind.String, 0),
                new TextLoader.Column("HashId", DataKind.Int32, 1),
                new TextLoader.Column("HashValue", DataKind.UInt64, new[] {new TextLoader.Range(2) }, keyCount: new KeyCount(long.MaxValue)),
                new TextLoader.Column("Counts", DataKind.Single, 3, 3 + labelCardinality - 1)
            };
            var countsData = new TextLoader(_host, new TextLoader.Options() { Columns = cols }).Load(new MultiFileSource(externalCountsFile));
            countsData = new ValueToKeyMappingEstimator(_host, "TableId").Fit(countsData).Transform(countsData);
            using (var cursor = countsData.GetRowCursor(countsData.Schema))
            {
                var tableIdGetter = cursor.GetGetter<uint>(countsData.Schema["TableId"]);
                var hashIdGetter = cursor.GetGetter<int>(countsData.Schema["HashId"]);
                var hashValueGetter = cursor.GetGetter<ulong>(countsData.Schema["HashValue"]);
                var countsGetter = cursor.GetGetter<VBuffer<float>>(countsData.Schema["Counts"]);
                VBuffer<float> counts = default;
                while (cursor.MoveNext())
                {
                    uint tableIndex = 0;
                    tableIdGetter(ref tableIndex);
                    if (tableIndex == 0 || tableIndex > labelCardinality)
                        continue;

                    int id = 0;
                    hashIdGetter(ref id);
                    _host.CheckDecode(id >= 0, "Invalid hash id.");

                    ulong key = 0;
                    hashValueGetter(ref key);
                    _host.CheckDecode(key <= long.MaxValue, "Invalid hash value.");

                    countsGetter(ref counts);

                    allBuilders[(int)(tableIndex - 1)].InsertOrUpdateRawCounts(id, (long)key, counts);
                }
            }
        }

        internal sealed class MultiCountTable : MultiCountTableBase
        {
            private readonly CountTableBase[][] _countTables;

            public override int ColCount => _countTables.Length;

            public override int[] SlotCount => _countTables.Select(ct => ct.Length).ToArray();

            public override ICountTable this[int iCol, int iSlot]
            {
                get
                {
                    Host.Check(0 <= iCol && iCol < ColCount, nameof(iCol));
                    Host.Check(0 <= iSlot && iSlot < SlotCount[iCol], nameof(iSlot));
                    return _countTables[iCol][iSlot];
                }
            }

            public MultiCountTable(IHostEnvironment env, CountTableBase[][] countTables)
                : base(env, LoaderSignature)
            {
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
                    loaderAssemblyName: typeof(MultiCountTable).Assembly.FullName);
            }

            public override void Save(ModelSaveContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel();
                ctx.SetVersionInfo(GetVersionInfo());

                // *** Binary format ***
                // number of columns
                // for each column, number of slots
                // Sub-models:
                // count tables (each in a separate folder)

                Host.Assert(_countTables.Length > 0);
                ctx.Writer.Write(_countTables.Length);

                for (int i = 0; i < _countTables.Length; i++)
                {
                    var size = _countTables[i].Length;
                    Host.Assert(size > 0);
                    ctx.Writer.Write(size);
                    for (int j = 0; j < size; j++)
                    {
                        var tableName = string.Format("Table_{0:000}_{1:000}", i, j);
                        ctx.SaveModel(_countTables[i][j], tableName);
                    }
                }
            }

            public MultiCountTable(IHostEnvironment env, ModelLoadContext ctx)
                : base(env, LoaderSignature)
            {
                Host.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel(GetVersionInfo());

                // *** Binary format ***
                // number of columns
                // for each column, number of slots
                // Sub-models:
                // count tables (each in a separate folder)

                int n = ctx.Reader.ReadInt32();
                Host.CheckDecode(n > 0);
                _countTables = new CountTableBase[n][];
                for (int i = 0; i < n; i++)
                {
                    var size = ctx.Reader.ReadInt32();
                    Host.CheckDecode(size > 0);
                    _countTables[i] = new CountTableBase[size];

                    for (int j = 0; j < size; j++)
                    {
                        var tableName = string.Format("Table_{0:000}_{1:000}", i, j);
                        ctx.LoadModel<CountTableBase, SignatureLoadModel>(Host, out _countTables[i][j], tableName);
                    }
                }
            }

            public override IDataView ToDataView()
            {
                int tableCount = 0;
                var tableIds = new List<int>();
                var hashIds = new List<int>();
                var hashValues = new List<ulong>();
                var counts = new List<float[]>();
                for (int i = 0; i < _countTables.Length; i++)
                {
                    var countTables = _countTables[i];
                    for (int j = 0; j < countTables.Length; j++)
                    {
                        var table = countTables[j];
                        var rowsAdded = table.AppendRows(hashIds, hashValues, counts);
                        tableIds.AddRange(Utils.CreateArray(rowsAdded, tableCount++));
                    }
                }
                var builder = new ArrayDataViewBuilder(Host);
                builder.AddColumn("TableId", NumberDataViewType.Int32, tableIds.ToArray());
                builder.AddColumn("HashId", NumberDataViewType.Int32, hashIds.ToArray());
                builder.AddColumn("HashValue", (ref VBuffer<ReadOnlyMemory<char>> dst) => { }, int.MaxValue, hashValues.ToArray());
                builder.AddColumn("Counts", NumberDataViewType.Single, counts.ToArray());
                return builder.GetDataView();
            }

            public override MultiCountTableBuilderBase ToBuilder(IHostEnvironment env)
            {
                return new ParallelMultiCountTableBuilder(env, this);
            }
        }
    }

    /// <summary>
    /// Implements the multi count table builder by creating one count table for everything
    /// </summary>
    internal sealed class BagMultiCountTableBuilder : MultiCountTableBuilderBase
    {
        private readonly IHost _host;
        private readonly InternalCountTableBuilderBase _builder;
        private readonly int _colCount;
        private readonly int[] _slotCount;

        public const string LoaderSignature = "BagMultiCountTableBuilder";

        public BagMultiCountTableBuilder(IHostEnvironment env, DataViewSchema.Column[] inputColumns, CountTableBuilderBase builder, long labelCardinality)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(inputColumns, nameof(inputColumns));
            _host = env.Register(LoaderSignature);

            // REVIEW: how to disallow non-zero garbage bin for bag dict count table? Or maybe just ignore?
            _builder = builder.GetInternalBuilder(labelCardinality);
            _colCount = inputColumns.Length;
            _slotCount = new int[_colCount];
            for (int i = 0; i < _colCount; i++)
                _slotCount[i] = inputColumns[i].Type.GetValueCount();
        }

        public BagMultiCountTableBuilder(IHostEnvironment env, MultiCountTable table)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(table, nameof(table));
            _host = env.Register(LoaderSignature);

            _builder = table.BaseTable.ToBuilder();
            _colCount = table.ColCount;
            _slotCount = new int[_colCount];
            table.SlotCount.CopyTo(_slotCount, 0);
        }

        public override void IncrementSlot(int iCol, int iSlot, uint key, uint labelKey)
        {
            var mixin = Hashing.MurmurRound((uint)iCol, (uint)iSlot);
            var newKey = Hashing.MurmurRound(mixin, key);
            _builder.Increment(newKey, labelKey);
        }

        public override MultiCountTableBase CreateMultiCountTable()
        {
            return new MultiCountTable(_host, _builder.CreateCountTable(), _colCount, _slotCount);
        }

        internal sealed class MultiCountTable : MultiCountTableBase
        {
            private readonly CountTableBase _baseTable;
            public CountTableBase BaseTable => _baseTable;

            public override int ColCount { get; }
            public override int[] SlotCount { get; }

            public override ICountTable this[int iCol, int iSlot]
            {
                get
                {
                    Host.Check(0 <= iCol && iCol < ColCount, nameof(iCol));
                    Host.Check(0 <= iSlot && iSlot < SlotCount[iCol], nameof(iSlot));
                    return new ProxyCountTable(iCol, iSlot, _baseTable);
                }
            }

            public MultiCountTable(IHostEnvironment env, CountTableBase baseTable, int colCount, int[] slotCount)
                : base(env, LoaderSignature)
            {
                _baseTable = baseTable;

                ColCount = colCount;
                SlotCount = slotCount;
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
                    loaderAssemblyName: typeof(MultiCountTable).Assembly.FullName);
            }

            public override void Save(ModelSaveContext ctx)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel();
                ctx.SetVersionInfo(GetVersionInfo());

                // *** Binary format ***
                // int: ColCount
                // int[]: SlotCount
                // count table (in a separate folder)

                ctx.Writer.Write(ColCount);
                ctx.Writer.WriteIntsNoCount(SlotCount);
                ctx.SaveModel(_baseTable, "BaseTable");
            }

            public MultiCountTable(IHostEnvironment env, ModelLoadContext ctx)
                : base(env, LoaderSignature)
            {
                Host.CheckValue(ctx, nameof(ctx));

                ctx.CheckAtModel(GetVersionInfo());

                // *** Binary format ***
                // int: ColCount
                // int[]: SlotCount
                // count table (in a separate folder)

                ColCount = ctx.Reader.ReadInt32();
                SlotCount = ctx.Reader.ReadIntArray(ColCount);
                ctx.LoadModel<CountTableBase, SignatureLoadModel>(Host, out _baseTable, "BaseTable");
            }

            public override IDataView ToDataView()
            {
                var tableIds = new List<int>();
                var hashIds = new List<int>();
                var hashValues = new List<ulong>();
                var counts = new List<float[]>();
                var rowsAdded = _baseTable.AppendRows(hashIds, hashValues, counts);
                tableIds.AddRange(Utils.CreateArray(rowsAdded, 0));
                var builder = new ArrayDataViewBuilder(Host);
                builder.AddColumn("TableId", NumberDataViewType.Int32, tableIds.ToArray());
                builder.AddColumn("HashId", NumberDataViewType.Int32, hashIds.ToArray());
                builder.AddColumn("HashValue", (ref VBuffer<ReadOnlyMemory<char>> dst) => { }, long.MaxValue, hashValues.ToArray());
                builder.AddColumn("Counts", NumberDataViewType.Single, counts.ToArray());
                return builder.GetDataView();
            }

            public override MultiCountTableBuilderBase ToBuilder(IHostEnvironment env)
            {
                return new BagMultiCountTableBuilder(env, this);
            }

            /// <summary>
            /// Mixes the column and slot index into the key, and then passes it to the wrapped count table
            /// </summary>
            private sealed class ProxyCountTable : ICountTable
            {
                private readonly uint _mixin;
                private readonly ICountTable _table;

                public const string LoaderSignature = "ProxyCountTable";

                public IReadOnlyCollection<float> GarbageCounts => _table.GarbageCounts;

                public ReadOnlySpan<double> PriorFrequencies => _table.PriorFrequencies;

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

                public float GarbageThreshold => 0;

                public int AppendRows(List<int> hashIds, List<ulong> hashValues, List<float[]> counts) => _table.AppendRows(hashIds, hashValues, counts);
            }
        }
    }
}