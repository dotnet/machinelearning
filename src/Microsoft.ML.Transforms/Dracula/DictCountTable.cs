// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(DictCountTableBuilder),
    typeof(DictCountTableBuilder.Arguments), typeof(SignatureCountTableBuilder),
    "Dictionary Based Count Table Builder",
    "Dictionary",
    "Dict")]

[assembly: LoadableClass(typeof(DictCountTable), null, typeof(SignatureLoadModel),
    "Dictionary Count Table",
    DictCountTable.LoaderSignature)]

[assembly: EntryPointModule(typeof(DictCountTableBuilder.Arguments))]

namespace Microsoft.ML.Transforms
{
    internal sealed class DictCountTable : CountTableBase
    {
        public const string LoaderSignature = "DictCountTable";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DICT  CT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(DictCountTable).Assembly.FullName);
        }

        public Dictionary<long, float>[] Tables { get; }

        public DictCountTable(Dictionary<long, float>[] counts, int labelCardinality, float[] priorCounts,
            float garbageThreshold, float[] garbageCounts)
            : base(labelCardinality, priorCounts, garbageThreshold, garbageCounts)
        {
            Contracts.CheckValue(counts, nameof(counts));
            Contracts.Check(counts.Length == labelCardinality, "Counts must be parallel to label cardinality");
            Contracts.Check(counts.All(x => x != null), "Count dictionaries must all exist");
            Tables = counts;
        }

        public static DictCountTable Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new DictCountTable(env, ctx);
        }

        private DictCountTable(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // foreach of the _labelCardinality dictionaries
            //     int: number N of elements in the dictionary.
            //     for each of the N elements:
            //         long: key
            //         Single: value

            Tables = new Dictionary<long, float>[LabelCardinality];
            for (int iTable = 0; iTable < LabelCardinality; iTable++)
            {
                Tables[iTable] = new Dictionary<long, float>();
                int cnt = ctx.Reader.ReadInt32();
                env.CheckDecode(cnt >= 0);
                for (int i = 0; i < cnt; i++)
                {
                    long key = ctx.Reader.ReadInt64();
                    env.CheckDecode(!Tables[iTable].ContainsKey(key));
                    var value = ctx.Reader.ReadSingle();
                    env.CheckDecode(value >= 0);
                    Tables[iTable].Add(key, value);
                }
            }
        }

        public override void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            base.Save(ctx);

            // *** Binary format ***
            // foreach of the _labelCardinality dictionaries
            //     int: number N of elements in the dictionary.
            //     for each of the N elements:
            //         long: key
            //         Single: value

            foreach (var table in Tables)
            {
                ctx.Writer.Write(table.Count);
                foreach (var pair in table)
                {
                    ctx.Writer.Write(pair.Key);
                    Contracts.Assert(pair.Value >= 0);
                    ctx.Writer.Write(pair.Value);
                }
            }
        }

        public override void GetCounts(long key, Span<float> counts)
        {
            Contracts.Check(counts.Length == LabelCardinality);
            for (int ilabel = 0; ilabel < LabelCardinality; ilabel++)
            {
                if (!Tables[ilabel].TryGetValue(key, out var count))
                    count = 0;

                counts[ilabel] = count;
            }
        }

        public override InternalCountTableBuilderBase ToBuilder(long labelCardinality)
        {
            return new DictCountTableBuilder.Builder(this, labelCardinality);
        }
    }

    internal sealed class DictCountTableBuilder : CountTableBuilderBase
    {
        public const string LoaderSignature = "DictCountTableBuilder";

        [TlcModule.Component(Name = "Dict", FriendlyName = "Dictionary Based Count Table Builder", Alias = "Dictionary",
            Desc = "Build a dictionary containing the exact count of each categorical feature value.")]
        public class Arguments : ICountTableBuilderFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Garbage threshold (counts below or equal to the threshold are assigned to the garbage bin)", ShortName = "gb")]
            public float GarbageThreshold;

            public CountTableBuilderBase CreateComponent(IHostEnvironment env)
            {
                return new DictCountTableBuilder(env, this);
            }
        }

        private readonly float _garbageThreshold;

        private DictCountTableBuilder(IHostEnvironment env, Arguments args)
            : this(Contracts.CheckRef(args, nameof(args)).GarbageThreshold)
        {
        }

        internal DictCountTableBuilder(float garbageThreshold)
        {
            Contracts.CheckParam(garbageThreshold >= 0, nameof(garbageThreshold), "Garbage threshold must be non-negative");
            _garbageThreshold = garbageThreshold;
        }

        internal override InternalCountTableBuilderBase GetInternalBuilder(long labelCardinality) => new Builder(labelCardinality, _garbageThreshold);

        public sealed class Builder : InternalCountTableBuilderBase
        {
            private readonly Dictionary<long, double>[] _tables;
            private readonly float _garbageThreshold;

            public Builder(long labelCardinality, float garbageThreshold)
                : base(labelCardinality)
            {
                _tables = new Dictionary<long, double>[LabelCardinality];
                for (int i = 0; i < LabelCardinality; i++)
                    _tables[i] = new Dictionary<long, double>();

                _garbageThreshold = garbageThreshold;
            }

            public Builder(DictCountTable table, long labelCardinality)
                : base(Math.Max(labelCardinality, table.LabelCardinality))
            {
                _tables = new Dictionary<long, double>[LabelCardinality];
                for (int i = 0; i < LabelCardinality; i++)
                {
                    _tables[i] = new Dictionary<long, double>();
                    if (i < table.LabelCardinality)
                    {
                        foreach (var kvp in table.Tables[i])
                            _tables[i][kvp.Key] = kvp.Value;
                    }
                }

                _garbageThreshold = table.GarbageThreshold;
            }

            internal override CountTableBase CreateCountTable()
            {
                var priorCounts = PriorCounts.Select(x => (float)x).ToArray();

                var singleTables = new Dictionary<long, float>[LabelCardinality];
                for (int iTable = 0; iTable < LabelCardinality; iTable++)
                    singleTables[iTable] = new Dictionary<long, float>();

                float[] garbageCounts = null;
                if (_garbageThreshold > 0)
                    ProcessGarbage(singleTables, out garbageCounts);
                else
                {
                    for (int iTable = 0; iTable < LabelCardinality; iTable++)
                    {
                        var dest = singleTables[iTable];
                        var src = _tables[iTable];
                        foreach (var pair in src)
                            dest[pair.Key] = (float)pair.Value;
                    }
                }

                return new DictCountTable(singleTables, LabelCardinality, priorCounts, _garbageThreshold, garbageCounts);
            }

            protected override void IncrementCore(long key, long labelKey)
            {
                if (!_tables[labelKey].TryGetValue(key, out var old))
                    old = 0;
                _tables[labelKey][key] = old + 1;
            }

            private void ProcessGarbage(Dictionary<long, float>[] outputTables, out float[] outputGarbageCounts)
            {
                // get all keys
                var keys = new HashSet<long>();
                foreach (var table in _tables)
                {
                    foreach (long key in table.Keys)
                        keys.Add(key);
                }

                var curCounts = new double[LabelCardinality];
                var garbageCounts = new double[LabelCardinality];
                foreach (var key in keys)
                {
                    double sumCounts = 0;

                    for (int i = 0; i < LabelCardinality; i++)
                    {
                        if (!_tables[i].TryGetValue(key, out curCounts[i]))
                            curCounts[i] = 0;
                        sumCounts += curCounts[i];
                    }

                    // if below threshold, accumulate to garbage counts, otherwise write actual counts to output table
                    if (sumCounts <= _garbageThreshold)
                    {
                        for (int i = 0; i < LabelCardinality; i++)
                            garbageCounts[i] += curCounts[i];
                    }
                    else
                    {
                        for (int i = 0; i < LabelCardinality; i++)
                        {
                            if (curCounts[i] > 0)
                                outputTables[i][key] = (float)curCounts[i];
                        }

                    }
                }

                outputGarbageCounts = garbageCounts.Select(x => (float)x).ToArray();
            }
        }
    }
}
