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

[assembly: LoadableClass(typeof(CMCountTableBuilder),
    typeof(CMCountTableBuilder.Options), typeof(SignatureCountTableBuilder),
    "Count Min Table Builder",
    "CMSketch",
    "CMTable")]

[assembly: LoadableClass(typeof(CMCountTable), null, typeof(SignatureLoadModel),
    "Count Min Table Executor",
    CMCountTable.LoaderSignature)]

[assembly: EntryPointModule(typeof(CMCountTableBuilder.Options))]

namespace Microsoft.ML.Transforms
{
    internal sealed class CMCountTable : CountTableBase
    {
        public const string LoaderSignature = "CMCountTable";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "COUNTMIN",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CMCountTable).Assembly.FullName);
        }

        public readonly int Depth; // Number of different hash functions
        public readonly int Width; // Hash space. May be any number, typically a power of 2

        public Dictionary<int, float>[][] Tables { get; }

        public CMCountTable(Dictionary<int, float>[][] tables, float[] priorCounts, int depth, int width)
            : base(Utils.Size(tables), priorCounts, 0, null)
        {
            Contracts.CheckValue(tables, nameof(tables));
            Contracts.Assert(LabelCardinality > 0);
            Contracts.Assert(Utils.Size(tables[0]) == depth);

            Depth = depth;
            Contracts.Check(Depth > 0, "depth must be positive");
            Contracts.Check(tables.All(x => Utils.Size(x) == Depth), "Depth must be the same for all labels");

            Width = width;
            Contracts.Check(Width > 0, "width must be positive");
            Contracts.Check(tables.All(t => t.All(t2 => t2.Max(kvp => kvp.Key) < Width)), "Keys must be between 0 and Width - 1");

            Tables = tables;
        }

        public static CMCountTable Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new CMCountTable(env, ctx);
        }

        private CMCountTable(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, LoaderSignature, ctx)
        {
            // *** Binary format ***
            // int: depth
            // int: width
            // for each of the _labelCardinality tables:
            //   for each of the _depth dictionaries
            //     int: the number of pairs in the dictionary
            //     for each pair:
            //       int: index
            //       float: value

            Depth = ctx.Reader.ReadInt32();
            env.CheckDecode(Depth > 0);
            Width = ctx.Reader.ReadInt32();
            env.CheckDecode(Width > 0);

            Tables = new Dictionary<int, float>[LabelCardinality][];
            for (int i = 0; i < LabelCardinality; i++)
            {
                Tables[i] = new Dictionary<int, float>[Depth];
                for (int j = 0; j < Depth; j++)
                {
                    var count = ctx.Reader.ReadInt32();
                    Tables[i][j] = new Dictionary<int, float>(count);
                    for (int k = 0; k < count; k++)
                    {
                        int index = ctx.Reader.ReadInt32();
                        float value = ctx.Reader.ReadSingle();
                        Tables[i][j].Add(index, value);
                    }
                }
            }
        }

        public override void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            base.Save(ctx);

            // *** Binary format ***
            // int: depth
            // int: width
            // for each of the _labelCardinality tables:
            //   for each of the _depth dictionaries
            //     int: the number of pairs in the dictionary
            //     for each pair:
            //       int: index
            //       float: value

            ctx.Writer.Write(Depth);
            ctx.Writer.Write(Width);

            for (int iLabel = 0; iLabel < LabelCardinality; iLabel++)
            {
                for (int iDepth = 0; iDepth < Depth; iDepth++)
                {
                    var dict = Tables[iLabel][iDepth];
                    ctx.Writer.Write(dict.Count);
                    foreach (var kvp in dict)
                    {
                        ctx.Writer.Write(kvp.Key);
                        ctx.Writer.Write(kvp.Value);
                    }
                }
            }
        }

        public override void GetCounts(long key, Span<float> counts)
        {
            Contracts.Assert(counts.Length == LabelCardinality);
            uint hash = Hashing.MurmurRound((uint)(key >> 32), (uint)key);
            for (int ilabel = 0; ilabel < LabelCardinality; ilabel++)
            {
                float minValue = -1;
                var table = Tables[ilabel];
                for (int idepth = 0; idepth < Depth; idepth++)
                {
                    int iwidth = (int)(Hashing.MixHash(Hashing.MurmurRound(hash, (uint)idepth)) % Width);
                    if (!table[idepth].TryGetValue(iwidth, out var count))
                        count = 0;
                    Contracts.Assert(count >= 0);
                    if (minValue > count || minValue < 0)
                        minValue = count;
                }
                counts[ilabel] = minValue;
            }
        }

        public override InternalCountTableBuilderBase ToBuilder(long labelCardinality)
        {
            return new CMCountTableBuilder.Builder(this, labelCardinality);
        }
    }

    internal sealed class CMCountTableBuilder : CountTableBuilderBase
    {
        private const int DepthLim = 100 + 1;
        public const string LoaderSignature = "CMCountTableBuilder";

        [TlcModule.Component(Name = "CMSketch", FriendlyName = "Count Min Table Builder", Alias = "CMTable",
            Desc = "Create the count table using the count-min sketch structure, which has a smaller memory footprint, at the expense of" +
            " some overcounting due to collisions.")]
        internal class Options : ICountTableBuilderFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Count-Min Sketch table depth", ShortName = "d")]
            public int Depth = 4;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Count-Min Sketch width", ShortName = "w")]
            public int Width = 1 << 23;

            public CountTableBuilderBase CreateComponent(IHostEnvironment env)
            {
                return new CMCountTableBuilder(env, this);
            }
        }

        private readonly int _depth;
        private readonly int _width;

        public CMCountTableBuilder(int depth = 4, int width = 1 << 23)
        {
            Contracts.Check(0 < depth && depth < DepthLim, "Depth out of range");
            Contracts.Check(0 < width, "Width out of range");
            _depth = depth;
            _width = width;
        }

        internal CMCountTableBuilder(IHostEnvironment env, Options options)
            : this(Contracts.CheckRef(options, nameof(options)).Depth, options.Width)
        {
        }

        internal override InternalCountTableBuilderBase GetInternalBuilder(long labelCardinality) => new Builder(labelCardinality, _depth, _width);

        internal sealed class Builder : InternalCountTableBuilderBase
        {
            private readonly int _depth;
            private readonly Dictionary<int, double>[][] _tables; // for each label and 0<=i<depth we keep a dictionary.
            private readonly int _width;

            public Builder(long labelCardinality, int depth, int width)
                : base(labelCardinality)
            {
                Contracts.Assert(0 < depth && depth < DepthLim);
                _depth = depth;

                Contracts.Assert(0 < width);
                _width = width;

                _tables = new Dictionary<int, double>[LabelCardinality][];
                for (int iLabel = 0; iLabel < LabelCardinality; iLabel++)
                {
                    _tables[iLabel] = new Dictionary<int, double>[_depth];
                    for (int iDepth = 0; iDepth < _depth; iDepth++)
                        _tables[iLabel][iDepth] = new Dictionary<int, double>();
                }
            }

            public Builder(CMCountTable table, long labelCardinality)
                : base(Math.Max(labelCardinality, table.LabelCardinality))
            {
                Contracts.AssertValue(table);

                _tables = new Dictionary<int, double>[LabelCardinality][];
                _depth = table.Depth;
                _width = table.Width;
                for (int iLabel = 0; iLabel < LabelCardinality; iLabel++)
                {
                    _tables[iLabel] = new Dictionary<int, double>[_depth];
                    for (int iDepth = 0; iDepth < _depth; iDepth++)
                    {
                        _tables[iLabel][iDepth] = new Dictionary<int, double>();
                        if (iLabel < table.LabelCardinality)
                        {
                            var oldDict = table.Tables[iLabel][iDepth];
                            foreach (var kvp in oldDict)
                                _tables[iLabel][iDepth].Add(kvp.Key, kvp.Value);
                        }
                    }
                }
            }

            internal override CountTableBase CreateCountTable()
            {
                var priorCounts = PriorCounts.Select(x => (float)x).ToArray();

                // copying / converting tables
                var tables = new Dictionary<int, float>[LabelCardinality][];
                for (int iLabel = 0; iLabel < LabelCardinality; iLabel++)
                {
                    tables[iLabel] = new Dictionary<int, float>[_depth];
                    for (int iDepth = 0; iDepth < _depth; iDepth++)
                    {
                        tables[iLabel][iDepth] = new Dictionary<int, float>();
                        foreach (var kvp in _tables[iLabel][iDepth])
                            tables[iLabel][iDepth].Add(kvp.Key, (float)kvp.Value);
                    }
                }

                return new CMCountTable(tables, priorCounts, _depth, _width);
            }

            protected override void IncrementCore(long key, long labelKey)
            {
                uint hash = Hashing.MurmurRound((uint)(key >> 32), (uint)key);
                for (int i = 0; i < _depth; i++)
                {
                    int idx = (int)(Hashing.MixHash(Hashing.MurmurRound(hash, (uint)i)) % _width);
                    if (!_tables[labelKey][i].ContainsKey(idx))
                        _tables[labelKey][i].Add(idx, 0);
                    _tables[labelKey][i][idx]++;
                }
            }
        }
    }
}
