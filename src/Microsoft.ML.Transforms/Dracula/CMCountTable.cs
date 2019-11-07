// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(CMCountTableBuilder),
    typeof(CMCountTableBuilder.Arguments), typeof(SignatureCountTableBuilder),
    "Count Min Table Builder",
    "CMSketch",
    "CMTable")]

[assembly: LoadableClass(typeof(CMCountTable), null, typeof(SignatureLoadModel),
    "Count Min Table Executor",
    CMCountTable.LoaderSignature)]

[assembly: EntryPointModule(typeof(CMCountTableBuilder.Arguments))]

namespace Microsoft.ML.Transforms
{
    internal sealed class CMCountTable : CountTableBase
    {
        public const string LoaderSignature = "CMCountTable";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "CM    CT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CMCountTable).Assembly.FullName);
        }

        private readonly int _depth; // Number of different hash functions
        private readonly int _width; // Hash space. May be any number, typically a power of 2
        private readonly float[][][] _tables; // dimensions: label cardinality * depth * width

        public CMCountTable(IHostEnvironment env, float[][][] tables, float[] priorCounts)
            : base(env, LoaderSignature, Utils.Size(tables), priorCounts, 0, null)
        {
            Host.CheckValue(tables, nameof(tables));
            Host.Assert(LabelCardinality > 0);

            _depth = Utils.Size(tables[0]);
            Host.Check(_depth > 0, "depth must be positive");
            Host.Check(tables.All(x => Utils.Size(x) == _depth), "Depth must be the same for all labels");

            _width = Utils.Size(tables[0][0]);
            Host.Check(_width > 0, "width must be positive");
            Host.Check(tables.All(t => t.All(t2 => Utils.Size(t2) == _width)), "Width must be the same for all depths");

            _tables = tables;
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
            //     bool: true iff table is saved in sparse format
            //     if sparse:
            //          for each of the _depth arrays, a sequence of (index, value) for non-zero values,
            //          with an index of -1 indicating the end of the array
            //     if dense:
            //          Single[][]: the count-min-sketch of dimensions _depth * _width.

            _depth = ctx.Reader.ReadInt32();
            Host.CheckDecode(_depth > 0);
            _width = ctx.Reader.ReadInt32();
            Host.CheckDecode(_width > 0);

            _tables = new float[LabelCardinality][][];
            for (int i = 0; i < LabelCardinality; i++)
            {
                bool isSparse = ctx.Reader.ReadBoolByte();

                _tables[i] = new float[_depth][];
                for (int j = 0; j < _depth; j++)
                {
                    if (!isSparse)
                        _tables[i][j] = ctx.Reader.ReadSingleArray(_width);
                    else
                    {
                        Single[] table;
                        _tables[i][j] = table = new Single[_width];
                        int pos = -1;
                        for (; ; )
                        {
                            int oldPos = pos;
                            pos = ctx.Reader.ReadInt32();
                            Host.CheckDecode(pos >= -1 && pos < _width);
                            if (pos < 0)
                                break;

                            Host.CheckDecode(pos > oldPos);
                            Single v = ctx.Reader.ReadSingle();
                            Contracts.CheckDecode(v >= 0);
                            table[pos] = v;
                        }
                    }
                }
            }
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());
            base.Save(ctx);

            // *** Binary format ***
            // int: depth
            // int: width
            // for each of the _labelCardinality tables:
            //     bool: true iff table is saved in sparse format
            //     if sparse:
            //          for each of the _depth arrays, a sequence of (index, value) for non-zero values,
            //          with an index of -1 indicating the end of the array
            //     if dense:
            //          Single[][]: the count-min-sketch of dimensions _depth * _width.

            ctx.Writer.Write(_depth);
            ctx.Writer.Write(_width);

            for (int iLabel = 0; iLabel < LabelCardinality; iLabel++)
            {
                var table = _tables[iLabel];
                bool isSparse = IsTableSparse(table);
                ctx.Writer.WriteBoolByte(isSparse);
                foreach (var array in table)
                {
                    if (!isSparse)
                        ctx.Writer.WriteSinglesNoCount(array);
                    else
                    {
                        for (int i = 0; i < _width; i++)
                        {
                            if (array[i] > 0)
                            {
                                ctx.Writer.Write(i);
                                ctx.Writer.Write(array[i]);
                            }
                        }
                        // end of sequence
                        ctx.Writer.Write((int)-1);
                    }
                }
            }
        }

        /// <summary>
        /// Inspects a portion of the count table for one label value and determines whether the
        /// tables can be saved in a sparse vs. dense format
        /// </summary>
        private bool IsTableSparse(float[][] table)
        {
            const int sampleSize = 10000;
            const Double sparseThreshold = 0.3;

            var widthSample = Math.Min(sampleSize, _width);
            int nonZero = 0;
            for (int iWidth = 0; iWidth < widthSample; iWidth++)
            {
                for (int iDepth = 0; iDepth < _depth; iDepth++)
                {
                    if (table[iDepth][iWidth] != 0)
                        nonZero++;
                }
            }

            return nonZero < sparseThreshold * widthSample * _depth;
        }

        public override void GetCounts(long key, Span<float> counts)
        {
            Host.Assert(counts.Length == LabelCardinality);
            uint hash = Hashing.MurmurRound((uint)(key >> 32), (uint)key);
            for (int ilabel = 0; ilabel < LabelCardinality; ilabel++)
            {
                Single minValue = -1;
                var table = _tables[ilabel];
                for (int idepth = 0; idepth < _depth; idepth++)
                {
                    int iwidth = (int)(Hashing.MixHash(Hashing.MurmurRound(hash, (uint)idepth)) % _width);
                    var count = table[idepth][iwidth];
                    Host.Assert(count >= 0);
                    if (minValue > count || minValue < 0)
                        minValue = count;
                }
                counts[ilabel] = minValue;
            }
        }

        //public override void GetRawCounts(RawCountKey key, Single[] counts)
        //{
        //    Host.AssertValue(counts);
        //    Host.Assert(Utils.Size(counts) == LabelCardinality);
        //    Host.Assert(key.HashId >= 0 && key.HashId < _depth);
        //    Host.Assert(key.HashValue >= 0 && key.HashValue < _width);
        //    int n = _tables.Length;
        //    for (int iLabel = 0; iLabel < n; iLabel++)
        //        counts[iLabel] = _tables[iLabel][key.HashId][key.HashValue];
        //}

        //public override IEnumerable<RawCountKey> AllRawCountKeys()
        //{
        //    for (int hashId = 0; hashId < _depth; hashId++)
        //    {
        //        for (long hashValue = 0; hashValue < _width; hashValue++)
        //        {
        //            bool allZero = true;
        //            for (int iLabel = 0; iLabel < _tables.Length; iLabel++)
        //                allZero = allZero && _tables[iLabel][hashId][hashValue] == 0;

        //            if (!allZero)
        //                yield return new RawCountKey(hashId, hashValue);
        //        }
        //    }
        //}
    }

    public sealed class CMCountTableBuilder : CountTableBuilderBase // ICountTableBuilder
    {
        private const int DepthLim = 100 + 1;
        public const string LoaderSignature = "CMCountTableBuilder";

        [TlcModule.Component(Name = "CMSketch", FriendlyName = "Count Min Table Builder", Alias = "CMTable",
            Desc = "Create the count table using the count-min sketch structure, which has a smaller memory footprint, at the expense of" +
            " some overcounting due to collisions.")]
        internal class Arguments : IComponentFactory<CountTableBuilderBase>
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

        private readonly IHost _host;
        private readonly int _depth;
        private readonly int _width;

        internal CMCountTableBuilder(IHostEnvironment env, Arguments args)
            : base()
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);

            _host.CheckValue(args, nameof(args));

            _host.Check(0 < args.Depth && args.Depth < DepthLim, "Depth out of range");
            _depth = args.Depth;

            _host.Check(0 < args.Width, "Width out of range");
            _width = args.Width;
        }

        internal override CountTableBuilderHelperBase GetBuilderHelper(long labelCardinality) => new Helper(_host, labelCardinality, _depth, _width);

        private sealed class Helper : CountTableBuilderHelperBase
        {
            private readonly int _depth;
            private readonly double[][][] _tables; // label cardinality * depth * width
            private readonly int _width;

            public Helper(IHostEnvironment env, long labelCardinality, int depth, int width)
                : base(env, nameof(CMCountTableBuilder), labelCardinality)
            {
                Host.Assert(0 < depth && depth < DepthLim);
                _depth = depth;

                Host.Assert(0 < width);
                _width = width;

                _tables = new double[LabelCardinality][][];
                for (int iLabel = 0; iLabel < LabelCardinality; iLabel++)
                {
                    _tables[iLabel] = new double[_depth][];
                    for (int iDepth = 0; iDepth < _depth; iDepth++)
                        _tables[iLabel][iDepth] = new double[_width];
                }
            }

            internal override ICountTable CreateCountTable()
            {
                var priorCounts = PriorCounts.Select(x => (float)x).ToArray();

                // copying / converting tables
                var tables = new float[LabelCardinality][][];
                for (int iLabel = 0; iLabel < LabelCardinality; iLabel++)
                {
                    tables[iLabel] = new float[_depth][];
                    for (int iDepth = 0; iDepth < _depth; iDepth++)
                    {
                        tables[iLabel][iDepth] = _tables[iLabel][iDepth].Select(input => (float)input).ToArray();
                    }
                }

                return new CMCountTable(Host, tables, priorCounts);
            }

            internal override double Increment(long key, long labelKey, double amount)
            {
                Host.Check(0 <= labelKey && labelKey < LabelCardinality);
                PriorCounts[labelKey] += amount;

                uint hash = Hashing.MurmurRound((uint)(key >> 32), (uint)key);
                double old = double.MaxValue;
                for (int i = 0; i < _depth; i++)
                {
                    int idx = (int)(Hashing.MixHash(Hashing.MurmurRound(hash, (uint)i)) % _width);
                    var cur = _tables[labelKey][i][idx];
                    if (old > cur)
                        old = cur;
                    _tables[labelKey][i][idx] += amount;
                }
                return old;
            }

            internal override void InsertOrUpdateRawCounts(int hashId, long hashValue, float[] counts)
            {
                Host.Check(Utils.Size(counts) == LabelCardinality);
                Host.Check(hashId >= 0 && hashId < _depth);
                Host.Check(hashValue >= 0 && hashValue < _width);
                for (int iLabel = 0; iLabel < LabelCardinality; iLabel++)
                {
                    if (counts[iLabel] >= 0)
                        _tables[iLabel][hashId][hashValue] = counts[iLabel];
                }
            }
        }
    }
}
