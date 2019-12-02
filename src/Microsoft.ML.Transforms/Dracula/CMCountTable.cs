// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
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

        public CMCountTable(float[][][] tables, float[] priorCounts)
            : base(Utils.Size(tables), priorCounts, 0, null)
        {
            Contracts.CheckValue(tables, nameof(tables));
            Contracts.Assert(LabelCardinality > 0);

            _depth = Utils.Size(tables[0]);
            Contracts.Check(_depth > 0, "depth must be positive");
            Contracts.Check(tables.All(x => Utils.Size(x) == _depth), "Depth must be the same for all labels");

            _width = Utils.Size(tables[0][0]);
            Contracts.Check(_width > 0, "width must be positive");
            Contracts.Check(tables.All(t => t.All(t2 => Utils.Size(t2) == _width)), "Width must be the same for all depths");

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
            env.CheckDecode(_depth > 0);
            _width = ctx.Reader.ReadInt32();
            env.CheckDecode(_width > 0);

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
                        float[] table;
                        _tables[i][j] = table = new float[_width];
                        int pos = -1;
                        for (; ; )
                        {
                            int oldPos = pos;
                            pos = ctx.Reader.ReadInt32();
                            env.CheckDecode(pos >= -1 && pos < _width);
                            if (pos < 0)
                                break;

                            env.CheckDecode(pos > oldPos);
                            var v = ctx.Reader.ReadSingle();
                            Contracts.CheckDecode(v >= 0);
                            table[pos] = v;
                        }
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
                        ctx.Writer.Write(-1);
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
            const double sparseThreshold = 0.3;

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
            Contracts.Assert(counts.Length == LabelCardinality);
            uint hash = Hashing.MurmurRound((uint)(key >> 32), (uint)key);
            for (int ilabel = 0; ilabel < LabelCardinality; ilabel++)
            {
                float minValue = -1;
                var table = _tables[ilabel];
                for (int idepth = 0; idepth < _depth; idepth++)
                {
                    int iwidth = (int)(Hashing.MixHash(Hashing.MurmurRound(hash, (uint)idepth)) % _width);
                    var count = table[idepth][iwidth];
                    Contracts.Assert(count >= 0);
                    if (minValue > count || minValue < 0)
                        minValue = count;
                }
                counts[ilabel] = minValue;
            }
        }

        public override int AppendRows(List<int> hashIds, List<ulong> hashValues, List<float[]> counts)
        {
            for (int i = 0; i < _depth; i++)
            {
                for (int j = 0; j < _width; j++)
                {
                    var countsCur = new float[LabelCardinality];
                    hashIds.Add(i);
                    hashValues.Add((ulong)j);
                    for (int label = 0; label < LabelCardinality; label++)
                        countsCur[label] = _tables[label][i][j];
                    counts.Add(countsCur);
                }
            }
            return _depth * _width;
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

        internal override InternalCountTableBuilderBase GetBuilderHelper(long labelCardinality) => new Builder(labelCardinality, _depth, _width);

        private sealed class Builder : InternalCountTableBuilderBase
        {
            private readonly int _depth;
            private readonly double[][][] _tables; // label cardinality * depth * width
            private readonly int _width;

            public Builder(long labelCardinality, int depth, int width)
                : base(labelCardinality)
            {
                Contracts.Assert(0 < depth && depth < DepthLim);
                _depth = depth;

                Contracts.Assert(0 < width);
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

                return new CMCountTable(tables, priorCounts);
            }

            protected override void IncrementCore(long key, long labelKey)
            {
                uint hash = Hashing.MurmurRound((uint)(key >> 32), (uint)key);
                for (int i = 0; i < _depth; i++)
                {
                    int idx = (int)(Hashing.MixHash(Hashing.MurmurRound(hash, (uint)i)) % _width);
                    _tables[labelKey][i][idx]++;
                }
            }

            internal override void InsertOrUpdateRawCounts(int hashId, long hashValue, in VBuffer<float> counts)
            {
                Contracts.Check(counts.Length == LabelCardinality);
                Contracts.Check(hashId >= 0 && hashId < _depth);
                Contracts.Check(hashValue >= 0 && hashValue < _width);
                int label = 0;
                foreach (var count in counts.DenseValues())
                {
                    if (count >= 0)
                        _tables[label][hashId][hashValue] = count;
                    label++;
                }
            }
        }
    }
}
