// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.Data
{
    // !!! WARNING !!!
    // This file contains the Single version for normalizers and is almost identical with NormalizeColumnDbl.cs
    // When making changes to one, use BeyondCompare or a similar tool to view diffs and propagate
    // appropriate changes to the other.
    using TFloat = Single;

    public static partial class AffineNormSerializationUtils
    {
        public static void SaveModel(ModelSaveContext ctx,
            int numFeatures, int[] indices, TFloat[] scales, TFloat[] offsets, bool saveText = false)
        {
            Contracts.AssertValue(ctx);
            ctx.CheckAtModel();
            Contracts.Check(numFeatures > 0);
            Contracts.CheckValueOrNull(indices);
            Contracts.CheckValue(scales, nameof(scales));
            Contracts.CheckValueOrNull(offsets);

            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(TFloat)
            // int: number of features (size)
            // int: number of indices morphed (morph: -1 means that we assume all are, zero means none are)
            // int[]: morphed indices (max(0, morph) of them)
            // int: number of scales (if morph >= 0, this should be morph, otherwise, should be size)
            // TFloat[]: scale values
            // int: number of offsets (zero if they are all zero, otherwise, should be morph or size - same as scales)
            // TFloat[]: offset values
            ctx.Writer.Write(sizeof(TFloat));
            ctx.Writer.Write(numFeatures);

            Contracts.Assert(offsets == null || offsets.Length == scales.Length);
            if (indices == null)
            {
                Contracts.Assert(scales.Length == numFeatures);
                ctx.Writer.Write(-1);
            }
            else
            {
                Contracts.Assert(indices.Length < numFeatures);
                Contracts.Assert(scales.Length == indices.Length);
                ctx.Writer.WriteIntArray(indices);
            }
            ctx.Writer.WriteSingleArray(scales);
            ctx.Writer.WriteSingleArray(offsets);

            if (saveText)
            {
                ctx.SaveTextStream("AffineNormalizer.txt",
                    writer =>
                    {
                        writer.WriteLine("NumNormalizationFeatures={0}", numFeatures);
                        if (indices == null)
                        {
                            for (int i = 0; i < numFeatures; i++)
                                writer.WriteLine("{0}\t{1}\t{2}", i, offsets != null ? offsets[i] : 0, scales[i]);
                        }
                        else
                        {
                            for (int ii = 0; ii < indices.Length; ii++)
                                writer.WriteLine("{0}\t{1}\t{2}", indices[ii], offsets != null ? offsets[ii] : 0,
                                    scales[ii]);
                        }
                        writer.WriteLine();
                    });
            }
        }

        public static void LoadModel(ModelLoadContext ctx, ref List<int> indicesShift,
            out int numFeatures, out TFloat[] scales, out TFloat[] offsets,
            out int[] indicesMorph, out TFloat[] scalesSparse, out TFloat[] offsetsSparse)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(TFloat)
            // int: number of features (size)
            // int: number of indices morphed (morph: -1 means that we assume all are, zero means none are)
            // int[]: morphed indices (max(0, morph) of them)
            // int: number of scales (if morph >= 0, this should be morph, otherwise, should be size)
            // TFloat[]: scale values
            // int: number of offsets (zero if they are all zero, otherwise, should be morph or size - same as scales)
            // TFloat[]: offset values

            int cbFloat = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(cbFloat == sizeof(TFloat));

            int size = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(size > 0);

            numFeatures = size;

            int morphCount = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(-1 <= morphCount & morphCount < size);

            if (indicesShift != null)
                indicesShift.Clear();
            if (morphCount == -1)
            {
                // Not using sparsity.
                indicesMorph = null;
                int scaleCount = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(scaleCount == size);
                scalesSparse = ctx.Reader.ReadSingleArray(scaleCount);
                int offsetCount = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(offsetCount == 0 || offsetCount == size);
                offsetsSparse = ctx.Reader.ReadSingleArray(offsetCount);

                scales = scalesSparse;
                offsets = offsetsSparse;
                for (int iv = 0; iv < scales.Length; iv++)
                {
                    TFloat scale = scales[iv];
                    Contracts.CheckDecode(!TFloat.IsNaN(scale));
                    if (offsets == null)
                        continue;
                    if (scale == 0)
                    {
                        offsets[iv] = 0;
                        continue;
                    }
                    TFloat offset = offsets[iv];
                    Contracts.CheckDecode(!TFloat.IsNaN(offset));
                    if (!(offset == 0))
                        Utils.Add(ref indicesShift, iv);
                }
            }
            else
            {
                // Using sparsity.
                indicesMorph = ctx.Reader.ReadIntArray(morphCount) ?? new int[0];

                int scaleCount = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(scaleCount == morphCount);
                scalesSparse = ctx.Reader.ReadSingleArray(scaleCount) ?? new TFloat[0];
                int offsetCount = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(offsetCount == 0 || offsetCount == morphCount);
                offsetsSparse = ctx.Reader.ReadSingleArray(offsetCount);

                // Construct the dense representations.
                scales = Utils.CreateArray<TFloat>(numFeatures, 1);
                offsets = offsetsSparse != null ? new TFloat[numFeatures] : null;
                int ivPrev = -1;
                for (int iiv = 0; iiv < indicesMorph.Length; iiv++)
                {
                    int iv = indicesMorph[iiv];
                    Contracts.CheckDecode(ivPrev < iv & iv < numFeatures);
                    ivPrev = iv;
                    TFloat scale = scales[iv] = scalesSparse[iiv];
                    Contracts.CheckDecode(!TFloat.IsNaN(scale));
                    if (offsetsSparse == null)
                        continue;
                    if (scale == 0)
                    {
                        offsetsSparse[iiv] = 0;
                        continue;
                    }
                    TFloat offset = offsets[iv] = offsetsSparse[iiv];
                    Contracts.CheckDecode(!TFloat.IsNaN(offset));
                    if (!(offset == 0))
                        Utils.Add(ref indicesShift, iv);
                }
            }

            Contracts.Assert(numFeatures > 0);
            Contracts.Assert(scalesSparse != null);
            Contracts.Assert(indicesMorph == null || indicesMorph.Length == scalesSparse.Length);
            Contracts.Assert(offsetsSparse == null || offsetsSparse.Length == scalesSparse.Length);
            Contracts.Assert((offsets == null) == (offsetsSparse == null));
        }
    }

    public static partial class BinNormSerializationUtils
    {
        public static void SaveModel(ModelSaveContext ctx, TFloat[][] binUpperBounds, bool saveText = false)
        {
            Contracts.AssertValue(ctx);

            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(TFloat)
            // int: number of bin upper bounds arrays = number of features
            // for each array:
            //     int: number of elements in bin upper bounds
            //     TFloat[]: bin upper bounds
            ctx.Writer.Write(sizeof(TFloat));

            ctx.Writer.Write(binUpperBounds.Length);
            foreach (var featureUpperBounds in binUpperBounds)
                ctx.Writer.WriteSingleArray(featureUpperBounds);

            if (saveText)
            {
                ctx.SaveTextStream("BinNormalizer.txt",
                    writer =>
                    {
                        writer.WriteLine("NumNormalizationFeatures={0}", binUpperBounds.Length);
                        for (int i = 0; i < binUpperBounds.Length; i++)
                        {
                            string pre = "";
                            for (int j = 0; j < binUpperBounds[i].Length - 1; j++)
                            {
                                writer.Write(pre);
                                pre = "\t";
                                writer.Write(binUpperBounds[i][j]);
                            }
                            writer.WriteLine();
                        }
                    });
            }
        }

        public static void LoadModel(ModelLoadContext ctx, out TFloat[][] binUpperBounds)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(TFloat)
            // int: number of bin upper bounds arrays = number of features
            // for each array:
            //     int: number of elements in bin upper bounds
            //     TFloat[]: bin upper bounds
            int cbFloat = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(cbFloat == sizeof(TFloat));

            // Core model
            int numFeatures = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(numFeatures > 0);
            binUpperBounds = new TFloat[numFeatures][];
            for (int i = 0; i < numFeatures; i++)
            {
                TFloat[] curUpperBounds = ctx.Reader.ReadSingleArray();
                Contracts.CheckDecode(Utils.Size(curUpperBounds) > 0);
                binUpperBounds[i] = curUpperBounds;
                for (int j = 1; j < curUpperBounds.Length; j++)
                    Contracts.CheckDecode(curUpperBounds[j - 1] < curUpperBounds[j]);
                Contracts.CheckDecode(curUpperBounds[curUpperBounds.Length - 1] == TFloat.PositiveInfinity);
            }
        }
    }

    public static partial class CdfNormSerializationUtils
    {
        public static void SaveModel(ModelSaveContext ctx, bool useLog, TFloat[] mean, TFloat[] stddev)
        {
            // *** Binary format ***
            // int: sizeof(TFloat)
            // bool: useLog
            // int: number of features (size)
            // TFloat[]: mean values
            // TFloat[]: stddev values
            ctx.Writer.Write(sizeof(TFloat));
            ctx.Writer.WriteBoolByte(useLog);
            ctx.Writer.Write(mean.Length);
            ctx.Writer.WriteSinglesNoCount(mean, mean.Length);
            ctx.Writer.WriteSinglesNoCount(stddev, mean.Length);

            ctx.SaveTextStream("CdfNormalizer.txt",
                writer =>
                {
                    writer.WriteLine("NumNormalizationFeatures={0}", mean.Length);
                    writer.WriteLine("Log={0}", useLog);
                    for (int i = 0; i < mean.Length; i++)
                        writer.WriteLine("{0}\t{1}", mean[i], stddev[i]);
                });
        }

        public static void LoadModel(ModelLoadContext ctx, int cv, out bool useLog, out TFloat[] mean, out TFloat[] stddev)
        {
            // *** Binary format ***
            // int: sizeof(TFloat)
            // bool: useLog
            // int: number of features (size)
            // TFloat[]: mean values
            // TFloat[]: stddev values

            int cbFloat = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(cbFloat == sizeof(TFloat));

            useLog = ctx.Reader.ReadBoolByte();

            int size = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(size > 0);
            if (size != cv)
                throw Contracts.Except("Normalizer expected {0} slots, but the input data column has {1} slots.", size, cv);
            mean = ctx.Reader.ReadSingleArray(size);
            stddev = ctx.Reader.ReadSingleArray(size);
        }
    }

    /// <summary>
    /// Base class for tracking min and max values for a vector valued column.
    /// It tracks min, max, number of non-sparse values (vCount) and number of ProcessValue() calls (trainCount).
    /// NaNs are ignored when updating min and max.
    /// </summary>
    public sealed class MinMaxSngAggregator : IColumnAggregator<VBuffer<TFloat>>
    {
        private readonly TFloat[] _min;
        private readonly TFloat[] _max;
        private readonly long[] _vCount;
        private long _trainCount;

        public MinMaxSngAggregator(int size)
        {
            Contracts.Check(size > 0);
            _min = new TFloat[size];
            _max = new TFloat[size];
            _vCount = new long[size];
            for (int i = 0; i < size; i++)
            {
                _min[i] = TFloat.PositiveInfinity;
                _max[i] = TFloat.NegativeInfinity;
            }
        }

        public TFloat[] Min
        {
            get { return _min; }
        }

        public TFloat[] Max
        {
            get { return _max; }
        }

        public long[] Count
        {
            get { return _vCount; }
        }

        public void ProcessValue(ref VBuffer<TFloat> value)
        {
            var size = _min.Length;
            Contracts.Check(value.Length == size);
            _trainCount++;
            var count = value.Count;
            Contracts.Assert(0 <= count & count <= size);
            if (count == 0)
                return;

            if (count == size)
            {
                var values = value.Values;
                for (int j = 0; j < count; j++)
                {
                    var val = values[j];
                    _vCount[j]++;
                    Update(j, val);
                }
            }
            else
            {
                var indices = value.Indices;
                var values = value.Values;
                for (int k = 0; k < count; k++)
                {
                    var val = values[k];
                    var j = indices[k];
                    _vCount[j]++;
                    Update(j, val);
                }
            }
        }

        public void Finish()
        {
            var size = _min.Length;
            for (int i = 0; i < size; i++)
            {
                if (_vCount[i] < _trainCount)
                    Update(i, 0);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Update(int j, TFloat val)
        {
            if (_max[j] < val)
                _max[j] = val;
            if (_min[j] > val)
                _min[j] = val;
        }
    }

    /// <summary>
    /// Class for computing the mean and variance for a vector valued column.
    /// It tracks the current mean and the M2 (sum of squared diffs of the values from the mean),
    /// the number of NaNs and the number of non-zero elements.
    /// Uses the algorithm described here: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    /// </summary>
    public sealed class MeanVarSngAggregator
    {
        private readonly bool _useLog;
        private readonly Double[] _mean;
        private readonly Double[] _m2;
        private readonly long[] _cnan;
        private readonly long[] _cnz;
        private long _trainCount;

        public MeanVarSngAggregator(int size, bool useLog)
        {
            _useLog = useLog;
            _mean = new Double[size];
            _m2 = new Double[size];
            if (!_useLog)
                _cnan = new long[size];
            _cnz = new long[size];
        }

        public long[] Counts
        {
            get { return _cnz; }
        }

        public Double[] Mean
        {
            get { return _mean; }
        }

        public Double[] StdDev
        {
            get { return _m2.Select((m2, i) => Math.Sqrt(m2 / _cnz[i])).ToArray(); }
        }

        public Double[] MeanSquareError
        {
            get { return _m2.Select((m2, i) => m2 / _cnz[i]).ToArray(); }
        }

        public Double[] M2
        {
            get { return _m2; }
        }

        public void ProcessValue(ref VBuffer<TFloat> value)
        {
            _trainCount++;
            var size = _mean.Length;
            var count = value.Count;
            Contracts.Assert(0 <= count & count <= size);
            if (count == 0)
                return;

            if (count == size)
            {
                var values = value.Values;
                for (int j = 0; j < count; j++)
                {
                    var origVal = values[j];
                    Update(j, origVal);
                }
            }
            else
            {
                var indices = value.Indices;
                var values = value.Values;
                for (int k = 0; k < count; k++)
                {
                    var origVal = values[k];
                    var j = indices[k];
                    Update(j, origVal);
                }
            }
        }

        public void Finish()
        {
            if (!_useLog)
            {
                for (int i = 0; i < _mean.Length; i++)
                {
                    Contracts.Assert(_trainCount >= _cnan[i] + _cnz[i]);
                    MeanVarUtils.AdjustForZeros(ref _mean[i], ref _m2[i], ref _cnz[i], _trainCount - _cnan[i] - _cnz[i]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Update(int j, TFloat origVal)
        {
            if (origVal == 0)
                return;
            var val = _useLog ? (TFloat)Math.Log(origVal) : origVal;
            if (!FloatUtils.IsFinite(val))
            {
                if (!_useLog)
                    _cnan[j]++;
                return;
            }

            _cnz[j]++;
            var delta = val - _mean[j];
            _mean[j] += delta / _cnz[j];
            var dm2 = delta * (val - _mean[j]);
            Contracts.Assert(dm2 >= 0);
            _m2[j] += dm2;
            Contracts.Assert(_m2[j] >= 0);
        }
    }

    public sealed partial class NormalizeTransform
    {
        internal abstract partial class AffineColumnFunction
        {
            public static IColumnFunction Create(IHost host, TFloat scale, TFloat offset)
            {
                return new Sng.ImplOne(host, scale, offset);
            }

            public static IColumnFunction Create(IHost host, TFloat[] scale, TFloat[] offset, int[] indicesNonZeroOffset)
            {
                return new Sng.ImplVec(host, scale, offset, indicesNonZeroOffset);
            }

            private static class Sng
            {
                // REVIEW: Should we have separate classes for offset==0 and/or scale==1?
                public sealed class ImplOne : ImplOne<TFloat>
                {
                    public ImplOne(IHost host, TFloat scale, TFloat offset)
                        : base(host, scale, offset)
                    {
                    }

                    public static new ImplOne Create(ModelLoadContext ctx, IHost host, ColumnType typeSrc)
                    {
                        host.Check(typeSrc.RawType == typeof(TFloat), "The column type must be R4.");
                        List<int> nz = null;
                        int cfeat;
                        TFloat[] scales;
                        TFloat[] offsets;
                        int[] indices;
                        TFloat[] scalesSparse;
                        TFloat[] offsetsSparse;

                        AffineNormSerializationUtils.LoadModel(ctx, ref nz, out cfeat, out scales, out offsets,
                            out indices, out scalesSparse, out offsetsSparse);
                        host.Assert(scales.Length == cfeat);
                        host.Assert(offsets == null || offsets.Length == cfeat);
                        host.Assert(Utils.Size(nz) == 0 || offsets != null);
                        if (cfeat != 1)
                            throw host.Except("Normalizer expected {0} slots, but the input data column has 1 slot.", cfeat);

                        return new ImplOne(host, scales[0], (offsets != null) ? offsets[0] : 0);
                    }

                    private void GetResult(ref TFloat input, ref TFloat value)
                    {
                        value = (input - Offset) * Scale;
                    }

                    public override void Save(ModelSaveContext ctx)
                    {
                        AffineNormSerializationUtils.SaveModel(ctx, 1, null, new[] { Scale }, new[] { Offset }, saveText: true);
                    }

                    public override JToken PfaInfo(BoundPfaContext ctx, JToken srcToken)
                        => PfaUtils.Call("*", PfaUtils.Call("-", srcToken, Offset), Scale);

                    public override bool OnnxInfo(OnnxContext ctx, OnnxNode node, int featureCount)
                    {
                        node.AddAttribute("offset", Enumerable.Repeat(Offset, featureCount));
                        node.AddAttribute("scale", Enumerable.Repeat(Scale, featureCount));
                        return true;
                    }

                    public override Delegate GetGetter(IRow input, int icol)
                    {
                        var getSrc = input.GetGetter<TFloat>(icol);
                        ValueGetter<TFloat> del =
                            (ref TFloat dst) =>
                            {
                                getSrc(ref dst);
                                GetResult(ref dst, ref dst);
                            };
                        return del;
                    }
                }

                // REVIEW: Does it make sense to have 3 separate classes for the 3 cases in GetResult?
                public sealed class ImplVec : ImplVec<TFloat>
                {
                    public ImplVec(IHost host, TFloat[] scale, TFloat[] offset, int[] indicesNonZeroOffset)
                        : base(host, scale, offset, indicesNonZeroOffset)
                    {
                    }

                    public static new ImplVec Create(ModelLoadContext ctx, IHost host, ColumnType typeSrc)
                    {
                        host.Check(typeSrc.ItemType.RawType == typeof(TFloat), "The column type must be vector of R4.");
                        int cv = Math.Max(1, typeSrc.VectorSize);
                        List<int> nz = null;
                        int cfeat;
                        TFloat[] scales;
                        TFloat[] offsets;
                        int[] indices;
                        TFloat[] scalesSparse;
                        TFloat[] offsetsSparse;

                        AffineNormSerializationUtils.LoadModel(ctx, ref nz, out cfeat, out scales, out offsets,
                            out indices, out scalesSparse, out offsetsSparse);
                        host.Assert(scales.Length == cfeat);
                        host.Assert(offsets == null || offsets.Length == cfeat);
                        host.Assert(Utils.Size(nz) == 0 || offsets != null);
                        if (cfeat != cv)
                            throw host.Except("Normalizer expected {0} slots, but the input data column has {1} slots.", cfeat, cv);

                        return new ImplVec(host, scales, offsets, (offsets != null && nz.Count < cv / 2) ? nz.ToArray() : null);
                    }

                    public override void Save(ModelSaveContext ctx)
                    {
                        AffineNormSerializationUtils.SaveModel(ctx, Scale.Length, null, Scale, Offset, saveText: true);
                    }

                    public override JToken PfaInfo(BoundPfaContext ctx, JToken srcToken)
                    {
                        var itemType = PfaUtils.Type.Double;
                        var arrType = PfaUtils.Type.Array(itemType);
                        var cellName = ctx.DeclareCell("AffNormScale", arrType, new JArray(Scale));
                        var scaleCell = PfaUtils.Cell(cellName);
                        if (Offset != null)
                        {
                            cellName = ctx.DeclareCell("AffNormOffset", arrType, new JArray(Offset));
                            var offsetCell = PfaUtils.Cell(cellName);
                            srcToken = PfaUtils.Call("a.zipmap", srcToken, offsetCell, PfaUtils.FuncRef(ctx.Pfa.EnsureSub(itemType)));
                        }
                        return PfaUtils.Call("a.zipmap", srcToken, scaleCell, PfaUtils.FuncRef(ctx.Pfa.EnsureMul(itemType)));
                    }

                    public override bool OnnxInfo(OnnxContext ctx, OnnxNode node, int featureCount)
                    {
                        if (Offset != null)
                            node.AddAttribute("offset", Offset);
                        else
                            node.AddAttribute("offset", Enumerable.Repeat<TFloat>(0, featureCount));

                        node.AddAttribute("scale", Scale);
                        return true;
                    }

                    public override Delegate GetGetter(IRow input, int icol)
                    {
                        var getSrc = input.GetGetter<VBuffer<TFloat>>(icol);
                        var bldr = new BufferBuilder<TFloat>(R4Adder.Instance);
                        ValueGetter<VBuffer<TFloat>> del;
                        if (Offset == null)
                        {
                            del = (ref VBuffer<TFloat> dst) =>
                            {
                                getSrc(ref dst);
                                Contracts.Check(dst.Length == Scale.Length);
                                FillValues(ref dst, bldr, Scale);
                                bldr.GetResult(ref dst);
                            };
                        }
                        else if (IndicesNonZeroOffset == null)
                        {
                            del = (ref VBuffer<TFloat> dst) =>
                            {
                                getSrc(ref dst);
                                Contracts.Check(dst.Length == Scale.Length);
                                FillValues(ref dst, bldr, Scale, Offset);
                                bldr.GetResult(ref dst);
                            };
                        }
                        else
                        {
                            del = (ref VBuffer<TFloat> dst) =>
                            {
                                getSrc(ref dst);
                                Contracts.Check(dst.Length == Scale.Length);
                                FillValues(ref dst, bldr, Scale, Offset, IndicesNonZeroOffset);
                                bldr.GetResult(ref dst);
                            };
                        }

                        return del;
                    }

                    // REVIEW: Change to normalize in place. when there are no offsets.
                    private static void FillValues(ref VBuffer<TFloat> input, BufferBuilder<TFloat> bldr, TFloat[] scale)
                    {
                        Contracts.Assert(input.Length == scale.Length);
                        int size = scale.Length;
                        int count = input.Count;
                        Contracts.Assert(0 <= count & count <= size);

                        // We always start with sparse, since we may make things sparser than the source.
                        bldr.Reset(size, dense: false);
                        if (count == 0)
                            return;

                        var values = input.Values;
                        if (count >= size)
                        {
                            for (int i = 0; i < size; i++)
                                bldr.AddFeature(i, values[i] * scale[i]);
                            return;
                        }

                        // The input is sparse.
                        var indices = input.Indices;
                        for (int ii = 0; ii < count; ii++)
                        {
                            int i = indices[ii];
                            Contracts.Assert(0 <= i & i < size);
                            bldr.AddFeature(i, values[ii] * scale[i]);
                        }
                    }

                    private static void FillValues(ref VBuffer<TFloat> input, BufferBuilder<TFloat> bldr, TFloat[] scale,
                        TFloat[] offset)
                    {
                        Contracts.Assert(input.Length == scale.Length);
                        int size = scale.Length;
                        int count = input.Count;
                        Contracts.Assert(0 <= count & count <= size);

                        // We always start with sparse, since we may make things sparser than the source.
                        bldr.Reset(size, dense: false);

                        if (count == 0)
                        {
                            for (int i = 0; i < size; i++)
                                bldr.AddFeature(i, -offset[i] * scale[i]);
                            return;
                        }

                        var values = input.Values;
                        if (count >= size)
                        {
                            for (int i = 0; i < size; i++)
                                bldr.AddFeature(i, (values[i] - offset[i]) * scale[i]);
                            return;
                        }

                        // The input is sparse.
                        var indices = input.Indices;
                        int ii = 0;
                        int ivSrc = indices[ii];
                        Contracts.Assert(ivSrc < size);
                        for (int ivDst = 0; ivDst < size; ivDst++)
                        {
                            Contracts.Assert(ivDst <= ivSrc & ivSrc <= size);
                            if (ivDst == ivSrc)
                            {
                                bldr.AddFeature(ivDst, (values[ii] - offset[ivDst]) * scale[ivDst]);
                                ivSrc = ++ii < count ? indices[ii] : size;
                                Contracts.Assert(ii == count || ivSrc < size);
                            }
                            else
                                bldr.AddFeature(ivDst, -offset[ivDst] * scale[ivDst]);
                        }
                    }

                    private static void FillValues(ref VBuffer<TFloat> input, BufferBuilder<TFloat> bldr, TFloat[] scale,
                        TFloat[] offset, int[] nz)
                    {
                        Contracts.Assert(input.Length == scale.Length);

                        int size = scale.Length;
                        int count = input.Count;
                        Contracts.Assert(0 <= count & count <= size);

                        // We always start with sparse, since we may make things sparser than the source.
                        bldr.Reset(size, dense: false);

                        if (count == 0)
                        {
                            foreach (int i in nz)
                                bldr.AddFeature(i, -offset[i] * scale[i]);
                            return;
                        }

                        var values = input.Values;
                        if (count >= size)
                        {
                            for (int i = 0; i < size; i++)
                                bldr.AddFeature(i, (values[i] - offset[i]) * scale[i]);
                            return;
                        }

                        // The input is sparse.
                        var indices = input.Indices;
                        int ii = 0;
                        int ivSrc = indices[ii];
                        int inz = 0;
                        int ivDst = nz[inz];
                        for (; ; )
                        {
                            Contracts.Assert(0 <= ivDst & ivDst <= size);
                            Contracts.Assert(0 <= ivSrc & ivSrc <= size);
                            Contracts.Assert(ii < count && ivSrc == indices[ii] || ii == count && ivSrc == size);
                            Contracts.Assert(inz < nz.Length && ivDst == nz[inz] || inz == nz.Length && ivDst == size);

                            int diff = ivSrc - ivDst;
                            if (diff > 0)
                            {
                                // Offset but no value
                                bldr.AddFeature(ivDst, -offset[ivDst] * scale[ivDst]);
                                ivDst = ++inz < nz.Length ? nz[inz] : size;
                            }
                            else if (diff < 0)
                            {
                                // Value but no offset
                                bldr.AddFeature(ivSrc, values[ii] * scale[ivSrc]);
                                ivSrc = ++ii < count ? indices[ii] : size;
                                Contracts.Assert((ii == count) == (ivSrc >= size));
                            }
                            else
                            {
                                Contracts.Assert(ivSrc == ivDst);
                                if (ivDst >= size)
                                    break;

                                bldr.AddFeature(ivDst, (values[ii] - offset[ivDst]) * scale[ivDst]);
                                ivSrc = ++ii < count ? indices[ii] : size;
                                Contracts.Assert((ii == count) == (ivSrc >= size));
                                ivDst = ++inz < nz.Length ? nz[inz] : size;
                                Contracts.Assert((inz == nz.Length) == (ivDst >= size));
                            }
                        }
                        Contracts.Assert(ii == count);
                        Contracts.Assert(inz == nz.Length);
                    }
                }
            }
        }

        internal abstract partial class CdfColumnFunction
        {
            public static IColumnFunction Create(IHost host, TFloat mean, TFloat stddev, bool useLog)
            {
                return new Sng.ImplOne(host, mean, stddev, useLog);
            }

            public static IColumnFunction Create(IHost host, TFloat[] mean, TFloat[] stddev, bool useLog)
            {
                return new Sng.ImplVec(host, mean, stddev, useLog);
            }

            private static class Sng
            {
                public sealed class ImplOne : ImplOne<TFloat>
                {
                    public ImplOne(IHost host, TFloat mean, TFloat stddev, bool useLog)
                        : base(host, mean, stddev, useLog)
                    {
                    }

                    public static new ImplOne Create(ModelLoadContext ctx, IHost host, ColumnType typeSrc)
                    {
                        host.Check(typeSrc.RawType == typeof(TFloat), "The column type must be R4.");
                        host.CheckValue(ctx, nameof(ctx));
                        ctx.CheckAtModel(GetVersionInfo());

                        bool useLog;
                        TFloat[] mean;
                        TFloat[] stddev;
                        CdfNormSerializationUtils.LoadModel(ctx, 1, out useLog, out mean, out stddev);

                        return new ImplOne(host, mean[0], stddev[0], useLog);
                    }

                    private void GetResult(ref TFloat input, ref TFloat value)
                    {
                        var val = UseLog ? (TFloat)Math.Log(input) : input;
                        if (!FloatUtils.IsFinite(val))
                        {
                            value = 0;
                            return;
                        }

                        value = CdfUtils.Cdf(val, Mean, Stddev);
                    }

                    public override void Save(ModelSaveContext ctx)
                    {
                        Contracts.AssertValue(ctx);
                        ctx.CheckAtModel();
                        ctx.SetVersionInfo(GetVersionInfo());

                        CdfNormSerializationUtils.SaveModel(ctx, UseLog, new[] { Mean }, new[] { Stddev });
                    }

                    public override Delegate GetGetter(IRow input, int icol)
                    {
                        if (Stddev <= TFloat.Epsilon)
                        {
                            ValueGetter<TFloat> trivial =
                                (ref TFloat dst) =>
                                {
                                    dst = 0;
                                };
                            return trivial;
                        }

                        var getSrc = input.GetGetter<TFloat>(icol);
                        ValueGetter<TFloat> del =
                            (ref TFloat dst) =>
                            {
                                getSrc(ref dst);
                                GetResult(ref dst, ref dst);
                            };
                        return del;
                    }
                }

                public sealed class ImplVec : ImplVec<TFloat>
                {
                    public ImplVec(IHost host, TFloat[] mean, TFloat[] stddev, bool useLog)
                        : base(host, mean, stddev, useLog)
                    {
                    }

                    public static new ImplVec Create(ModelLoadContext ctx, IHost host, ColumnType typeSrc)
                    {
                        host.Check(typeSrc.ItemType.RawType == typeof(TFloat), "The column type must be vector of R4.");
                        int cv = Math.Max(1, typeSrc.VectorSize);

                        host.CheckValue(ctx, nameof(ctx));
                        ctx.CheckAtModel(GetVersionInfo());

                        bool useLog;
                        TFloat[] mean;
                        TFloat[] stddev;
                        CdfNormSerializationUtils.LoadModel(ctx, cv, out useLog, out mean, out stddev);

                        return new ImplVec(host, mean, stddev, useLog);
                    }

                    public override void Save(ModelSaveContext ctx)
                    {
                        Contracts.AssertValue(ctx);
                        ctx.CheckAtModel();
                        ctx.SetVersionInfo(GetVersionInfo());

                        CdfNormSerializationUtils.SaveModel(ctx, UseLog, Mean, Stddev);
                    }

                    public override Delegate GetGetter(IRow input, int icol)
                    {
                        var getSrc = input.GetGetter<VBuffer<TFloat>>(icol);
                        var bldr = new BufferBuilder<TFloat>(R4Adder.Instance);
                        ValueGetter<VBuffer<TFloat>> del;
                        del = (ref VBuffer<TFloat> dst) =>
                        {
                            getSrc(ref dst);
                            Host.Check(dst.Length == Mean.Length);
                            FillValues(ref dst, bldr, Mean, Stddev, UseLog);
                            bldr.GetResult(ref dst);
                        };

                        return del;
                    }

                    private static void FillValues(ref VBuffer<TFloat> input, BufferBuilder<TFloat> bldr, TFloat[] mean,
                        TFloat[] stddev, bool useLog)
                    {
                        Contracts.Assert(input.Length == mean.Length);
                        int size = mean.Length;
                        int count = input.Count;
                        Contracts.Assert(0 <= count & count <= size);

                        // We always start with sparse, since we may make things sparser than the source.
                        bldr.Reset(size, dense: false);

                        if (count == 0)
                            return;

                        var values = input.Values;
                        if (count >= size)
                        {
                            for (int i = 0; i < size; i++)
                            {
                                var sigma = stddev[i];
                                if (sigma > TFloat.Epsilon)
                                {
                                    var val = useLog ? (TFloat)Math.Log(values[i]) : values[i];
                                    if (FloatUtils.IsFinite(val))
                                        bldr.AddFeature(i, CdfUtils.Cdf(val, mean[i], sigma));
                                }
                            }
                            return;
                        }

                        // The input is sparse.
                        var indices = input.Indices;
                        for (int ii = 0; ii < indices.Length; ii++)
                        {
                            var ivDst = indices[ii];
                            var sigma = stddev[ivDst];
                            if (sigma > TFloat.Epsilon)
                            {
                                var val = useLog ? (TFloat)Math.Log(values[ii]) : values[ii];
                                if (FloatUtils.IsFinite(val))
                                    bldr.AddFeature(ivDst, CdfUtils.Cdf(val, mean[ivDst], sigma));
                            }
                        }
                    }
                }
            }
        }

        internal abstract partial class BinColumnFunction
        {
            public static IColumnFunction Create(IHost host, TFloat[] binUpperBounds, bool fixZero)
            {
                return new Sng.ImplOne(host, binUpperBounds, fixZero);
            }

            public static IColumnFunction Create(IHost host, TFloat[][] binUpperBounds, bool fixZero)
            {
                return new Sng.ImplVec(host, binUpperBounds, fixZero);
            }

            private static class Sng
            {
                public sealed class ImplOne : BinColumnFunction, NormalizerTransformer.IBinData<TFloat>
                {
                    private readonly TFloat[] _binUpperBounds;
                    private readonly TFloat _den;
                    private readonly TFloat _offset;

                    ImmutableArray<TFloat> NormalizerTransformer.IBinData<TFloat>.UpperBounds => ImmutableArray.Create(_binUpperBounds);

                    public ImplOne(IHost host, TFloat[] binUpperBounds, bool fixZero)
                        : base(host)
                    {
                        _binUpperBounds = binUpperBounds;
                        _den = Math.Max(1, _binUpperBounds.Length - 1);
                        if (fixZero)
                            _offset = _binUpperBounds.FindIndexSorted(0) / _den;
                        Host.Assert(0 <= _offset & _offset <= 1);
                    }

                    public static new ImplOne Create(ModelLoadContext ctx, IHost host, ColumnType typeSrc)
                    {
                        host.Check(typeSrc.RawType == typeof(TFloat), "The column type must be R4.");
                        host.CheckValue(ctx, nameof(ctx));
                        ctx.CheckAtModel(GetVersionInfo());

                        // *** Binary format ***
                        // Byte: fixZero bool
                        bool fixZero = ctx.Reader.ReadBoolByte();

                        TFloat[][] binUpperBounds = null;
                        if (!ctx.TryProcessSubModel("BinNormalizer",
                            c => BinNormSerializationUtils.LoadModel(c, out binUpperBounds)))
                        {
                            throw host.ExceptDecode();
                        }
                        if (binUpperBounds.Length != 1)
                            throw host.Except("Normalizer expected {0} slots, but the input data column has 1 slot.", binUpperBounds.Length);

                        return new ImplOne(host, binUpperBounds[0], fixZero);
                    }

                    public override void Save(ModelSaveContext ctx)
                    {
                        Contracts.AssertValue(ctx);
                        ctx.CheckAtModel();
                        ctx.SetVersionInfo(GetVersionInfo());

                        // *** Binary format ***
                        // Byte: fixZero bool
                        ctx.Writer.WriteBoolByte(_offset != 0);

                        ctx.SaveSubModel("BinNormalizer",
                            c => BinNormSerializationUtils.SaveModel(c, new[] { _binUpperBounds }, saveText: true));
                    }

                    public override Delegate GetGetter(IRow input, int icol)
                    {
                        var getSrc = input.GetGetter<TFloat>(icol);
                        ValueGetter<TFloat> del =
                            (ref TFloat dst) =>
                            {
                                getSrc(ref dst);
                                GetResult(ref dst, ref dst);
                            };
                        return del;
                    }

                    private void GetResult(ref TFloat input, ref TFloat value)
                    {
                        value = BinUtils.GetValue(ref input, _binUpperBounds, _den, _offset);
                    }
                }

                public sealed class ImplVec : BinColumnFunction, NormalizerTransformer.IBinData<ImmutableArray<TFloat>>
                {
                    private readonly TFloat[][] _binUpperBounds;
                    private readonly TFloat[] _den;
                    private readonly TFloat[] _offset;

                    ImmutableArray<ImmutableArray<TFloat>> NormalizerTransformer.IBinData<ImmutableArray<TFloat>>.UpperBounds
                        => _binUpperBounds.Select(b => ImmutableArray.Create(b)).ToImmutableArray();

                    public ImplVec(IHost host, TFloat[][] binUpperBounds, bool fixZero)
                        : base(host)
                    {
                        _binUpperBounds = binUpperBounds;
                        _den = new TFloat[_binUpperBounds.Length];
                        for (int i = 0; i < _binUpperBounds.Length; i++)
                            _den[i] = Math.Max(1, _binUpperBounds[i].Length - 1);
                        if (fixZero)
                        {
                            _offset = new TFloat[_binUpperBounds.Length];
                            bool any = false;
                            for (int i = 0; i < _binUpperBounds.Length; i++)
                            {
                                _offset[i] = _binUpperBounds[i].FindIndexSorted(0) / _den[i];
                                Host.Assert(0 <= _offset[i] & _offset[i] <= 1);
                                any |= _offset[i] != 0;
                            }
                            if (!any)
                                _offset = null;
                        }
                    }

                    public static new ImplVec Create(ModelLoadContext ctx, IHost host, ColumnType typeSrc)
                    {
                        host.Check(typeSrc.ItemType.RawType == typeof(TFloat), "The column type must be vector of R4.");
                        int cv = Math.Max(1, typeSrc.VectorSize);
                        host.CheckValue(ctx, nameof(ctx));
                        ctx.CheckAtModel(GetVersionInfo());

                        // *** Binary format ***
                        // Byte: fixZero bool
                        bool fixZero = ctx.Reader.ReadBoolByte();

                        TFloat[][] binUpperBounds = null;
                        if (!ctx.TryProcessSubModel("BinNormalizer",
                            c => BinNormSerializationUtils.LoadModel(c, out binUpperBounds)))
                        {
                            throw host.ExceptDecode();
                        }
                        if (binUpperBounds.Length != cv)
                            throw host.Except("Normalizer expected {0} slots, but the input data column has {1} slots.", binUpperBounds.Length, cv);

                        return new ImplVec(host, binUpperBounds, fixZero);
                    }

                    public override void Save(ModelSaveContext ctx)
                    {
                        Contracts.AssertValue(ctx);
                        ctx.CheckAtModel();
                        ctx.SetVersionInfo(GetVersionInfo());

                        // *** Binary format ***
                        // Byte: fixZero bool
                        ctx.Writer.WriteBoolByte(_offset != null);

                        ctx.SaveSubModel("BinNormalizer", c => BinNormSerializationUtils.SaveModel(c, _binUpperBounds, saveText: true));
                    }

                    public override Delegate GetGetter(IRow input, int icol)
                    {
                        var getSrc = input.GetGetter<VBuffer<TFloat>>(icol);
                        var bldr = new BufferBuilder<TFloat>(R4Adder.Instance);
                        ValueGetter<VBuffer<TFloat>> del =
                            (ref VBuffer<TFloat> dst) =>
                            {
                                getSrc(ref dst);
                                Host.Check(dst.Length == _binUpperBounds.Length);
                                GetResult(ref dst, ref dst, bldr);
                            };
                        return del;
                    }

                    private void GetResult(ref VBuffer<TFloat> input, ref VBuffer<TFloat> value, BufferBuilder<TFloat> bldr)
                    {
                        Contracts.Assert(input.Length == _binUpperBounds.Length);
                        int size = _binUpperBounds.Length;
                        int count = input.Count;
                        Contracts.Assert(0 <= count & count <= size);

                        // We always start with sparse, since we may make things sparser than the source.
                        bldr.Reset(size, dense: false);
                        if (count == 0)
                        {
                            bldr.GetResult(ref value);
                            return;
                        }

                        var values = input.Values;
                        if (count >= size)
                        {
                            if (_offset != null)
                            {
                                for (int i = 0; i < size; i++)
                                    bldr.AddFeature(i, BinUtils.GetValue(ref values[i], _binUpperBounds[i], _den[i], _offset[i]));
                            }
                            else
                            {
                                for (int i = 0; i < size; i++)
                                    bldr.AddFeature(i, BinUtils.GetValue(ref values[i], _binUpperBounds[i], _den[i]));
                            }
                            bldr.GetResult(ref value);
                            return;
                        }

                        // The input is sparse.
                        if (_offset != null)
                        {
                            var indices = input.Indices;
                            int ii = 0;
                            int ivSrc = indices[ii];
                            Contracts.Assert(ivSrc < size);
                            TFloat zero = 0;
                            for (int ivDst = 0; ivDst < size; ivDst++)
                            {
                                Contracts.Assert(ivDst <= ivSrc & ivSrc <= size);
                                if (ivDst == ivSrc)
                                {
                                    bldr.AddFeature(ivDst,
                                        BinUtils.GetValue(ref values[ii], _binUpperBounds[ivDst], _den[ivDst], _offset[ivDst]));
                                    ivSrc = ++ii < count ? indices[ii] : size;
                                    Contracts.Assert(ii == count || ivSrc < size);
                                }
                                else
                                    bldr.AddFeature(ivDst,
                                        BinUtils.GetValue(ref zero, _binUpperBounds[ivDst], _den[ivDst], _offset[ivDst]));
                            }
                        }
                        else
                        {
                            var indices = input.Indices;
                            for (int ii = 0; ii < count; ii++)
                            {
                                int i = indices[ii];
                                Contracts.Assert(0 <= i & i < size);
                                bldr.AddFeature(i, BinUtils.GetValue(ref values[ii], _binUpperBounds[i], _den[i]));
                            }
                        }

                        bldr.GetResult(ref value);
                    }
                }
            }
        }

        internal static partial class MinMaxUtils
        {
            public static void ComputeScaleAndOffset(bool fixZero, TFloat max, TFloat min, out TFloat scale, out TFloat offset)
            {
                if (fixZero)
                    ComputeScaleAndOffsetFixZero(max, min, out scale, out offset);
                else
                    ComputeScaleAndOffset(max, min, out scale, out offset);
            }

            private static void ComputeScaleAndOffset(TFloat max, TFloat min, out TFloat scale, out TFloat offset)
            {
                Contracts.Assert(!TFloat.IsNaN(min));
                Contracts.Assert(!TFloat.IsNaN(max));

                // If the column has only NaNs, or has no rows at all, then min==infinity and max==-infinity. In all
                // other cases, min<=max.
                Contracts.Assert(min <= max || (TFloat.IsPositiveInfinity(min) && TFloat.IsNegativeInfinity(max)));

                // In the case where max <= min, the slot contains no useful information (since it is either constant, or
                // is all NaNs, or has no rows), so we force it to zero.
                // Note that setting scale to zero effectively maps finite values to zero,
                // but infinities and NaN to NaN.
                // REVIEW: If min <= 0 and max >= 0, then why not fix zero for this slot and simply scale by 1 / max(abs(..))?
                // We could even be more aggressive about it, and fix zero if 0 < min < max <= 2 * min.
                // Then the common case where features are in the range [1, N] (and integer valued) wouldn't subtract 1 every time....
                if (!(max > min))
                    scale = offset = 0;
                else if ((scale = 1 / (max - min)) == 0)
                    offset = 0;
                else
                    offset = min;
                Contracts.Assert(0 <= scale & scale < TFloat.PositiveInfinity);
            }

            private static void ComputeScaleAndOffsetFixZero(TFloat max, TFloat min, out TFloat scale, out TFloat offset)
            {
                Contracts.Assert(!TFloat.IsNaN(min));
                Contracts.Assert(!TFloat.IsNaN(max));

                // If the column has only NaNs, or has no rows at all, then min==infinity and max==-infinity. In all
                // other cases, min<=max.
                Contracts.Assert(min <= max || (TFloat.IsPositiveInfinity(min) && TFloat.IsNegativeInfinity(max)));

                // In the case where max <= min, the slot contains no useful information (since it is either constant, or
                // is all NaNs, or has no rows), so we force it to zero.
                // Note that setting scale to zero effectively maps finite values to zero,
                // but infinities and NaN to NaN.
                offset = 0;
                if (!(max > min))
                    scale = 0;
                else
                    scale = 1 / Math.Max(Math.Abs(max), Math.Abs(min));
                Contracts.Assert(0 <= scale & scale < TFloat.PositiveInfinity);
            }
        }

        internal static partial class MeanVarUtils
        {
            public static void ComputeScaleAndOffset(Double mean, Double stddev, out TFloat scale, out TFloat offset)
            {
                Contracts.Assert(!Double.IsNaN(mean));
                Contracts.Assert(stddev >= 0);

                // In the case where stdev==0, the slot contains no useful information (since it is constant),
                // so we force it to zero. Note that setting scale to zero effectively maps finite values to zero,
                // but infinities and NaN to NaN.
                if (stddev == 0)
                    scale = offset = 0;
                else if ((scale = 1 / (TFloat)stddev) == 0)
                    offset = 0;
                else
                    offset = (TFloat)mean;
                Contracts.Assert(0 <= scale & scale < TFloat.PositiveInfinity);
            }

            public static void ComputeScaleAndOffsetFixZero(Double mean, Double meanSquaredError, out TFloat scale, out TFloat offset)
            {
                Contracts.Assert(!Double.IsNaN(mean));
                Contracts.Assert(meanSquaredError >= 0);

                // In the case where stdev==0, the slot contains no useful information (since it is constant),
                // so we force it to zero. Note that setting scale to zero effectively maps finite values to zero,
                // but infinities and NaN to NaN.
                offset = 0;
                if (meanSquaredError == 0)
                    scale = 0;
                else
                    scale = 1 / (TFloat)Math.Sqrt(meanSquaredError + mean * mean);
                Contracts.Assert(0 <= scale & scale < TFloat.PositiveInfinity);
            }
        }

        private static partial class CdfUtils
        {
            public static TFloat Cdf(TFloat input, TFloat mean, TFloat stddev)
            {
                // REVIEW: This should be changed to call the AML stats library.
                // Temporarily, it does the following:
                // Using CDF(x) = 0.5 ( 1 + erf( ( x - mu ) / ( sigma * sqrt(2) ) ) )
                // Also using an approximation for erf(x) from https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
                var x = (input - mean) / stddev;
                var x2 = x * x / 2;
                const TFloat a = (TFloat)0.147;
                var ax2 = a * x2;
                return (TFloat)(0.5 + 0.5 * Math.Sign(x) * Math.Sqrt(1 - Math.Exp(-x2 * (4 / Math.PI + ax2) / (1 + ax2))));
            }
        }

        internal static partial class BinUtils
        {
            public static TFloat GetValue(ref TFloat input, TFloat[] binUpperBounds, TFloat den, TFloat offset)
            {
                if (TFloat.IsNaN(input))
                    return input;
                int binIdx = binUpperBounds.FindIndexSorted(0, binUpperBounds.Length - 1, input);
                Contracts.Check(binIdx < binUpperBounds.Length);
                var value = binIdx / den - offset;
                Contracts.Assert(-1 <= value & value <= 1);
                return value;
            }

            public static TFloat GetValue(ref TFloat input, TFloat[] binUpperBounds, TFloat den)
            {
                if (TFloat.IsNaN(input))
                    return input;
                int binIdx = binUpperBounds.FindIndexSorted(0, binUpperBounds.Length - 1, input);
                Contracts.Check(binIdx < binUpperBounds.Length);
                var value = binIdx / den;
                Contracts.Assert(0 <= value & value <= 1);
                return value;
            }
        }

        private static class Sng
        {
            public abstract class MinMaxOneColumnFunctionBuilderBase : OneColumnFunctionBuilderBase<TFloat>
            {
                protected readonly bool Fix;
                protected readonly MinMaxSngAggregator Aggregator;
                private VBuffer<TFloat> _buffer;

                protected MinMaxOneColumnFunctionBuilderBase(IHost host, long lim, bool fix, ValueGetter<TFloat> getSrc)
                    : base(host, lim, getSrc)
                {
                    Fix = fix;
                    Aggregator = new MinMaxSngAggregator(1);
                    _buffer = new VBuffer<TFloat>(1, new TFloat[1]);
                }

                protected override bool ProcessValue(ref TFloat val)
                {
                    if (!base.ProcessValue(ref val))
                        return false;
                    _buffer.Values[0] = val;
                    Aggregator.ProcessValue(ref _buffer);
                    return true;
                }
            }

            public sealed class MinMaxOneColumnFunctionBuilder : MinMaxOneColumnFunctionBuilderBase
            {
                private MinMaxOneColumnFunctionBuilder(IHost host, long lim, bool fix, ValueGetter<TFloat> getSrc)
                    : base(host, lim, fix, getSrc)
                {
                }

                public static IColumnFunctionBuilder Create(Normalizer.MinMaxColumn column, IHost host, ColumnType srcType,
                    ValueGetter<TFloat> getter)
                {
                    host.CheckUserArg(column.MaxTrainingExamples > 1, nameof(column.MaxTrainingExamples), "Must be greater than 1");
                    return new MinMaxOneColumnFunctionBuilder(host, column.MaxTrainingExamples, column.FixZero, getter);
                }

                public override IColumnFunction CreateColumnFunction()
                {
                    Aggregator.Finish();
                    TFloat scale;
                    TFloat offset;
                    MinMaxUtils.ComputeScaleAndOffset(Fix, Aggregator.Max[0], Aggregator.Min[0], out scale, out offset);

                    return AffineColumnFunction.Create(Host, scale, offset);
                }
            }

            public abstract class MinMaxVecColumnFunctionBuilderBase : VecColumnFunctionBuilderBase<TFloat>
            {
                protected readonly MinMaxSngAggregator Aggregator;
                protected readonly bool Fix;

                protected MinMaxVecColumnFunctionBuilderBase(IHost host, int cv, long lim, bool fix, ValueGetter<VBuffer<TFloat>> getSrc)
                    : base(host, lim, getSrc)
                {
                    Fix = fix;
                    Aggregator = new MinMaxSngAggregator(cv);
                }

                protected override bool ProcessValue(ref VBuffer<TFloat> buffer)
                {
                    if (!base.ProcessValue(ref buffer))
                        return false;
                    var size = Aggregator.Min.Length;
                    if (buffer.Length != size)
                        throw Host.Except("Normalizer expected {0} slots but got {1}", size, buffer.Length);
                    Aggregator.ProcessValue(ref buffer);
                    return true;
                }
            }

            public sealed class MinMaxVecColumnFunctionBuilder : MinMaxVecColumnFunctionBuilderBase
            {
                private MinMaxVecColumnFunctionBuilder(IHost host, int cv, long lim, bool fix,
                    ValueGetter<VBuffer<TFloat>> getSrc)
                    : base(host, cv, lim, fix, getSrc)
                {
                }

                public static IColumnFunctionBuilder Create(Normalizer.MinMaxColumn column, IHost host, ColumnType srcType,
                    ValueGetter<VBuffer<TFloat>> getter)
                {
                    host.CheckUserArg(column.MaxTrainingExamples > 1, nameof(column.MaxTrainingExamples), "Must be greater than 1");
                    var cv = srcType.ValueCount;
                    return new MinMaxVecColumnFunctionBuilder(host, cv, column.MaxTrainingExamples, column.FixZero, getter);
                }

                public override IColumnFunction CreateColumnFunction()
                {
                    Aggregator.Finish();
                    var cv = Aggregator.Min.Length;
                    // These are ignored if fix is true.
                    int lim = cv / 2;
                    var nz = new List<int>();

                    for (int i = 0; i < cv; i++)
                    {
                        MinMaxUtils.ComputeScaleAndOffset(Fix, Aggregator.Max[i], Aggregator.Min[i], out Aggregator.Max[i], out Aggregator.Min[i]);
                        if (Aggregator.Min[i] != 0 && nz.Count < lim)
                            nz.Add(i);
                    }

                    var min = Aggregator.Min;
                    // Note: There is a special case when cv == 1. In this case lim == 0, so nz will be empty regardless
                    // of whether the offset is non-zero.
                    Host.Assert((lim == 0) == (cv == 1));
                    int[] indicesNonZeroOffset = null;
                    if (Fix)
                        min = null;
                    else if (cv == 1)
                    {
                        if (min[0] == 0)
                            min = null;
                    }
                    else if (nz.Count == 0)
                        min = null;
                    else if (nz.Count < lim)
                        indicesNonZeroOffset = nz.ToArray();

                    return AffineColumnFunction.Create(Host, Aggregator.Max, min, indicesNonZeroOffset);
                }
            }

            public sealed class MeanVarOneColumnFunctionBuilder : OneColumnFunctionBuilderBase<TFloat>
            {
                private readonly bool _useLog;
                private readonly bool _useCdf;
                private readonly bool _fix;
                private readonly MeanVarSngAggregator _aggregator;
                private VBuffer<TFloat> _buffer;

                private MeanVarOneColumnFunctionBuilder(IHost host, long lim, bool fix, ValueGetter<TFloat> getSrc, bool useLog, bool useCdf)
                    : base(host, lim, getSrc)
                {
                    _useLog = useLog;
                    _useCdf = useCdf;
                    _fix = fix;
                    _aggregator = new MeanVarSngAggregator(1, useLog);
                    _buffer = new VBuffer<TFloat>(1, new TFloat[1]);
                }

                public static IColumnFunctionBuilder Create(Normalizer.MeanVarColumn column, IHost host, ColumnType srcType,
                    ValueGetter<TFloat> getter)
                {
                    host.CheckUserArg(column.MaxTrainingExamples > 1, nameof(column.MaxTrainingExamples), "Must be greater than 1");
                    return new MeanVarOneColumnFunctionBuilder(host, column.MaxTrainingExamples, column.FixZero, getter, false, column.UseCdf);
                }

                public static IColumnFunctionBuilder Create(Normalizer.LogMeanVarColumn column, IHost host, ColumnType srcType,
                    ValueGetter<TFloat> getter)
                {
                    var lim = column.MaxTrainingExamples;
                    host.CheckUserArg(lim > 1, nameof(column.MaxTrainingExamples), "Must be greater than 1");
                    return new MeanVarOneColumnFunctionBuilder(host, lim, false, getter, true, column.UseCdf);
                }

                protected override bool ProcessValue(ref TFloat origVal)
                {
                    if (!base.ProcessValue(ref origVal))
                        return false;
                    _buffer.Values[0] = origVal;
                    _aggregator.ProcessValue(ref _buffer);
                    return true;
                }

                public override IColumnFunction CreateColumnFunction()
                {
                    _aggregator.Finish();
                    if (_useCdf)
                        return CreateCdfColumnFunction();
                    return CreateAffineColumnFunction();
                }

                private IColumnFunction CreateAffineColumnFunction()
                {
                    Contracts.Assert(_aggregator.M2[0] >= 0);
                    if (_aggregator.M2[0] == 0)
                        return AffineColumnFunction.Create(Host, (TFloat)0, (TFloat)0);
                    TFloat scale;
                    TFloat offset;
                    if (_fix)
                        MeanVarUtils.ComputeScaleAndOffsetFixZero(_aggregator.Mean[0], _aggregator.MeanSquareError[0], out scale, out offset);
                    else
                        MeanVarUtils.ComputeScaleAndOffset(_aggregator.Mean[0], _aggregator.StdDev[0], out scale, out offset);

                    return AffineColumnFunction.Create(Host, scale, offset);
                }

                private IColumnFunction CreateCdfColumnFunction()
                {
                    Contracts.Assert(_aggregator.M2[0] >= 0);
                    if (_aggregator.M2[0] == 0 || _aggregator.Counts[0] == 0)
                        return CdfColumnFunction.Create(Host, (TFloat)0, (TFloat)0, _useLog);

                    return CdfColumnFunction.Create(Host, (TFloat)_aggregator.Mean[0], (TFloat)_aggregator.StdDev[0], _useLog);
                }
            }

            public sealed class MeanVarVecColumnFunctionBuilder : VecColumnFunctionBuilderBase<TFloat>
            {
                private readonly bool _fix;
                private readonly bool _useLog;
                private readonly bool _useCdf;
                private readonly MeanVarSngAggregator _aggregator;

                private MeanVarVecColumnFunctionBuilder(IHost host, int cv, long lim, bool fix,
                    ValueGetter<VBuffer<TFloat>> getSrc, bool useLog, bool useCdf)
                    : base(host, lim, getSrc)
                {
                    _aggregator = new MeanVarSngAggregator(cv, useLog);
                    _fix = fix;
                    _useLog = useLog;
                    _useCdf = useCdf;
                }

                public static IColumnFunctionBuilder Create(Normalizer.MeanVarColumn column, IHost host, ColumnType srcType,
                    ValueGetter<VBuffer<TFloat>> getter)
                {
                    host.CheckUserArg(column.MaxTrainingExamples > 1, nameof(column.MaxTrainingExamples), "Must be greater than 1");
                    var cv = srcType.ValueCount;
                    return new MeanVarVecColumnFunctionBuilder(host, cv, column.MaxTrainingExamples, column.FixZero, getter, false, column.UseCdf);
                }

                public static IColumnFunctionBuilder Create(Normalizer.LogMeanVarColumn column, IHost host, ColumnType srcType,
                    ValueGetter<VBuffer<TFloat>> getter)
                {
                    var lim = column.MaxTrainingExamples;
                    host.CheckUserArg(lim > 1, nameof(column.MaxTrainingExamples), "Must be greater than 1");
                    var cv = srcType.ValueCount;
                    return new MeanVarVecColumnFunctionBuilder(host, cv, lim, false, getter, true, column.UseCdf);
                }

                protected override bool ProcessValue(ref VBuffer<TFloat> buffer)
                {
                    if (!base.ProcessValue(ref buffer))
                        return false;

                    _aggregator.ProcessValue(ref buffer);
                    return true;
                }

                public override IColumnFunction CreateColumnFunction()
                {
                    _aggregator.Finish();
                    if (_useCdf)
                        return CreateCdfColumnFunction();
                    return CreateAffineColumnFunction();
                }

                private IColumnFunction CreateAffineColumnFunction()
                {
                    int cv = _aggregator.Mean.Length;
                    // These are ignored if fix is true.
                    int lim = cv / 2;
                    var nz = new List<int>();

                    var scale = new TFloat[cv];
                    var offset = new TFloat[cv];

                    for (int i = 0; i < cv; i++)
                    {
                        Contracts.Assert(_aggregator.M2[i] >= 0);
                        if (_aggregator.M2[i] == 0)
                        {
                            scale[i] = offset[i] = 0;
                            continue;
                        }
                        if (_fix)
                            MeanVarUtils.ComputeScaleAndOffsetFixZero(_aggregator.Mean[i], _aggregator.MeanSquareError[i], out scale[i], out offset[i]);
                        else
                            MeanVarUtils.ComputeScaleAndOffset(_aggregator.Mean[i], _aggregator.StdDev[i], out scale[i], out offset[i]);
                        if (offset[i] != 0 && nz.Count < lim)
                            nz.Add(i);
                    }

                    // Note: There is a special case when cv == 1. In this case lim == 0, so nz will be empty regardless
                    // of whether the offset is non-zero.
                    Host.Assert((lim == 0) == (cv == 1));
                    int[] indicesNonZeroOffset = null;
                    if (_fix)
                        offset = null;
                    else if (cv == 1)
                    {
                        if (offset[0] == 0)
                            offset = null;
                    }
                    else if (nz.Count == 0)
                        offset = null;
                    else if (nz.Count < lim)
                        indicesNonZeroOffset = nz.ToArray();

                    return AffineColumnFunction.Create(Host, scale, offset, indicesNonZeroOffset);
                }

                private IColumnFunction CreateCdfColumnFunction()
                {
                    int cv = _aggregator.Mean.Length;

                    var mean = new TFloat[cv];
                    var stddev = new TFloat[cv];

                    for (int i = 0; i < cv; i++)
                    {
                        Contracts.Assert(_aggregator.M2[i] >= 0);
                        if (_aggregator.M2[i] == 0 || _aggregator.Counts[i] == 0)
                        {
                            mean[i] = stddev[i] = 0;
                            continue;
                        }
                        mean[i] = (TFloat)_aggregator.Mean[i];
                        stddev[i] = (TFloat)_aggregator.StdDev[i];
                    }

                    return CdfColumnFunction.Create(Host, mean, stddev, _useLog);
                }
            }

            public sealed class BinOneColumnFunctionBuilder : OneColumnFunctionBuilderBase<TFloat>
            {
                private readonly bool _fix;
                private readonly int _numBins;
                private List<TFloat> _values;

                private BinOneColumnFunctionBuilder(IHost host, long lim, bool fix, int numBins, ValueGetter<TFloat> getSrc)
                    : base(host, lim, getSrc)
                {
                    _fix = fix;
                    _numBins = numBins;
                    _values = new List<TFloat>();
                }

                public static IColumnFunctionBuilder Create(Normalizer.BinningColumn column, IHost host, ColumnType srcType,
                    ValueGetter<TFloat> getter)
                {
                    var lim = column.MaxTrainingExamples;
                    host.CheckUserArg(lim > 1, nameof(column.MaxTrainingExamples), "Must be greater than 1");
                    bool fix = column.FixZero;
                    var numBins = column.NumBins;
                    host.CheckUserArg(numBins > 1, nameof(column.NumBins), "Must be greater than 1");
                    return new BinOneColumnFunctionBuilder(host, lim, fix, numBins, getter);
                }

                protected override bool ProcessValue(ref TFloat val)
                {
                    if (!base.ProcessValue(ref val))
                        return false;
                    if (val != 0)
                        _values.Add(val);
                    return true;
                }

                public override IColumnFunction CreateColumnFunction()
                {
                    var binFinder = new GreedyBinFinder();
                    var numZeroes = checked((int)(Lim - Rem - _values.Count));
                    _values.RemoveAll(TFloat.IsNaN);
                    var binUpperBounds = binFinder.FindBins(_numBins, _values, numZeroes);
                    return BinColumnFunction.Create(Host, binUpperBounds, _fix);
                }
            }

            public sealed class BinVecColumnFunctionBuilder : VecColumnFunctionBuilderBase<TFloat>
            {
                private readonly bool _fix;
                private readonly int _numBins;
                private List<TFloat>[] _values;

                private BinVecColumnFunctionBuilder(IHost host, int cv, long lim, bool fix, int numBins,
                    ValueGetter<VBuffer<TFloat>> getSrc)
                    : base(host, lim, getSrc)
                {
                    _fix = fix;
                    _numBins = numBins;
                    _values = new List<TFloat>[cv];
                    for (int i = 0; i < cv; i++)
                    {
                        _values[i] = new List<TFloat>();
                    }
                }

                public static IColumnFunctionBuilder Create(Normalizer.BinningColumn column, IHost host, ColumnType srcType,
                    ValueGetter<VBuffer<TFloat>> getter)
                {
                    var lim = column.MaxTrainingExamples;
                    host.CheckUserArg(lim > 1, nameof(column.MaxTrainingExamples), "Must be greater than 1");
                    bool fix = column.FixZero;
                    var numBins = column.NumBins;
                    host.CheckUserArg(numBins > 1, nameof(column.NumBins), "Must be greater than 1");
                    var cv = srcType.ValueCount;
                    return new BinVecColumnFunctionBuilder(host, cv, lim, fix, numBins, getter);
                }

                protected override bool ProcessValue(ref VBuffer<TFloat> buffer)
                {
                    if (!base.ProcessValue(ref buffer))
                        return false;

                    int size = _values.Length;
                    Host.Check(buffer.Length == size);

                    int count = buffer.Count;
                    Host.Assert(0 <= count & count <= size);
                    if (count == 0)
                        return true;

                    if (count == size)
                    {
                        var values = buffer.Values;
                        for (int j = 0; j < count; j++)
                            _values[j].Add(values[j]);
                    }
                    else
                    {
                        var indices = buffer.Indices;
                        var values = buffer.Values;
                        for (int k = 0; k < count; k++)
                        {
                            var val = values[k];
                            var j = indices[k];
                            _values[j].Add(val);
                        }
                    }
                    return true;
                }

                public override IColumnFunction CreateColumnFunction()
                {
                    var binFinder = new GreedyBinFinder();
                    var count = _values.Length;
                    var binUpperBounds = new TFloat[count][];
                    for (int i = 0; i < count; i++)
                    {
                        var numZeroes = checked((int)(Lim - Rem - _values[i].Count));
                        _values[i].RemoveAll(TFloat.IsNaN);
                        binUpperBounds[i] = binFinder.FindBins(_numBins, _values[i], numZeroes);
                    }
                    return BinColumnFunction.Create(Host, binUpperBounds, _fix);
                }
            }

            public sealed class SupervisedBinOneColumnFunctionBuilder : OneColumnSupervisedBinFunctionBuilderBase<TFloat>
            {
                private readonly bool _fix;
                private readonly int _numBins;
                private readonly int _minBinSize;

                private SupervisedBinOneColumnFunctionBuilder(IHost host, long lim, bool fix, int numBins, int minBinSize, int valueColumnId, int labelColumnId, IRow dataRow)
                    : base(host, lim, valueColumnId, labelColumnId, dataRow)
                {
                    _fix = fix;
                    _numBins = numBins;
                    _minBinSize = minBinSize;
                }

                protected override bool AcceptColumnValue(ref TFloat colValue)
                {
                    return !TFloat.IsNaN(colValue);
                }

                public override IColumnFunction CreateColumnFunction()
                {
                    var binFinder = new SupervisedBinFinder();
                    var binUpperBounds = binFinder.FindBins(_numBins, _minBinSize, LabelCardinality, ColValues, Labels);
                    return BinColumnFunction.Create(Host, binUpperBounds, _fix);
                }

                public static IColumnFunctionBuilder Create(SupervisedBinArguments args, IHost host, int argsColumnIndex, int valueColumnId, int labelColumnId, IRow dataRow)
                {
                    var lim = args.Column[argsColumnIndex].MaxTrainingExamples ?? args.MaxTrainingExamples;
                    host.CheckUserArg(lim > 1, nameof(args.MaxTrainingExamples), "Must be greater than 1");
                    bool fix = args.Column[argsColumnIndex].FixZero ?? args.FixZero;
                    var numBins = args.Column[argsColumnIndex].NumBins ?? args.NumBins;
                    host.CheckUserArg(numBins > 1, nameof(args.NumBins), "Must be greater than 1");
                    host.CheckUserArg(args.MinBinSize > 0, nameof(args.MinBinSize), "Must be positive");
                    return new SupervisedBinOneColumnFunctionBuilder(host, lim, fix, numBins, args.MinBinSize, valueColumnId, labelColumnId, dataRow);
                }
            }

            public sealed class SupervisedBinVecColumnFunctionBuilder : VecColumnSupervisedBinFunctionBuilderBase<TFloat>
            {
                private readonly bool _fix;
                private readonly int _numBins;
                private readonly int _minBinSize;

                private SupervisedBinVecColumnFunctionBuilder(IHost host, long lim, bool fix, int numBins, int minBinSize, int valueColumnId, int labelColumnId, IRow dataRow)
                    : base(host, lim, valueColumnId, labelColumnId, dataRow)
                {
                    _fix = fix;
                    _numBins = numBins;
                    _minBinSize = minBinSize;
                }

                protected override bool AcceptColumnValue(ref VBuffer<TFloat> colValuesBuffer)
                {
                    return !colValuesBuffer.Values.Any(TFloat.IsNaN);
                }

                public override IColumnFunction CreateColumnFunction()
                {
                    var binFinder = new SupervisedBinFinder();
                    TFloat[][] binUpperBounds = new TFloat[ColumnSlotCount][];
                    for (int i = 0; i < ColumnSlotCount; i++)
                        binUpperBounds[i] = binFinder.FindBins(_numBins, _minBinSize, LabelCardinality, ColValues[i], Labels);
                    return BinColumnFunction.Create(Host, binUpperBounds, _fix);
                }

                public static IColumnFunctionBuilder Create(SupervisedBinArguments args, IHost host, int argsColumnIndex, int valueColumnId, int labelColumnId, IRow dataRow)
                {
                    var lim = args.Column[argsColumnIndex].MaxTrainingExamples ?? args.MaxTrainingExamples;
                    host.CheckUserArg(lim > 1, nameof(args.MaxTrainingExamples), "Must be greater than 1");
                    bool fix = args.Column[argsColumnIndex].FixZero ?? args.FixZero;
                    var numBins = args.Column[argsColumnIndex].NumBins ?? args.NumBins;
                    host.CheckUserArg(numBins > 1, nameof(args.NumBins), "Must be greater than 1");
                    host.CheckUserArg(args.MinBinSize > 0, nameof(args.MinBinSize), "Must be positive");
                    return new SupervisedBinVecColumnFunctionBuilder(host, lim, fix, numBins, args.MinBinSize, valueColumnId, labelColumnId, dataRow);
                }
            }
        }
    }
}
